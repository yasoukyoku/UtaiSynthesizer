//! SF2 解析器(阶段2)。SF2 2.04 子集。
//!
//! RIFF 'sfbk' 三段:INFO(名称)/ sdta(16-bit PCM)/ pdta
//! (phdr→pbag→pgen / inst→ibag→igen / shdr)。
//! preset zone(pgen)与 instrument zone(igen)合并成 Region;
//! preset 层覆盖 instrument 层,keyRange/velRange 取交集。
//! modulator(pmod/imod)按白名单烘焙进 Region:velocity → initialFilterFc、
//! CC1 → vibLfoToPitch/modLfoToPitch(3-4);曲线近似线性。
//! 立体声对(left/right + sampleLink)合并成双声道 SampleData。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::synth::{
    LoadedInstrument, LoopMode, Region, SampleData, sf2_fc_cents_to_hz, sf2_q_cb_to_linear,
};

/// 生成器操作码(仅本引擎支持的子集)。
/// 未引用的常量是 SF2 2.04 操作码表的文档性成员,保留以便对照规范。
#[allow(dead_code)]
mod gen {
    pub const START_ADDRS_OFFSET: u16 = 0;
    pub const END_ADDRS_OFFSET: u16 = 1;
    pub const STARTLOOP_ADDRS_OFFSET: u16 = 2;
    pub const ENDLOOP_ADDRS_OFFSET: u16 = 3;
    pub const START_ADDRS_COARSE_OFFSET: u16 = 4;
    pub const END_ADDRS_COARSE_OFFSET: u16 = 12;
    pub const PAN: u16 = 17;
    pub const DELAY_VOLENV: u16 = 33;
    pub const ATTACK_VOLENV: u16 = 34;
    pub const HOLD_VOLENV: u16 = 35;
    pub const DECAY_VOLENV: u16 = 36;
    pub const SUSTAIN_VOLENV: u16 = 37;
    pub const RELEASE_VOLENV: u16 = 38;
    pub const INSTRUMENT: u16 = 41;
    pub const KEY_RANGE: u16 = 43;
    pub const VEL_RANGE: u16 = 44;
    pub const STARTLOOP_ADDRS_COARSE_OFFSET: u16 = 45;
    pub const INITIAL_ATTENUATION: u16 = 48;
    pub const COARSE_TUNE: u16 = 51;
    pub const FINE_TUNE: u16 = 52;
    pub const SAMPLE_ID: u16 = 53;
    pub const SAMPLE_MODES: u16 = 54;
    pub const SCALE_TUNING: u16 = 56;
    pub const OVERRIDING_ROOT_KEY: u16 = 58;
    pub const INITIAL_FILTER_FC: u16 = 590;
    pub const INITIAL_FILTER_Q: u16 = 591;
}

/// 采样类型(shdr sampleType 低 4 位)。
#[allow(dead_code)]
mod sample_type {
    pub const MONO: u16 = 1;
    pub const RIGHT: u16 = 2;
    pub const LEFT: u16 = 4;
    pub const LINKED: u16 = 8;
}

#[derive(Debug, Clone)]
pub struct Sf2PresetInfo {
    pub bank: u16,
    pub program: u16,
    pub name: String,
}

/// 解析好的 SF2:presets + 每个 preset 可直接构造乐器。
pub struct Sf2Font {
    pub name: String,
    pub presets: Vec<Sf2Preset>,
}

pub struct Sf2Preset {
    pub info: Sf2PresetInfo,
    pub regions: Vec<Region>,
    pub samples: Vec<Arc<SampleData>>,
}

impl Sf2Preset {
    pub fn to_instrument(&self) -> LoadedInstrument {
        LoadedInstrument {
            regions: self.regions.clone(),
            samples: self.samples.clone(),
        }
    }
}

struct Gen {
    op: u16,
    amount: i16,
}

/// SF2 modulator 记录(pmod/imod,10 字节/条)。
/// 3-4 只按 (src, dst) 白名单在 region 构建时烘焙,不做运行时调制引擎。
struct Mod {
    src: u16,
    dst: u16,
    amount: i16,
    amt_src: u16,
    transport: u16,
}

struct Zone {
    gens: Vec<Gen>,
    mods: Vec<Mod>,
}

struct Shdr {
    name: String,
    start: u32,
    end: u32,
    start_loop: u32,
    end_loop: u32,
    rate: u32,
    original_pitch: u8,
    pitch_fraction: u8,
    link: u16,
    ty: u16,
}

struct Bytes<'a> {
    d: &'a [u8],
}

impl<'a> Bytes<'a> {
    fn u16(&self, off: usize) -> u16 {
        let b = self.d.get(off..off + 2).unwrap_or(&[0, 0]);
        u16::from_le_bytes([b[0], b[1]])
    }
    fn u32(&self, off: usize) -> u32 {
        let b = self.d.get(off..off + 4).unwrap_or(&[0, 0, 0, 0]);
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn i16(&self, off: usize) -> i16 {
        self.u16(off) as i16
    }
    fn str20(&self, off: usize) -> String {
        let raw = self.d.get(off..off + 20).unwrap_or(&[0; 20]);
        let end = raw.iter().position(|&c| c == 0).unwrap_or(20);
        String::from_utf8_lossy(&raw[..end]).trim_end().to_string()
    }
}

/// 解析 SF2 文件。
pub fn parse_sf2(path: &Path) -> Result<Sf2Font, String> {
    let data = std::fs::read(path).map_err(|e| format!("读取 SF2 失败 '{}': {}", path.display(), e))?;
    let b = Bytes { d: &data };
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"sfbk" {
        return Err("不是有效的 SF2(RIFF/sfbk)文件".into());
    }

    // 遍历顶层 LIST chunk
    let mut info_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("SF2").to_string();
    let mut sdta: &[u8] = &[];
    let mut pdta: &[u8] = &[];
    // pdta 在文件中的绝对起点:pdta 子 chunk 的偏移基准(见下 chunks 注释)。
    let mut pdta_off = 0usize;
    let mut pos = 12usize;
    while pos + 8 <= data.len() {
        let ck = &data[pos..pos + 4];
        let size = b.u32(pos + 4) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + size).min(data.len());
        if ck == b"LIST" && body_start + 4 <= body_end {
            let form = &data[body_start..body_start + 4];
            let sub = &data[body_start + 4..body_end];
            match form {
                b"INFO" => {
                    let ib = Bytes { d: sub };
                    let mut p = 0usize;
                    while p + 8 <= sub.len() {
                        let cid = &sub[p..p + 4];
                        let csize = ib.u32(p + 4) as usize;
                        let body = sub.get(p + 8..(p + 8 + csize).min(sub.len())).unwrap_or(&[]);
                        if cid == b"INAM" {
                            let end = body.iter().position(|&c| c == 0).unwrap_or(body.len());
                            info_name = String::from_utf8_lossy(&body[..end]).to_string();
                        }
                        p += 8 + csize + (csize & 1); // chunk 2 字节对齐
                    }
                }
                b"sdta" => {
                    // smpl 子 chunk(前 4 字节是 "smpl")
                    if sub.len() >= 8 && &sub[0..4] == b"smpl" {
                        sdta = &sub[8..];
                    }
                }
                b"pdta" => {
                    pdta = sub;
                    pdta_off = body_start + 4;
                }
                _ => {}
            }
        }
        pos = body_start + size + (size & 1);
    }
    if sdta.is_empty() {
        return Err("SF2 缺少采样数据(sdta/smpl)".into());
    }
    if pdta.is_empty() {
        return Err("SF2 缺少结构数据(pdta)".into());
    }

    // pdta 子 chunk 定位。
    // 偏移统一存**文件绝对偏移**(pdta_off + 相对偏移):后续 phdr/pbag/… 的读取
    // 全部走文件级 Bytes `b`,基准必须一致——曾因存相对偏移导致全表读错、preset 全空。
    let mut chunks: HashMap<&[u8], (usize, usize)> = HashMap::new();
    {
        let pb = Bytes { d: pdta };
        let mut p = 0usize;
        while p + 8 <= pdta.len() {
            let cid: &[u8] = &pdta[p..p + 4];
            let csize = pb.u32(p + 4) as usize;
            let end = (p + 8 + csize).min(pdta.len());
            chunks.insert(cid, (pdta_off + p + 8, pdta_off + end));
            p = p + 8 + csize + (csize & 1);
        }
    }
    let get = |id: &[u8]| -> (usize, usize) { *chunks.get(id).unwrap_or(&(0, 0)) };

    // phdr:38 字节/记录(最后一条 EOP 终端)
    let (ph_s, ph_e) = get(b"phdr");
    let mut presets_meta: Vec<(String, u16, u16, usize)> = Vec::new();
    let mut p = ph_s;
    while p + 38 <= ph_e {
        let name = b.str20(p);
        let program = b.u16(p + 20);
        let bank = b.u16(p + 22);
        let bag_idx = b.u16(p + 24) as usize;
        if name.starts_with("EOP") && p + 38 == ph_e {
            break;
        }
        presets_meta.push((name, bank, program, bag_idx));
        p += 38;
    }

    // pmod/imod:10 字节/记录(src, dst, amount, amtSrc, transport)
    let (pm_s, pm_e) = get(b"pmod");
    let (im_s, im_e) = get(b"imod");
    let read_mods = |s: usize, e: usize| -> Vec<Mod> {
        let mut out = Vec::new();
        let mut p = s;
        while p + 10 <= e {
            out.push(Mod {
                src: b.u16(p),
                dst: b.u16(p + 2),
                amount: b.i16(p + 4),
                amt_src: b.u16(p + 6),
                transport: b.u16(p + 8),
            });
            p += 10;
        }
        out
    };

    // pbag:4 字节/记录(genIdx, modIdx)
    let (pb_s, pb_e) = get(b"pbag");
    let pbag_pair = |i: usize| -> (usize, usize) {
        let o = pb_s + i * 4;
        (b.u16(o) as usize, b.u16(o + 2) as usize)
    };

    // pgen:4 字节/记录
    let (pg_s, pg_e) = get(b"pgen");
    let read_gens = |s: usize, e: usize| -> Vec<Gen> {
        let mut out = Vec::new();
        let mut p = s;
        while p + 4 <= e {
            out.push(Gen {
                op: b.u16(p),
                amount: b.i16(p + 2),
            });
            p += 4;
        }
        out
    };

    // inst:22 字节/记录
    let (in_s, in_e) = get(b"inst");
    let mut instruments: Vec<(String, usize)> = Vec::new();
    let mut p = in_s;
    while p + 22 <= in_e {
        let name = b.str20(p);
        let bag_idx = b.u16(p + 20) as usize;
        if name.starts_with("EOI") && p + 22 == in_e {
            break;
        }
        instruments.push((name, bag_idx));
        p += 22;
    }

    // ibag / igen
    let (ib_s, ib_e) = get(b"ibag");
    let ibag_pair = |i: usize| -> (usize, usize) {
        let o = ib_s + i * 4;
        (b.u16(o) as usize, b.u16(o + 2) as usize)
    };
    let (ig_s, ig_e) = get(b"igen");

    // shdr:46 字节/记录(最后一条 EOS 终端)
    let (sh_s, sh_e) = get(b"shdr");
    let mut shdrs: Vec<Shdr> = Vec::new();
    let mut p = sh_s;
    while p + 46 <= sh_e {
        let s = Shdr {
            name: b.str20(p),
            start: b.u32(p + 20),
            end: b.u32(p + 24),
            start_loop: b.u32(p + 28),
            end_loop: b.u32(p + 32),
            rate: b.u32(p + 36),
            original_pitch: data.get(p + 40).copied().unwrap_or(60),
            pitch_fraction: data.get(p + 41).copied().unwrap_or(0),
            link: b.u16(p + 42),
            ty: b.u16(p + 44),
        };
        if s.name.starts_with("EOS") && p + 46 == sh_e {
            break;
        }
        shdrs.push(s);
        p += 46;
    }

    // 预解析 instrument zones(乐器 → zones)
    let ibag_count = (ib_e - ib_s) / 4;
    let mut inst_zones: Vec<Vec<Zone>> = Vec::with_capacity(instruments.len());
    for (i, (_name, bag_start)) in instruments.iter().enumerate() {
        let bag_end = instruments
            .get(i + 1)
            .map(|(_, nb)| *nb)
            .unwrap_or_else(|| ibag_count.saturating_sub(1).max(*bag_start));
        let mut zones = Vec::new();
        for bi in *bag_start..bag_end.min(ibag_count) {
            let (g_start, m_start) = ibag_pair(bi);
            let (g_end, m_end) = if bi + 1 < ibag_count {
                ibag_pair(bi + 1)
            } else {
                ((ig_e - ig_s) / 4, (im_e - im_s) / 10)
            };
            zones.push(Zone {
                gens: read_gens(ig_s + g_start * 4, ig_s + g_end * 4),
                mods: read_mods(im_s + m_start * 10, im_s + m_end * 10),
            });
        }
        inst_zones.push(zones);
    }

    // 采样缓存(shdr index → SampleData;立体声对合并)
    let mut sample_cache: HashMap<usize, Arc<SampleData>> = HashMap::new();
    let sdta_frames = (sdta.len() / 2) as u32;
    let build_sample = |idx: usize| -> Option<Arc<SampleData>> {
        let sh = shdrs.get(idx)?;
        let ty = sh.ty & 0x7FFF & 0x000F;
        // 循环点(相对采样起点;end_loop <= start_loop 视为无循环)
        let lp = if sh.end_loop > sh.start_loop {
            Some((sh.start_loop.saturating_sub(sh.start), sh.end_loop.saturating_sub(sh.start)))
        } else {
            None
        };
        // 立体声:left(4) 为主体,link 指向 right(2)
        if ty == sample_type::LEFT {
            let right_idx = sh.link as usize;
            let rsh = shdrs.get(right_idx)?;
            let left = slice_i16(sdta, sh.start, sh.end, sdta_frames);
            let right = slice_i16(sdta, rsh.start, rsh.end, sdta_frames);
            Some(Arc::new(SampleData {
                left,
                right: Some(right),
                rate: if sh.rate > 0 { sh.rate } else { 44100 },
                loop_range: lp,
                root_key: Some(sh.original_pitch.min(127)),
                fine_tune_cents: sh.pitch_fraction as f32 / 256.0,
            }))
        } else if ty == sample_type::RIGHT {
            // right 样本:若其 link 是 left(配对主体),跳过(由配对处理);
            // 否则当独立 mono 用
            let lsh = shdrs.get(sh.link as usize)?;
            if (lsh.ty & 0x7FFF & 0x000F) == sample_type::LEFT {
                return None;
            }
            None
        } else {
            let left = slice_i16(sdta, sh.start, sh.end, sdta_frames);
            Some(Arc::new(SampleData {
                left,
                right: None,
                rate: if sh.rate > 0 { sh.rate } else { 44100 },
                loop_range: lp,
                root_key: Some(sh.original_pitch.min(127)),
                fine_tune_cents: sh.pitch_fraction as f32 / 256.0,
            }))
        }
    };

    let pseudo_path = |idx: usize| -> PathBuf {
        PathBuf::from(format!("sf2://{}#{}", path.display(), idx))
    };

    // preset 解析
    let pbag_count = (pb_e - pb_s) / 4;
    let mut presets: Vec<Sf2Preset> = Vec::new();
    for (i, (name, bank, program, bag_start)) in presets_meta.iter().enumerate() {
        let bag_end = presets_meta
            .get(i + 1)
            .map(|(_, _, _, nb)| *nb)
            .unwrap_or(pbag_count.saturating_sub(1));
        // preset zones(gens + mods)
        let mut pzones: Vec<(Vec<Gen>, Vec<Mod>)> = Vec::new();
        for bi in *bag_start..bag_end.min(pbag_count) {
            let (g_start, m_start) = pbag_pair(bi);
            let (g_end, m_end) = if bi + 1 < pbag_count {
                pbag_pair(bi + 1)
            } else {
                ((pg_e - pg_s) / 4, (pm_e - pm_s) / 10)
            };
            pzones.push((
                read_gens(pg_s + g_start * 4, pg_s + g_end * 4),
                read_mods(pm_s + m_start * 10, pm_s + m_end * 10),
            ));
        }
        if pzones.is_empty() {
            continue;
        }
        // global zone = 无 instrument gen 的 zone(取第一个)
        let pglobal: &[Gen] = pzones
            .iter()
            .find(|(g, _)| !g.iter().any(|g| g.op == gen::INSTRUMENT))
            .map(|(g, _)| g.as_slice())
            .unwrap_or(&[]);

        let mut regions: Vec<Region> = Vec::new();
        let mut samples: Vec<Arc<SampleData>> = Vec::new();
        for (pz, pmods) in &pzones {
            let Some(inst_idx) = pz.iter().find(|g| g.op == gen::INSTRUMENT).map(|g| g.amount as usize) else {
                continue;
            };
            let Some(izones) = inst_zones.get(inst_idx) else { continue };
            let iglobal: &[Gen] = izones
                .iter()
                .find(|z| !z.gens.iter().any(|g| g.op == gen::SAMPLE_ID))
                .map(|z| z.gens.as_slice())
                .unwrap_or(&[]);

            for iz in izones {
                let Some(sample_idx) = iz.gens.iter().find(|g| g.op == gen::SAMPLE_ID).map(|g| g.amount as usize) else {
                    continue;
                };
                if sample_idx >= shdrs.len() {
                    continue;
                }
                // 采样(缓存;right 跳过时 zone 也跳过)
                if !sample_cache.contains_key(&sample_idx) {
                    let s = build_sample(sample_idx);
                    sample_cache.insert(sample_idx, s.unwrap_or_else(|| Arc::new(SampleData {
                        left: Vec::new(),
                        right: None,
                        rate: 44100,
                        loop_range: None,
                        root_key: None,
                        fine_tune_cents: 0.0,
                    })));
                }
                let sample = sample_cache.get(&sample_idx).unwrap();
                if sample.left.is_empty() {
                    continue; // right 配对样本或空采样
                }
                let sh = &shdrs[sample_idx];

                // generator 查找:preset local > preset global > inst local > inst global
                let lookup = |op: u16| -> Option<i16> {
                    pz.iter().find(|g| g.op == op)
                        .map(|g| g.amount)
                        .or_else(|| pglobal.iter().find(|g| g.op == op).map(|g| g.amount))
                        .or_else(|| iz.gens.iter().find(|g| g.op == op).map(|g| g.amount))
                        .or_else(|| iglobal.iter().find(|g| g.op == op).map(|g| g.amount))
                };
                // range 类 generator(单层):lo/hi 两字节
                let zone_range = |gens: &[Gen], op: u16| -> Option<(u8, u8)> {
                    let amt = gens.iter().find(|g| g.op == op).map(|g| g.amount)?;
                    let lo = (amt & 0xFF) as u8;
                    let hi = ((amt >> 8) & 0xFF) as u8;
                    Some((lo.min(hi), lo.max(hi)))
                };
                let prange = |op: u16| -> Option<(u8, u8)> {
                    zone_range(pz, op).or_else(|| zone_range(pglobal, op))
                };
                let irange = |op: u16| -> Option<(u8, u8)> {
                    zone_range(&iz.gens, op).or_else(|| zone_range(iglobal, op))
                };

                // key/vel 区间:preset 层 ∩ instrument 层。
                // 两层均未定义 = 全域;显式给出但交集空 = 丢弃 zone。
                let (key_lo, key_hi) = match intersect_range(prange(gen::KEY_RANGE), irange(gen::KEY_RANGE)) {
                    Some((lo, hi)) => (lo, hi),
                    None => {
                        if prange(gen::KEY_RANGE).is_none() && irange(gen::KEY_RANGE).is_none() {
                            (0, 127)
                        } else {
                            continue;
                        }
                    }
                };
                let (vel_lo, vel_hi) = match intersect_range(prange(gen::VEL_RANGE), irange(gen::VEL_RANGE)) {
                    Some((lo, hi)) => (lo, hi),
                    None => {
                        if prange(gen::VEL_RANGE).is_none() && irange(gen::VEL_RANGE).is_none() {
                            (0, 127)
                        } else {
                            continue;
                        }
                    }
                };

                // 地址(含 coarse×32768 偏移)
                let off = |op: u16, coarse_op: u16| -> u32 {
                    lookup(op).unwrap_or(0) as u32 + (lookup(coarse_op).unwrap_or(0) as i64 * 32768).max(0) as u32
                };
                let start = sh.start + off(gen::START_ADDRS_OFFSET, gen::START_ADDRS_COARSE_OFFSET);
                let end = (sh.end as i64
                    + lookup(gen::END_ADDRS_OFFSET).unwrap_or(0) as i64
                    + lookup(gen::END_ADDRS_COARSE_OFFSET).unwrap_or(0) as i64 * 32768)
                    .max(start as i64 + 1) as u32;
                let loop_s = sh.start_loop as i64
                    + lookup(gen::STARTLOOP_ADDRS_OFFSET).unwrap_or(0) as i64
                    + lookup(gen::STARTLOOP_ADDRS_COARSE_OFFSET).unwrap_or(0) as i64 * 32768;
                let loop_e = sh.end_loop as i64 + lookup(gen::ENDLOOP_ADDRS_OFFSET).unwrap_or(0) as i64;

                // 包络(timecents → 秒;-32768 视为 0)
                let tc = |op: u16, default: i16| -> f32 {
                    let v = lookup(op).unwrap_or(default);
                    if v <= -32768 {
                        0.0
                    } else {
                        2f32.powf(v as f32 / 1200.0)
                    }
                };
                let sustain_cb = lookup(gen::SUSTAIN_VOLENV).unwrap_or(0).max(0);
                let sustain = 10f32.powf(-sustain_cb as f32 / 200.0).clamp(0.0, 1.0);
                let attenuation = lookup(gen::INITIAL_ATTENUATION).unwrap_or(0).max(0) as f32
                    + iz.gens.iter().find(|g| g.op == gen::INITIAL_ATTENUATION).map(|g| g.amount.max(0) as f32).unwrap_or(0.0)
                    + pz.iter().find(|g| g.op == gen::INITIAL_ATTENUATION).map(|g| g.amount.max(0) as f32).unwrap_or(0.0);

                let pan = lookup(gen::PAN).unwrap_or(0).clamp(-500, 500) as f32 / 500.0;
                let scale = lookup(gen::SCALE_TUNING).unwrap_or(100);
                let root = match lookup(gen::OVERRIDING_ROOT_KEY) {
                    Some(r) if (0..=127).contains(&r) => r as u8,
                    _ => sh.original_pitch.min(127),
                };

                let mut region = Region {
                    sample_path: pseudo_path(sample_idx),
                    key_lo,
                    key_hi,
                    vel_lo,
                    vel_hi,
                    keycenter: Some(root),
                    track: scale != 0,
                    transpose: lookup(gen::COARSE_TUNE).unwrap_or(0) as i32,
                    tune: lookup(gen::FINE_TUNE).unwrap_or(0) as f32,
                    volume_db: 0.0,
                    pan,
                    veltrack: 100.0,
                    attack: tc(gen::ATTACK_VOLENV, -12000),
                    hold: tc(gen::HOLD_VOLENV, -12000),
                    decay: tc(gen::DECAY_VOLENV, -12000),
                    sustain,
                    release: tc(gen::RELEASE_VOLENV, -12000),
                    loop_mode: match lookup(gen::SAMPLE_MODES).unwrap_or(0) {
                        1 => LoopMode::Continuous,
                        3 => LoopMode::Sustain,
                        _ => LoopMode::NoLoop,
                    },
                    // 循环/端点相对采样起点(SampleData 已从 start 切片)
                    loop_start: if loop_e > loop_s {
                        Some((loop_s - sh.start as i64).max(0) as u32)
                    } else {
                        None
                    },
                    loop_end: if loop_e > loop_s {
                        Some((loop_e - sh.start as i64).max(0) as u32)
                    } else {
                        None
                    },
                    end: Some((end as i64 - sh.start as i64).max(1) as u32),
                    offset: 0,
                    ..Default::default()
                };
                // SF2 modulator 白名单应用(3-4):vel→initialFilterFc(累加 cents)、
                // CC1→vibLfoToPitch/modLfoToPitch(后写覆盖);amt_src 非零的忽略。
                // zero 终端记录(src=0)不匹配任何白名单,自然无害。
                let mut vel2filter_cents = 0.0f32;
                let mut vib_depth = 50.0f32;
                for m in pmods.iter().chain(iz.mods.iter()) {
                    if m.amt_src != 0 {
                        continue;
                    }
                    let is_vel = (m.src & 0x0080) == 0 && (m.src & 0x007F) == 2;
                    let is_cc1 = (m.src & 0x0080) != 0 && (m.src & 0x007F) == 1;
                    if is_vel && m.dst == gen::INITIAL_FILTER_FC {
                        vel2filter_cents += m.amount as f32;
                    } else if is_cc1 && (m.dst == 0 || m.dst == 1) {
                        vib_depth = m.amount as f32;
                    }
                }
                region.vel2filter_cents = vel2filter_cents;
                region.vib_depth_cents = vib_depth;
                region.attenuation_cb = attenuation;
                // 低通滤波器(gen 590/591):absolute cents → Hz,Q cB → 线性
                if let Some(c) = lookup(gen::INITIAL_FILTER_FC) {
                    region.cutoff_hz = Some(sf2_fc_cents_to_hz(c.max(0) as f32));
                }
                region.resonance_q =
                    sf2_q_cb_to_linear(lookup(gen::INITIAL_FILTER_Q).unwrap_or(0).max(0) as f32);
                // 采样自带的循环(切好片后的 loop_range)与 zone 循环并存:
                // zone 显式 loop 优先,否则用采样自带(在 synth 里合成)
                if region.loop_start.is_none() {
                    if let Some((ls, le)) = sample.loop_range {
                        region.loop_start = Some(ls);
                        region.loop_end = Some(le);
                    }
                }
                // 循环越界保护:循环点必须在采样内
                let frames = sample.frame_count();
                if let (Some(ls), Some(le)) = (region.loop_start, region.loop_end) {
                    if le <= ls || le > frames {
                        region.loop_start = None;
                        region.loop_end = None;
                    }
                }
                if let Some(e) = region.end {
                    if e > frames {
                        region.end = Some(frames);
                    }
                }
                regions.push(region);
                samples.push(sample.clone());
            }
        }
        if regions.is_empty() {
            continue;
        }
        presets.push(Sf2Preset {
            info: Sf2PresetInfo {
                bank: *bank,
                program: *program,
                name: name.clone(),
            },
            regions,
            samples,
        });
    }

    if presets.is_empty() {
        return Err("SF2 无可用的 preset(zone 合并后为空)".into());
    }
    Ok(Sf2Font {
        name: info_name,
        presets,
    })
}

/// ranges 区间相交(None = 全域;空交集返回 None)。
fn intersect_range(a: Option<(u8, u8)>, b: Option<(u8, u8)>) -> Option<(u8, u8)> {
    match (a, b) {
        (Some((al, ah)), Some((bl, bh))) => {
            let lo = al.max(bl);
            let hi = ah.min(bh);
            if lo <= hi {
                Some((lo, hi))
            } else {
                None
            }
        }
        (x, None) | (None, x) => x,
    }
}

/// 从 smpl(16-bit LE)切一段 f32。
fn slice_i16(sdta: &[u8], start: u32, end: u32, total_frames: u32) -> Vec<f32> {
    let s = (start as usize).min(total_frames as usize);
    let e = (end as usize).min(total_frames as usize).max(s);
    let mut out = Vec::with_capacity(e - s);
    for i in s..e {
        let off = i * 2;
        if off + 1 >= sdta.len() {
            break;
        }
        let v = ((sdta[off] as u16) | ((sdta[off + 1] as u16) << 8)) as i16;
        out.push(v as f32 / 32768.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 子 chunk 写入:id + size + body(2 字节对齐)。
    fn ck(out: &mut Vec<u8>, id: &[u8], body: &[u8]) {
        out.extend_from_slice(id);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        if body.len() & 1 == 1 {
            out.push(0);
        }
    }

    fn list(form: &[u8], body: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"LIST");
        v.extend_from_slice(&((body.len() + 4) as u32).to_le_bytes());
        v.extend_from_slice(form);
        v.extend_from_slice(body);
        v
    }

    fn rec_name(name: &str) -> Vec<u8> {
        let mut v = [0u8; 20];
        v[..name.len()].copy_from_slice(name.as_bytes());
        v.to_vec()
    }

    fn gens(list: &[(u16, i16)]) -> Vec<u8> {
        let mut v = Vec::new();
        for (op, a) in list {
            v.extend_from_slice(&op.to_le_bytes());
            v.extend_from_slice(&a.to_le_bytes());
        }
        v
    }

    /// modulator 记录:10 字节/条(src, dst, amount, amtSrc, transport)。
    fn mods(list: &[(u16, u16, i16, u16, u16)]) -> Vec<u8> {
        let mut v = Vec::new();
        for (src, dst, amt, amt_src, transport) in list {
            v.extend_from_slice(&src.to_le_bytes());
            v.extend_from_slice(&dst.to_le_bytes());
            v.extend_from_slice(&amt.to_le_bytes());
            v.extend_from_slice(&amt_src.to_le_bytes());
            v.extend_from_slice(&transport.to_le_bytes());
        }
        v
    }

    /// 构造一个最小合法 SF2:1 preset / 1 zone / 1 mono 样本(key 60,无循环)。
    fn minimal_sf2() -> Vec<u8> {
        // INFO
        let mut info = Vec::new();
        ck(&mut info, b"INAM", b"TestFont\0");
        // sdta:16 帧 16-bit 正弦
        let frames = 16u32;
        let mut smpl = Vec::new();
        for i in 0..frames {
            let v = ((i as f32 * 0.4).sin() * 10000.0) as i16;
            smpl.extend_from_slice(&v.to_le_bytes());
        }
        let mut sdta = Vec::new();
        ck(&mut sdta, b"smpl", &smpl);

        // pdta
        let mut pdta = Vec::new();
        // phdr:preset(Piano) + EOP 终端
        let mut phdr = Vec::new();
        phdr.extend_from_slice(&rec_name("Piano"));
        phdr.extend_from_slice(&[0u16, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()); // program/bank/bagNdx
        phdr.extend_from_slice(&[0u32, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()); // library/genre/morph
        phdr.extend_from_slice(&rec_name("EOP"));
        phdr.extend_from_slice(&[0u16, 0, 1].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()); // EOP bagNdx = 1(pbag 记录数)
        phdr.extend_from_slice(&[0u32, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        ck(&mut pdta, b"phdr", &phdr);
        // pbag:bag0(genIdx 0, modIdx 0)+ 终点(genIdx 2, modIdx 1)
        ck(&mut pdta, b"pbag", &gens(&[(0, 0), (2, 1)]));
        // pmod:1 条全零 terminal
        ck(&mut pdta, b"pmod", &vec![0u8; 10]);
        // pgen:instrument=0 + terminal
        ck(&mut pdta, b"pgen", &gens(&[(41, 0), (0, 0)]));
        // inst:乐器 + EOI 终端
        let mut inst = Vec::new();
        inst.extend_from_slice(&rec_name("Piano Inst"));
        inst.extend_from_slice(&0u16.to_le_bytes()); // bagNdx
        inst.extend_from_slice(&rec_name("EOI"));
        inst.extend_from_slice(&1u16.to_le_bytes()); // EOI bagNdx = 1(ibag 记录数)
        ck(&mut pdta, b"inst", &inst);
        // ibag:bag0(genIdx 0, modIdx 0)+ 终点(genIdx 5, modIdx 3 = 2 条 mod + terminal)
        ck(&mut pdta, b"ibag", &gens(&[(0, 0), (5, 3)]));
        // imod:vel→initialFilterFc(-1200 cents)+ CC1→vibLfoToPitch(80 cents)+ terminal(3-4)
        ck(&mut pdta, b"imod", &mods(&[
            (0x0502, 590, -1200, 0, 0),
            (0x0081, 1, 80, 0, 0),
            (0, 0, 0, 0, 0),
        ]));
        // igen:keyRange(60,60) / sampleID 0 / attack / release / terminal
        ck(&mut pdta, b"igen", &gens(&[
            (43, 15420),  // keyRange lo=60 hi=60
            (53, 0),      // sampleID
            (34, -12000), // attackVolEnv
            (38, -12000), // releaseVolEnv
            (0, 0),       // terminal
        ]));
        // shdr:样本(Sine16)+ EOS 终端
        let mut shdr = Vec::new();
        for (name, start, end) in [("Sine16", 0u32, frames), ("EOS", frames, frames)] {
            shdr.extend_from_slice(&rec_name(name));
            shdr.extend_from_slice(&start.to_le_bytes());
            shdr.extend_from_slice(&end.to_le_bytes());
            shdr.extend_from_slice(&0u32.to_le_bytes()); // startLoop(0 → 无循环)
            shdr.extend_from_slice(&0u32.to_le_bytes()); // endLoop
            shdr.extend_from_slice(&44100u32.to_le_bytes());
            shdr.push(60); // originalPitch
            shdr.push(0); // pitchFraction
            shdr.extend_from_slice(&0u16.to_le_bytes()); // link
            shdr.extend_from_slice(&1u16.to_le_bytes()); // type=mono
        }
        ck(&mut pdta, b"shdr", &shdr);

        // 顶层 RIFF 组装
        let mut body = Vec::new();
        body.extend_from_slice(&list(b"INFO", &info));
        body.extend_from_slice(&list(b"sdta", &sdta));
        body.extend_from_slice(&list(b"pdta", &pdta));
        let mut riff = Vec::new();
        riff.extend_from_slice(b"RIFF");
        riff.extend_from_slice(&((body.len() + 4) as u32).to_le_bytes());
        riff.extend_from_slice(b"sfbk");
        riff.extend_from_slice(&body);
        riff
    }

    #[test]
    fn parse_minimal_sf2() {
        let bytes = minimal_sf2();
        let dir = std::env::temp_dir().join("muno_sf2_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("minimal.sf2");
        std::fs::write(&path, &bytes).unwrap();

        let font = parse_sf2(&path).expect("解析失败");
        assert_eq!(font.name, "TestFont");
        assert_eq!(font.presets.len(), 1);
        let p = &font.presets[0];
        assert_eq!(p.info.program, 0);
        assert_eq!(p.info.bank, 0);
        assert_eq!(p.regions.len(), 1);
        let r = &p.regions[0];
        assert_eq!((r.key_lo, r.key_hi), (60, 60));
        assert_eq!(r.keycenter, Some(60));
        // 3-4:modulator 白名单烘焙进 region
        assert_eq!(r.vel2filter_cents, -1200.0);
        assert_eq!(r.vib_depth_cents, 80.0);
        assert_eq!(p.samples[0].frame_count(), 16);
        assert_eq!(p.samples[0].rate, 44100);
        assert!((p.samples[0].left[0] - 0.0).abs() < 1e-6);
        assert!(p.samples[0].left.iter().any(|s| s.abs() > 0.1), "样本数据非零");
        // 能构造乐器并渲染出声
        let instr = p.to_instrument();
        let synth = super::super::synth::Synth::new(&instr, 44100.0);
        let mut events = vec![super::super::synth::MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 }];
        let out = synth.render_offline(&mut events, 0.2);
        assert!(out.iter().any(|s| s.abs() > 1e-4), "应渲染出声音");
    }
}
