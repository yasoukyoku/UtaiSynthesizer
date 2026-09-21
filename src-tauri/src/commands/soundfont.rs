//! 音源管理 + 乐器轨渲染命令(阶段1+2)。

use std::path::PathBuf;
use std::sync::Arc;

use tauri::State;

use crate::audio::soundfont::{self, RenderBend, RenderCc, RenderLayer, RenderNote, SoundfontEntry};
use crate::AppState;

fn sfonts_dir(state: &AppState) -> PathBuf {
    soundfont::soundfonts_dir(&state.app_dir)
}

#[tauri::command]
pub async fn list_soundfonts(state: State<'_, Arc<AppState>>) -> Result<Vec<SoundfontEntry>, String> {
    let dir = sfonts_dir(&state);
    Ok(soundfont::scan_soundfonts(&dir))
}

/// 导入音源(前端用对话框选好路径后调用)。
#[tauri::command]
pub async fn import_soundfont(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<SoundfontEntry, String> {
    let dir = sfonts_dir(&state);
    let src = PathBuf::from(&path);
    if !src.exists() {
        return Err(format!("路径不存在: {}", path));
    }
    let mut entry = soundfont::import_soundfont(&dir, &src)?;
    // 复扫补 preset 列表
    let all = soundfont::scan_soundfonts(&dir);
    if let Some(found) = all.into_iter().find(|e| e.id == entry.id) {
        entry = found;
    }
    Ok(entry)
}

#[tauri::command]
pub async fn delete_soundfont(state: State<'_, Arc<AppState>>, id: String) -> Result<(), String> {
    let dir = sfonts_dir(&state);
    soundfont::delete_soundfont(&dir, &id)
}

/// 打开音源目录(资源管理器)。fire-and-forget,同 logs::open_log_dir 模式。
#[tauri::command]
pub async fn open_soundfonts_dir(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let dir = sfonts_dir(&state);
    std::fs::create_dir_all(&dir).map_err(|e| format!("创建目录失败: {}", e))?;
    #[cfg(windows)]
    let spawned = std::process::Command::new("explorer").arg(&dir).spawn();
    #[cfg(target_os = "macos")]
    let spawned = std::process::Command::new("open").arg(&dir).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let spawned = std::process::Command::new("xdg-open").arg(&dir).spawn();
    if let Err(e) = spawned {
        tracing::warn!("open_soundfonts_dir: failed to open {}: {}", dir.display(), e);
    }
    Ok(dir.to_string_lossy().to_string())
}

/// 渲染乐器轨(离线 bake → WAV 缓存文件,前端 WebAudio 播放)。
/// 返回 WAV 绝对路径;缓存在 cache_dir/render 下,
/// 键 = hash(font+preset+notes+cc+bend+layers+backend)。
#[tauri::command]
pub async fn render_soundfont_notes(
    state: State<'_, Arc<AppState>>,
    font_id: String,
    preset_id: String,
    notes: Vec<RenderNote>,
    cc: Option<Vec<RenderCc>>,
    bend: Option<Vec<RenderBend>>,
    // 3-1 分层音源:非空时按层渲染并混合(font_id/preset_id 仅用于缓存键与占位)。
    layers: Option<Vec<RenderLayer>>,
    sample_rate: Option<u32>,
    // 3-9 FluidSynth 通用化:"builtin"(默认,内置合成器) | "fluidsynth"(CLI 对照组)。
    backend: Option<String>,
) -> Result<String, String> {
    let dir = sfonts_dir(&state);
    let render_dir = state.cache_dir.join("render");
    std::fs::create_dir_all(&render_dir).map_err(|e| format!("创建缓存目录失败: {}", e))?;
    let ccs = cc.unwrap_or_default();
    let bends = bend.unwrap_or_default();
    let layers = layers.unwrap_or_default();
    // 3-9 后端归一:空/未指定 → builtin;fluidsynth 不支持分层混合(层是内置渲染器概念)。
    let backend = backend
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("builtin");
    if backend == "fluidsynth" && !layers.is_empty() {
        return Err("SOUNDFONT_BACKEND_LAYERS_UNSUPPORTED".into());
    }
    // 缓存键必须涵盖全部影响输出的输入(CC/弯音变化 → 不同 WAV)
    let mut key = format!("{}|{}|{}", font_id, preset_id, notes.len());
    for n in &notes {
        key.push_str(&format!("|{},{},{},{}", n.start, n.dur, n.key, n.vel));
    }
    for c in &ccs {
        key.push_str(&format!("|c{},{},{}", c.start, c.cc, c.value));
    }
    for b in &bends {
        key.push_str(&format!("|b{},{}", b.start, b.value));
    }
    for l in &layers {
        key.push_str(&format!("|L{},{},{},{}", l.font_id, l.preset_id, l.gain, l.pan));
    }
    // 后端参与缓存键:仅 fluidsynth 追加标记(fold-away,builtin 既有缓存键不变)。
    if backend == "fluidsynth" {
        key.push_str("|bfs");
    }
    let hash = xxhash_rust::xxh3::xxh3_64(key.as_bytes());
    let out_path = render_dir.join(format!("sf_{}.wav", hash));
    if out_path.is_file() {
        return Ok(out_path.to_string_lossy().to_string());
    }
    let _guard = state.begin_task("render");
    let sr = sample_rate.unwrap_or(44100);
    if backend == "fluidsynth" {
        // FluidSynth 走 MIDI 文件中转(秒 → 960 tick/秒),渲染完即删临时 .mid。
        let midi_tmp = render_dir.join(format!("sf_{}.mid", hash));
        crate::audio::fluidsynth::render_notes_to_wav(
            &dir,
            &font_id,
            &preset_id,
            &notes,
            &ccs,
            &bends,
            sr,
            &out_path,
            &state.amt_models_dir,
            &midi_tmp,
        )?;
    } else if layers.is_empty() {
        soundfont::render_to_wav(&dir, &font_id, &preset_id, &notes, &ccs, &bends, sr, &out_path)?;
    } else {
        soundfont::render_layers_to_wav(&dir, &layers, &notes, &ccs, &bends, sr, &out_path)?;
    }
    Ok(out_path.to_string_lossy().to_string())
}

/// 试听单音(前端按键/点击音符时调用)。
/// 返回 `None` = 已由 cpal 实时引擎直接发声;`Some(path)` = 实时不可用,
/// 回退返回 WAV 缓存文件路径(前端 WebAudio 播放)。
#[tauri::command]
pub async fn audition_soundfont_note(
    state: State<'_, Arc<AppState>>,
    font_id: String,
    preset_id: String,
    key: u8,
    vel: Option<u8>,
    dur: Option<f64>,
) -> Result<Option<String>, String> {
    let dir = sfonts_dir(&state);
    let key = key.min(127);
    let vel = vel.unwrap_or(100).min(127);
    let dur = dur.unwrap_or(1.0).clamp(0.05, 8.0);
    // Muno 阶段2:优先走 cpal 常驻实时流(乐器已缓存时零文件 I/O)。
    // 引擎不可用(无设备/格式不支持)→ 回退 WAV;乐器加载失败 → 直接报错
    // (WAV 路径同样要加载该乐器,回退无意义)。
    match crate::audio::audio_output::try_rt_audition(&dir, &font_id, &preset_id, key, vel, dur as f32) {
        Ok(crate::audio::audio_output::RtAudition::Played) => return Ok(None),
        Ok(crate::audio::audio_output::RtAudition::Unavailable) => {}
        Err(e) => return Err(e),
    }
    let render_dir = state.cache_dir.join("render");
    std::fs::create_dir_all(&render_dir).map_err(|e| format!("创建缓存目录失败: {}", e))?;
    // 文件名含 vel/dur:同名缓存直接复用,重复抓同一音符零重渲染;参数变了则各存一份。
    let out_path = render_dir.join(format!(
        "note_{}_{}_{}_{}_{}.wav",
        sanitize_file(&font_id),
        sanitize_file(&preset_id),
        key,
        vel,
        (dur * 1000.0).round() as u32
    ));
    if out_path.is_file() {
        return Ok(Some(out_path.to_string_lossy().to_string()));
    }
    let note = RenderNote {
        start: 0.0,
        dur,
        key,
        vel,
    };
    soundfont::render_to_wav(&dir, &font_id, &preset_id, &[note], &[], &[], 44100, &out_path)?;
    Ok(Some(out_path.to_string_lossy().to_string()))
}

fn sanitize_file(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            ':' => '-',
            _ => '_',
        })
        .collect()
}
