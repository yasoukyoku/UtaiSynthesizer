/**
 * MultiTrackStudio - 多轨工作室
 *
 * ACE-Step v1.5（Base 多轨 / XL-Turbo 快速任务）：
 *   1. 叠加音轨 (lego)      — 在已有音频上叠加一条新乐器轨道（12 种轨道可选）
 *   2. 分轨分离 (extract)   — 从成品音频中分离出指定轨道
 *   3. 人声转伴奏 (complete) — 上传人声干声，自动补全匹配的伴奏（Vocal2BGM）
 *   4. 风格翻唱 (cover)     — 保留原曲旋律结构，转换为目标风格
 *   5. 局部重绘 (repaint)   — 只重新生成 [start, end) 区间的内容
 *
 * YuE-2 3B（符号乐谱 ABC 方案，YuE2 无独立分轨模块）：
 *   6. YuE2 乐谱 — 新乐器轨道（按原曲 BPM/风格生成独奏轨 + ABC→MIDI）/ 乐谱续写
 */

import { useEffect, useMemo, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { readTextFile } from "@tauri-apps/plugin-fs";
import { songGenerate, listSongModels, type SongOutput } from "../../lib/backendSong";
import "./MultiTrackStudio.css";

export interface StudioSong {
  uuid: string;
  label?: string;
  audioPath?: string;
  abcPath?: string;
  durationSec: number;
  bpm?: number;
  styleHint?: string;
  modelId?: string;
  /** 原曲歌词：供创作助手「翻唱助手」预填（可修改） */
  lyrics?: string;
}

export interface MultiTrackResult {
  outputs: SongOutput[];
  label: string;
  durationSec: number;
  modelId: string;
  wantMidi: boolean;
}

export interface MultiTrackStudioProps {
  songs: StudioSong[];
  initialAudioPath?: string;
  /** 规划 9.1：带任务预选打开（消除"重新选一遍"） */
  initialTask?: Tab;
  onClose: () => void;
  /** 保存为新的历史歌曲 */
  onGenerated: (r: MultiTrackResult) => void;
  /** 直接把结果音频落到工程轨道 */
  onSendToTrack: (audioPath: string, label: string, durationSec: number) => void;
}

const TRACK_NAMES = [
  "woodwinds", "brass", "fx", "synth", "strings", "percussion",
  "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals",
];

const TRACK_LABELS: Record<string, string> = {
  woodwinds: "木管 Woodwinds",
  brass: "铜管 Brass",
  fx: "特效 FX",
  synth: "合成器 Synth",
  strings: "弦乐 Strings",
  percussion: "打击乐 Percussion",
  keyboard: "键盘 Keyboard",
  guitar: "吉他 Guitar",
  bass: "贝斯 Bass",
  drums: "鼓 Drums",
  backing_vocals: "和声 Backing Vocals",
  vocals: "人声 Vocals",
};

/** YuE2 独奏轨道生成用的乐器（符号乐谱方案） */
const YUE2_INSTRUMENTS = [
  { value: "piano", zh: "钢琴" },
  { value: "acoustic guitar", zh: "木吉他" },
  { value: "electric guitar", zh: "电吉他" },
  { value: "bass guitar", zh: "贝斯" },
  { value: "drums", zh: "鼓组" },
  { value: "strings ensemble", zh: "弦乐组" },
  { value: "synth pad", zh: "合成器 Pad" },
  { value: "saxophone", zh: "萨克斯" },
  { value: "erhu", zh: "二胡" },
  { value: "flute", zh: "长笛" },
];

/** 风格翻唱预设（点击追加到描述） */
const COVER_STYLE_PRESETS = [
  { label: "流行", tag: "pop" },
  { label: "摇滚", tag: "rock" },
  { label: "电子", tag: "electronic dance" },
  { label: "爵士", tag: "jazz" },
  { label: "民谣", tag: "folk acoustic" },
  { label: "说唱", tag: "hip-hop rap" },
  { label: "古风", tag: "chinese traditional" },
  { label: "轻音乐", tag: "light instrumental" },
  { label: "Lo-Fi", tag: "lo-fi chillhop" },
  { label: "管弦", tag: "orchestral cinematic" },
];

type Tab = "lego" | "extract" | "vocal2bgm" | "cover" | "repaint" | "yue2score";

export type { Tab as StudioTab };

const TABS: { id: Tab; title: string; desc: string }[] = [
  { id: "lego", title: "叠加音轨", desc: "在当前音频上叠加一条新轨道（鼓/贝斯/人声…），模型自动对齐原曲调式、BPM 与律动" },
  { id: "extract", title: "分轨分离", desc: "从成品音频中分离出指定轨道（Stem 分离，Base 模型原生提取）" },
  { id: "vocal2bgm", title: "人声转伴奏", desc: "上传人声干声，自动生成匹配的伴奏（Vocal2BGM）" },
  { id: "cover", title: "风格翻唱", desc: "保留原曲旋律与结构，把整体风格转换为目标曲风" },
  { id: "repaint", title: "局部重绘", desc: "只重新生成选定时间区间的内容，区间外保持原样" },
  { id: "yue2score", title: "YuE2 乐谱", desc: "YuE-2 无独立分轨模块，通过符号乐谱（ABC）实现：生成单乐器轨 / 续写乐谱，可转 MIDI" },
];

/** 每个 tab 对应的模型（与模型列表显示逻辑一致） */
const TAB_MODEL: Record<Tab, { id: string; name: string; desc: string }> = {
  lego: { id: "acestep-v1.5", name: "ACE-Step V1.5 (Base)", desc: "多轨任务专用 · 50 步推理" },
  extract: { id: "acestep-v1.5", name: "ACE-Step V1.5 (Base)", desc: "原生分轨提取 · 50 步推理" },
  vocal2bgm: { id: "acestep-v1.5", name: "ACE-Step V1.5 (Base)", desc: "音轨补全 · 50 步推理" },
  cover: { id: "acestep-v1.5", name: "ACE-Step V1.5 (XL-Turbo)", desc: "快速风格转换 · 8 步推理" },
  repaint: { id: "acestep-v1.5", name: "ACE-Step V1.5 (XL-Turbo)", desc: "快速局部重绘 · 8 步推理" },
  yue2score: { id: "yue2-3b", name: "YuE-2 3B", desc: "符号乐谱 ABC → 音频 + MIDI" },
};

const BASE_FILES_PREFIX = "acestep-v1.5/checkpoints/acestep-v15-base/";
const YUE2_FILES_PREFIX = "yue2-3b/";

export function MultiTrackStudio({ songs, initialAudioPath, initialTask, onClose, onGenerated, onSendToTrack }: MultiTrackStudioProps) {
  const [tab, setTab] = useState<Tab>(initialTask ?? "lego");
  const [modelInstalled, setModelInstalled] = useState<{ base: boolean; yue2: boolean } | null>(null);

  const audioOptions = useMemo(() => songs.filter((s) => s.audioPath), [songs]);
  const [srcPath, setSrcPath] = useState<string>(
    initialAudioPath && initialAudioPath.length > 0 ? initialAudioPath : (audioOptions[0]?.audioPath ?? ""),
  );

  // lego / extract
  const [track, setTrack] = useState("drums");
  const [extractTrack, setExtractTrack] = useState("vocals");
  // cover
  const [coverPrompt, setCoverPrompt] = useState("");
  const [coverStrength, setCoverStrength] = useState(1.0);
  // repaint
  const [repaintStart, setRepaintStart] = useState("0");
  const [repaintEnd, setRepaintEnd] = useState("10");
  const [repaintPrompt, setRepaintPrompt] = useState("");
  const [repaintLyrics, setRepaintLyrics] = useState("");
  // yue2
  const [yue2Mode, setYue2Mode] = useState<"instrument" | "continue">("instrument");
  const [yue2Instrument, setYue2Instrument] = useState("piano");
  const [yue2StyleText, setYue2StyleText] = useState("");
  const [yue2Abc, setYue2Abc] = useState("");
  const [yue2SongUuid, setYue2SongUuid] = useState("");

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MultiTrackResult | null>(null);

  const srcSong = audioOptions.find((s) => s.audioPath === srcPath);
  const activeTab = TABS.find((t) => t.id === tab)!;
  const activeModel = TAB_MODEL[tab];

  // 检查 Base / YuE2 模型是否已安装（安装清单来自后端）
  useEffect(() => {
    let cancelled = false;
    listSongModels()
      .then((files) => {
        if (cancelled) return;
        const base = files.some((f) => f.filename.startsWith(BASE_FILES_PREFIX));
        const yue2 = files.some((f) => f.filename.startsWith(YUE2_FILES_PREFIX));
        setModelInstalled({ base, yue2 });
      })
      .catch(() => { if (!cancelled) setModelInstalled({ base: true, yue2: true }); });
    return () => { cancelled = true; };
  }, []);

  function switchTab(t: Tab) {
    setTab(t);
    setError(null);
    setResult(null);
  }

  async function pickExternalFile() {
    const picked = await open({
      multiple: false,
      filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac", "ogg", "m4a"] }],
    });
    if (typeof picked === "string" && picked) {
      setSrcPath(picked);
    }
  }

  async function loadAbcFromSong() {
    const song = songs.find((s) => s.uuid === yue2SongUuid);
    if (!song?.abcPath) {
      setError("所选歌曲没有 ABC 乐谱文件（仅 YuE-2 生成的歌曲带有 .abc）");
      return;
    }
    try {
      const text = await readTextFile(song.abcPath);
      setYue2Abc(text);
      setError(null);
    } catch (e) {
      setError(`读取 ABC 文件失败：${String(e)}`);
    }
  }

  function validate(): string | null {
    if (tab === "yue2score") {
      if (yue2Mode === "continue" && !yue2Abc.trim()) return "请粘贴或加载要续写的 ABC 乐谱";
      return null;
    }
    if (!srcPath) return "请先选择源音频（可从歌曲列表选择，或选择本地文件）";
    if (tab === "repaint") {
      const s = Number(repaintStart), e = Number(repaintEnd);
      if (Number.isNaN(s) || Number.isNaN(e)) return "重绘区间必须是数字（秒）";
      if (s < 0 || e <= s) return "重绘区间无效：需要 0 ≤ 起点 < 终点";
      const dur = srcSong?.durationSec ?? 0;
      if (dur > 0 && e > dur) return `重绘终点 ${e}s 超出源音频时长 ${dur}s`;
    }
    return null;
  }

  async function handleGenerate() {
    const v = validate();
    if (v) { setError(v); return; }
    setBusy(true);
    setError(null);
    setResult(null);

    try {
      let outputs: SongOutput[];
      let label: string;
      let durationSec: number;
      let modelId: string;
      let wantMidi = false;

      if (tab === "yue2score") {
        // ── YuE-2 符号乐谱方案 ──
        modelId = "yue2-3b";
        wantMidi = true;
        let prompt: string;
        if (yue2Mode === "instrument") {
          // 新乐器轨道：沿用源歌曲的 BPM/风格，生成单乐器独奏（模型对齐节奏靠 BPM 提示）
          const inst = YUE2_INSTRUMENTS.find((i) => i.value === yue2Instrument);
          const bpmPart = srcSong?.bpm ? `${srcSong.bpm} BPM, ` : "";
          const stylePart = (yue2StyleText.trim() || srcSong?.styleHint || "");
          prompt = `solo ${yue2Instrument} instrumental track, ${bpmPart}${stylePart}`.trim();
          label = `${srcSong?.label || "song"}_yue2_${inst?.zh || yue2Instrument}`;
        } else {
          // 乐谱续写：把已有 ABC 作为条件传给 YuE-2，按描述延展
          prompt = yue2StyleText.trim() || "continue the score in the same style, seamless extension";
          label = `abc_continue_${Date.now().toString(36)}`;
        }
        const baseName = srcSong?.label?.replace(/[\\/:*?"<>|]/g, "_") || "yue2";
        outputs = await songGenerate({
          model: modelId,
          task: "text2music",
          song_name: `${baseName}_yue2_${yue2Mode}`,
          prompt,
          lyrics: "[Instrumental]",
          audio_duration: 0,
          format: "wav",
          output_dir: "",
          want_midi: true,
          ...(yue2Mode === "continue" ? { abc: yue2Abc } : {}),
        });
        const pa = outputs.find((o) => o.audio_path);
        if (!pa?.audio_path) throw new Error("生成完成但没有返回音频文件");
        durationSec = srcSong?.durationSec && yue2Mode === "instrument" ? srcSong.durationSec : 60;
      } else {
        // ── ACE-Step 任务 ──
        modelId = "acestep-v1.5";
        const taskType = tab === "vocal2bgm" ? "complete" : tab;
        const trackName = tab === "lego" ? track : tab === "extract" ? extractTrack : undefined;
        const baseName = srcPath.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || "src";
        label = `${baseName}_${taskType}${trackName ? `_${trackName}` : ""}`;
        const prompt = tab === "cover" ? coverPrompt.trim() : tab === "repaint" ? repaintPrompt.trim()
          : tab === "lego" ? "" : "";

        outputs = await songGenerate({
          model: modelId,
          task: taskType,
          src_audio_path: srcPath,
          track_name: trackName,
          prompt,
          lyrics: tab === "repaint" ? repaintLyrics : "",
          song_name: label,
          audio_duration: 0, // direct-conditioning 任务自动取源音频时长
          format: "wav",
          output_dir: "",
          ...(tab === "cover" ? { audio_cover_strength: coverStrength } : {}),
          ...(tab === "repaint" ? { repaint_start: Number(repaintStart), repaint_end: Number(repaintEnd) } : {}),
        });
        const pa = outputs.find((o) => o.audio_path);
        if (!pa?.audio_path) throw new Error("生成完成但没有返回音频文件");
        durationSec = srcSong?.durationSec ?? 60;
      }

      setResult({ outputs, label, durationSec, modelId, wantMidi });
    } catch (err) {
      const msg = typeof err === "string" ? err : err instanceof Error ? err.message : JSON.stringify(err);
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  const resultAudio = result?.outputs.find((o) => o.audio_path)?.audio_path;
  const resultMidi = result?.outputs.find((o) => o.midi_path)?.midi_path;
  const modelWarning =
    (tab === "yue2score" && modelInstalled && !modelInstalled.yue2) ? "YuE-2 3B 模型尚未下载，请先到「资源管理 → 生成歌曲」下载。" :
    (tab !== "yue2score" && modelInstalled && !modelInstalled.base) ? "ACE-Step V1.5 (Base) 模型尚未下载（约 4.8GB），请先到「资源管理 → 生成歌曲」下载。" :
    null;

  return (
    <div className="mts-overlay" onClick={onClose}>
      <div className="mts-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="mts-header">
          <div className="mts-title">🎛 多轨工作室</div>
          <div className="mts-subtitle">模型层面分层生成 / 叠加 · 非可视化 DAW 编辑器</div>
          <button className="mts-close" onClick={onClose}>✕</button>
        </div>

        <div className="mts-tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`mts-tab ${tab === t.id ? "active" : ""}`}
              onClick={() => switchTab(t.id)}
            >
              {t.title}
            </button>
          ))}
        </div>
        <div className="mts-tab-desc">{activeTab.desc}</div>

        {/* 模型显示行：随任务自动切换，默认选中 */}
        <div className="mts-model-row">
          <label className="mts-label">使用模型</label>
          <select className="mts-select mts-model-select" value={activeModel.id} disabled>
            <option value={activeModel.id}>{activeModel.name}</option>
          </select>
          <span className="mts-model-desc">{activeModel.desc}</span>
        </div>
        {modelWarning && <div className="mts-warn">⚠ {modelWarning}</div>}

        <div className="mts-body">
          {tab !== "yue2score" && (
            <div className="mts-field">
              <label className="mts-label">源音频</label>
              <div className="mts-src-row">
                <select className="mts-select" value={srcPath} onChange={(e) => setSrcPath(e.target.value)}>
                  {audioOptions.length === 0 && <option value="">（歌曲列表暂无音频）</option>}
                  {audioOptions.map((s) => (
                    <option key={s.uuid} value={s.audioPath}>
                      {s.label || "Untitled"}（{s.durationSec}s）
                    </option>
                  ))}
                  {srcPath && !audioOptions.some((s) => s.audioPath === srcPath) && (
                    <option value={srcPath}>本地文件：{srcPath.split(/[\\/]/).pop()}</option>
                  )}
                </select>
                <button className="mts-btn mts-btn-ghost" onClick={pickExternalFile} disabled={busy}>
                  选择本地文件
                </button>
              </div>
              <div className="mts-hint">支持 WAV / MP3 / FLAC / OGG / M4A，建议 44.1kHz+；分轨分离 / 叠加 / 补全 / 翻唱 / 重绘都会保持源音频时长。</div>
            </div>
          )}

          {tab === "lego" && (
            <div className="mts-field">
              <label className="mts-label">要叠加的轨道</label>
              <select className="mts-select" value={track} onChange={(e) => setTrack(e.target.value)} disabled={busy}>
                {TRACK_NAMES.map((t) => (
                  <option key={t} value={t}>{TRACK_LABELS[t]}</option>
                ))}
              </select>
              <div className="mts-hint">生成结果可「发送到轨道」，新轨道与原曲调式/BPM 自动对齐；音量平衡在主时间线的轨道推子上调节。</div>
            </div>
          )}

          {tab === "extract" && (
            <div className="mts-field">
              <label className="mts-label">要分离出的轨道</label>
              <select className="mts-select" value={extractTrack} onChange={(e) => setExtractTrack(e.target.value)} disabled={busy}>
                {TRACK_NAMES.map((t) => (
                  <option key={t} value={t}>{TRACK_LABELS[t]}</option>
                ))}
              </select>
              <div className="mts-hint">每次分离一条轨道；分离结果保存到歌曲列表后，可直接作为「风格翻唱」或「局部重绘」的源音频联动使用。</div>
            </div>
          )}

          {tab === "vocal2bgm" && (
            <div className="mts-hint">
              选择一条人声干声作为源音频，模型会自动补全与其调式/BPM 匹配的伴奏（ drums / bass / 和声等全乐器）。
            </div>
          )}

          {tab === "cover" && (
            <>
              <div className="mts-field">
                <label className="mts-label">风格预设（点击追加）</label>
                <div className="mts-chips">
                  {COVER_STYLE_PRESETS.map((p) => (
                    <button
                      key={p.tag}
                      className="mts-chip"
                      onClick={() => setCoverPrompt((v) => (v.includes(p.tag) ? v : `${v ? v + ", " : ""}${p.tag}`))}
                      disabled={busy}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="mts-field">
                <label className="mts-label">目标风格描述</label>
                <textarea
                  className="mts-textarea"
                  rows={2}
                  placeholder="例如：energetic rock, distorted guitar, powerful drums"
                  value={coverPrompt}
                  onChange={(e) => setCoverPrompt(e.target.value)}
                  disabled={busy}
                />
              </div>
              <div className="mts-field">
                <label className="mts-label">风格转换强度：{coverStrength.toFixed(2)}</label>
                <input
                  type="range" min={0.3} max={1} step={0.05}
                  value={coverStrength}
                  onChange={(e) => setCoverStrength(Number(e.target.value))}
                  disabled={busy}
                />
                <div className="mts-hint">1.0 = 完全按新风格重演绎；调低则保留更多原曲音色细节。输出为 48kHz WAV。</div>
              </div>
            </>
          )}

          {tab === "repaint" && (
            <>
              <div className="mts-field mts-repaint-range">
                <div>
                  <label className="mts-label">起点（秒）</label>
                  <input className="mts-input" value={repaintStart}
                    onChange={(e) => setRepaintStart(e.target.value.replace(/[^0-9.]/g, ""))} disabled={busy} />
                </div>
                <div>
                  <label className="mts-label">终点（秒）</label>
                  <input className="mts-input" value={repaintEnd}
                    onChange={(e) => setRepaintEnd(e.target.value.replace(/[^0-9.]/g, ""))} disabled={busy} />
                </div>
              </div>
              <div className="mts-field">
                <label className="mts-label">区间新内容描述</label>
                <textarea
                  className="mts-textarea"
                  rows={2}
                  placeholder="例如：drum fill buildup into the chorus"
                  value={repaintPrompt}
                  onChange={(e) => setRepaintPrompt(e.target.value)}
                  disabled={busy}
                />
              </div>
              <div className="mts-field">
                <label className="mts-label">区间歌词（可选，含人声时填写）</label>
                <textarea
                  className="mts-textarea"
                  rows={2}
                  value={repaintLyrics}
                  onChange={(e) => setRepaintLyrics(e.target.value)}
                  disabled={busy}
                />
              </div>
              <div className="mts-hint">区间外音频保持原样；渲染时只对 [起点, 终点) 做扩散重绘，速度与区间长度成正比。</div>
            </>
          )}

          {tab === "yue2score" && (
            <>
              <div className="mts-mode-row">
                <button className={`mts-mode-btn ${yue2Mode === "instrument" ? "active" : ""}`}
                  onClick={() => { setYue2Mode("instrument"); setError(null); setResult(null); }} disabled={busy}>
                  新乐器轨道
                </button>
                <button className={`mts-mode-btn ${yue2Mode === "continue" ? "active" : ""}`}
                  onClick={() => { setYue2Mode("continue"); setError(null); setResult(null); }} disabled={busy}>
                  乐谱续写
                </button>
              </div>

              {yue2Mode === "instrument" && (
                <>
                  <div className="mts-field">
                    <label className="mts-label">乐器类型</label>
                    <select className="mts-select" value={yue2Instrument}
                      onChange={(e) => setYue2Instrument(e.target.value)} disabled={busy}>
                      {YUE2_INSTRUMENTS.map((i) => (
                        <option key={i.value} value={i.value}>{i.zh}（{i.value}）</option>
                      ))}
                    </select>
                  </div>
                  <div className="mts-field">
                    <label className="mts-label">风格补充描述（可选，默认沿用源歌曲风格与 BPM）</label>
                    <input className="mts-input" value={yue2StyleText}
                      onChange={(e) => setYue2StyleText(e.target.value)}
                      placeholder="例如：warm acoustic ballad" disabled={busy} />
                  </div>
                  <div className="mts-hint">
                    技术方案：YuE-2 无独立分轨模块，此处按源歌曲 BPM/风格生成单乐器独奏（同步依赖 BPM 提示对齐），
                    同时产出 ABC 乐谱与 MIDI；生成后「发送到轨道」即可与原曲叠放，音量在轨道推子上平衡。
                  </div>
                </>
              )}

              {yue2Mode === "continue" && (
                <>
                  <div className="mts-field">
                    <label className="mts-label">从历史歌曲加载 ABC（仅 YuE-2 生成的歌曲）</label>
                    <div className="mts-src-row">
                      <select className="mts-select" value={yue2SongUuid}
                        onChange={(e) => setYue2SongUuid(e.target.value)} disabled={busy}>
                        <option value="">（选择歌曲）</option>
                        {songs.filter((s) => s.abcPath).map((s) => (
                          <option key={s.uuid} value={s.uuid}>{s.label || "Untitled"}</option>
                        ))}
                      </select>
                      <button className="mts-btn mts-btn-ghost" onClick={loadAbcFromSong} disabled={busy || !yue2SongUuid}>
                        加载
                      </button>
                    </div>
                  </div>
                  <div className="mts-field">
                    <label className="mts-label">ABC 乐谱（可直接粘贴编辑）</label>
                    <textarea
                      className="mts-textarea mts-mono"
                      rows={6}
                      placeholder={"X:1\nM:4/4\nL:1/8\nK:C\n|:C2D2E2G2:|"}
                      value={yue2Abc}
                      onChange={(e) => setYue2Abc(e.target.value)}
                      disabled={busy}
                    />
                  </div>
                  <div className="mts-field">
                    <label className="mts-label">续写方向描述</label>
                    <input className="mts-input" value={yue2StyleText}
                      onChange={(e) => setYue2StyleText(e.target.value)}
                      placeholder="例如：以同样的风格继续发展，加入副歌推进" disabled={busy} />
                  </div>
                  <div className="mts-hint">
                    续写逻辑：把已有 ABC 作为结构条件传给 YuE-2 的 CoT 规划层，模型在乐谱骨架上延展并重新渲染音频，
                    同时输出新的 ABC + MIDI（与 ACE-Step 的音频补全不同，这是乐谱层的续写）。
                  </div>
                </>
              )}
            </>
          )}

          <button className="mts-btn mts-btn-primary" onClick={handleGenerate} disabled={busy}>
            {busy ? "生成中…请耐心等待" :
              tab === "lego" ? "生成叠加轨道" :
              tab === "extract" ? "开始分离" :
              tab === "vocal2bgm" ? "生成伴奏" :
              tab === "cover" ? "开始翻唱" :
              tab === "repaint" ? "重绘区间" :
              "生成乐谱轨道"}
          </button>

          {busy && (
            <div className="mts-hint">
              {activeModel.desc}；首次加载模型需要更长时间，进度详情见控制台日志。
            </div>
          )}

          {error && <div className="mts-error">{error}</div>}

          {result && resultAudio && (
            <div className="mts-result">
              <div className="mts-result-title">✅ 生成完成：{result.label}</div>
              <audio controls src={convertFileSrc(resultAudio)} className="mts-audio" />
              {(resultMidi || result.outputs.find((o) => o.abc_path)?.abc_path) && (
                <div className="mts-hint">
                  {resultMidi ? "已附带 MIDI 文件；" : ""}
                  {result.outputs.find((o) => o.abc_path)?.abc_path ? "已附带 ABC 乐谱文件；" : ""}
                  保存到歌曲列表后可在歌曲详情中使用。
                </div>
              )}
              <div className="mts-result-actions">
                <button className="mts-btn mts-btn-primary" onClick={() => onGenerated(result)}>
                  保存到歌曲列表
                </button>
                <button
                  className="mts-btn mts-btn-ghost"
                  onClick={() => onSendToTrack(resultAudio, result.label, result.durationSec)}
                >
                  发送到轨道
                </button>
                <button className="mts-btn mts-btn-ghost" onClick={() => setResult(null)}>关闭</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
