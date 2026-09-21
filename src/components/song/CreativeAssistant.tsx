/**
 * CreativeAssistant - 创作助手（简化入口）
 *
 * 三个高频创作流，统一到一个"少滚动、参数分主次"的面板里（规划 A/B/C/D 简化）：
 *   B. 单乐器叠加 — 选源歌曲 + 选乐器 → 生成并叠加（ACE-Step lego）
 *   C. 乐谱续写   — 载入历史 ABC → 续写方向 → 生成多个候选 → 试听/采用并继续（YuE-2）
 *   D. 翻唱助手   — 源歌曲 → 分离人声 → 改歌词 → 选风格 → 生成（ACE-Step cover）
 *
 * 复用后端既有任务管线（songGenerate），不新增模型/命令；
 * 生成结果统一走 onGenerated（存入歌曲列表）/ onSendToTrack（落到工程轨道）。
 */

import { useEffect, useMemo, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { readTextFile } from "@tauri-apps/plugin-fs";
import { songGenerate, listSongModels, type SongOutput } from "../../lib/backendSong";
import type { StudioSong, MultiTrackResult } from "./MultiTrackStudio";
import "./CreativeAssistant.css";

export type CreativeTab = "instrument" | "score" | "cover";

export interface CreativeAssistantProps {
  songs: StudioSong[];
  initialTab?: CreativeTab;
  initialAudioPath?: string;
  onClose: () => void;
  /** 保存为新的历史歌曲 */
  onGenerated: (r: MultiTrackResult) => void;
  /** 直接把结果音频落到工程轨道 */
  onSendToTrack: (audioPath: string, label: string, durationSec: number) => void;
  /** 打开 MIDI 编辑器（结果带 MIDI 时可用） */
  onOpenMidi?: (midiPath: string, label: string) => void;
}

const BASE_MODEL = "acestep-v1.5";
const YUE2_MODEL = "yue2-3b";
const BASE_FILES_PREFIX = "acestep-v1.5/checkpoints/acestep-v15-base/";
const YUE2_FILES_PREFIX = "yue2-3b/";

const TABS: { id: CreativeTab; title: string; desc: string }[] = [
  { id: "instrument", title: "🎸 叠加乐器", desc: "给现有歌曲加一条乐器轨（鼓/贝斯/吉他…），模型自动对齐原曲调式与 BPM" },
  { id: "score", title: "🎼 乐谱续写", desc: "载入历史 ABC 乐谱，按你的方向生成多个延展候选，试听后采用并继续写下去" },
  { id: "cover", title: "🎤 翻唱助手", desc: "一键流程：分离人声 → 改歌词 → 选风格 → 生成，自动沿用原曲 BPM 与曲式" },
];

/** 常用乐器（点击即选），value 为 ACE-Step 轨道名 */
const INSTRUMENTS = [
  { value: "drums", zh: "🥁 鼓组" },
  { value: "bass", zh: "🎸 贝斯" },
  { value: "guitar", zh: "🎸 吉他" },
  { value: "keyboard", zh: "🎹 键盘" },
  { value: "strings", zh: "🎻 弦乐" },
  { value: "synth", zh: "🎛 合成器" },
  { value: "brass", zh: "🎺 铜管" },
  { value: "woodwinds", zh: "🎷 木管" },
  { value: "percussion", zh: "🪘 打击乐" },
  { value: "fx", zh: "✨ 特效" },
  { value: "backing_vocals", zh: "🎤 和声" },
  { value: "vocals", zh: "🎤 人声" },
];

/** 翻唱风格预设（点击追加到描述） */
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

interface Candidate {
  id: string;
  label: string;
  outputs: SongOutput[];
  audioPath?: string;
  abcPath?: string;
  midiPath?: string;
  durationSec: number;
}

function baseNameOf(p: string): string {
  return p.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || "src";
}

export function CreativeAssistant({
  songs,
  initialTab,
  initialAudioPath,
  onClose,
  onGenerated,
  onSendToTrack,
  onOpenMidi,
}: CreativeAssistantProps) {
  const [tab, setTab] = useState<CreativeTab>(initialTab ?? "instrument");
  const audioOptions = useMemo(() => songs.filter((s) => s.audioPath), [songs]);
  const abcOptions = useMemo(() => songs.filter((s) => s.abcPath), [songs]);

  const [srcPath, setSrcPath] = useState<string>(initialAudioPath ?? audioOptions[0]?.audioPath ?? "");
  const srcSong = audioOptions.find((s) => s.audioPath === srcPath);

  // B：单乐器叠加
  const [instrument, setInstrument] = useState("drums");
  const [instStyle, setInstStyle] = useState("");

  // C：乐谱续写
  const [abcSongUuid, setAbcSongUuid] = useState("");
  const [abcText, setAbcText] = useState("");
  const [contStyle, setContStyle] = useState("");
  const [candidateCount, setCandidateCount] = useState(2);
  const [candidates, setCandidates] = useState<Candidate[]>([]);

  // D：翻唱助手
  const [vocalPath, setVocalPath] = useState<string | null>(null);
  const [useVocalSource, setUseVocalSource] = useState(true);
  const [coverLyrics, setCoverLyrics] = useState("");
  const [coverPrompt, setCoverPrompt] = useState("");
  const [coverStrength, setCoverStrength] = useState(0.8);

  const [busy, setBusy] = useState(false);
  const [progressText, setProgressText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [result, setResult] = useState<MultiTrackResult | null>(null);
  const [modelInstalled, setModelInstalled] = useState<{ base: boolean; yue2: boolean } | null>(null);

  useEffect(() => {
    let cancelled = false;
    listSongModels()
      .then((files) => {
        if (cancelled) return;
        setModelInstalled({
          base: files.some((f) => f.filename.startsWith(BASE_FILES_PREFIX)),
          yue2: files.some((f) => f.filename.startsWith(YUE2_FILES_PREFIX)),
        });
      })
      .catch(() => { if (!cancelled) setModelInstalled({ base: true, yue2: true }); });
    return () => { cancelled = true; };
  }, []);

  // 切换源歌曲时，用原曲歌词预填翻唱歌词（"自动记忆"原曲内容）
  useEffect(() => {
    if (srcSong?.lyrics && !coverLyrics.trim()) setCoverLyrics(srcSong.lyrics);
    setVocalPath(null);
    // 仅在切换源歌曲时同步
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [srcPath]);

  function switchTab(t: CreativeTab) {
    if (busy) return;
    setTab(t);
    setError(null);
    setStatus(null);
    setResult(null);
  }

  function resetOutputs() {
    setError(null);
    setStatus(null);
    setResult(null);
    setCandidates([]);
  }

  async function loadAbcFromSong(uuid: string) {
    const song = songs.find((s) => s.uuid === uuid);
    if (!song?.abcPath) {
      setError("所选歌曲没有 ABC 乐谱文件（仅 YuE-2 生成的歌曲带有 .abc）");
      return;
    }
    try {
      const text = await readTextFile(song.abcPath);
      setAbcText(text);
      setError(null);
      setStatus(`已载入乐谱：${song.label || "Untitled"}`);
    } catch (e) {
      setError(`读取 ABC 文件失败：${String(e)}`);
    }
  }

  // ── B：单乐器叠加 ────────────────────────────────────────────────────────
  async function generateInstrument() {
    if (!srcPath) { setError("请先选择源歌曲（或到歌曲列表生成一首）"); return; }
    setBusy(true);
    resetOutputs();
    try {
      const label = `${baseNameOf(srcPath)}_lego_${instrument}`;
      const outputs = await songGenerate({
        model: BASE_MODEL,
        task: "lego",
        src_audio_path: srcPath,
        track_name: instrument,
        prompt: instStyle.trim(),
        lyrics: "",
        song_name: label,
        audio_duration: 0,
        format: "wav",
        output_dir: "",
      });
      if (!outputs.find((o) => o.audio_path)?.audio_path) throw new Error("生成完成但没有返回音频文件");
      setResult({ outputs, label, durationSec: srcSong?.durationSec ?? 60, modelId: BASE_MODEL, wantMidi: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  // ── C：乐谱续写（多候选）─────────────────────────────────────────────────
  async function generateCandidates() {
    if (!abcText.trim()) { setError("请先加载或粘贴要续写的 ABC 乐谱"); return; }
    setBusy(true);
    resetOutputs();
    const list: Candidate[] = [];
    try {
      for (let i = 0; i < candidateCount; i++) {
        setProgressText(`候选 ${i + 1}/${candidateCount} 生成中…`);
        const label = `abc_continue_${Date.now().toString(36)}_${i + 1}`;
        const outputs = await songGenerate({
          model: YUE2_MODEL,
          task: "text2music",
          song_name: label,
          prompt: contStyle.trim() || "continue the score in the same style, seamless extension",
          lyrics: "[Instrumental]",
          audio_duration: 0,
          format: "wav",
          output_dir: "",
          want_midi: true,
          abc: abcText,
        });
        const audioPath = outputs.find((o) => o.audio_path)?.audio_path;
        if (!audioPath) throw new Error("生成完成但没有返回音频文件");
        list.push({
          id: crypto.randomUUID(),
          label,
          outputs,
          audioPath,
          abcPath: outputs.find((o) => o.abc_path)?.abc_path,
          midiPath: outputs.find((o) => o.midi_path)?.midi_path,
          durationSec: 60,
        });
        setCandidates([...list]);
      }
      setStatus(`已生成 ${list.length} 个候选，试听后「采用并继续」`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setProgressText(null);
    }
  }

  /** 采用某候选的乐谱作为新的续写起点 → 形成"继续写"的循环 */
  async function adoptCandidate(c: Candidate) {
    if (!c.abcPath) { setError("该候选没有附带 ABC 乐谱文件"); return; }
    try {
      const text = await readTextFile(c.abcPath);
      setAbcText(text);
      setCandidates([]);
      setStatus("已采用该候选乐谱作为新的续写起点，可继续生成下一段");
      setError(null);
    } catch (e) {
      setError(`读取候选乐谱失败：${String(e)}`);
    }
  }

  // ── D：翻唱助手 ──────────────────────────────────────────────────────────
  async function extractVocals() {
    if (!srcPath) { setError("请先选择源歌曲"); return; }
    setBusy(true);
    setError(null);
    setStatus(null);
    setVocalPath(null);
    try {
      const label = `${baseNameOf(srcPath)}_vocals`;
      const outputs = await songGenerate({
        model: BASE_MODEL,
        task: "extract",
        src_audio_path: srcPath,
        track_name: "vocals",
        prompt: "",
        lyrics: "",
        song_name: label,
        audio_duration: 0,
        format: "wav",
        output_dir: "",
      });
      const vp = outputs.find((o) => o.audio_path)?.audio_path;
      if (!vp) throw new Error("分离完成但没有返回人声文件");
      setVocalPath(vp);
      setStatus(`人声已分离：${vp.split(/[\\/]/).pop()}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function generateCover() {
    const coverSrc = useVocalSource && vocalPath ? vocalPath : srcPath;
    if (!coverSrc) { setError("请先选择源歌曲（或先分离人声）"); return; }
    setBusy(true);
    resetOutputs();
    try {
      const label = `${baseNameOf(srcPath)}_cover`;
      const outputs = await songGenerate({
        model: BASE_MODEL,
        task: "cover",
        src_audio_path: coverSrc,
        prompt: coverPrompt.trim(),
        lyrics: coverLyrics,
        song_name: label,
        audio_duration: 0,
        format: "wav",
        output_dir: "",
        audio_cover_strength: coverStrength,
      });
      if (!outputs.find((o) => o.audio_path)?.audio_path) throw new Error("生成完成但没有返回音频文件");
      setResult({ outputs, label, durationSec: srcSong?.durationSec ?? 60, modelId: BASE_MODEL, wantMidi: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const activeTab = TABS.find((t) => t.id === tab) ?? TABS[0]!;
  const resultAudio = result?.outputs.find((o) => o.audio_path)?.audio_path;
  const resultMidi = result?.outputs.find((o) => o.midi_path)?.midi_path;
  const modelWarning =
    (tab === "score" && modelInstalled && !modelInstalled.yue2)
      ? "YuE-2 3B 模型尚未下载，请先到「资源管理 → 生成歌曲」下载。"
      : (tab !== "score" && modelInstalled && !modelInstalled.base)
        ? "ACE-Step V1.5 (Base) 模型尚未下载（约 4.8GB），请先到「资源管理 → 生成歌曲」下载。"
        : null;

  return (
    <div className="ca-overlay" onClick={onClose}>
      <div className="ca-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="ca-header">
          <div className="ca-title">🪄 创作助手</div>
          <div className="ca-subtitle">把常用的三步创作，收进一个面板里</div>
          <button className="ca-close" onClick={onClose}>✕</button>
        </div>

        <div className="ca-tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`ca-tab ${tab === t.id ? "active" : ""}`}
              onClick={() => switchTab(t.id)}
              disabled={busy}
            >
              {t.title}
            </button>
          ))}
        </div>
        <div className="ca-tab-desc">{activeTab.desc}</div>
        {modelWarning && <div className="ca-warn">⚠ {modelWarning}</div>}

        <div className="ca-body">
          {/* ── B 单乐器叠加 ── */}
          {tab === "instrument" && (
            <>
              <div className="ca-field">
                <label className="ca-label">源歌曲</label>
                <select className="ca-select" value={srcPath} onChange={(e) => setSrcPath(e.target.value)} disabled={busy}>
                  {audioOptions.length === 0 && <option value="">（歌曲列表暂无音频）</option>}
                  {audioOptions.map((s) => (
                    <option key={s.uuid} value={s.audioPath}>
                      {s.label || "Untitled"}（{s.durationSec}s{s.bpm ? ` · ${s.bpm}BPM` : ""}）
                    </option>
                  ))}
                </select>
              </div>
              <div className="ca-field">
                <label className="ca-label">添加乐器</label>
                <div className="ca-chips">
                  {INSTRUMENTS.map((i) => (
                    <button
                      key={i.value}
                      className={`ca-chip ${instrument === i.value ? "active" : ""}`}
                      onClick={() => setInstrument(i.value)}
                      disabled={busy}
                    >
                      {i.zh}
                    </button>
                  ))}
                </div>
              </div>
              <details className="ca-advanced">
                <summary>高级：风格补充（可选）</summary>
                <input
                  className="ca-input"
                  value={instStyle}
                  onChange={(e) => setInstStyle(e.target.value)}
                  placeholder="例如：warm jazz brush drums, laid back"
                  disabled={busy}
                />
              </details>
              <button className="ca-btn ca-btn-primary" onClick={generateInstrument} disabled={busy || !srcPath}>
                {busy ? "生成中…请耐心等待" : "生成并叠加"}
              </button>
            </>
          )}

          {/* ── C 乐谱续写 ── */}
          {tab === "score" && (
            <>
              <div className="ca-field">
                <label className="ca-label">从历史歌曲加载乐谱</label>
                <div className="ca-row">
                  <select
                    className="ca-select"
                    value={abcSongUuid}
                    onChange={(e) => setAbcSongUuid(e.target.value)}
                    disabled={busy}
                  >
                    <option value="">（选择带 ABC 的歌曲）</option>
                    {abcOptions.map((s) => (
                      <option key={s.uuid} value={s.uuid}>{s.label || "Untitled"}</option>
                    ))}
                  </select>
                  <button
                    className="ca-btn ca-btn-ghost"
                    onClick={() => loadAbcFromSong(abcSongUuid)}
                    disabled={busy || !abcSongUuid}
                  >
                    载入
                  </button>
                </div>
                {abcOptions.length === 0 && (
                  <div className="ca-hint">暂无带 ABC 的歌曲：先用 YuE-2 生成一首会附带 .abc 乐谱。</div>
                )}
              </div>
              <div className="ca-field">
                <label className="ca-label">ABC 乐谱（可直接粘贴编辑）</label>
                <textarea
                  className="ca-textarea ca-mono"
                  rows={5}
                  placeholder={"X:1\nM:4/4\nL:1/8\nK:C\n|:C2D2E2G2:|"}
                  value={abcText}
                  onChange={(e) => setAbcText(e.target.value)}
                  disabled={busy}
                />
              </div>
              <div className="ca-row">
                <div className="ca-field ca-grow">
                  <label className="ca-label">续写方向</label>
                  <input
                    className="ca-input"
                    value={contStyle}
                    onChange={(e) => setContStyle(e.target.value)}
                    placeholder="例如：以同样风格继续发展，加入副歌推进"
                    disabled={busy}
                  />
                </div>
                <div className="ca-field ca-field-narrow">
                  <label className="ca-label">候选数量</label>
                  <select
                    className="ca-select"
                    value={candidateCount}
                    onChange={(e) => setCandidateCount(Number(e.target.value))}
                    disabled={busy}
                  >
                    <option value={1}>1 个</option>
                    <option value={2}>2 个</option>
                    <option value={3}>3 个</option>
                  </select>
                </div>
              </div>
              <button className="ca-btn ca-btn-primary" onClick={generateCandidates} disabled={busy || !abcText.trim()}>
                {busy ? (progressText ?? "生成中…") : "生成候选"}
              </button>

              {candidates.length > 0 && (
                <div className="ca-candidates">
                  {candidates.map((c, idx) => (
                    <div key={c.id} className="ca-candidate">
                      <div className="ca-candidate-head">候选 {idx + 1}</div>
                      {c.audioPath && <audio controls src={convertFileSrc(c.audioPath)} className="ca-audio" />}
                      <div className="ca-candidate-actions">
                        <button className="ca-btn ca-btn-primary ca-btn-sm" onClick={() => adoptCandidate(c)}>
                          ✓ 采用并继续
                        </button>
                        <button
                          className="ca-btn ca-btn-ghost ca-btn-sm"
                          onClick={() => onSendToTrack(c.audioPath!, c.label, c.durationSec)}
                        >
                          ➕ 发送到轨道
                        </button>
                        {c.midiPath && onOpenMidi && (
                          <button
                            className="ca-btn ca-btn-ghost ca-btn-sm"
                            onClick={() => onOpenMidi(c.midiPath!, c.label)}
                          >
                            🎹 MIDI 编辑
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {/* ── D 翻唱助手 ── */}
          {tab === "cover" && (
            <>
              <div className="ca-field">
                <label className="ca-label">源歌曲</label>
                <select className="ca-select" value={srcPath} onChange={(e) => setSrcPath(e.target.value)} disabled={busy}>
                  {audioOptions.length === 0 && <option value="">（歌曲列表暂无音频）</option>}
                  {audioOptions.map((s) => (
                    <option key={s.uuid} value={s.audioPath}>
                      {s.label || "Untitled"}（{s.durationSec}s{s.bpm ? ` · ${s.bpm}BPM` : ""}）
                    </option>
                  ))}
                </select>
                {srcSong?.bpm && (
                  <div className="ca-hint">已自动记忆原曲 BPM：{srcSong.bpm}，翻唱将沿用该速度与曲式结构。</div>
                )}
              </div>

              <div className="ca-step">
                <span className="ca-step-no">①</span>
                <div className="ca-step-body">
                  <button className="ca-btn ca-btn-ghost" onClick={extractVocals} disabled={busy || !srcPath}>
                    {vocalPath ? "重新分离人声" : "分离人声"}
                  </button>
                  <label className="ca-check">
                    <input
                      type="checkbox"
                      checked={useVocalSource && !!vocalPath}
                      onChange={(e) => setUseVocalSource(e.target.checked)}
                      disabled={busy || !vocalPath}
                    />
                    用分离出的人声作为翻唱源（推荐）
                  </label>
                </div>
              </div>

              <div className="ca-step">
                <span className="ca-step-no">②</span>
                <div className="ca-step-body ca-grow">
                  <label className="ca-label">歌词（可修改）</label>
                  <textarea
                    className="ca-textarea"
                    rows={4}
                    value={coverLyrics}
                    onChange={(e) => setCoverLyrics(e.target.value)}
                    placeholder="留空则生成纯音乐；可粘贴或改写原曲歌词"
                    disabled={busy}
                  />
                </div>
              </div>

              <div className="ca-step">
                <span className="ca-step-no">③</span>
                <div className="ca-step-body ca-grow">
                  <label className="ca-label">目标风格（点击追加）</label>
                  <div className="ca-chips">
                    {COVER_STYLE_PRESETS.map((p) => (
                      <button
                        key={p.tag}
                        className="ca-chip"
                        onClick={() => setCoverPrompt((v) => (v.includes(p.tag) ? v : `${v ? v + ", " : ""}${p.tag}`))}
                        disabled={busy}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                  <textarea
                    className="ca-textarea"
                    rows={2}
                    value={coverPrompt}
                    onChange={(e) => setCoverPrompt(e.target.value)}
                    placeholder="例如：energetic rock, distorted guitar, powerful drums"
                    disabled={busy}
                  />
                </div>
              </div>

              <details className="ca-advanced">
                <summary>高级：风格转换强度 {coverStrength.toFixed(2)}</summary>
                <input
                  type="range" min={0.3} max={1} step={0.05}
                  value={coverStrength}
                  onChange={(e) => setCoverStrength(Number(e.target.value))}
                  disabled={busy}
                />
                <div className="ca-hint">1.0 = 完全按新风格重演绎；调低保留更多原曲音色。</div>
              </details>

              <div className="ca-step">
                <span className="ca-step-no">④</span>
                <div className="ca-step-body">
                  <button
                    className="ca-btn ca-btn-primary"
                    onClick={generateCover}
                    disabled={busy || !srcPath}
                  >
                    {busy ? "生成中…请耐心等待" : "生成翻唱"}
                  </button>
                </div>
              </div>
            </>
          )}

          {error && <div className="ca-error">{error}</div>}
          {status && <div className="ca-status">{status}</div>}

          {result && resultAudio && (
            <div className="ca-result">
              <div className="ca-result-title">✅ 生成完成：{result.label}</div>
              <audio controls src={convertFileSrc(resultAudio)} className="ca-audio" />
              <div className="ca-candidate-actions">
                <button className="ca-btn ca-btn-primary ca-btn-sm" onClick={() => onGenerated(result)}>
                  保存到歌曲列表
                </button>
                <button
                  className="ca-btn ca-btn-ghost ca-btn-sm"
                  onClick={() => onSendToTrack(resultAudio, result.label, result.durationSec)}
                >
                  发送到轨道
                </button>
                {resultMidi && onOpenMidi && (
                  <button className="ca-btn ca-btn-ghost ca-btn-sm" onClick={() => onOpenMidi(resultMidi, result.label)}>
                    🎹 MIDI 编辑
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
