import { useState, useCallback, useRef, useEffect } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useSoundfontStore } from "../../store/soundfont";
import { useHistoryStore } from "../../store/history";
import { useVoiceModelStore } from "../../store/voice-models";
import { useRecording } from "../../lib/audio/recorder";
import { VOCAL_LANGUAGES, langById } from "../../lib/vocal/languages";
import { VolumeFader, formatDb, formatPan } from "../common/VolumeFader";
import { OutLevelMeter } from "../common/OutLevelMeter";
import { updateTrackVolume, updateTrackPan, updateTrackAudibility } from "../../lib/audio/playback";
import { FADER_MIN_DB, FADER_MAX_DB } from "../../lib/constants";
import { trackTypeCssVar } from "../../lib/trackColors";
import "./TrackInspectorPanel.css";

/** Studio Pro 风格底部 docked Inspector 面板
 *  v3: 删掉顶部 AI 按钮组(与右键菜单/底部栏重复) 与假"输入路由"行;
 *      录音按钮放大醒目; 音源/人物声集中在顶部一行;
 *      主区 = 平衡旋钮 + 竖直音量推子(带 dB 刻度) + 实时电平表 + 插入槽位/发送;
 *      全局效果总线移入"🎛 控制台"(MixerConsole 全轨道混音台)。
 */
const PLUGIN_CATALOG = [
  { id: "eq",         name: "EQ 均衡器",   icon: "🎚️" },
  { id: "compressor", name: "压缩器",      icon: "📦" },
  { id: "reverb",     name: "混响",        icon: "🌊" },
  { id: "delay",      name: "延迟",        icon: "⏱️" },
  { id: "chorus",     name: "合唱",        icon: "🎭" },
  { id: "limiter",    name: "限制器",      icon: "🔒" },
];

const loadInspectorHeight = () => {
  try { return parseInt(localStorage.getItem("utai.inspectorHeight") || "280", 10) || 280; }
  catch { return 280; }
};

export function TrackInspectorPanel() {
  const tracks = useProjectStore((s) => s.tracks);
  const inspectorTrackId = useAppStore((s) => s.inspectorTrackId);
  const closeInspector = useAppStore((s) => s.closeInspector);
  const openInspector = useAppStore((s) => s.openInspector);
  const updateTrack = useProjectStore((s) => s.updateTrack);
  const setTrackFxSend = useProjectStore((s) => s.setTrackFxSend);
  const setTrackSoundfont = useProjectStore((s) => s.setTrackSoundfont);
  const refreshFonts = useSoundfontStore((s) => s.refresh);
  const fonts = useSoundfontStore((s) => s.fonts);
  // 录音态 + 歌手/语言 (从轨道头移入参数面板)
  const recording = useRecording((s) => (inspectorTrackId ? s.recordingTrackId === inspectorTrackId : false));
  const voiceModels = useVoiceModelStore((s) => s.models);
  const setVocalParams = useProjectStore((s) => s.setVocalParams);
  const [insertOpen, setInsertOpen] = useState(false);
  const [activeSlot, setActiveSlot] = useState(0); // -1 = 「添加新插槽」追加模式
  const [panelHeight, setPanelHeight] = useState(loadInspectorHeight);
  const dockRef = useRef<HTMLElement | null>(null);
  const handleRef = useRef<HTMLDivElement | null>(null);
  const targetHRef = useRef(panelHeight);
  const rafRef = useRef<number | null>(null);
  const draggingRef = useRef(false);
  const startYRef = useRef(0);
  const startHRef = useRef(0);
  const activePointerIdRef = useRef<number | null>(null);

  // Keep DOM in sync with ref during normal (non-drag) renders
  useEffect(() => { targetHRef.current = panelHeight; }, [panelHeight]);

  // Lazily refresh soundfonts ONCE when they're empty. Must be in useEffect, NOT render body —
  // calling zustand setters during render triggers React's "Cannot update while rendering" error.
  useEffect(() => {
    if (fonts.length === 0) refreshFonts().catch(() => {});
  }, [fonts.length, refreshFonts]);

  const track = tracks.find((t) => t.id === inspectorTrackId) ?? null;
  if (!track) return null;

  const currentFont = track.soundfont
    ? fonts.find((f) => f.id === track.soundfont?.fontId)
    : null;

  // 插件槽位数据：从 Track.pluginSlots 读（动态数量）。
  // 旧工程存的是 6 个 null 的占位数组, 这里过滤成纯字符串列表 → 槽位随插入自动增长。
  const plugins = (track.pluginSlots ?? []).filter((s): s is string => typeof s === "string");

  const updatePlugins = (slots: string[]) => {
    useHistoryStore.getState().beginTransaction();
    updateTrack(track.id, { pluginSlots: slots });
    useHistoryStore.getState().commitTransaction();
  };

  // activeSlot >= 0 → 替换该槽位; activeSlot === -1 → 追加一个新插槽
  const onInsertPlugin = (pluginId: string) => {
    const next = [...plugins];
    if (activeSlot >= 0) next[activeSlot] = pluginId;
    else next.push(pluginId);
    updatePlugins(next);
    setInsertOpen(false);
  };

  // 动态槽位: 清除 = 直接删掉这一格 (后面的插槽前移)
  const onClearSlot = (slotIdx: number) => {
    const next = [...plugins];
    next.splice(slotIdx, 1);
    updatePlugins(next);
  };

  // 音量/平衡: store + 播放引擎双写 (与轨道头推子完全一致, 播放中拖动即时生效)
  // 历史事务由 VolumeFader 的 onGestureStart/End 统一包裹, 这里不再嵌套 begin/commit
  // (旧代码每次 mousemove 各开一层事务, 与手势外层事务嵌套错配, 撤销历史被刷屏)
  const updateVolume = useCallback((v: number) => {
    updateTrack(track.id, { volumeDb: v });
    updateTrackVolume(track.id, v);
  }, [track.id, updateTrack]);

  const updatePan = useCallback((v: number) => {
    updateTrack(track.id, { pan: v });
    updateTrackPan(track.id, v);
  }, [track.id, updateTrack]);

  const toggleMute = () => {
    useHistoryStore.getState().beginTransaction();
    updateTrack(track.id, { muted: !track.muted });
    useHistoryStore.getState().commitTransaction();
    // 播放中立即反映静音状态 (与轨道头 M/S 按钮一致)
    updateTrackAudibility(useProjectStore.getState().tracks);
  };
  const toggleSolo = () => {
    useHistoryStore.getState().beginTransaction();
    updateTrack(track.id, { solo: !track.solo });
    useHistoryStore.getState().commitTransaction();
    updateTrackAudibility(useProjectStore.getState().tracks);
  };

  const onFontChange = (fontId: string) => {
    const font = fonts.find((f) => f.id === fontId);
    if (!font) { setTrackSoundfont(track.id, undefined); return; }
    const firstPreset = font.presets[0];
    useHistoryStore.getState().beginTransaction();
    setTrackSoundfont(track.id, {
      fontId: font.id,
      presetId: firstPreset?.id ?? "0:0",
      presetName: firstPreset?.name ?? font.name,
    });
    useHistoryStore.getState().commitTransaction();
  };

  // 旋钮指针角度: pan∈[-1,1] → 指针从左(-90°)经上(0°=居中)到右(+90°)。
  // 旧公式 -90*(1-pan) 把范围压在 -180°..0°, 居中时指左、最右时朝上, 视觉完全错位。
  const panDeg = track.pan * 90;

  // 平衡旋钮: 按住左右拖动调节 (旧版只有一个点击回中, 鼠标拖不动)
  // 灵敏度 1px = 0.01 pan; 单击不动 = 回正中; 拖动期间只开一次撤销事务
  const onPanPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const startX = e.clientX;
    const startPan = track.pan;
    let moved = false;
    useHistoryStore.getState().beginTransaction();
    const onMove = (ev: PointerEvent) => {
      const dx = ev.clientX - startX;
      if (!moved && Math.abs(dx) < 3) return;
      moved = true;
      const next = Math.max(-1, Math.min(1, Math.round((startPan + dx / 100) * 100) / 100));
      updatePan(next);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      if (!moved) updatePan(0); // 单击 = 回到正中 (保留原交互)
      useHistoryStore.getState().commitTransaction();
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, [track.pan, updatePan]);

  // ── Height resize drag — Pointer Events + rAF direct DOM write ──
  // 关键点: mousemove 只更新 ref, rAF 里直接写 dock.style.height,
  // 全程不触发 React re-render, 只在 pointerup 时 commit 到 state.
  const flushRaf = useCallback(() => {
    if (!rafRef.current) return;
    rafRef.current = null;
    const dock = dockRef.current;
    if (dock) dock.style.height = `${targetHRef.current}px`;
  }, []);

  const onResizePointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const el = handleRef.current;
    if (el && el.setPointerCapture) {
      try { el.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    }
    activePointerIdRef.current = e.pointerId;
    draggingRef.current = true;
    startYRef.current = e.clientY;
    startHRef.current = targetHRef.current;
    document.body.style.cursor = "ns-resize";
    document.body.style.userSelect = "none";
    document.body.style.touchAction = "none";
    el?.classList.add("is-dragging");

    const onMove = (ev: PointerEvent) => {
      if (!draggingRef.current) return;
      if (activePointerIdRef.current !== null && ev.pointerId !== activePointerIdRef.current) return;
      const dy = startYRef.current - ev.clientY;
      const next = Math.max(180, Math.min(560, startHRef.current + dy));
      targetHRef.current = next;
      if (!rafRef.current) rafRef.current = requestAnimationFrame(flushRaf);
    };

    const finish = () => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      activePointerIdRef.current = null;
      // Cancel any pending rAF and commit last value
      if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
      flushRaf();
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.body.style.touchAction = "";
      handleRef.current?.classList.remove("is-dragging");
      // Persist to localStorage + commit state (only ONCE at end)
      const final = Math.round(targetHRef.current);
      localStorage.setItem("utai.inspectorHeight", String(final));
      setPanelHeight((prev) => (prev !== final ? final : prev));
    };

    const onUp = (ev: PointerEvent) => {
      if (activePointerIdRef.current !== null && ev.pointerId !== activePointerIdRef.current) return;
      const el2 = handleRef.current;
      if (el2 && el2.releasePointerCapture) {
        try { el2.releasePointerCapture(ev.pointerId); } catch { /* ignore */ }
      }
      finish();
    };

    const onCancel = () => finish();

    // pointer capture ensures we get move/up even when pointer leaves the handle
    el?.addEventListener("pointermove", onMove);
    el?.addEventListener("pointerup", onUp);
    el?.addEventListener("pointercancel", onCancel);
    // Also bind on window as safety net (capture already handles it, but belt-and-suspenders)
    window.addEventListener("pointerup", finish);
    window.addEventListener("blur", finish);

    // Cleanup is tied to pointerup/cancel — but also protect against unmount mid-drag
    (handleRef.current as any & { __cleanupResize?: () => void }).__cleanupResize = () => {
      el?.removeEventListener("pointermove", onMove);
      el?.removeEventListener("pointerup", onUp);
      el?.removeEventListener("pointercancel", onCancel);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("blur", finish);
    };
  }, [flushRaf]);

  // Close insert popover when clicking outside
  useEffect(() => {
    if (!insertOpen) return;
    const onDocClick = () => setInsertOpen(false);
    const t = setTimeout(() => document.addEventListener("click", onDocClick), 0);
    return () => { clearTimeout(t); document.removeEventListener("click", onDocClick); };
  }, [insertOpen]);

  return (
    <aside
      ref={dockRef}
      className="inspector-dock"
      role="dialog"
      aria-label="轨道属性"
      style={{ height: panelHeight }}
    >
      {/* ── RESIZE HANDLE (top edge) — Pointer Events + setPointerCapture ── */}
      <div
        ref={handleRef}
        className="inspector-resize-handle"
        onPointerDown={onResizePointerDown}
        title="拖动调整高度"
      />

      {/* ── HEADER — 只留关闭 + 轨道名 (静音/独奏/音量数字移到录音行, 控制台入口移到主页底栏) ── */}
      <div className="inspector-header">
        <button className="inspector-close" onClick={closeInspector} title="关闭">✕</button>
        <span className="inspector-track-name" title={track.name}>{track.name}</span>
        <div className="inspector-header-spacer" />
      </div>

      {/* ── 录音(大按钮) + 静音/独奏(同尺寸) + 音量数字 + 歌手/语言/音源 ── */}
      <div className="inspector-rec-row">
        <button
          className={`inspector-rec-btn inspector-rec-big ${recording ? "recording" : ""}`}
          onClick={() => { void useRecording.getState().toggle(track.id); }}
          title={recording ? "停止录音" : "开始在此轨道录音"}
        >
          {recording ? "■ 停止录音" : "● 录音"}
        </button>
        <button
          className={`inspector-state-btn inspector-state-big ${track.muted ? "active-mute" : ""}`}
          onClick={toggleMute}
          title="静音这条轨道"
        >静音</button>
        <button
          className={`inspector-state-btn inspector-state-big ${track.solo ? "active-solo" : ""}`}
          onClick={toggleSolo}
          title="独奏 — 只听这条轨道"
        >独奏</button>
        <span className="inspector-db-readout" title="当前轨道音量">{formatDb(track.volumeDb, FADER_MIN_DB)}</span>
        {track.trackType === "vocal" && (() => {
          const singers = [...voiceModels.sovits, ...voiceModels.rvc];
          const currentSinger = singers.find((s) => s.name === track.voiceModel) ?? null;
          return (
            <>
              <span className="inspector-field-label">🎤 人物声音</span>
              <select
                className="inspector-input-select"
                value={track.voiceModel ?? ""}
                onChange={(e) => {
                  const m = singers.find((s) => s.name === e.target.value);
                  useHistoryStore.getState().beginTransaction();
                  updateTrack(track.id, {
                    voiceModel: m?.name,
                    voiceModelAvatar: m?.avatar_path ? convertFileSrc(m.avatar_path) : undefined,
                  });
                  useHistoryStore.getState().commitTransaction();
                }}
                title="选择歌手模型 (翻唱人物卡)"
              >
                <option value="">🎤 选择歌手…</option>
                {singers.map((m) => (
                  <option key={`${m.model_type}-${m.path}`} value={m.name}>{m.name}</option>
                ))}
              </select>
              {currentSinger?.avatar_path && (
                <img
                  className="inspector-singer-avatar"
                  src={convertFileSrc(currentSinger.avatar_path)}
                  alt={currentSinger.name}
                  title={currentSinger.name}
                />
              )}
              <span className="inspector-field-label">🌐 语言</span>
              <select
                className="inspector-input-select inspector-lang-select"
                value={track.vocalParams?.langId ?? 0}
                onChange={(e) => setVocalParams(track.id, { langId: +e.target.value })}
                title={`发音语言: ${langById(track.vocalParams?.langId ?? 0).code.toUpperCase()}`}
              >
                {VOCAL_LANGUAGES.map((l) => (
                  <option key={l.id} value={l.id}>{l.short} — {l.code}</option>
                ))}
              </select>
            </>
          );
        })()}
        {track.trackType === "instrument" && (
          <>
            <span className="inspector-field-label">🎹 音源</span>
            <select
              className="inspector-input-select"
              value={track.soundfont?.fontId ?? ""}
              onChange={(e) => onFontChange(e.target.value)}
              title="替换这条 MIDI 轨的乐器音源"
            >
              <option value="">— 默认合成 —</option>
              {fonts.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.name} ({f.format.toUpperCase()})
                </option>
              ))}
            </select>
            {currentFont && currentFont.presets.length > 1 && (
              <select
                className="inspector-input-select"
                value={track.soundfont?.presetId ?? ""}
                onChange={(e) => {
                  if (!track.soundfont) return;
                  const preset = currentFont.presets.find((p) => p.id === e.target.value);
                  useHistoryStore.getState().beginTransaction();
                  setTrackSoundfont(track.id, {
                    ...track.soundfont,
                    presetId: e.target.value,
                    presetName: preset?.name ?? e.target.value,
                  });
                  useHistoryStore.getState().commitTransaction();
                }}
                title="音色 (音源内的预设)"
              >
                {currentFont.presets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name || p.id}</option>
                ))}
              </select>
            )}
            {/* 3-9 渲染引擎(SF2 专属):内置合成器 vs FluidSynth CLI 对照组;换音源自动重置。 */}
            {currentFont?.format === "sf2" && track.soundfont && (
              <select
                className="inspector-input-select"
                value={track.soundfont.backend === "fluidsynth" ? "fluidsynth" : "builtin"}
                onChange={(e) => {
                  const cur = track.soundfont;
                  if (!cur) return;
                  useHistoryStore.getState().beginTransaction();
                  setTrackSoundfont(track.id, {
                    fontId: cur.fontId,
                    presetId: cur.presetId,
                    presetName: cur.presetName,
                    ...(e.target.value === "fluidsynth" ? { backend: "fluidsynth" as const } : {}),
                  });
                  useHistoryStore.getState().commitTransaction();
                }}
                title="渲染引擎(仅 SF2 音源可选)"
              >
                <option value="builtin">⚙ 内置合成器</option>
                <option value="fluidsynth">🎛 FluidSynth</option>
              </select>
            )}
          </>
        )}
      </div>

      {/* ── MAIN AREA — 通道条: 平衡 | 音量推子(dB刻度) | 实时电平 | 插入+发送 ── */}
      <div className="inspector-main">
        {/* PAN — 左右声道平衡, 按住拖动 / 点击回中 */}
        <div className="inspector-pan-group">
          <span className="inspector-pan-label">平衡 (左·右)</span>
          <div
            className="inspector-pan-knob"
            onPointerDown={onPanPointerDown}
            title="左右声道平衡 — 按住左右拖动, 单击回到正中"
          >
            <div
              className="inspector-pan-indicator"
              style={{ transform: `translateX(-50%) rotate(${panDeg}deg)` }}
            />
          </div>
          <span className="inspector-pan-value">{formatPan(track.pan)}</span>
        </div>

        {/* VOLUME — 竖直推子 + dB 刻度 + 分轨实时电平表 (电平与音量融合在同一推子台) */}
        <div className="inspector-volume-group">
          <span className="inspector-volume-label">音量</span>
          <div className="inspector-volume-faderbox">
            <div className="inspector-fader-scale" aria-hidden>
              {[FADER_MAX_DB, 0, -12, FADER_MIN_DB].map((db) => {
                const r = (db - FADER_MIN_DB) / (FADER_MAX_DB - FADER_MIN_DB);
                // 钳制到盒内 — 旧版 +6 标签会越过推子台顶部, 飘出一个小方块
                const clamped = Math.min(0.94, Math.max(0.03, r));
                return <span key={db} style={{ bottom: `${clamped * 100}%` }}>{db > 0 ? `+${db}` : db}</span>;
              })}
            </div>
            <VolumeFader
              value={track.volumeDb}
              min={FADER_MIN_DB}
              max={FADER_MAX_DB}
              orientation="vertical"
              width={28}
              height={150}
              onChange={updateVolume}
              onGestureStart={() => useHistoryStore.getState().beginTransaction()}
              onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
              tip="轨道音量 (上下拖动)"
            />
            <OutLevelMeter trackId={track.id} width={14} height={150} />
          </div>
        </div>

        {/* RIGHT SIDE: 槽1固定音源槽 + 动态效果器插槽 (随添加增长) */}
        <div className="inspector-plugins-area" onClick={(e) => e.stopPropagation()}>
          <div className="inspector-plugin-insert-row">
            {/* 槽 1 — 固定音源插槽: 显示资源管理下载的音源, MIDI 轨播放时输出该音源的声音 */}
            <div
              className="inspector-plugin-slot inspector-font-slot"
              title="固定音源插槽 — 在上方「音源」下拉里更换; MIDI 轨播放时输出这个音源的声音"
            >
              {track.soundfont
                ? <span>🎹 {currentFont?.name ?? "已选音源"}</span>
                : <span>🎹 音源 · 默认合成</span>}
            </div>
            {plugins.map((pluginId, i) => {
              const p = PLUGIN_CATALOG.find((pc) => pc.id === pluginId);
              return (
                <div
                  key={i}
                  className={`inspector-plugin-slot ${activeSlot === i ? "inspector-plugin-slot-active" : ""}`}
                  onClick={() => { setActiveSlot(i); setInsertOpen(true); }}
                  title={p ? `${p.name} — 点击更换/移除` : `效果器插槽 ${i + 1}（空）— 点击插入效果器`}
                >
                  {p ? (
                    <span>
                      {p.icon} {p.name}
                      <button
                        className="inspector-plugin-clear"
                        onClick={(e) => { e.stopPropagation(); onClearSlot(i); }}
                        title="移除"
                      >×</button>
                    </span>
                  ) : (
                    <span style={{ color: "var(--text-tertiary, #666)" }}>＋ 空槽 {i + 1}</span>
                  )}
                </div>
              );
            })}
            <div
              className="inspector-plugin-slot inspector-plugin-add"
              onClick={() => { setActiveSlot(-1); setInsertOpen(true); }}
              title="添加一个新效果器插槽"
            >
              <span>＋ 添加效果器</span>
            </div>
          </div>

          {/* Insert popover menu — 点已有槽位 = 替换, 点「＋ 添加效果器」= 追加新插槽 */}
          {insertOpen && (
            <div className="inspector-insert-menu">
              <div className="inspector-insert-menu-title">
                {activeSlot >= 0
                  ? `替换插槽 ${activeSlot + 1}（${plugins[activeSlot] ? PLUGIN_CATALOG.find(p => p.id === plugins[activeSlot])?.name : "空"}）`
                  : "添加新效果器插槽"}
              </div>
              {PLUGIN_CATALOG.map((p) => (
                <button
                  key={p.id}
                  className="inspector-insert-item"
                  onClick={() => onInsertPlugin(p.id)}
                >
                  <span className="inspector-insert-item-icon">{p.icon}</span>
                  <span>{p.name}</span>
                </button>
              ))}
              {plugins[activeSlot] && (
                <button
                  className="inspector-insert-item inspector-insert-clear"
                  onClick={() => onClearSlot(activeSlot)}
                >
                  🗑️ 清除此槽位
                </button>
              )}
            </div>
          )}

          {/* ── SENDS: 混响 / 延迟 发送量 — 竖直小推子 (与音量推子同操作方式) ── */}
          <div className="inspector-sends-row">
            <span className="inspector-sends-label">🎛 发送</span>
            <div className="inspector-send-vert" title="混响发送 — 发到全局混响总线的量 (上下拖动)">
              <VolumeFader
                value={track.reverbSend ?? 0}
                min={0}
                max={1}
                orientation="vertical"
                width={22}
                height={84}
                step={0.01}
                onChange={(v) => setTrackFxSend(track.id, "reverbSend", v)}
                onGestureStart={() => useHistoryStore.getState().beginTransaction()}
                onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
                tip="混响发送 (上下拖动)"
              />
              <span className="inspector-send-vert-name">🌊 {Math.round((track.reverbSend ?? 0) * 100)}%</span>
            </div>
            <div className="inspector-send-vert" title="延迟发送 — 发到全局延迟总线的量 (上下拖动)">
              <VolumeFader
                value={track.delaySend ?? 0}
                min={0}
                max={1}
                orientation="vertical"
                width={22}
                height={84}
                step={0.01}
                onChange={(v) => setTrackFxSend(track.id, "delaySend", v)}
                onGestureStart={() => useHistoryStore.getState().beginTransaction()}
                onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
                tip="延迟发送 (上下拖动)"
              />
              <span className="inspector-send-vert-name">⏱ {Math.round((track.delaySend ?? 0) * 100)}%</span>
            </div>
            <button
              className="inspector-sends-reset"
              onClick={() => {
                useHistoryStore.getState().beginTransaction();
                setTrackFxSend(track.id, "reverbSend", 0);
                setTrackFxSend(track.id, "delaySend", 0);
                useHistoryStore.getState().commitTransaction();
              }}
              title="清零所有发送"
            >× 清零</button>
          </div>
        </div>
      </div>

      {/* ── TRACK SWITCH TABS — 动态: 有几条轨道就显示几个, 颜色跟随轨道类型 ── */}
      <div className="inspector-track-tabs">
        {tracks.map((t, i) => {
          const color = trackTypeCssVar(t.trackType); // 统一走 trackColors（与 TrackList/主题 --track-* 一致）
          const isActive = t.id === inspectorTrackId;
          return (
            <button
              key={t.id}
              className={`inspector-track-tab ${isActive ? "active" : ""}`}
              onClick={() => openInspector(t.id)}
              style={{
                borderColor: isActive ? color : "transparent",
                borderLeftColor: isActive ? color : color,
                borderLeftWidth: isActive ? "4px" : "3px",
              }}
              title={t.name}
            >
              <span style={{ color, fontWeight: 700, marginRight: 4 }}>{i + 1}</span>
              {t.name}
            </button>
          );
        })}
        {tracks.length === 0 && (
          <span style={{ color: "var(--text-tertiary, #666)", fontSize: 11, padding: "4px 10px" }}>
            工程里还没有轨道
          </span>
        )}
      </div>
    </aside>
  );
}
