import { useRef, useCallback, useEffect } from "react";
import { Toolbar } from "./Toolbar";
import { TrackList } from "./TrackList";
import { TimelineRuler } from "./TimelineRuler";
import { Arrangement } from "./Arrangement";
import { HScrollbar } from "./HScrollbar";
import { useAppStore } from "../../store/app";
import { useProjectStore, useTimeAxis } from "../../store/project";
import { PIXELS_PER_TICK, TRACK_ADD_FOOTER } from "../../lib/constants";
import { saveSetting } from "../../lib/settings";
import { computeTotalTracksHeight, computeTotalTicks } from "../../lib/trackLayout";
import { runAutoArrange, runChordMidi, resolveMelodyTrackId } from "../../lib/arrangement/autoArrange";
import { blankTrack } from "../../lib/trackFactory";
import { useTranslation } from "react-i18next";
import "./DawView.css";

export function DawView() {
  const { t } = useTranslation();
  // scrollX/scrollY are intentionally NOT subscribed here — horizontal/vertical scroll must not
  // re-render the whole DAW subtree. The canvases (Arrangement/TimelineRuler) self-subscribe and
  // repaint imperatively; HScrollbar/TrackList self-subscribe their own scroll value. vZoom is
  // likewise NOT subscribed (zoom gestures would re-render the subtree); it's read via getState.
  const zoom = useAppStore((s) => s.zoom);
  const canvasWidth = useAppStore((s) => s.canvasWidth);
  const canvasHeight = useAppStore((s) => s.canvasHeight);
  const setCanvasWidth = useAppStore((s) => s.setCanvasWidth);
  const setCanvasHeight = useAppStore((s) => s.setCanvasHeight);
  // 轨头列宽:拖拽分隔条时 rAF 合帧写 store(与画布 ResizeObserver 的重渲染节奏一致),
  // pointerup 才持久化 localStorage。
  const headerWidth = useAppStore((s) => s.trackHeaderWidth);
  const tracks = useProjectStore((s) => s.tracks);
  const addTrack = useProjectStore((s) => s.addTrack);
  const timeAxis = useTimeAxis();
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  // Vertical-zoom wheel events are coalesced into one setVZoom per frame (high-res wheels fire many
  // per frame; applying each = many re-renders/redraws = sluggish).
  const vZoomFactorRef = useRef(1);
  const vZoomRafRef = useRef(0);
  useEffect(() => () => cancelAnimationFrame(vZoomRafRef.current), []);

  // ── 轨头列宽拖拽(横向调整):pointer capture + rAF 合帧 ──────────────────────
  const headerDragRef = useRef<{ startX: number; startW: number } | null>(null);
  const headerWDragRafRef = useRef(0);
  useEffect(() => () => cancelAnimationFrame(headerWDragRafRef.current), []);
  const onHeaderDividerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    const st = useAppStore.getState();
    headerDragRef.current = { startX: e.clientX, startW: st.trackHeaderWidth };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    const onMove = (ev: PointerEvent) => {
      const d = headerDragRef.current;
      if (!d) return;
      const next = d.startW + (ev.clientX - d.startX);
      if (!headerWDragRafRef.current) {
        headerWDragRafRef.current = requestAnimationFrame(() => {
          headerWDragRafRef.current = 0;
          useAppStore.getState().setTrackHeaderWidth(next);
        });
      }
    };
    const onUp = () => {
      headerDragRef.current = null;
      if (headerWDragRafRef.current) { cancelAnimationFrame(headerWDragRafRef.current); headerWDragRafRef.current = 0; }
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      // 拖完一次性持久化最终值
      saveSetting("utai.trackHeaderWidth", useAppStore.getState().trackHeaderWidth);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, []);

  useEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setCanvasWidth(entry.contentRect.width);
        setCanvasHeight(entry.contentRect.height);
      }
    });
    observer.observe(el);
    setCanvasWidth(el.clientWidth);
    setCanvasHeight(el.clientHeight);
    return () => observer.disconnect();
  }, [setCanvasWidth, setCanvasHeight]);

  // Keep scrollY within content bounds: when the track stack is shorter than the viewport (few
  // tracks, or vertical-zoom shrank it), don't allow scrolling past it. vZoom read via getState
  // (its own change sites clamp); this covers track add/remove + viewport resize.
  useEffect(() => {
    const st = useAppStore.getState();
    const maxY = Math.max(0, computeTotalTracksHeight(tracks, st.vZoom) + TRACK_ADD_FOOTER - canvasHeight);
    if (st.scrollY > maxY) st.setScroll(st.scrollX, maxY);
  }, [tracks, canvasHeight]);

  // ── AI 全局快捷键 ───────────────────────────────────────────────
  // Ctrl+Shift+D → AI 自动编曲 (默认流行+中性情绪, 可 Ctrl+Z 撤销)
  // Ctrl+Shift+C → AI 生成和弦 (默认参数)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // 只在 timeline pane 激活时响应 (避免在其他 dialog/workflow 里误触)
      if (useAppStore.getState().activePane !== "timeline") return;
      // 输入框/文本框/select 里不响应
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (!(e.ctrlKey && e.shiftKey)) return;

      if (e.key.toLowerCase() === "d") {
        e.preventDefault();
        const tid = resolveMelodyTrackId();
        if (!tid) {
          useAppStore.getState().showToast("选一条旋律/乐器轨先 (或右键那条轨)", "info");
          return;
        }
        runAutoArrange(tid, "pop", "neutral").then((ok) => {
          if (ok) useAppStore.getState().showToast("✅ 已生成默认伴奏 (Ctrl+Z 撤销, Ctrl+Shift+D 再次生成, Ctrl+Shift+A 打开编曲面板)", "success");
        });
      } else if (e.key.toLowerCase() === "c") {
        e.preventDefault();
        const tid = resolveMelodyTrackId();
        if (!tid) {
          useAppStore.getState().showToast("选一条旋律/乐器轨先 (或右键那条轨)", "info");
          return;
        }
        runChordMidi(tid, { chordsPerBar: 1, style: "POP_STANDARD", key: "auto" }).then((ok) => {
          if (ok) useAppStore.getState().showToast("✅ 和弦轨已生成 (Ctrl+Z 撤销)", "success");
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const totalWidth = computeTotalTicks(tracks, timeAxis) * PIXELS_PER_TICK * zoom;

  // Wheel over the track-header column: Alt or Ctrl → vertical (track-height) zoom; plain →
  // vertical scroll, clamped so a partially-filled column can't scroll its content out of view.
  const handleTrackListWheel = useCallback((e: React.WheelEvent) => {
    e.stopPropagation();
    const st = useAppStore.getState();
    if (e.altKey || e.ctrlKey) {
      e.preventDefault(); // stop the browser's Ctrl+wheel page-zoom / Alt+wheel default
      vZoomFactorRef.current *= e.deltaY > 0 ? 0.9 : 1.1;
      if (!vZoomRafRef.current) {
        vZoomRafRef.current = requestAnimationFrame(() => {
          vZoomRafRef.current = 0;
          const s = useAppStore.getState();
          s.setVZoom(s.vZoom * vZoomFactorRef.current);
          vZoomFactorRef.current = 1;
          const maxY = Math.max(0, computeTotalTracksHeight(useProjectStore.getState().tracks, useAppStore.getState().vZoom) + TRACK_ADD_FOOTER - s.canvasHeight);
          s.setScroll(s.scrollX, Math.min(s.scrollY, maxY));
        });
      }
      return;
    }
    const maxY = Math.max(0, computeTotalTracksHeight(useProjectStore.getState().tracks, st.vZoom) + TRACK_ADD_FOOTER - st.canvasHeight);
    st.setScroll(st.scrollX, Math.max(0, Math.min(maxY, st.scrollY + e.deltaY)));
  }, []);

  return (
    <div
      className="daw-view"
      onPointerDownCapture={() => useAppStore.getState().setActivePane("timeline")}
    >
      <Toolbar />
      <div className="daw-grid">
        <div className="daw-corner" style={{ width: headerWidth }}>
          <button 
            className="corner-btn" 
            onClick={() => addTrack(blankTrack(crypto.randomUUID(), `Audio ${useProjectStore.getState().tracks.length + 1}`, "audio"))}
            title={t("toolbar.addTrack")}
          >
            +
          </button>
          <button 
            className="corner-btn" 
            onClick={async () => {
              const { open } = await import("@tauri-apps/plugin-dialog");
              const file = await open({
                multiple: false,
                filters: [{ name: "Audio", extensions: ["mp3", "wav", "ogg", "flac", "m4a", "aac"] }]
              });
              if (file) {
                const { importAudioToNewTrack } = await import("../../lib/audio/import");
                await importAudioToNewTrack(file as string, useProjectStore.getState().playheadTick);
              }
            }}
            title="上传音频文件"
          >
            🎵
          </button>
          <button 
            className="corner-btn" 
            onClick={() => {
              const track = blankTrack(crypto.randomUUID(), `Instrument ${useProjectStore.getState().tracks.length + 1}`, "instrument");
              addTrack(track);
            }}
            title="新建MIDI轨道"
          >
            D
          </button>
        </div>
        <TimelineRuler />
        <div className="daw-tracklist-wrap" onWheel={handleTrackListWheel}>
          <TrackList width={headerWidth} />
        </div>
        {/* 轨头列 ↔ 画布 的竖向分隔条:拖拽调轨头列宽(Studio Pro 式) */}
        <div className="daw-header-divider" onPointerDown={onHeaderDividerDown} title="拖拽调整轨道面板宽度" />
        <div className="daw-canvas-container" ref={canvasContainerRef}>
          <Arrangement />
        </div>
        <div className="daw-scrollbar-corner" style={{ width: headerWidth }} />
        <HScrollbar
          totalWidth={totalWidth}
          viewWidth={canvasWidth}
          onChange={(x) => { const st = useAppStore.getState(); st.setScroll(x, st.scrollY); }}
        />
      </div>
    </div>
  );
}
