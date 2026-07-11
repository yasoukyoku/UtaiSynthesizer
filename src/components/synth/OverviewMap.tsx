import { useRef, useEffect, useCallback, useState } from "react";
import { useProjectStore, useTimeAxis } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { PIXELS_PER_TICK } from "../../lib/constants";
import { ticksToMs } from "../../lib/audio/laneOps";
import { computeTotalTicks, segmentPlaysLanes, segmentLaneSumPeaks, laneSumSig } from "../../lib/trackLayout";
import { getWaveformCache, blitWaveform } from "../../lib/waveformCache";
import { rgba, ACCENT_RGB, TRACK_RGB } from "../../lib/trackColors";
import { drawPlayhead, CANVAS_BORDER } from "../../lib/canvasDraw";
import "./OverviewMap.css";

export function OverviewMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // The minimap repaints on scroll/playback to move its viewport rect + playhead. The WAVEFORM is the
  // only expensive part and changes only at known times (segments move/add/remove, peaks load, a track's
  // mute/solo flips), so it's cached in an offscreen and rebuilt only when its content key changes — every
  // scroll/playback frame just blits the cache and draws the cheap overlay (viewport + playhead).
  const tracks = useProjectStore((s) => s.tracks);
  const timeAxis = useTimeAxis();
  const tempo = useProjectStore((s) => s.tempo);
  const playheadTick = useProjectStore((s) => s.playheadTick);
  const setPlayhead = useProjectStore((s) => s.setPlayhead);
  const scrollX = useAppStore((s) => s.scrollX);
  const zoom = useAppStore((s) => s.zoom);
  const canvasWidth = useAppStore((s) => s.canvasWidth);
  const audioFiles = useAudioStore((s) => s.audioFiles);

  // SAME basis as DawView's scroll width, so the viewport box + drag map 1:1 to the real scroll range
  // (a smaller minimap range made the box fill the map on short projects → seek/scroll both dead).
  const totalTicks = computeTotalTicks(tracks, timeAxis);

  const waveRef = useRef<OffscreenCanvas | null>(null);
  const waveKeyRef = useRef("");
  const [cursor, setCursor] = useState("pointer");

  // Drag state: "viewport" = scrubbing the window box (scroll), "playhead" = moving the playhead.
  const dragRef = useRef<{ mode: "viewport" | "playhead"; offsetX: number; viewW: number; ppt: number } | null>(null);

  // Where the viewport box sits, in minimap CSS pixels (depends on scroll/zoom/canvas size).
  const viewportRect = useCallback(
    (viewW: number) => {
      const ppt = PIXELS_PER_TICK * zoom;
      const startX = (scrollX / ppt / totalTicks) * viewW;
      const endX = ((scrollX + canvasWidth) / ppt / totalTicks) * viewW;
      return { ppt, startX, endX };
    },
    [scrollX, zoom, canvasWidth, totalTicks],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const localX = e.clientX - rect.left;
      const { ppt, startX, endX } = viewportRect(rect.width);

      if (localX >= startX && localX <= endX) {
        // Grab the window box → drag to scroll (AU-style).
        dragRef.current = { mode: "viewport", offsetX: localX - startX, viewW: rect.width, ppt };
        setCursor("grabbing");
      } else {
        // Seek the playhead (works during playback via the seeking flag → Toolbar reschedules).
        dragRef.current = { mode: "playhead", offsetX: 0, viewW: rect.width, ppt };
        setPlayhead(Math.max(0, Math.round((localX / rect.width) * totalTicks)));
        if (useAudioStore.getState().isPlaying) useAudioStore.getState().setSeeking(true);
      }
    },
    [viewportRect, totalTicks, setPlayhead],
  );

  // Hover cursor: "grab" over the window box (draggable), "pointer" elsewhere (click to seek).
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (dragRef.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const localX = e.clientX - rect.left;
      const { startX, endX } = viewportRect(rect.width);
      setCursor(localX >= startX && localX <= endX ? "grab" : "pointer");
    },
    [viewportRect],
  );

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = dragRef.current;
      if (!d) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const localX = e.clientX - canvas.getBoundingClientRect().left;
      if (d.mode === "playhead") {
        // If playback was started mid-drag (Space), pin the rAF advance so it doesn't fight the drag
        // and so mouseup reschedules from the new position.
        if (useAudioStore.getState().isPlaying && !useAudioStore.getState().seeking) {
          useAudioStore.getState().setSeeking(true);
        }
        setPlayhead(Math.max(0, Math.round((localX / d.viewW) * totalTicks)));
      } else {
        const newStartX = localX - d.offsetX;
        const maxScrollX = Math.max(0, totalTicks * d.ppt - useAppStore.getState().canvasWidth);
        const newScrollX = Math.max(0, Math.min(maxScrollX, (newStartX / d.viewW) * totalTicks * d.ppt));
        useAppStore.getState().setScroll(newScrollX, useAppStore.getState().scrollY);
      }
    };
    const onUp = () => {
      const d = dragRef.current;
      if (!d) return;
      dragRef.current = null;
      // After a viewport drag the pointer is still over the box → keep the grab cursor; a playhead
      // seek ends outside the box → pointer. (Next real mousemove recomputes precisely anyway.)
      setCursor(d.mode === "viewport" ? "grab" : "pointer");
      if (d.mode === "playhead" && useAudioStore.getState().seeking) {
        useAudioStore.getState().setSeeking(false);
      }
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, [setPlayhead, totalTicks]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = devicePixelRatio;
    const { width, height } = canvas.getBoundingClientRect();
    const cw = Math.round(width * dpr);
    const ch = Math.round(height * dpr);
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }

    // ── Waveform cache: the TRUE MIXDOWN preview — only audible tracks (mute excluded; solo
    //    isolates), and per segment the REAL source: the sub-lane sum (row mutes + slice recipes,
    //    via THE shared segmentPlaysLanes/segmentLaneSumPeaks) when lanes play, the original audio
    //    only when it actually plays (no ready lanes / playOriginal). Rebuilt only when its content
    //    key changes — not on scroll/playback. ──
    const hasSolo = tracks.some((t) => t.solo);
    const waveKey = `${cw}:${ch}:${totalTicks}:${tempo}:${hasSolo ? 1 : 0}:${Object.keys(audioFiles).length}:${tracks
      .map((t) => `${t.muted ? 1 : 0}${t.solo ? 1 : 0}|${t.segments.map((s) => (s.content.type === "audioClip"
        // srcPeaks length: an existing audioFiles entry gaining peaks doesn't change the key count.
        // s.loading: a clip of an ALREADY-decoded source flips loading→false with every other term
        // identical — without this the cached bitmap blits forever and the clip never appears.
        // segmentPlaysLanes + laneSumSig: the source switch (playOriginal / lanes turning ready) and
        // every lane-sum input (lane peaks, row mutes, slice recipes) must rebake the preview.
        ? `${s.startTick}.${s.durationTicks}.${s.loading ? 1 : 0}.${s.content.offsetMs}.${s.content.sourcePath}.${audioFiles[s.content.sourcePath]?.peaks.length ?? 0}.${segmentPlaysLanes(t, s) ? 1 : 0}.${laneSumSig(t, s)}`
        // ② vocal bake: rebake the preview when the rendered stem lands/changes (peaks length + path + dur).
        : (s.content.type === "notes" && s.processedOutputs?.length
          ? `n${s.startTick}.${s.durationTicks}.${s.loading ? 1 : 0}.${s.processedOutputs.map((o) => `${o.audioPath}:${o.totalDurationMs}:${o.offsetMs ?? 0}:${o.waveformPeaks?.length ?? 0}`).join("+")}`
          : ""))).join(",")}`)
      .join(";")}`;
    const sizeChanged = !waveRef.current || waveRef.current.width !== cw || waveRef.current.height !== ch;
    if (sizeChanged || waveKeyRef.current !== waveKey) {
      if (sizeChanged) waveRef.current = new OffscreenCanvas(cw, ch);
      const wc = waveRef.current!.getContext("2d")!;
      wc.setTransform(dpr, 0, 0, dpr, 0, 0);
      wc.fillStyle = "#0a0e1a";
      wc.fillRect(0, 0, width, height);
      // Overlapping clips (e.g. a split clip whose halves are resized into each other) would otherwise
      // draw two translucent waveforms on top of each other in the overlap, reading as a darker
      // "doubled" smear. "lighten" makes overlaps take the per-pixel MAX instead — a clean overview
      // showing the louder of the overlapping clips, with no doubling. Reset to source-over after.
      wc.globalCompositeOperation = "lighten";
      const waveColor = rgba(TRACK_RGB.audio, 0.6);
      for (const track of tracks) {
        if (track.muted || (hasSolo && !track.solo)) continue; // not audible → excluded from the overview
        for (const seg of track.segments) {
          if (seg.loading) continue;
          if (seg.content.type === "audioClip") {
            if (seg.content.totalDurationMs <= 0) continue;
            const sx = (seg.startTick / totalTicks) * width;
            const sw = (seg.durationTicks / totalTicks) * width;
            // Draw the segment's SLICE of the source (offset → offset+duration), via the shared waveform
            // cache — NOT the whole source. Otherwise a split clip draws the entire waveform in each
            // half, garbling the minimap at the split point. (The lane-sum stem spans the whole source,
            // so the SAME window ratios apply to both branches.)
            const offMs = seg.content.offsetMs;
            const totalMs = seg.content.totalDurationMs;
            const segMs = ticksToMs(seg.durationTicks, tempo);
            const startRatio = offMs / totalMs;
            const endRatio = Math.min(1, (offMs + segMs) / totalMs);
            // WHAT MIXES IS WHAT SHOWS: lanes' real audible sum when they are the source, else the
            // original audio. Same predicate + sum + exact-sig cache id as the arrangement's main row.
            let wave: OffscreenCanvas | null = null;
            const sumPeaks = segmentPlaysLanes(track, seg) ? segmentLaneSumPeaks(track, seg) : null;
            if (sumPeaks) {
              wave = getWaveformCache(`lanesum:${track.id}:${seg.id}:${laneSumSig(track, seg)}`, sumPeaks, waveColor);
            } else {
              const audio = audioFiles[seg.content.sourcePath];
              if (audio && audio.peaks.length) wave = getWaveformCache(seg.content.sourcePath, audio.peaks, waveColor);
            }
            if (wave) blitWaveform(wc, wave, sx, 0, sw, height, startRatio, endRatio, width);
          } else if (seg.content.type === "notes" && seg.processedOutputs && seg.processedOutputs.length > 0) {
            // ② Vocal bake: the rendered stem windowed [offset, offset+seg] into the box (offset from a notes
            // SPLIT, else 0 = whole stem; mirrors the arrangement sub-lane window). Vocal hue to read distinct.
            const sx = (seg.startTick / totalTicks) * width;
            const sw = (seg.durationTicks / totalTicks) * width;
            const segMs = ticksToMs(seg.durationTicks, tempo);
            const vColor = rgba(TRACK_RGB.vocal, 0.6);
            for (const out of seg.processedOutputs) {
              if (!out.waveformPeaks || out.waveformPeaks.length === 0 || out.totalDurationMs <= 0) continue;
              const off = Math.max(0, out.offsetMs ?? 0); // ② split: window into the SAME stem (off 0 = un-split)
              const startRatio = Math.min(1, off / out.totalDurationMs);
              const endRatio = Math.min(1, (off + segMs) / out.totalDurationMs);
              const wave = getWaveformCache(out.audioPath, out.waveformPeaks, vColor);
              if (wave) blitWaveform(wc, wave, sx, 0, sw, height, startRatio, endRatio, width);
            }
          }
        }
      }
      wc.globalCompositeOperation = "source-over";
      waveKeyRef.current = waveKey;
    }

    // ── Blit the cached waveform, then draw the cheap overlay on top. ──
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(waveRef.current!, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const ppt = PIXELS_PER_TICK * zoom;
    const viewStartRatio = scrollX / ppt / totalTicks;
    const viewEndRatio = (scrollX + canvasWidth) / ppt / totalTicks;
    ctx.fillStyle = rgba(ACCENT_RGB, 0.1);
    ctx.fillRect(viewStartRatio * width, 0, (viewEndRatio - viewStartRatio) * width, height);
    ctx.strokeStyle = rgba(ACCENT_RGB, 0.4);
    ctx.lineWidth = 1;
    ctx.strokeRect(viewStartRatio * width, 0, (viewEndRatio - viewStartRatio) * width, height);

    const phx = (playheadTick / totalTicks) * width;
    drawPlayhead(ctx, { x: phx, height, line: true, lineWidth: 1 });

    ctx.strokeStyle = CANVAS_BORDER;
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, width, height);
  }, [tracks, audioFiles, totalTicks, tempo, playheadTick, scrollX, zoom, canvasWidth]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Observe size once (redraw via the latest draw without rebuilding the observer each frame).
  const drawRef = useRef(draw);
  drawRef.current = draw;
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => drawRef.current());
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="overview-map"
      style={{ cursor }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
    />
  );
}
