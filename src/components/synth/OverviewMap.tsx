import { useRef, useEffect, useCallback } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { TICKS_PER_BEAT, PIXELS_PER_TICK } from "../../lib/constants";
import "./OverviewMap.css";

export function OverviewMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { tracks, playheadTick, setPlayhead, timeSignature } = useProjectStore();
  const { zoom, scrollX } = useAppStore();
  const { audioFiles } = useAudioStore();

  const totalTicks = Math.max(
    TICKS_PER_BEAT * timeSignature[0] * 4,
    ...tracks.flatMap((t) => t.segments.map((s) => s.startTick + s.durationTicks))
  );

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      setPlayhead(Math.round(ratio * totalTicks));
    },
    [totalTicks, setPlayhead]
  );

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    ctx.fillStyle = "#0a0e1a";
    ctx.fillRect(0, 0, width, height);

    // Draw combined waveform overview
    const midY = height / 2;
    const amp = height / 2 - 1;

    for (const track of tracks) {
      for (const seg of track.segments) {
        if (seg.content.type !== "audioClip") continue;
        const audio = audioFiles[seg.content.sourcePath];
        if (!audio || !audio.peaks.length) continue;

        const segStart = seg.startTick / totalTicks;
        const segEnd = (seg.startTick + seg.durationTicks) / totalTicks;
        const sx = segStart * width;
        const sw = (segEnd - segStart) * width;

        ctx.strokeStyle = "rgba(96,165,250,0.6)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let px = 0; px < sw; px++) {
          const peakIdx = Math.floor((px / sw) * audio.peaks.length);
          const peak = audio.peaks[Math.min(peakIdx, audio.peaks.length - 1)] ?? 0;
          const drawX = sx + px;
          ctx.moveTo(drawX, midY - peak * amp);
          ctx.lineTo(drawX, midY + peak * amp);
        }
        ctx.stroke();
      }
    }

    // Viewport indicator
    const ppt = PIXELS_PER_TICK * zoom;
    const viewStartRatio = scrollX / ppt / totalTicks;
    // Estimate visible ticks from the arrangement canvas width (approximate)
    const visibleTicks = 800 / ppt;
    const viewEndRatio = (scrollX / ppt + visibleTicks) / totalTicks;

    ctx.fillStyle = "rgba(57,197,187,0.1)";
    ctx.fillRect(viewStartRatio * width, 0, (viewEndRatio - viewStartRatio) * width, height);
    ctx.strokeStyle = "rgba(57,197,187,0.4)";
    ctx.lineWidth = 1;
    ctx.strokeRect(viewStartRatio * width, 0, (viewEndRatio - viewStartRatio) * width, height);

    // Playhead
    const phRatio = playheadTick / totalTicks;
    const phx = phRatio * width;
    ctx.strokeStyle = "#ff6b9d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(phx, 0);
    ctx.lineTo(phx, height);
    ctx.stroke();

    // Border
    ctx.strokeStyle = "#2a3a5c";
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, width, height);
  }, [tracks, audioFiles, totalTicks, playheadTick, zoom, scrollX]);

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  return <canvas ref={canvasRef} className="overview-map" onClick={handleClick} />;
}
