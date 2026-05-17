import { useRef, useEffect, useCallback } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import "./Arrangement.css";

const TRACK_HEIGHT = 48;
const PIXELS_PER_TICK = 0.15;
const TICKS_PER_BEAT = 480;

export function Arrangement() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { tracks, timeSignature, playheadTick } = useProjectStore();
  const { zoom, scrollX, scrollY } = useAppStore();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const ppt = PIXELS_PER_TICK * zoom;
    const ticksPerBar = TICKS_PER_BEAT * timeSignature[0];

    // Background
    ctx.fillStyle = getComputedStyle(canvas).getPropertyValue("--bg-base").trim() || "#0d1220";
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    const startTick = Math.floor(scrollX / ppt);
    const endTick = Math.ceil((scrollX + width) / ppt);

    for (let tick = startTick - (startTick % TICKS_PER_BEAT); tick < endTick; tick += TICKS_PER_BEAT) {
      const x = tick * ppt - scrollX;
      const isBar = tick % ticksPerBar === 0;

      ctx.strokeStyle = isBar
        ? "rgba(57, 197, 187, 0.22)"
        : "rgba(57, 197, 187, 0.06)";
      ctx.lineWidth = isBar ? 1 : 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      if (isBar) {
        const barNum = Math.floor(tick / ticksPerBar) + 1;
        ctx.fillStyle = "rgba(139, 158, 194, 0.5)";
        ctx.font = "10px var(--font-mono, monospace)";
        ctx.fillText(String(barNum), x + 3, 12);
      }
    }

    // Track lanes
    for (let i = 0; i < tracks.length; i++) {
      const y = i * TRACK_HEIGHT - scrollY;

      ctx.strokeStyle = "rgba(30, 42, 69, 0.8)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y + TRACK_HEIGHT);
      ctx.lineTo(width, y + TRACK_HEIGHT);
      ctx.stroke();

      // Draw segments
      const track = tracks[i];
      if (track) {
        for (const seg of track.segments) {
          const sx = seg.startTick * ppt - scrollX;
          const sw = seg.durationTicks * ppt;
          const sy = y + 4;
          const sh = TRACK_HEIGHT - 8;

          const colors = {
            vocal: "rgba(57, 197, 187, 0.3)",
            audio: "rgba(96, 165, 250, 0.3)",
            instrument: "rgba(167, 139, 250, 0.3)",
          };
          const borderColors = {
            vocal: "rgba(57, 197, 187, 0.6)",
            audio: "rgba(96, 165, 250, 0.6)",
            instrument: "rgba(167, 139, 250, 0.6)",
          };

          ctx.fillStyle = colors[track.trackType] ?? colors.audio;
          ctx.strokeStyle = borderColors[track.trackType] ?? borderColors.audio;
          ctx.lineWidth = 1;

          ctx.beginPath();
          ctx.roundRect(sx, sy, sw, sh, 3);
          ctx.fill();
          ctx.stroke();
        }
      }
    }

    // Playhead
    const phx = playheadTick * ppt - scrollX;
    ctx.strokeStyle = "#ff6b9d";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(phx, 0);
    ctx.lineTo(phx, height);
    ctx.stroke();

    // Playhead triangle
    ctx.fillStyle = "#ff6b9d";
    ctx.beginPath();
    ctx.moveTo(phx - 5, 0);
    ctx.lineTo(phx + 5, 0);
    ctx.lineTo(phx, 6);
    ctx.closePath();
    ctx.fill();
  }, [tracks, zoom, scrollX, scrollY, playheadTick, timeSignature]);

  useEffect(() => {
    draw();
    const observer = new ResizeObserver(draw);
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [draw]);

  return (
    <div className="arrangement">
      <canvas ref={canvasRef} className="arrangement-canvas" />
    </div>
  );
}
