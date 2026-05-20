import { useRef, useEffect, useCallback } from "react";
import { useProjectStore } from "../../store/project";
import { TICKS_PER_BEAT, PIXELS_PER_TICK } from "../../lib/constants";
import "./TimelineRuler.css";

interface Props {
  scrollX: number;
  zoom: number;
}

export function TimelineRuler({ scrollX, zoom }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { tempo, timeSignature, setPlayhead } = useProjectStore();
  const dragging = useRef(false);
  const ppt = PIXELS_PER_TICK * zoom;

  const clickToTick = useCallback(
    (clientX: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return 0;
      const rect = canvas.getBoundingClientRect();
      return Math.max(0, Math.round((clientX - rect.left + scrollX) / ppt));
    },
    [scrollX, ppt],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      dragging.current = true;
      setPlayhead(clickToTick(e.clientX));
    },
    [clickToTick, setPlayhead],
  );

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      setPlayhead(clickToTick(e.clientX));
    };
    const onUp = () => {
      dragging.current = false;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, [clickToTick, setPlayhead]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const ticksPerBar = TICKS_PER_BEAT * timeSignature[0];
    const secsPerBar = (60.0 / tempo) * timeSignature[0];

    ctx.fillStyle = "#1a2236";
    ctx.fillRect(0, 0, width, height);

    const startTick = Math.floor(scrollX / ppt);
    const endTick = Math.ceil((scrollX + width) / ppt);
    const startBar = Math.floor(startTick / ticksPerBar);
    const endBar = Math.ceil(endTick / ticksPerBar);

    for (let tick = startBar * ticksPerBar; tick < endTick; tick += TICKS_PER_BEAT) {
      const x = tick * ppt - scrollX;
      const isBar = tick % ticksPerBar === 0;
      ctx.strokeStyle = isBar ? "rgba(57, 197, 187, 0.4)" : "rgba(57, 197, 187, 0.15)";
      ctx.lineWidth = isBar ? 1 : 0.5;
      ctx.beginPath();
      ctx.moveTo(x, isBar ? 0 : height - 6);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    for (let bar = startBar; bar <= endBar; bar++) {
      const tick = bar * ticksPerBar;
      const x = tick * ppt - scrollX;
      ctx.fillStyle = "#e8ecf4";
      ctx.font = "bold 10px monospace";
      ctx.fillText(String(bar + 1), x + 3, 10);
      ctx.fillStyle = "#556b94";
      ctx.font = "9px monospace";
      ctx.fillText(formatTime(bar * secsPerBar), x + 3, 20);
    }

    // Playhead marker on ruler
    const { playheadTick } = useProjectStore.getState();
    const phx = playheadTick * ppt - scrollX;
    if (phx >= 0 && phx <= width) {
      ctx.fillStyle = "#ff6b9d";
      ctx.beginPath();
      ctx.moveTo(phx - 5, height);
      ctx.lineTo(phx + 5, height);
      ctx.lineTo(phx, height - 6);
      ctx.closePath();
      ctx.fill();
    }

    ctx.strokeStyle = "#2a3a5c";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 0.5);
    ctx.lineTo(width, height - 0.5);
    ctx.stroke();
  }, [scrollX, ppt, tempo, timeSignature]);

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  // Redraw when playhead changes
  const playheadTick = useProjectStore((s) => s.playheadTick);
  useEffect(() => { draw(); }, [playheadTick, draw]);

  return (
    <canvas
      ref={canvasRef}
      className="timeline-ruler"
      style={{ cursor: "pointer" }}
      onMouseDown={handleMouseDown}
    />
  );
}

function formatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}
