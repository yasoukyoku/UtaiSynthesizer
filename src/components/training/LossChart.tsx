/**
 * Loss-curve canvas chart (no chart lib in this repo — hand-drawn, angular house
 * style). Series toggle via legend chips; best-step marker; PNG export through
 * the imperative handle (RunStep owns the export button + save dialog).
 */
import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import type { StepPoint } from "../../store/training";

export interface LossChartHandle {
  toPngBlob: () => Promise<Blob | null>;
}

const SERIES: { key: string; varName: string; fallback: string }[] = [
  { key: "mel", varName: "--accent-primary", fallback: "#39c5bb" },
  { key: "g_total", varName: "--accent-secondary", fallback: "#8b5cf6" },
  { key: "d_total", varName: "--accent-tertiary", fallback: "#ff6b9d" },
];

const PAD_L = 44;
const PAD_R = 10;
const PAD_T = 8;
const PAD_B = 18;

export const LossChart = forwardRef<
  LossChartHandle,
  { history: StepPoint[]; bestStep?: number | null; height?: number }
>(function LossChart({ history, bestStep, height = 220 }, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  const [enabled, setEnabled] = useState<Record<string, boolean>>({
    mel: true,
    g_total: true,
    d_total: true,
  });

  useImperativeHandle(ref, () => ({
    toPngBlob: () =>
      new Promise<Blob | null>((resolve) => {
        const canvas = canvasRef.current;
        if (!canvas) return resolve(null);
        canvas.toBlob((b) => resolve(b), "image/png");
      }),
  }));

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width < 60) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const css = getComputedStyle(document.documentElement);
    const col = (v: string, fb: string) => css.getPropertyValue(v).trim() || fb;
    const bg = col("--bg-surface", "#131a2b");
    const grid = col("--border-subtle", "#232c44");
    const textCol = col("--text-muted", "#556b94");
    const successCol = col("--color-success", "#4ade80");

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    const plotW = width - PAD_L - PAD_R;
    const plotH = height - PAD_T - PAD_B;
    const active = SERIES.filter((s) => enabled[s.key]);
    const pts = history;
    if (!pts.length || !active.length || plotW < 10) {
      return;
    }

    const x0 = pts[0]!.step;
    const x1 = Math.max(pts[pts.length - 1]!.step, x0 + 1);
    let yMax = 0;
    for (const p of pts) {
      for (const s of active) {
        const v = p.losses[s.key];
        if (v !== undefined && isFinite(v)) yMax = Math.max(yMax, v);
      }
    }
    if (yMax <= 0) yMax = 1;
    yMax *= 1.05;

    const xOf = (step: number) => PAD_L + ((step - x0) / (x1 - x0)) * plotW;
    const yOf = (v: number) => PAD_T + plotH - (v / yMax) * plotH;

    // grid + y labels
    ctx.strokeStyle = grid;
    ctx.fillStyle = textCol;
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const v = (yMax / 4) * i;
      const y = Math.round(yOf(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(PAD_L, y);
      ctx.lineTo(width - PAD_R, y);
      ctx.stroke();
      ctx.fillText(v >= 100 ? v.toFixed(0) : v.toFixed(1), 4, y + 3);
    }
    // x labels
    ctx.textAlign = "center";
    for (const step of [x0, Math.round((x0 + x1) / 2), x1]) {
      ctx.fillText(String(step), xOf(step), height - 5);
    }
    ctx.textAlign = "left";

    // per-pixel bucket average, then polyline
    for (const s of active) {
      const buckets = new Map<number, { sum: number; n: number }>();
      for (const p of pts) {
        const v = p.losses[s.key];
        if (v === undefined || !isFinite(v)) continue;
        const px = Math.round(xOf(p.step));
        const b = buckets.get(px);
        if (b) {
          b.sum += v;
          b.n += 1;
        } else {
          buckets.set(px, { sum: v, n: 1 });
        }
      }
      const xs = [...buckets.keys()].sort((a, b) => a - b);
      if (!xs.length) continue;
      ctx.strokeStyle = col(s.varName, s.fallback);
      ctx.lineWidth = 1.25;
      if (xs.length === 1) {
        // a one-point polyline is invisible — draw a dot
        const b = buckets.get(xs[0]!)!;
        ctx.fillStyle = col(s.varName, s.fallback);
        ctx.fillRect(xs[0]! - 1, yOf(b.sum / b.n) - 1, 3, 3);
        continue;
      }
      ctx.beginPath();
      xs.forEach((px, i) => {
        const b = buckets.get(px)!;
        const y = yOf(b.sum / b.n);
        if (i === 0) ctx.moveTo(px, y);
        else ctx.lineTo(px, y);
      });
      ctx.stroke();
    }

    // best marker
    if (bestStep != null && bestStep >= x0 && bestStep <= x1) {
      const x = Math.round(xOf(bestStep)) + 0.5;
      ctx.strokeStyle = successCol;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, PAD_T);
      ctx.lineTo(x, PAD_T + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = successCol;
      ctx.fillText("best", Math.min(x + 3, width - 34), PAD_T + 10);
    }
  }, [history, width, height, enabled, bestStep]);

  return (
    <div className="loss-chart" ref={wrapRef}>
      <canvas ref={canvasRef} />
      <div className="loss-chart-legend">
        {SERIES.map((s) => (
          <button
            key={s.key}
            className={`loss-legend-chip ${enabled[s.key] ? "on" : ""}`}
            onClick={() =>
              setEnabled((e) => ({ ...e, [s.key]: !e[s.key] }))
            }
          >
            <span
              className="loss-legend-swatch"
              style={{ background: `var(${s.varName}, ${s.fallback})` }}
            />
            {s.key}
          </button>
        ))}
      </div>
    </div>
  );
});
