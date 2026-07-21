import { useRef, useCallback, useEffect } from "react";
import "./VolumeFader.css";

interface Props {
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  /** Fired once when a drag begins (mousedown) — open an undo transaction here. */
  onGestureStart?: () => void;
  /** Fired once when a drag ends (mouseup) — commit the undo transaction here. */
  onGestureEnd?: () => void;
  width?: number;
  /** Drag quantization step. Default 0.5 (dB); pass e.g. 0.1 for a −1..1 pan fader. */
  step?: number;
  /** Fill geometry: "left" (volume: min→thumb) or "center" (pan: zero-notch→thumb). */
  fillFrom?: "left" | "center";
  /** Tooltip formatter. Default renders dB ("+1.5 dB"); pass `formatPan` for a pan fader. */
  format?: (v: number) => string;
  /** S73b: 说明性 tip,与数值合成进同一个 root title(外包 div 的 title 会被这里的 title 遮蔽,
   *  所以说明必须从这个单一 title 源走)。 */
  tip?: string;
}

/** Pan tooltip text ("L50" / "C" / "R50") for a −1..1 fader — the ONE pan display convention. */
export function formatPan(v: number): string {
  if (v === 0) return "C";
  return v < 0 ? `L${Math.round(-v * 100)}` : `R${Math.round(v * 100)}`;
}

/** Volume-fader dB text ("-∞ dB" at/below the floor, else "+1.5 dB" / "-6.0 dB"). The ONE dB
 *  formatter shared by the fader tooltip AND the always-on TrackList numeric readout. */
export function formatDb(v: number, min: number): string {
  return v <= min ? "-∞ dB" : `${v > 0 ? "+" : ""}${v.toFixed(1)} dB`;
}

export function VolumeFader({ value, min, max, onChange, onGestureStart, onGestureEnd, width = 48, step = 0.5, fillFrom = "left", format, tip }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  // Everything the document listeners need lives in refs so calcValue/the listener effect are STABLE
  // (empty deps). If calcValue depended on `value` (a per-frame controlled prop), the listener effect
  // would tear down + re-create on every drag frame — and its cleanup would fire mid-drag.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const onStartRef = useRef(onGestureStart);
  onStartRef.current = onGestureStart;
  const onEndRef = useRef(onGestureEnd);
  onEndRef.current = onGestureEnd;
  const minRef = useRef(min);
  minRef.current = min;
  const maxRef = useRef(max);
  maxRef.current = max;
  const stepRef = useRef(step);
  stepRef.current = step;
  const valueRef = useRef(value);
  valueRef.current = value;

  const ratio = (value - min) / (max - min);

  const calcValue = useCallback((clientX: number) => {
    const el = trackRef.current;
    if (!el) return valueRef.current;
    const rect = el.getBoundingClientRect();
    const r = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const raw = minRef.current + r * (maxRef.current - minRef.current);
    // Quantize to `step`, then snap away float dust (0.1-steps yield 0.30000000000000004).
    return Math.round((Math.round(raw / stepRef.current) * stepRef.current) * 1000) / 1000;
  }, []);

  const handleDown = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      e.preventDefault();
      // Commit any focused-field transaction first (e.g. the BPM input) so a field edit-session and
      // this fader drag don't collapse into one undo step. preventDefault above suppresses the default
      // focus shift, so blur explicitly.
      (document.activeElement as HTMLElement | null)?.blur?.();
      dragging.current = true;
      onStartRef.current?.(); // open the undo transaction BEFORE the first value write
      onChangeRef.current(calcValue(e.clientX));
    },
    [calcValue],
  );

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      onChangeRef.current(calcValue(e.clientX));
    };
    const onUp = () => {
      if (!dragging.current) return;
      dragging.current = false;
      onEndRef.current?.(); // commit the whole drag as one undo step (before-press → on-release)
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      // If the fader unmounts mid-drag (e.g. its track is deleted), still close the gesture so the
      // undo transaction opened in handleDown can't be left open. This effect mounts ONCE (stable
      // calcValue), so the cleanup only runs on true unmount — never mid-drag.
      if (dragging.current) {
        dragging.current = false;
        onEndRef.current?.();
      }
    };
  }, [calcValue]);

  const zeroRatio = (0 - min) / (max - min);
  // "center" (pan): fill spans zero-notch→thumb, either side; "left" (volume): min→thumb as before.
  const fillLeft = fillFrom === "center" ? Math.min(ratio, zeroRatio) : 0;
  const fillWidth = fillFrom === "center" ? Math.abs(ratio - zeroRatio) : ratio;

  return (
    <div
      className="vol-fader"
      ref={trackRef}
      style={{ width }}
      onMouseDown={handleDown}
      onClick={(e) => e.stopPropagation()}
      // Default dB formatter: the fader BOTTOM reads −∞ (mute — see FADER_MIN_DB); a custom `format`
      // (the pan fader) is never affected.
      title={`${tip ? `${tip} — ` : ""}${format ? format(value) : formatDb(value, min)}`}
    >
      <div className="vol-track" />
      <div className="vol-zero" style={{ left: `${zeroRatio * 100}%` }} />
      <div className="vol-fill" style={{ left: `${fillLeft * 100}%`, width: `${fillWidth * 100}%` }} />
      <div className="vol-thumb" style={{ left: `${ratio * 100}%` }} />
    </div>
  );
}
