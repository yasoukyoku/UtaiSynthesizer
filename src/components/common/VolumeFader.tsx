import { useRef, useCallback, useEffect } from "react";
import "./VolumeFader.css";

interface Props {
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  width?: number;
}

export function VolumeFader({ value, min, max, onChange, width = 48 }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const ratio = (value - min) / (max - min);

  const calcValue = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el) return value;
      const rect = el.getBoundingClientRect();
      const r = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const raw = min + r * (max - min);
      return Math.round(raw * 2) / 2;
    },
    [min, max, value],
  );

  const handleDown = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      e.preventDefault();
      dragging.current = true;
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
      dragging.current = false;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, [calcValue]);

  const zeroRatio = (0 - min) / (max - min);

  return (
    <div
      className="vol-fader"
      ref={trackRef}
      style={{ width }}
      onMouseDown={handleDown}
      onClick={(e) => e.stopPropagation()}
      title={`${value > 0 ? "+" : ""}${value.toFixed(1)} dB`}
    >
      <div className="vol-track" />
      <div className="vol-zero" style={{ left: `${zeroRatio * 100}%` }} />
      <div className="vol-fill" style={{ width: `${ratio * 100}%` }} />
      <div className="vol-thumb" style={{ left: `${ratio * 100}%` }} />
    </div>
  );
}
