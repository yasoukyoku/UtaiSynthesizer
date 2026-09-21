import { useEffect, useRef } from "react";
import { getAnalyserFor, getTrackAnalyser } from "../../lib/audio/effectsBus";
import { getContext } from "../../lib/audio/playback";
import "./OutLevelMeter.css";

interface Props {
  width?: number;
  height?: number;
  /** 传入轨道 id 时显示该轨的实时电平 (需该轨播放过一次); 不传则显示总输出电平 */
  trackId?: string;
}

/** 实时电平表 — 读取 AnalyserNode 的波形峰值, 播放时跳动.
 *  rAF 循环直接写 DOM height, 不经过 React state (每帧 setState 会卡死). */
export function OutLevelMeter({ width = 14, height = 150, trackId }: Props) {
  const fillRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let raf = 0;
    let buf: Uint8Array<ArrayBuffer> | null = null;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const fill = fillRef.current;
      if (!fill) return;
      const analyser = trackId ? getTrackAnalyser(getContext(), trackId) : getAnalyserFor(getContext());
      if (!analyser) { fill.style.height = "0%"; return; }
      if (!buf || buf.length !== analyser.fftSize) {
        buf = new Uint8Array(new ArrayBuffer(analyser.fftSize));
      }
      const data = buf;
      analyser.getByteTimeDomainData(data);
      let peak = 0;
      for (const s of data) {
        const v = Math.abs(s - 128) / 128;
        if (v > peak) peak = v;
      }
      fill.style.height = `${Math.min(100, peak * 100)}%`;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="out-level-meter" style={{ width, height }} title="总输出实时电平">
      <div ref={fillRef} className="out-level-fill" />
    </div>
  );
}
