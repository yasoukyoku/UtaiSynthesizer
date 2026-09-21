import { jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useRef } from "react";
import { getContext } from "../../lib/audio/playback";
import { getAnalyserFor } from "../../lib/audio/effectsBus";
export function SpectrumAnalyzer({ height = 80, compact = false }) {
    const canvasRef = useRef(null);
    const rafRef = useRef(0);
    useEffect(() => {
        let disposed = false;
        const draw = () => {
            if (disposed)
                return;
            if (!canvasRef.current)
                return;
            const canvas = canvasRef.current;
            const dpr = window.devicePixelRatio || 1;
            const w = canvas.clientWidth || 240;
            const h = canvas.clientHeight || height;
            if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
                canvas.width = w * dpr;
                canvas.height = h * dpr;
            }
            const ctx2d = canvas.getContext("2d");
            if (!ctx2d)
                return;
            ctx2d.save();
            ctx2d.scale(dpr, dpr);
            ctx2d.fillStyle = "rgba(0,0,0,0.25)";
            ctx2d.fillRect(0, 0, w, h);
            // 获取全局 AnalyserNode（懒加载：首次播放后才存在）
            let analyser = null;
            try {
                analyser = getAnalyserFor(getContext());
            }
            catch {
                // AudioContext 还没初始化
            }
            if (analyser) {
                try {
                    const buf = new Uint8Array(analyser.frequencyBinCount);
                    analyser.getByteFrequencyData(buf);
                    const barCount = compact ? 32 : 64;
                    const step = Math.max(1, Math.floor(buf.length / barCount));
                    if (step <= 0)
                        return;
                    const barW = w / barCount;
                    for (let i = 0; i < barCount; i++) {
                        let sum = 0;
                        for (let j = 0; j < step; j++)
                            sum += buf[i * step + j] ?? 0;
                        const v = sum / step;
                        const bh = (v / 255) * h;
                        if (bh < 1)
                            continue; // 跳过静音
                        const grad = ctx2d.createLinearGradient(0, h - bh, 0, h);
                        grad.addColorStop(0, "#39c5bb");
                        grad.addColorStop(0.5, "#8b5cf6");
                        grad.addColorStop(1, "#ff6b9d");
                        ctx2d.fillStyle = grad;
                        ctx2d.fillRect(i * barW + 1, h - bh, Math.max(1, barW - 2), bh);
                    }
                }
                catch {
                    // Analyser 可能被断开，跳过本帧
                }
            }
            else {
                // 还没开始播放，显示提示或保持空白
                ctx2d.fillStyle = "rgba(148,163,184,0.5)";
                ctx2d.font = "11px inherit";
                ctx2d.textAlign = "center";
                ctx2d.fillText("▶ 播放时显示频谱", w / 2, h / 2 + 4);
            }
            ctx2d.restore();
            if (!disposed)
                rafRef.current = requestAnimationFrame(draw);
        };
        draw();
        return () => {
            disposed = true;
            cancelAnimationFrame(rafRef.current);
        };
    }, [compact, height]);
    return (_jsx("canvas", { ref: canvasRef, style: {
            width: "100%",
            height,
            borderRadius: 4,
            border: "1px solid var(--border-default, #2a3a5c)",
            background: "var(--bg-deep)",
            display: "block",
        }, title: "\uD83D\uDCCA \u9891\u8C31\u5206\u6790\u4EEA" }));
}
