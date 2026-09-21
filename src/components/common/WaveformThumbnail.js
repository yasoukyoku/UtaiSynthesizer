import { jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useRef } from "react";
/**
 * Waveform thumbnail — Web Audio API decode + downsample.
 * No Rust dependency — works in browser & Tauri.
 */
export function WaveformThumbnail({ src, width, height, color = "#a78bfa" }) {
    const canvasRef = useRef(null);
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !src)
            return;
        const ctx = canvas.getContext("2d");
        if (!ctx)
            return;
        let cancelled = false;
        ctx.fillStyle = "transparent";
        ctx.fillRect(0, 0, width, height);
        // Web Audio decode (works in Tauri via file:// URLs too)
        const actx = new (window.AudioContext || window.webkitAudioContext)();
        fetch(src)
            .then((r) => r.arrayBuffer())
            .then((ab) => actx.decodeAudioData(ab))
            .then((buf) => {
            if (cancelled || !ctx)
                return;
            // mono mixdown, downsample to width points
            const ch0 = buf.getChannelData(0);
            const ch1 = buf.numberOfChannels > 1 ? buf.getChannelData(1) : ch0;
            const step = Math.max(1, Math.floor(ch0.length / width));
            const peaks = new Float32Array(width);
            for (let i = 0; i < width; i++) {
                let max = 0;
                const start = i * step;
                for (let j = 0; j < step; j++) {
                    const v = (Math.abs(ch0[start + j] ?? 0) + Math.abs(ch1[start + j] ?? 0)) * 0.5;
                    if (v > max)
                        max = v;
                }
                peaks[i] = max;
            }
            // draw
            ctx.fillStyle = color;
            const amp = height * 0.45;
            const mid = height / 2;
            for (let i = 0; i < width; i++) {
                const h = peaks[i] * amp;
                ctx.fillRect(i, mid - h, 1, h * 2);
            }
        })
            .catch(() => { })
            .finally(() => { try {
            void actx.close();
        }
        catch { /* noop */ } });
        return () => { cancelled = true; };
    }, [src + ":" + width, width, height, color]);
    return (_jsx("canvas", { ref: canvasRef, width: width, height: height, style: { width, height, display: "block" } }));
}
