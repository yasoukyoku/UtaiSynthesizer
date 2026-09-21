import { jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useRef } from "react";
import { useAudioStore } from "../../store/audio";
import { useProjectStore } from "../../store/project";
import { trackTypeCssVar } from "../../lib/trackColors";
import { ticksToMs } from "../../lib/audio/laneOps";
const CLIP_THRESHOLD = 0.95; // 爆音阈值: 峰值 ≥ 95% 满量程视为削波 → 电平条顶部变红
const CLIP_COLOR = "#ef4444";
const BAR_W = 4; // "非常细的一个条"
/** 轨道头电平表(Studio Pro 式): 暂停时完全留白(无静态波形/无框架)。
 *  播放 — 左右并排两根极细竖条: 左排=左通道、右排=右通道, 各 4px 宽、高度占满轨道头,
 *  随音量上下跳动(底边锚定向上生长); 左右各自独立采样(peaksL/peaksR),
 *  音量增益与声像(pan)分别作用于对应通道, 爆音(≥95% 满量程)时该条顶部变红。
 *  颜色跟随轨道色。 */
export function TrackWaveform({ track, height }) {
    const canvasRef = useRef(null);
    const audioFiles = useAudioStore((s) => s.audioFiles);
    const isPlaying = useAudioStore((s) => s.isPlaying);
    const clips = [];
    for (const seg of track.segments) {
        if (seg.content.type !== "audioClip" || seg.loading)
            continue;
        const c = seg.content;
        if (c.totalDurationMs <= 0)
            continue;
        clips.push({
            src: c.sourcePath,
            offsetMs: c.offsetMs,
            durMs: c.totalDurationMs,
            startTick: seg.startTick,
            durTicks: seg.durationTicks,
            stretch: c.stretch ?? 1,
        });
    }
    const hasData = clips.some((c) => audioFiles[c.src]);
    const colorVar = track.color || trackTypeCssVar(track.trackType);
    // rAF 循环里读不到最新 props → 音量/声像走 ref;每帧峰值归一化系数缓存(每源每通道一次)
    const volRef = useRef(track.volumeDb);
    volRef.current = track.volumeDb;
    const panRef = useRef(track.pan ?? 0);
    panRef.current = track.pan ?? 0;
    const peakMaxRef = useRef(new Map());
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const W = canvas.clientWidth || 48;
        const H = canvas.clientHeight || height;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(W * dpr));
        canvas.height = Math.max(1, Math.round(H * dpr));
        const ctx = canvas.getContext("2d");
        if (!ctx)
            return;
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, W, H);
        if (!hasData || !isPlaying)
            return; // 暂停: 只保留上面 clearRect — 完全留白
        const rgb = resolveRgb(colorVar);
        /** 采样某通道 peaks 在 atMs 处 ±winMs 窗口的平均峰值(归一化后)。 */
        const sampleAt = (peaks, maxKey, srcMs, atMs, winMs, gain) => {
            let peakMax = peakMaxRef.current.get(maxKey) ?? 0;
            if (peakMax === 0) {
                for (const p of peaks)
                    if (p > peakMax)
                        peakMax = p;
                peakMaxRef.current.set(maxKey, peakMax);
            }
            const norm = peakMax > 0 ? 1 / peakMax : 1;
            const center = Math.floor((atMs / srcMs) * peaks.length);
            const span = Math.max(1, Math.floor((winMs / srcMs) * peaks.length));
            let sum = 0;
            let n = 0;
            for (let k = center - span; k <= center + span; k++) {
                if (k >= 0 && k < peaks.length) {
                    sum += peaks[k] ?? 0;
                    n++;
                }
            }
            const avg = n > 0 ? sum / n : 0;
            return Math.max(0, Math.min(1, avg * norm * gain));
        };
        // ── 播放中: Studio Pro 式左右并排双竖条(左=L / 右=R), 高度占满轨道头 ──
        const GAP = 6; // 两根竖条之间的间距
        const bxL = W / 2 - GAP / 2 - BAR_W; // 左排 = 左通道
        const bxR = W / 2 + GAP / 2; // 右排 = 右通道
        const WIN_MS = 25; // 采样窗口(毫秒) — 小窗跟随瞬态, 大窗更平滑
        // 电平表惯性质感: 起音快(跳上去灵敏), 释放慢(落下来缓) — 每通道独立
        const smooth = { L: 0, R: 0 };
        let raf = 0;
        const loop = () => {
            ctx.clearRect(0, 0, W, H);
            const st = useProjectStore.getState();
            const tick = st.playheadTick;
            const gain = Math.pow(10, volRef.current / 20);
            const pan = panRef.current;
            const gainL = gain * (pan <= 0 ? 1 : 1 - pan); // 声像右偏 → 左声道衰减
            const gainR = gain * (pan >= 0 ? 1 : 1 + pan); // 声像左偏 → 右声道衰减
            const seg = clips.find((c) => tick >= c.startTick && tick < c.startTick + c.durTicks);
            const audio = seg ? audioFiles[seg.src] : undefined;
            let aL = 0.05 + 0.03 * Math.sin(performance.now() / 140); // 无音频时的低幅呼吸
            let aR = aL;
            if (seg && audio) {
                const peaksL = audio.peaksL.length > 0 ? audio.peaksL : audio.peaks;
                const peaksR = audio.peaksR.length > 0 ? audio.peaksR : audio.peaks;
                const relMs = ticksToMs(tick - seg.startTick, st.tempo) / seg.stretch;
                const atMs = seg.offsetMs + relMs;
                aL = sampleAt(peaksL, `${seg.src}:L`, audio.durationMs, atMs, WIN_MS, gainL);
                aR = sampleAt(peaksR, `${seg.src}:R`, audio.durationMs, atMs, WIN_MS, gainR);
            }
            // 起音快 / 释放慢
            smooth.L = smooth.L + (aL - smooth.L) * (aL > smooth.L ? 0.5 : 0.15);
            smooth.R = smooth.R + (aR - smooth.R) * (aR > smooth.R ? 0.5 : 0.15);
            // 两根竖条: 底边锚定在轨道头底部, 随音量向上生长(高度占满整个轨头)
            ctx.fillStyle = rgb;
            const hL = Math.max(1, smooth.L * (H - 2));
            ctx.fillRect(bxL, H - 1 - hL, BAR_W, hL);
            const hR = Math.max(1, smooth.R * (H - 2));
            ctx.fillRect(bxR, H - 1 - hR, BAR_W, hR);
            // 爆音 → 该条顶部变红
            if (smooth.L >= CLIP_THRESHOLD) {
                ctx.fillStyle = CLIP_COLOR;
                ctx.fillRect(bxL, 1, BAR_W, 3);
            }
            if (smooth.R >= CLIP_THRESHOLD) {
                ctx.fillStyle = CLIP_COLOR;
                ctx.fillRect(bxR, 1, BAR_W, 3);
            }
            raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(raf);
    }, [clips.map((c) => `${c.src}|${c.offsetMs}|${c.durMs}|${c.startTick}|${c.durTicks}|${c.stretch}`).join(";"), hasData, colorVar, height, audioFiles, isPlaying]);
    return _jsx("canvas", { ref: canvasRef, className: "track-waveform-canvas", style: { width: 48, height } });
}
/** 把 "var(--track-vocal)" 或 "#hex" 解析成可用的 fill 颜色。canvas 不认 CSS 变量。 */
function resolveRgb(color) {
    if (!color)
        return "#6b7280";
    if (color.startsWith("#"))
        return color;
    const m = getComputedStyle(document.documentElement).getPropertyValue(color.slice(4, -1)).trim();
    return m || "#6b7280";
}
