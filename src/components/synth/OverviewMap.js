import { jsx as _jsx } from "react/jsx-runtime";
import { useRef, useEffect, useCallback, useState } from "react";
import { useProjectStore, useTimeAxis } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { PIXELS_PER_TICK } from "../../lib/constants";
import { segmentSourceWindowMs, ticksToMs } from "../../lib/audio/laneOps";
import { computeTotalTicks, segmentPlaysLanes, segmentLaneSumPeaks, laneSumSig } from "../../lib/trackLayout";
import { getWaveformCache, blitWaveform } from "../../lib/waveformCache";
import { rgba, ACCENT_RGB, TRACK_RGB } from "../../lib/trackColors";
import { drawPlayhead, CANVAS_BORDER, canvasThemeVars } from "../../lib/canvasDraw";
import { getContext } from "../../lib/audio/playback";
import { getAnalyserFor } from "../../lib/audio/effectsBus";
import "./OverviewMap.css";
export function OverviewMap() {
    const canvasRef = useRef(null);
    // The minimap repaints on scroll/playback to move its viewport rect + playhead. The WAVEFORM is the
    // only expensive part and changes only at known times (segments move/add/remove, peaks load, a track's
    // mute/solo flips), so it's cached in an offscreen and rebuilt only when its content key changes — every
    // scroll/playback frame just blits the cache and draws the cheap overlay (viewport + playhead).
    const tracks = useProjectStore((s) => s.tracks);
    const timeAxis = useTimeAxis();
    const tempo = useProjectStore((s) => s.tempo);
    const playheadTick = useProjectStore((s) => s.playheadTick);
    const setPlayhead = useProjectStore((s) => s.setPlayhead);
    const scrollX = useAppStore((s) => s.scrollX);
    const zoom = useAppStore((s) => s.zoom);
    const canvasWidth = useAppStore((s) => s.canvasWidth);
    const audioFiles = useAudioStore((s) => s.audioFiles);
    const isPlaying = useAudioStore((s) => s.isPlaying);
    // SAME basis as DawView's scroll width, so the viewport box + drag map 1:1 to the real scroll range
    // (a smaller minimap range made the box fill the map on short projects → seek/scroll both dead).
    const totalTicks = computeTotalTicks(tracks, timeAxis);
    const waveRef = useRef(null);
    const waveKeyRef = useRef("");
    const [cursor, setCursor] = useState("pointer");
    const animationFrameRef = useRef(0);
    const [waveformAmplitudes, setWaveformAmplitudes] = useState([]);
    const analyserRef = useRef(null);
    const spectrumPeaksRef = useRef([]);
    // Drag state: "viewport" = scrubbing the window box (scroll), "playhead" = moving the playhead.
    const dragRef = useRef(null);
    // Where the viewport box sits, in minimap CSS pixels (depends on scroll/zoom/canvas size).
    const viewportRect = useCallback((viewW) => {
        const ppt = PIXELS_PER_TICK * zoom;
        const startX = (scrollX / ppt / totalTicks) * viewW;
        const endX = ((scrollX + canvasWidth) / ppt / totalTicks) * viewW;
        return { ppt, startX, endX };
    }, [scrollX, zoom, canvasWidth, totalTicks]);
    const handleMouseDown = useCallback((e) => {
        if (e.button !== 0)
            return;
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        const localX = e.clientX - rect.left;
        const { ppt, startX, endX } = viewportRect(rect.width);
        const viewportWidth = endX - startX;
        const clickMargin = Math.max(4, viewportWidth * 0.1);
        if (localX >= startX - clickMargin && localX <= endX + clickMargin) {
            dragRef.current = { mode: "viewport", offsetX: localX - startX, viewW: rect.width, ppt };
            setCursor("grabbing");
        }
        else {
            dragRef.current = { mode: "playhead", offsetX: 0, viewW: rect.width, ppt };
            setPlayhead(Math.max(0, Math.round((localX / rect.width) * totalTicks)));
            if (useAudioStore.getState().isPlaying)
                useAudioStore.getState().setSeeking(true);
        }
    }, [viewportRect, totalTicks, setPlayhead]);
    // Hover cursor: "grab" over the window box (draggable), "pointer" elsewhere (click to seek).
    const handleMouseMove = useCallback((e) => {
        if (dragRef.current)
            return;
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        const localX = e.clientX - rect.left;
        const { startX, endX } = viewportRect(rect.width);
        const viewportWidth = endX - startX;
        const clickMargin = Math.max(4, viewportWidth * 0.1);
        setCursor(localX >= startX - clickMargin && localX <= endX + clickMargin ? "grab" : "pointer");
    }, [viewportRect]);
    useEffect(() => {
        const onMove = (e) => {
            const d = dragRef.current;
            if (!d)
                return;
            const canvas = canvasRef.current;
            if (!canvas)
                return;
            const localX = e.clientX - canvas.getBoundingClientRect().left;
            if (d.mode === "playhead") {
                // If playback was started mid-drag (Space), pin the rAF advance so it doesn't fight the drag
                // and so mouseup reschedules from the new position.
                if (useAudioStore.getState().isPlaying && !useAudioStore.getState().seeking) {
                    useAudioStore.getState().setSeeking(true);
                }
                setPlayhead(Math.max(0, Math.round((localX / d.viewW) * totalTicks)));
            }
            else {
                const newStartX = localX - d.offsetX;
                const maxScrollX = Math.max(0, totalTicks * d.ppt - useAppStore.getState().canvasWidth);
                const newScrollX = Math.max(0, Math.min(maxScrollX, (newStartX / d.viewW) * totalTicks * d.ppt));
                useAppStore.getState().setScroll(newScrollX, useAppStore.getState().scrollY);
            }
        };
        const onUp = () => {
            const d = dragRef.current;
            if (!d)
                return;
            dragRef.current = null;
            // After a viewport drag the pointer is still over the box → keep the grab cursor; a playhead
            // seek ends outside the box → pointer. (Next real mousemove recomputes precisely anyway.)
            setCursor(d.mode === "viewport" ? "grab" : "pointer");
            if (d.mode === "playhead" && useAudioStore.getState().seeking) {
                useAudioStore.getState().setSeeking(false);
            }
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        return () => {
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
        };
    }, [setPlayhead, totalTicks]);
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext("2d");
        if (!ctx)
            return;
        const dpr = devicePixelRatio;
        const { width, height } = canvas.getBoundingClientRect();
        const cw = Math.round(width * dpr);
        const ch = Math.round(height * dpr);
        if (canvas.width !== cw || canvas.height !== ch) {
            canvas.width = cw;
            canvas.height = ch;
        }
        // ── Waveform cache: the TRUE MIXDOWN preview — only audible tracks (mute excluded; solo
        //    isolates), and per segment the REAL source: the sub-lane sum (row mutes + slice recipes,
        //    via THE shared segmentPlaysLanes/segmentLaneSumPeaks) when lanes play, the original audio
        //    only when it actually plays (no ready lanes / playOriginal). Rebuilt only when its content
        //    key changes — not on scroll/playback. ──
        const hasSolo = tracks.some((t) => t.solo);
        const waveKey = `${cw}:${ch}:${totalTicks}:${tempo}:${hasSolo ? 1 : 0}:${Object.keys(audioFiles).length}:${tracks
            .map((t) => `${t.muted ? 1 : 0}${t.solo ? 1 : 0}|${t.segments.map((s) => (s.content.type === "audioClip"
            // srcPeaks length: an existing audioFiles entry gaining peaks doesn't change the key count.
            // s.loading: a clip of an ALREADY-decoded source flips loading→false with every other term
            // identical — without this the cached bitmap blits forever and the clip never appears.
            // segmentPlaysLanes + laneSumSig: the source switch (playOriginal / lanes turning ready) and
            // every lane-sum input (lane peaks, row mutes, slice recipes) must rebake the preview.
            // S59: the stretch factor changes the source window (segMs/r) → must rebake the preview.
            ? `${s.startTick}.${s.durationTicks}.${s.loading ? 1 : 0}.${s.content.offsetMs}.${s.content.stretch ?? 1}.${s.content.sourcePath}.${audioFiles[s.content.sourcePath]?.peaks.length ?? 0}.${segmentPlaysLanes(t, s) ? 1 : 0}.${laneSumSig(t, s)}`
            // ② vocal bake: rebake the preview when the rendered stem lands/changes (peaks length + path + dur).
            : (s.content.type === "notes" && s.processedOutputs?.length
                ? `n${s.startTick}.${s.durationTicks}.${s.loading ? 1 : 0}.${s.processedOutputs.map((o) => `${o.audioPath}:${o.totalDurationMs}:${o.offsetMs ?? 0}:${o.waveformPeaks?.length ?? 0}`).join("+")}`
                : ""))).join(",")}`)
            .join(";")}`;
        const sizeChanged = !waveRef.current || waveRef.current.width !== cw || waveRef.current.height !== ch;
        if (sizeChanged || waveKeyRef.current !== waveKey) {
            if (sizeChanged)
                waveRef.current = new OffscreenCanvas(cw, ch);
            const wc = waveRef.current.getContext("2d");
            wc.setTransform(dpr, 0, 0, dpr, 0, 0);
            const themeVars = canvasThemeVars();
            wc.fillStyle = themeVars.bgBase;
            wc.fillRect(0, 0, width, height);
            // Overlapping clips (e.g. a split clip whose halves are resized into each other) would otherwise
            // draw two translucent waveforms on top of each other in the overlap, reading as a darker
            // "doubled" smear. Use separate passes for audio and vocal to prevent overlap confusion.
            wc.globalCompositeOperation = "source-over";
            const amplitudes = [];
            const waveColor = rgba(TRACK_RGB.audio, 0.7);
            for (const track of tracks) {
                if (track.muted || (hasSolo && !track.solo))
                    continue; // not audible → excluded from the overview
                for (const seg of track.segments) {
                    if (seg.loading)
                        continue;
                    if (seg.content.type === "audioClip") {
                        if (seg.content.totalDurationMs <= 0)
                            continue;
                        const sx = (seg.startTick / totalTicks) * width;
                        const sw = (seg.durationTicks / totalTicks) * width;
                        // Draw the segment's SLICE of the source (offset → offset+duration), via the shared waveform
                        // cache — NOT the whole source. Otherwise a split clip draws the entire waveform in each
                        // half, garbling the minimap at the split point. (The lane-sum stem spans the whole source,
                        // so the SAME window ratios apply to both branches.)
                        const offMs = seg.content.offsetMs;
                        const totalMs = seg.content.totalDurationMs;
                        const segMs = segmentSourceWindowMs(seg, tempo); // S59: window length in SOURCE ms (÷ stretch)
                        const startRatio = offMs / totalMs;
                        const endRatio = Math.min(1, (offMs + segMs) / totalMs);
                        // WHAT MIXES IS WHAT SHOWS: lanes' real audible sum when they are the source, else the
                        // original audio. Same predicate + sum + exact-sig cache id as the arrangement's main row.
                        let wave = null;
                        const sumPeaks = segmentPlaysLanes(track, seg) ? segmentLaneSumPeaks(track, seg) : null;
                        if (sumPeaks) {
                            wave = getWaveformCache(`lanesum:${track.id}:${seg.id}:${laneSumSig(track, seg)}`, sumPeaks, waveColor);
                            const maxAmp = Math.max(...sumPeaks);
                            amplitudes.push(maxAmp);
                        }
                        else {
                            const audio = audioFiles[seg.content.sourcePath];
                            if (audio && audio.peaks.length) {
                                wave = getWaveformCache(seg.content.sourcePath, audio.peaks, waveColor);
                                const maxAmp = Math.max(...audio.peaks);
                                amplitudes.push(maxAmp);
                            }
                        }
                        if (wave)
                            blitWaveform(wc, wave, sx, 0, sw, height, startRatio, endRatio, width);
                    }
                    else if (seg.content.type === "notes" && seg.processedOutputs && seg.processedOutputs.length > 0) {
                        // ② Vocal bake: the rendered stem windowed [offset, offset+seg] into the box (offset from a notes
                        // SPLIT, else 0 = whole stem; mirrors the arrangement sub-lane window). Vocal hue to read distinct.
                        const sx = (seg.startTick / totalTicks) * width;
                        const sw = (seg.durationTicks / totalTicks) * width;
                        const segMs = ticksToMs(seg.durationTicks, tempo);
                        const vColor = rgba(TRACK_RGB.vocal, 0.85);
                        for (const out of seg.processedOutputs) {
                            if (!out.waveformPeaks || out.waveformPeaks.length === 0 || out.totalDurationMs <= 0)
                                continue;
                            const off = Math.max(0, out.offsetMs ?? 0);
                            const startRatio = Math.min(1, off / out.totalDurationMs);
                            const endRatio = Math.min(1, (off + segMs) / out.totalDurationMs);
                            const wave = getWaveformCache(out.audioPath, out.waveformPeaks, vColor);
                            if (wave) {
                                blitWaveform(wc, wave, sx, 0, sw, height * 0.75, startRatio, endRatio, width);
                                const maxAmp = Math.max(...out.waveformPeaks);
                                amplitudes.push(maxAmp);
                            }
                        }
                    }
                }
            }
            wc.globalCompositeOperation = "source-over";
            waveKeyRef.current = waveKey;
            setWaveformAmplitudes(amplitudes);
        }
        // ── Blit the cached waveform, then draw the cheap overlay on top. ──
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.drawImage(waveRef.current, 0, 0);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // ── 频谱显示：在波形后面显示实时频谱分析 ──
        if (isPlaying) {
            try {
                const audioContext = getContext();
                if (!analyserRef.current) {
                    analyserRef.current = getAnalyserFor(audioContext);
                    if (analyserRef.current) {
                        analyserRef.current.fftSize = 256;
                    }
                }
                const analyser = analyserRef.current;
                if (analyser) {
                    // 每帧创建新的 Uint8Array 避免类型冲突
                    const freqData = new Uint8Array(analyser.frequencyBinCount);
                    analyser.getByteFrequencyData(freqData);
                    const barCount = 64;
                    const step = Math.max(1, Math.floor(freqData.length / barCount));
                    const barWidth = width / barCount;
                    // 底部留边距，频谱不贴底；整体高度缩小
                    const specBase = height - 3;
                    const specMax = height * 0.68;
                    // 峰值帽状态（每根柱子独立下落）
                    const peaks = spectrumPeaksRef.current;
                    if (peaks.length !== barCount) {
                        peaks.length = 0;
                        for (let i = 0; i < barCount; i++)
                            peaks.push(0);
                    }
                    ctx.globalCompositeOperation = "screen";
                    for (let i = 0; i < barCount; i++) {
                        let sum = 0;
                        for (let j = 0; j < step; j++)
                            sum += freqData[i * step + j] ?? 0;
                        const value = Math.min(1, ((sum / step) / 255) * 1.25);
                        const barH = value * specMax;
                        const x = i * barWidth;
                        // 彩虹色系：橙黄 → 绿 → 青 → 蓝 → 紫 → 品红，越靠高频越绚丽
                        const hue = (i / barCount) * 320 + 20;
                        if (barH > 1) {
                            // 垂直渐变：底部深浓 → 顶部霓虹亮色
                            const grad = ctx.createLinearGradient(0, specBase, 0, specBase - barH);
                            grad.addColorStop(0, `hsla(${hue}, 100%, 38%, 0.95)`);
                            grad.addColorStop(0.55, `hsla(${hue}, 100%, 52%, 0.9)`);
                            grad.addColorStop(1, `hsla(${(hue + 50) % 360}, 100%, 72%, 0.98)`);
                            ctx.fillStyle = grad;
                            ctx.shadowBlur = 9;
                            ctx.shadowColor = `hsla(${hue}, 100%, 55%, 0.85)`;
                            ctx.fillRect(x, specBase - barH, barWidth - 0.5, barH);
                        }
                        // 下落的峰值帽：与柱身不同色，霓虹白亮
                        const peakNow = Math.max(peaks[i] ?? 0, value);
                        const peakFall = Math.max(0, peakNow - 0.01);
                        peaks[i] = peakFall;
                        const capY = specBase - peakFall * specMax - 3;
                        if (peakFall > 0.02) {
                            ctx.shadowBlur = 6;
                            ctx.shadowColor = `hsla(${(hue + 50) % 360}, 100%, 70%, 0.9)`;
                            ctx.fillStyle = `hsla(${(hue + 50) % 360}, 100%, 78%, 0.95)`;
                            ctx.fillRect(x, capY, barWidth - 0.5, 2);
                        }
                    }
                    ctx.shadowBlur = 0;
                    ctx.globalCompositeOperation = "source-over";
                }
            }
            catch (e) {
                // 如果获取频谱失败，静默处理
            }
        }
        // ── 波形跳动效果：根据当前播放位置的振幅动态缩放波形高度 ──
        if (isPlaying && waveformAmplitudes.length > 0 && waveRef.current) {
            const playRatio = playheadTick / totalTicks;
            const ampIndex = Math.floor(playRatio * waveformAmplitudes.length);
            const currentAmp = waveformAmplitudes[ampIndex] || 0;
            const normalizedAmp = Math.min(1, currentAmp / 255);
            // 在当前播放位置绘制跳动的波形叠加效果
            const jumpScale = 1 + normalizedAmp * 0.4; // 最多放大40%，跳动更明显
            const phx = playRatio * width;
            const jumpWidth = width * 0.08; // 增加跳动区域宽度到8%
            ctx.save();
            ctx.globalCompositeOperation = "lighter";
            ctx.globalAlpha = normalizedAmp * 0.6; // 提高叠加透明度
            // 在播放头位置绘制放大的波形切片
            const srcX = Math.max(0, phx - jumpWidth / 2);
            const srcW = Math.min(width - srcX, jumpWidth);
            if (srcW > 0) {
                const destY = (height - height * jumpScale) / 2;
                const destH = height * jumpScale;
                ctx.drawImage(waveRef.current, srcX * dpr, 0, srcW * dpr, ch, srcX, destY, srcW, destH);
            }
            ctx.restore();
        }
        const ppt = PIXELS_PER_TICK * zoom;
        const viewStartRatio = scrollX / ppt / totalTicks;
        const viewEndRatio = (scrollX + canvasWidth) / ppt / totalTicks;
        const rectX = viewStartRatio * width;
        const rectWidth = (viewEndRatio - viewStartRatio) * width;
        const radius = 4; // 圆角半径
        // 视口框加深加亮，播放时频谱跳动背景下也能看清 - 使用圆角矩形
        ctx.fillStyle = rgba(ACCENT_RGB, 0.28);
        ctx.beginPath();
        ctx.roundRect(rectX, 0, rectWidth, height, radius);
        ctx.fill();
        ctx.strokeStyle = rgba(ACCENT_RGB, 0.95);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(rectX, 0, rectWidth, height, radius);
        ctx.stroke();
        // 顶部加一条高亮线，强化可见度
        ctx.fillStyle = rgba(ACCENT_RGB, 0.9);
        ctx.fillRect(rectX + radius, 0, rectWidth - radius * 2, 1.5);
        ctx.fillRect(rectX + radius, height - 1.5, rectWidth - radius * 2, 1.5);
        const phx = (playheadTick / totalTicks) * width;
        drawPlayhead(ctx, { x: phx, height, line: true, lineWidth: 1 });
        ctx.strokeStyle = CANVAS_BORDER;
        ctx.lineWidth = 1;
        ctx.strokeRect(0, 0, width, height);
    }, [tracks, audioFiles, totalTicks, tempo, playheadTick, scrollX, zoom, canvasWidth, isPlaying, waveformAmplitudes]);
    useEffect(() => {
        draw();
        // Animation loop for smooth playback effects
        if (isPlaying) {
            const animate = () => {
                draw();
                animationFrameRef.current = requestAnimationFrame(animate);
            };
            animationFrameRef.current = requestAnimationFrame(animate);
            return () => {
                if (animationFrameRef.current)
                    cancelAnimationFrame(animationFrameRef.current);
            };
        }
    }, [draw, isPlaying]);
    // Observe size once (redraw via the latest draw without rebuilding the observer each frame).
    const drawRef = useRef(draw);
    drawRef.current = draw;
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const observer = new ResizeObserver(() => drawRef.current());
        observer.observe(canvas);
        return () => observer.disconnect();
    }, []);
    return (_jsx("canvas", { ref: canvasRef, className: "overview-map", style: { cursor }, onMouseDown: handleMouseDown, onMouseMove: handleMouseMove }));
}
