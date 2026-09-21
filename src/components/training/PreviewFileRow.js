import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * One audio row with inline preview — play / pause / scrub, plus an optional remove button.
 *
 * Extracted from the data step (S76 批 5a-fix) so the project detail page can offer the same
 * 试听 on the files a project ALREADY holds. Two identical preview implementations were the
 * alternative, and the preview is the fiddly part: a decode that outlives its gesture, a file
 * that disappears while decoding, and a singleton player that three screens preempt each other
 * on (see `previewPlayer`'s consumer contract).
 */
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { readFile } from "@tauri-apps/plugin-fs";
import { open } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../../store/app";
import { backendErrorMessage, isBusyError } from "../../lib/backendError";
import { fmtDur } from "../../lib/constants";
import { exportOneAudioFileToFolder, laneExportErrorMessage } from "../../lib/audio/exportLaneAudio";
import { preview } from "../common/previewPlayer";
import { Scrubber } from "../common/Scrubber";
/**
 * Owns the singleton preview player for as long as the calling screen is mounted.
 *
 * `stillPresent` is asked AFTER the decode resolves: a long file takes seconds, in which time
 * the user may have removed it (or switched projects). Without it the row would start playing
 * audio that is no longer listed anywhere.
 */
export function useFilePreview(stillPresent) {
    const [playingPath, setPlayingPath] = useState(null);
    const [loadingPath, setLoadingPath] = useState(null);
    const [paused, setPaused] = useState(false);
    const [pos, setPos] = useState(0);
    const rafRef = useRef(null);
    const playTokenRef = useRef(0);
    // Read through a ref: the callbacks below are created once and would otherwise capture the
    // first render's predicate (which closes over a stale file list).
    const presentRef = useRef(stillPresent);
    presentRef.current = stillPresent;
    const stopTicker = () => {
        if (rafRef.current != null)
            cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
    };
    const runTicker = () => {
        stopTicker();
        const tick = () => {
            setPos(preview.position);
            rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);
    };
    const reset = () => {
        stopTicker();
        setPlayingPath(null);
        setPaused(false);
        setPos(0);
    };
    useEffect(() => {
        // consumer contract: stop whatever another screen left running, THEN take onEnd
        preview.stop();
        preview.onEnd = reset;
        return () => {
            preview.onEnd = null;
            stopTicker();
            preview.stop();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    const toggle = async (path) => {
        if (playingPath === path) {
            if (paused) {
                preview.resume();
                setPaused(false);
                runTicker();
            }
            else {
                preview.pause();
                setPaused(true);
                stopTicker();
            }
            return;
        }
        preview.stop();
        stopTicker();
        const token = ++playTokenRef.current;
        setPlayingPath(path);
        setLoadingPath(path);
        setPaused(false);
        setPos(0);
        try {
            // Read the file directly (fs scope allows **) and decode on the player's own context.
            // Deliberately NOT `load_audio_file`: that re-decodes, extracts waveform peaks nobody
            // needs here and writes a cache WAV — the multi-second stall on a 35-minute file.
            const bytes = await readFile(path);
            const buffer = await preview.decode(bytes);
            if (token !== playTokenRef.current)
                return; // superseded by a newer gesture
            if (presentRef.current && !presentRef.current(path)) {
                setPlayingPath(null);
                setLoadingPath(null);
                return;
            }
            await preview.play(path, buffer);
            setLoadingPath(null);
            runTicker();
        }
        catch (e) {
            if (token !== playTokenRef.current)
                return;
            preview.stop();
            setPlayingPath(null);
            setLoadingPath(null);
            useAppStore
                .getState()
                .showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
        }
    };
    return {
        playingPath,
        loadingPath,
        paused,
        pos,
        activeDur: preview.duration || 0,
        toggle: (p) => void toggle(p),
        seek: (frac) => {
            preview.seek(frac);
            setPos(preview.position);
        },
        reset,
        stopIfPlaying: (p) => {
            if (playingPath === p) {
                // Invalidate any in-flight decode/play for this path: the bytes may already be in
                // memory, and leaving the token valid would let the coroutine start playing a file that
                // is being removed (the exact race the `stillPresent` contract exists to prevent).
                playTokenRef.current += 1;
                preview.stop();
                reset();
            }
        },
    };
}
export function PreviewFileRow({ p, path, name, title, lead, meta, onRemove, }) {
    const { t } = useTranslation();
    const isActive = p.playingPath === path;
    const isLoading = p.loadingPath === path;
    const isPlaying = isActive && !p.paused && !isLoading;
    // 试听行的「下载」按钮：把该音频文件(原样复制,不重编码)存入用户自选的文件夹。
    const download = async () => {
        const out = await open({ directory: true, title: t("tracks.exportTracks") });
        if (!out || typeof out !== "string")
            return;
        try {
            await exportOneAudioFileToFolder({ label: name, sourcePath: path }, out, name);
            useAppStore
                .getState()
                .showToast(`${t("tracks.exportDone")} 1 ${t("tracks.exportCopied")} ${out}`, "success");
        }
        catch (e) {
            useAppStore.getState().showToast(laneExportErrorMessage(e), "error");
        }
    };
    return (_jsxs("div", { className: "training-file-row", title: title ?? path, children: [_jsxs("div", { className: "training-file-main", children: [_jsx("button", { className: `training-file-play ${isPlaying ? "on" : ""} ${isLoading ? "loading" : ""}`, onClick: () => p.toggle(path), disabled: isLoading, title: isLoading
                            ? t("training.loadingPreview")
                            : isPlaying
                                ? t("training.pausePreview")
                                : t("training.preview"), children: isLoading ? "◌" : isPlaying ? "❚❚" : "▶" }), _jsx("button", { className: "training-file-dl", onClick: (e) => {
                            e.stopPropagation();
                            void download();
                        }, title: t("tracks.exportTracks"), children: "\u2B07" }), lead, _jsx("span", { className: "training-file-name tproj-ds-file", children: name }), _jsx("span", { className: "training-file-dur", children: isActive && p.activeDur > 0
                            ? `${fmtDur(p.pos)} / ${fmtDur(p.activeDur)}`
                            : (meta ?? "--:--") }), onRemove && (_jsx("button", { className: "training-file-remove", onClick: () => {
                            // stop first: the row (and its scrubber) is about to unmount
                            p.stopIfPlaying(path);
                            onRemove();
                        }, title: t("training.remove"), children: "X" }))] }), isActive && p.activeDur > 0 && (_jsx(Scrubber, { className: "training-scrubber-slot", value: p.pos / p.activeDur, onSeek: p.seek }))] }));
}
