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
import { useAppStore } from "../../store/app";
import { backendErrorMessage, isBusyError } from "../../lib/backendError";
import { fmtDur } from "../../lib/constants";
import { preview } from "../common/previewPlayer";
import { Scrubber } from "../common/Scrubber";

export interface FilePreview {
  /** Path currently taken over by the player — null when nothing is playing. */
  playingPath: string | null;
  loadingPath: string | null;
  paused: boolean;
  /** Seconds into the active file, driven by a rAF ticker while playing. */
  pos: number;
  /** Decoded duration of the active file (0 until it decodes). */
  activeDur: number;
  toggle: (path: string) => void;
  /** Seek AND publish the new position — while paused there is no ticker to do it, so a
   *  seek-only call would leave the scrubber head behind until playback resumes. */
  seek: (frac: number) => void;
  /** Reset local state — the player does NOT fire onEnd for an explicit stop. */
  reset: () => void;
  /** Stop + reset only if `path` is the one playing (call BEFORE unmounting its row). */
  stopIfPlaying: (path: string) => void;
}

/**
 * Owns the singleton preview player for as long as the calling screen is mounted.
 *
 * `stillPresent` is asked AFTER the decode resolves: a long file takes seconds, in which time
 * the user may have removed it (or switched projects). Without it the row would start playing
 * audio that is no longer listed anywhere.
 */
export function useFilePreview(stillPresent?: (path: string) => boolean): FilePreview {
  const [playingPath, setPlayingPath] = useState<string | null>(null);
  const [loadingPath, setLoadingPath] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [pos, setPos] = useState(0);
  const rafRef = useRef<number | null>(null);
  const playTokenRef = useRef(0);
  // Read through a ref: the callbacks below are created once and would otherwise capture the
  // first render's predicate (which closes over a stale file list).
  const presentRef = useRef(stillPresent);
  presentRef.current = stillPresent;

  const stopTicker = () => {
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
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

  const toggle = async (path: string) => {
    if (playingPath === path) {
      if (paused) {
        preview.resume();
        setPaused(false);
        runTicker();
      } else {
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
      if (token !== playTokenRef.current) return; // superseded by a newer gesture
      if (presentRef.current && !presentRef.current(path)) {
        setPlayingPath(null);
        setLoadingPath(null);
        return;
      }
      await preview.play(path, buffer);
      setLoadingPath(null);
      runTicker();
    } catch (e) {
      if (token !== playTokenRef.current) return;
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
        preview.stop();
        reset();
      }
    },
  };
}

export function PreviewFileRow({
  p,
  path,
  name,
  title,
  lead,
  meta,
  onRemove,
}: {
  p: FilePreview;
  /** Absolute path handed to the reader — NOT necessarily what `name` shows. */
  path: string;
  name: string;
  title?: string;
  /** Optional chip before the name (e.g. which singer this file belongs to). */
  lead?: React.ReactNode;
  /** Right-hand readout while NOT playing; the live position replaces it during playback. */
  meta?: React.ReactNode;
  onRemove?: () => void;
}) {
  const { t } = useTranslation();
  const isActive = p.playingPath === path;
  const isLoading = p.loadingPath === path;
  const isPlaying = isActive && !p.paused && !isLoading;
  return (
    <div className="training-file-row" title={title ?? path}>
      <div className="training-file-main">
        <button
          className={`training-file-play ${isPlaying ? "on" : ""} ${isLoading ? "loading" : ""}`}
          onClick={() => p.toggle(path)}
          disabled={isLoading}
          title={
            isLoading
              ? t("training.loadingPreview")
              : isPlaying
                ? t("training.pausePreview")
                : t("training.preview")
          }
        >
          {isLoading ? "◌" : isPlaying ? "❚❚" : "▶"}
        </button>
        {lead}
        <span className="training-file-name tproj-ds-file">{name}</span>
        <span className="training-file-dur">
          {isActive && p.activeDur > 0
            ? `${fmtDur(p.pos)} / ${fmtDur(p.activeDur)}`
            : (meta ?? "--:--")}
        </span>
        {onRemove && (
          <button
            className="training-file-remove"
            onClick={() => {
              // stop first: the row (and its scrubber) is about to unmount
              p.stopIfPlaying(path);
              onRemove();
            }}
            title={t("training.remove")}
          >
            X
          </button>
        )}
      </div>
      {isActive && p.activeDur > 0 && (
        <Scrubber
          className="training-scrubber-slot"
          value={p.pos / p.activeDur}
          onSeek={p.seek}
        />
      )}
    </div>
  );
}
