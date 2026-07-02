import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

export interface AudioTrackData {
  filePath: string;
  playbackPath: string;
  durationMs: number;
  sampleRate: number;
  peaks: number[];
}

interface AudioState {
  audioFiles: Record<string, AudioTrackData>;
  /** Source paths with a decode currently IN FLIGHT (peaks not yet available). Drives the arrangement's
   *  "加载中…" indicator for both a fresh import and an opened project whose peaks are still loading. */
  loadingPaths: string[];
  isPlaying: boolean;
  /** True while the user is dragging the playhead during playback (suppresses the rAF
   *  auto-advance so the drag isn't clobbered; on release playback reschedules). */
  seeking: boolean;
  /** Bumped when a committed edit (clip move/resize/delete) changes segment timing during playback.
   *  The Toolbar watches it and reschedules the Web Audio graph from the current playhead — already
   *  scheduled sources can't be moved, so without this the old layout keeps playing until replay. */
  scheduleVersion: number;

  loadAudioFile: (filePath: string) => Promise<AudioTrackData>;
  setPlaying: (playing: boolean) => void;
  setSeeking: (seeking: boolean) => void;
  bumpSchedule: () => void;
}

export const useAudioStore = create<AudioState>((set, get) => ({
  audioFiles: {},
  loadingPaths: [],
  isPlaying: false,
  seeking: false,
  scheduleVersion: 0,

  loadAudioFile: async (filePath) => {
    const existing = get().audioFiles[filePath];
    if (existing) return existing;

    set((s) => ({
      loadingPaths: s.loadingPaths.includes(filePath) ? s.loadingPaths : [...s.loadingPaths, filePath],
    }));
    try {
      const info = await invoke<{
        duration_ms: number;
        sample_rate: number;
        channels: number;
        peaks: number[];
        playback_path: string;
      }>("load_audio_file", { path: filePath });

      const data: AudioTrackData = {
        filePath,
        playbackPath: info.playback_path,
        durationMs: info.duration_ms,
        sampleRate: info.sample_rate,
        peaks: info.peaks,
      };

      set((s) => ({
        audioFiles: { ...s.audioFiles, [filePath]: data },
        loadingPaths: s.loadingPaths.filter((p) => p !== filePath),
      }));
      return data;
    } catch (e) {
      // Clear the in-flight marker on failure too, so a missing/bad file doesn't spin the indicator forever.
      set((s) => ({ loadingPaths: s.loadingPaths.filter((p) => p !== filePath) }));
      throw e;
    }
  },

  setPlaying: (playing) => set({ isPlaying: playing }),
  setSeeking: (seeking) => set({ seeking }),
  bumpSchedule: () => set((s) => ({ scheduleVersion: s.scheduleVersion + 1 })),
}));
