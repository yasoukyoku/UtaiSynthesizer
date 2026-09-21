import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { useProjectStore } from "./project";
import type { Track } from "../types/project";

export interface AudioTrackData {
  filePath: string;
  playbackPath: string;
  durationMs: number;
  sampleRate: number;
  peaks: number[];
  /** 左/右通道峰值包络(与 peaks 同帧率同对齐)。旧解码缓存/单声道 → 回退为混音 peaks。 */
  peaksL: number[];
  peaksR: number[];
}

interface AudioState {
  audioFiles: Record<string, AudioTrackData>;
  /** Source paths with a decode currently IN FLIGHT (peaks not yet available). Drives the arrangement's
   *  "加载中…" indicator for both a fresh import and an opened project whose peaks are still loading. */
  loadingPaths: string[];
  isPlaying: boolean;
  /** S60-3: play was REQUESTED but the schedule isn't sounding yet (auto-render + stretch
   *  regeneration + buffer decode). Drives the Play button's "preparing" look — the first
   *  play of a big project used to sit in this window for seconds with ZERO feedback,
   *  which reads as a hang (§user: never let the app look frozen). */
  preparing: boolean;
  /** True while the user is dragging the playhead during playback (suppresses the rAF
   *  auto-advance so the drag isn't clobbered; on release playback reschedules). */
  seeking: boolean;
  /** Bumped when a committed edit (clip move/resize/delete) changes segment timing during playback.
   *  The Toolbar watches it and reschedules the Web Audio graph from the current playhead — already
   *  scheduled sources can't be moved, so without this the old layout keeps playing until replay. */
  scheduleVersion: number;

  loadAudioFile: (filePath: string) => Promise<AudioTrackData>;
  setPlaying: (playing: boolean) => void;
  setPreparing: (preparing: boolean) => void;
  setSeeking: (seeking: boolean) => void;
  bumpSchedule: () => void;
  pruneUnusedAudioCache: () => void;
}

export const useAudioStore = create<AudioState>((set, get) => ({
  audioFiles: {},
  loadingPaths: [],
  isPlaying: false,
  preparing: false,
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
        peaks_l?: number[];
        peaks_r?: number[];
        playback_path: string;
      }>("load_audio_file", { path: filePath });

      const data: AudioTrackData = {
        filePath,
        playbackPath: info.playback_path,
        durationMs: info.duration_ms,
        sampleRate: info.sample_rate,
        peaks: info.peaks,
        peaksL: info.peaks_l && info.peaks_l.length > 0 ? info.peaks_l : info.peaks,
        peaksR: info.peaks_r && info.peaks_r.length > 0 ? info.peaks_r : info.peaks,
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
  setPreparing: (preparing) => set({ preparing }),
  setSeeking: (seeking) => set({ seeking }),
  bumpSchedule: () => set((s) => ({ scheduleVersion: s.scheduleVersion + 1 })),

  pruneUnusedAudioCache: () => {
    const projectStore = useProjectStore.getState();
    const currentTracks: Track[] = projectStore.tracks;
    
    const usedPaths = new Set<string>();
    currentTracks.forEach((track: Track) => {
      track.segments?.forEach((segment) => {
        if (segment.content.type === 'audioClip') {
          usedPaths.add(segment.content.sourcePath);
        }
        segment.processedOutputs?.forEach((output) => {
          usedPaths.add(output.audioPath);
        });
      });
    });

    set((s) => {
      const newAudioFiles: Record<string, AudioTrackData> = {};
      let prunedCount = 0;
      
      Object.entries(s.audioFiles).forEach(([path, data]) => {
        if (usedPaths.has(path)) {
          newAudioFiles[path] = data;
        } else {
          prunedCount++;
        }
      });

      if (prunedCount > 0) {
        console.log(`[AudioCache] Pruned ${prunedCount} unused audio file(s) from cache`);
      }

      return { audioFiles: newAudioFiles };
    });
  },
}));
