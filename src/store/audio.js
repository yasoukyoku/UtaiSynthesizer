import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { useProjectStore } from "./project";
export const useAudioStore = create((set, get) => ({
    audioFiles: {},
    loadingPaths: [],
    isPlaying: false,
    preparing: false,
    seeking: false,
    scheduleVersion: 0,
    loadAudioFile: async (filePath) => {
        const existing = get().audioFiles[filePath];
        if (existing)
            return existing;
        set((s) => ({
            loadingPaths: s.loadingPaths.includes(filePath) ? s.loadingPaths : [...s.loadingPaths, filePath],
        }));
        try {
            const info = await invoke("load_audio_file", { path: filePath });
            const data = {
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
        }
        catch (e) {
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
        const currentTracks = projectStore.tracks;
        const usedPaths = new Set();
        currentTracks.forEach((track) => {
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
            const newAudioFiles = {};
            let prunedCount = 0;
            Object.entries(s.audioFiles).forEach(([path, data]) => {
                if (usedPaths.has(path)) {
                    newAudioFiles[path] = data;
                }
                else {
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
