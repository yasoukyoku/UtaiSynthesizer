// 规划 13.3：统一歌曲任务队列 —— 当前任务 + 等待队列（≤3）+ 统一进度 + 取消句柄。
// 歌曲弹窗 / 节点引擎 / 原创向导都从这里提交任务（串行执行），彻底消除
// "多处各自 generating 布尔值"的状态不一致；后端单任务由前端队列保证。
import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { runSongTask, } from "../lib/song/runSongTask";
export const MAX_WAITING = 3;
let seq = 0;
// 串行执行链：前一个任务失败不阻塞后续任务。
let chain = Promise.resolve();
export const useSongTaskQueue = create((set, get) => ({
    current: null,
    waiting: [],
    progress: null,
    taskError: null,
    generatingEntryId: null,
    enqueue: (req) => {
        if (get().waiting.length >= MAX_WAITING) {
            return Promise.reject(new Error(`等待队列已满（最多 ${MAX_WAITING} 个任务）`));
        }
        const id = `songq_${Date.now().toString(36)}_${(seq++).toString(36)}`;
        const item = {
            id,
            task: req.task,
            label: req.label,
            entryId: req.entryId,
        };
        set((s) => ({ waiting: [...s.waiting, item] }));
        const run = async () => {
            set((s) => ({
                current: item,
                waiting: s.waiting.filter((w) => w.id !== id),
                generatingEntryId: req.entryId ?? s.generatingEntryId,
            }));
            try {
                return await runSongTask(req.task, req.payload, {
                    onProgress: (p) => {
                        set({ progress: p });
                        req.onProgress?.(p);
                    },
                });
            }
            finally {
                set((s) => ({
                    current: s.current?.id === id ? null : s.current,
                    progress: null,
                    generatingEntryId: req.entryId && s.generatingEntryId === req.entryId ? null : s.generatingEntryId,
                }));
            }
        };
        const queued = chain.then(run, run);
        chain = queued.catch(() => { });
        return queued;
    },
    cancelCurrent: async () => {
        if (!get().current)
            return;
        try {
            await invoke("song_cancel");
        }
        catch {
            // sidecar 已退出等场景：忽略，runSongTask 会自然 settle
        }
    },
    setTaskError: (error) => set({ taskError: error }),
    setGeneratingEntryId: (id) => set({ generatingEntryId: id }),
}));
/** 非 hook 场景（引擎/向导）直接提交任务 */
export function enqueueSongTask(req) {
    return useSongTaskQueue.getState().enqueue(req);
}
