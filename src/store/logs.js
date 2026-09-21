import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
let pollInterval = null;
export const useLogStore = create((set, get) => ({
    entries: [],
    logDir: "",
    polling: false,
    lastTimestamp: "",
    fetchRecent: async () => {
        try {
            const logs = await invoke("get_recent_logs", { count: 500 });
            const last = logs.length > 0 ? logs[logs.length - 1].timestamp : "";
            set({ entries: logs, lastTimestamp: last });
        }
        catch {
            // ignore
        }
    },
    fetchNew: async () => {
        const { lastTimestamp } = get();
        if (!lastTimestamp) {
            return get().fetchRecent();
        }
        try {
            const newLogs = await invoke("get_logs_since", { after: lastTimestamp });
            if (newLogs.length > 0) {
                set((s) => {
                    const combined = [...s.entries, ...newLogs];
                    const trimmed = combined.length > 2000 ? combined.slice(-2000) : combined;
                    return {
                        entries: trimmed,
                        lastTimestamp: newLogs[newLogs.length - 1].timestamp,
                    };
                });
            }
        }
        catch {
            // ignore
        }
    },
    fetchLogDir: async () => {
        try {
            const dir = await invoke("get_log_file_path");
            set({ logDir: dir });
        }
        catch {
            // ignore
        }
    },
    startPolling: () => {
        if (pollInterval)
            return;
        get().fetchRecent();
        get().fetchLogDir();
        pollInterval = setInterval(() => get().fetchNew(), 1000);
        set({ polling: true });
    },
    stopPolling: () => {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
        set({ polling: false });
    },
    clear: () => set({ entries: [], lastTimestamp: "" }),
}));
