import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

export interface LogEntry {
  timestamp: string;
  level: string;
  module: string;
  message: string;
}

interface LogStore {
  entries: LogEntry[];
  logDir: string;
  polling: boolean;
  lastTimestamp: string;

  fetchRecent: () => Promise<void>;
  fetchNew: () => Promise<void>;
  fetchLogDir: () => Promise<void>;
  startPolling: () => void;
  stopPolling: () => void;
  clear: () => void;
}

let pollInterval: ReturnType<typeof setInterval> | null = null;

export const useLogStore = create<LogStore>((set, get) => ({
  entries: [],
  logDir: "",
  polling: false,
  lastTimestamp: "",

  fetchRecent: async () => {
    try {
      const logs = await invoke<LogEntry[]>("get_recent_logs", { count: 500 });
      const last = logs.length > 0 ? logs[logs.length - 1]!.timestamp : "";
      set({ entries: logs, lastTimestamp: last });
    } catch {
      // ignore
    }
  },

  fetchNew: async () => {
    const { lastTimestamp } = get();
    if (!lastTimestamp) {
      return get().fetchRecent();
    }
    try {
      const newLogs = await invoke<LogEntry[]>("get_logs_since", { after: lastTimestamp });
      if (newLogs.length > 0) {
        set((s) => {
          const combined = [...s.entries, ...newLogs];
          const trimmed = combined.length > 2000 ? combined.slice(-2000) : combined;
          return {
            entries: trimmed,
            lastTimestamp: newLogs[newLogs.length - 1]!.timestamp,
          };
        });
      }
    } catch {
      // ignore
    }
  },

  fetchLogDir: async () => {
    try {
      const dir = await invoke<string>("get_log_file_path");
      set({ logDir: dir });
    } catch {
      // ignore
    }
  },

  startPolling: () => {
    if (pollInterval) return;
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
