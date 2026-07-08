import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { relative } from "node:path";

const host = process.env.TAURI_DEV_HOST;

// Repo root (this file's dir). The non-frontend trees to keep the watcher OFF live at the ROOT —
// anchor to it so the ignore does NOT also swallow same-named dirs under src/ (e.g.
// src/components/training/, src/lib/models/), which a bare `**/training/**` glob silently did →
// HMR served stale modules for those files (S46 pitfall root cause).
const ROOT = fileURLToPath(new URL(".", import.meta.url));
const IGNORED_ROOT_DIRS =
  /^(src-tauri|data|training|converter|runtime|models|python)([\\/]|$)/;

export default defineConfig(async () => ({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: "ws", host, port: 1421 } : undefined,
    // Ignore every non-frontend tree — vite otherwise watches the WHOLE repo root:
    // installing a runtime pack (10k+ files under data/runtimes) OOM'd Node's 4 GB
    // heap and took the entire dev server down (S42 live crash), and the .venv /
    // model dirs are GB-scale watcher churn for zero HMR value.
    watch: {
      // Ignore ONLY the ROOT non-frontend trees — vite otherwise watches the whole repo root:
      // installing a runtime pack (10k+ files under data/runtimes) OOM'd Node's 4 GB heap (S42
      // live crash), and .venv/model dirs are GB-scale churn for zero HMR value. Root-anchored so
      // it never ignores src/components/training/ or src/lib/models/ (a bare `**/training/**`
      // glob did → stale HMR). Poll on top (Windows + Tauri-spawned vite miss native fs events for
      // tool writes); polling now only covers src/, so CPU stays low.
      usePolling: true,
      interval: 300,
      ignored: (p: string) =>
        IGNORED_ROOT_DIRS.test(relative(ROOT, p).replace(/\\/g, "/")),
    },
  },
}));
