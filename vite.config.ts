import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const host = process.env.TAURI_DEV_HOST;

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
      ignored: [
        "**/src-tauri/**",
        "**/data/**",
        "**/training/**",
        "**/converter/**",
        "**/runtime/**",
        "**/models/**",
        "**/python/**",
      ],
    },
  },
}));
