import { useEffect, useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import { useDraggable } from "../../lib/useDraggable";
import "./Settings.css";

interface HardwareInfo {
  gpu_name: string;
  cuda_available: boolean;
  directml_available: boolean;
  current_device: string;
}

interface CudaProgress {
  stage: string;
  progress: number;
  message: string;
}

export function Settings({ onClose }: { onClose: () => void }) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const { pos, startDrag } = useDraggable(() => ({ x: 72, y: 84 }));
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [device, setDevice] = useState("auto");
  const [saving, setSaving] = useState(false);
  const [cudaReady, setCudaReady] = useState(false);
  const [cudaDownloading, setCudaDownloading] = useState(false);
  const [cudaProgress, setCudaProgress] = useState<CudaProgress | null>(null);
  const [cudaError, setCudaError] = useState<string | null>(null);
  const [cudaJustInstalled, setCudaJustInstalled] = useState(false);
  const [dataDir, setDataDir] = useState("");
  const [relocating, setRelocating] = useState(false);
  const [relocateMsg, setRelocateMsg] = useState<string | null>(null);

  useEffect(() => {
    invoke<HardwareInfo>("get_hardware_info").then(setHw).catch(() => {});
    invoke<string>("get_device_preference").then(setDevice).catch(() => {});
    invoke<boolean>("is_cuda_runtime_ready").then(setCudaReady).catch(() => {});
    invoke<string>("get_data_dir").then(setDataDir).catch(() => {});
  }, []);

  const handleRelocate = useCallback(async () => {
    const dir = await open({ directory: true, multiple: false, title: "Choose data directory" });
    if (!dir || typeof dir !== "string") return;
    setRelocating(true);
    setRelocateMsg(null);
    try {
      await invoke("migrate_data_dir", { newDir: dir });
      setDataDir(dir);
      setRelocateMsg("migrated");
    } catch (e) {
      setRelocateMsg(`error: ${e}`);
    } finally {
      setRelocating(false);
    }
  }, []);

  useEffect(() => {
    const unlisten = listen<CudaProgress>("cuda-download-progress", (e) => {
      setCudaProgress(e.payload);
      if (e.payload.stage === "done") {
        setCudaDownloading(false);
        setCudaReady(true);
        setCudaJustInstalled(true);
      }
    });
    return () => { unlisten.then((f) => f()); };
  }, []);

  const handleLangChange = (value: string) => {
    i18n.changeLanguage(value);
    try { localStorage.setItem("lang", value); } catch { /* ignore */ }
  };

  const handleDeviceChange = useCallback(async (value: string) => {
    setDevice(value);
    setSaving(true);
    try {
      await invoke("set_device_preference", { device: value });
    } catch (e) {
      console.error("Failed to save device preference:", e);
    }
    setSaving(false);
  }, []);

  const handleCudaDownload = useCallback(async () => {
    setCudaDownloading(true);
    setCudaError(null);
    setCudaProgress({ stage: "start", progress: 0, message: "Starting..." });
    try {
      await invoke("download_cuda_runtime");
    } catch (e) {
      setCudaError(String(e));
      setCudaDownloading(false);
    }
  }, []);

  const L = (key: string) => {
    const map: Record<string, Record<string, string>> = {
      title: { zh: "设置", en: "Settings", ja: "設定" },
      language: { zh: "界面语言", en: "Language", ja: "表示言語" },
      hardware: { zh: "硬件", en: "Hardware", ja: "ハードウェア" },
      gpu: { zh: "显卡", en: "GPU", ja: "GPU" },
      cuda: { zh: "CUDA 可用", en: "CUDA Available", ja: "CUDA 利用可能" },
      directml: { zh: "DirectML 可用", en: "DirectML Available", ja: "DirectML 利用可能" },
      epLabel: { zh: "推理设备", en: "Inference Device", ja: "推論デバイス" },
      auto: { zh: "自动", en: "Auto", ja: "自動" },
      cudaOpt: { zh: "CUDA (NVIDIA GPU)", en: "CUDA (NVIDIA GPU)", ja: "CUDA (NVIDIA GPU)" },
      dmlOpt: { zh: "DirectML (通用 GPU)", en: "DirectML (Any GPU)", ja: "DirectML (汎用 GPU)" },
      cpuOpt: { zh: "CPU", en: "CPU", ja: "CPU" },
      saved: { zh: "已保存", en: "Saved", ja: "保存済み" },
      note: { zh: "切换设备后需要重启应用才能生效。", en: "Restart the app after changing device.", ja: "デバイス変更後、アプリを再起動してください。" },
      yes: { zh: "是", en: "Yes", ja: "はい" },
      no: { zh: "否", en: "No", ja: "いいえ" },
      cudaRuntime: { zh: "CUDA 运行时", en: "CUDA Runtime", ja: "CUDAランタイム" },
      cudaInstalled: { zh: "已就绪", en: "Ready", ja: "準備完了" },
      cudaNotInstalled: { zh: "未安装", en: "Not Installed", ja: "未インストール" },
      cudaDownload: { zh: "下载 CUDA 运行时", en: "Download CUDA Runtime", ja: "CUDAランタイムをダウンロード" },
      cudaDownloading: { zh: "下载中...", en: "Downloading...", ja: "ダウンロード中..." },
      cudaNote: { zh: "需要 CUDA 12 Toolkit。下载 ORT CUDA DLLs (~200MB) + cuDNN (~400MB)。", en: "Requires CUDA 12 Toolkit. Downloads ORT CUDA DLLs (~200MB) + cuDNN (~400MB).", ja: "CUDA 12 Toolkit が必要です。ORT CUDA DLLs + cuDNN をダウンロードします。" },
      cudaRestart: { zh: "下载完成，重启应用后生效。", en: "Download complete. Restart to activate.", ja: "ダウンロード完了。再起動で有効になります。" },
      storage: { zh: "存储位置", en: "Storage", ja: "保存場所" },
      dataDir: { zh: "数据目录（模型 + 缓存）", en: "Data folder (models + cache)", ja: "データフォルダ（モデル + キャッシュ）" },
      relocate: { zh: "更改并迁移…", en: "Change & migrate…", ja: "変更して移行…" },
      relocating: { zh: "迁移中…", en: "Migrating…", ja: "移行中…" },
      relocated: { zh: "已迁移，重启后生效（旧数据保留，确认无误后可手动删除）", en: "Migrated — restart to apply (old data kept; delete it manually once confirmed)", ja: "移行完了 — 再起動で有効（旧データは保持）" },
      dataDirNote: { zh: "默认在程序目录旁，避免占用 C 盘。模型/缓存会很大，可换到其他盘。", en: "Defaults next to the program (off C:). Models/cache grow large — point this at another drive.", ja: "既定はプログラム横（Cドライブ外）。" },
    };
    return map[key]?.[lang] ?? map[key]?.en ?? key;
  };

  return (
    <aside className="settings-panel" style={{ left: pos.x, top: pos.y }}>
      <div className="panel-header" onMouseDown={startDrag}>
        <span className="panel-title">{L("title")}</span>
        <button className="panel-close" onClick={onClose}>X</button>
      </div>

      <div className="settings-content">
        <section className="settings-section">
          <h3 className="settings-section-title">{L("language")}</h3>
          <div className="settings-field">
            <select value={lang} onChange={(e) => handleLangChange(e.target.value)}>
              <option value="zh">简体中文</option>
              <option value="en">English</option>
              <option value="ja">日本語</option>
            </select>
          </div>
        </section>

        <section className="settings-section">
          <h3 className="settings-section-title">{L("hardware")}</h3>

          {hw && (
            <div className="settings-hw-info">
              <div className="settings-row" style={{ flexDirection: "column", alignItems: "flex-start", gap: 2 }}>
                <span className="settings-label">{L("gpu")}</span>
                {hw.gpu_name.split(", ").map((name, i) => (
                  <span key={i} className="settings-value" style={{ maxWidth: "100%" }}>{name}</span>
                ))}
              </div>
              <div className="settings-row">
                <span className="settings-label">{L("cuda")}</span>
                <span className={`settings-badge ${hw.cuda_available ? "ok" : "no"}`}>
                  {hw.cuda_available ? L("yes") : L("no")}
                </span>
              </div>
              <div className="settings-row">
                <span className="settings-label">{L("directml")}</span>
                <span className={`settings-badge ${hw.directml_available ? "ok" : "no"}`}>
                  {hw.directml_available ? L("yes") : L("no")}
                </span>
              </div>
            </div>
          )}

          <div className="settings-field">
            <label>{L("epLabel")}</label>
            <select value={device} onChange={(e) => handleDeviceChange(e.target.value)}>
              <option value="auto">{L("auto")}</option>
              <option value="cuda" disabled={!hw?.cuda_available}>{L("cudaOpt")}</option>
              <option value="directml" disabled={!hw?.directml_available}>{L("dmlOpt")}</option>
              <option value="cpu">{L("cpuOpt")}</option>
            </select>
            {saving && <span className="settings-saving">...</span>}
          </div>

          <p className="settings-note">{L("note")}</p>
        </section>

        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("storage")}</h3>
          <div className="settings-field" style={{ flexDirection: "column", alignItems: "flex-start", gap: 2 }}>
            <label>{L("dataDir")}</label>
            <span className="settings-value" style={{ maxWidth: "100%", wordBreak: "break-all", fontSize: 11 }}>{dataDir || "…"}</span>
          </div>
          <div className="settings-field">
            <button className="settings-btn" onClick={handleRelocate} disabled={relocating} style={{ padding: "5px 12px", cursor: "pointer" }}>
              {relocating ? L("relocating") : L("relocate")}
            </button>
          </div>
          {relocateMsg === "migrated" && <p className="settings-note">{L("relocated")}</p>}
          {relocateMsg?.startsWith("error") && <p className="settings-note" style={{ color: "#f87171" }}>{relocateMsg}</p>}
          <p className="settings-note">{L("dataDirNote")}</p>
        </section>

        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("cudaRuntime")}</h3>

          <div className="settings-hw-info">
            <div className="settings-row">
              <span className="settings-label">{L("cudaRuntime")}</span>
              <span className={`settings-badge ${cudaReady ? "ok" : "no"}`}>
                {cudaReady ? L("cudaInstalled") : L("cudaNotInstalled")}
              </span>
            </div>
          </div>

          {!cudaReady && !cudaDownloading && (
            <>
              <button className="settings-download-btn" onClick={handleCudaDownload}>
                {L("cudaDownload")}
              </button>
              <p className="settings-note">{L("cudaNote")}</p>
            </>
          )}

          {cudaDownloading && cudaProgress && (
            <div className="settings-progress">
              <div className="settings-progress-bar">
                <div
                  className="settings-progress-fill"
                  style={{ width: `${Math.round(cudaProgress.progress * 100)}%` }}
                />
              </div>
              <span className="settings-progress-text">{cudaProgress.message}</span>
            </div>
          )}

          {cudaError && (
            <p className="settings-error">{cudaError}</p>
          )}

          {cudaJustInstalled && (
            <p className="settings-note" style={{ color: "#4ade80" }}>{L("cudaRestart")}</p>
          )}
        </section>
      </div>
    </aside>
  );
}
