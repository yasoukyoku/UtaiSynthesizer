import { useEffect, useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import { useAppStore } from "../../store/app";
import { useDraggable } from "../../lib/useDraggable";
import "./Settings.css";

interface GpuAdapter {
  name: string;
  vendor: string; // "nvidia" | "amd" | "intel" | "other"
}

interface HardwareInfo {
  gpu_name: string;
  cuda_available: boolean;
  directml_available: boolean;
  current_device: string;
  gpus: GpuAdapter[];
  recommended_variant: string;
}

interface CudaProgress {
  stage: string;
  progress: number;
  message: string;
}

/** Mirror of Rust `pyenv::PackStatus` (meta flattened) — list_runtime_packs entry. */
interface RuntimePack {
  id: string;
  variant: string;
  label: string;
  python: string;
  torch: string;
  disk_bytes: number;
  path: string;
  envtest?: { overall?: string } | null;
}

interface RuntimeCatalogItem {
  id: string;
  variant: string;
  label: string;
  download_bytes: number;
  disk_bytes: number;
  experimental: boolean;
  downloadable: boolean;
  installed: boolean;
}

interface RuntimeEnvInfo {
  root: string;
  root_ascii_ok: boolean;
  packs: RuntimePack[];
  catalog: RuntimeCatalogItem[];
  /** Backend busy flags — the component rebuilds its state from these on (re)mount,
   *  so closing/reopening the panel mid-install keeps the progress + cancel UI alive. */
  installing: boolean;
  envtest_running: boolean;
}

interface PyenvProgress {
  id: string;
  phase: string;
  progress: number;
  message: string;
}

const fmtGB = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);

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
  const showConfirm = useAppStore((s) => s.showConfirm);
  const [rt, setRt] = useState<RuntimeEnvInfo | null>(null);
  const [rtBusy, setRtBusy] = useState(false);
  const [rtProgress, setRtProgress] = useState<PyenvProgress | null>(null);
  const [rtError, setRtError] = useState<string | null>(null);
  const [rtNotice, setRtNotice] = useState<string | null>(null);
  const [envtesting, setEnvtesting] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const refreshRuntime = useCallback(() => {
    invoke<RuntimeEnvInfo>("get_runtime_env_info")
      .then((info) => {
        setRt(info);
        // Rebuild busy state from the backend (panel may have been closed and
        // reopened mid-install — component state alone would strand the cancel
        // button and mislabel every other button as available).
        setRtBusy(info.installing);
        setEnvtesting((prev) => {
          if (info.envtest_running) return prev ?? "__backend__";
          return prev === "__backend__" ? null : prev;
        });
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    invoke<HardwareInfo>("get_hardware_info").then(setHw).catch(() => {});
    invoke<string>("get_device_preference").then(setDevice).catch(() => {});
    invoke<boolean>("is_cuda_runtime_ready").then(setCudaReady).catch(() => {});
    invoke<string>("get_data_dir").then(setDataDir).catch(() => {});
    refreshRuntime();
  }, [refreshRuntime]);

  useEffect(() => {
    const unlisten = listen<PyenvProgress>("pyenv-progress", (e) => {
      setRtProgress(e.payload);
      if (e.payload.phase === "done" || e.payload.phase === "error") {
        setRtBusy(false);
        if (e.payload.phase === "error") setRtError(e.payload.message);
        // The done message can carry a REAL verdict（"已安装，但自检未通过：…"）—
        // it must survive the progress bar disappearing, not vanish with it.
        if (e.payload.phase === "done") setRtNotice(e.payload.message);
        refreshRuntime();
      }
    });
    return () => { unlisten.then((f) => f()); };
  }, [refreshRuntime]);

  useEffect(() => {
    // Envtest lifecycle channel: the final {type:"done"} is the ONLY signal a
    // backend-started (or another-instance-started) self-test has ended — without
    // this, a panel that entered the "__backend__" sentinel could never leave it.
    const unlisten = listen<{ id: string; event: { type?: string } }>("pyenv-envtest", (e) => {
      if (e.payload?.event?.type === "done") refreshRuntime();
    });
    return () => { unlisten.then((f) => f()); };
  }, [refreshRuntime]);

  const handleRtDownload = useCallback(async (id: string) => {
    setRtBusy(true);
    setRtError(null);
    setRtNotice(null);
    setRtProgress(null);
    try {
      await invoke("download_runtime_pack", { id });
    } catch (e) {
      setRtError(String(e));
    } finally {
      setRtBusy(false);
      refreshRuntime();
    }
  }, [refreshRuntime]);

  const handleRtLocalInstall = useCallback(async () => {
    const file = await open({
      multiple: false,
      title: "runtime pack (.tar.zst / .part01)",
      filters: [{ name: "Runtime pack", extensions: ["zst", "part01"] }],
    });
    if (!file || typeof file !== "string") return;
    setRtBusy(true);
    setRtError(null);
    setRtNotice(null);
    setRtProgress(null);
    try {
      await invoke("install_runtime_pack_local", { path: file });
    } catch (e) {
      setRtError(String(e));
    } finally {
      setRtBusy(false);
      refreshRuntime();
    }
  }, [refreshRuntime]);

  const handleRtEnvtest = useCallback(async (id: string) => {
    setEnvtesting(id);
    setRtError(null);
    setRtNotice(null);
    try {
      await invoke("run_pack_envtest", { id });
    } catch (e) {
      setRtError(String(e));
    } finally {
      setEnvtesting(null);
      refreshRuntime();
    }
  }, [refreshRuntime]);

  const handleRtDelete = useCallback(async (id: string) => {
    const choice = await showConfirm({
      title: L("rtDeleteTitle"),
      body: `${id}\n${L("rtDeleteBody")}`,
      buttons: [
        { id: "cancel", label: L("rtCancelBtn") },
        { id: "del", label: L("rtDelete"), kind: "danger" },
      ],
    });
    if (choice !== "del") return;
    // Visible busy state for the whole removal (a 1 GB tree takes seconds — with
    // no feedback users assume a hang and click again).
    setDeleting(id);
    setRtError(null);
    setRtNotice(null);
    try {
      await invoke("delete_runtime_pack", { id });
    } catch (e) {
      setRtError(String(e));
    } finally {
      setDeleting(null);
      refreshRuntime();
    }
  }, [refreshRuntime, showConfirm, lang]);

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
      rtTitle: { zh: "训练环境（内嵌 Python 运行时）", en: "Training Runtime (embedded Python)", ja: "トレーニング環境（内蔵 Python）" },
      rtRoot: { zh: "运行时目录", en: "Runtime folder", ja: "ランタイムフォルダ" },
      rtAsciiWarn: { zh: "数据目录路径含非英文字符——内嵌 Python/torch 在此类路径下会加载失败。请先在「存储位置」迁移到纯英文路径（如 D:\\UtaiData）。", en: "Data folder path contains non-ASCII characters — the embedded Python/torch will fail to load there. Migrate the data folder to an ASCII-only path first.", ja: "データフォルダに非 ASCII 文字が含まれています。内蔵 Python/torch が読み込めないため、英数字のみのパスへ移行してください。" },
      rtTestPass: { zh: "自检通过", en: "Self-test passed", ja: "セルフテスト合格" },
      rtTestFail: { zh: "自检未通过", en: "Self-test failed", ja: "セルフテスト不合格" },
      rtTestNone: { zh: "未自检", en: "Not tested", ja: "未テスト" },
      rtTest: { zh: "自检", en: "Self-test", ja: "セルフテスト" },
      rtTesting: { zh: "自检中…", en: "Testing…", ja: "テスト中…" },
      rtDelete: { zh: "删除", en: "Delete", ja: "削除" },
      rtDeleting: { zh: "删除中…", en: "Deleting…", ja: "削除中…" },
      rtDeleteTitle: { zh: "删除运行时包", en: "Delete runtime pack", ja: "ランタイムパックを削除" },
      rtDeleteBody: { zh: "该运行时包将从磁盘删除（之后可重新下载或从本地文件安装）。", en: "The pack will be removed from disk (it can be re-downloaded or re-installed later).", ja: "ディスクから削除されます（後で再ダウンロード/再インストール可能）。" },
      rtCancelBtn: { zh: "取消", en: "Cancel", ja: "キャンセル" },
      rtDownload: { zh: "下载", en: "Download", ja: "ダウンロード" },
      rtNotPublished: { zh: "在线包尚未发布——可用「从本地文件安装」。", en: "Online pack not published yet — use “Install from local file”.", ja: "オンライン版は未公開——「ローカルから」をご利用ください。" },
      rtLocalInstall: { zh: "从本地文件安装…", en: "Install from local file…", ja: "ローカルからインストール…" },
      rtCancel: { zh: "取消安装", en: "Cancel install", ja: "インストール中止" },
      rtExperimental: { zh: "实验性", en: "Experimental", ja: "実験的" },
      rtNote: { zh: "模型转换与训练使用此内嵌运行时（无需系统 Python）。GPU 训练包（NVIDIA 20-50 系 / AMD / Intel）按阶段加入。", en: "Model conversion and training run on this embedded runtime (no system Python needed). GPU packs (NVIDIA 20-50 / AMD / Intel) arrive in stages.", ja: "モデル変換とトレーニングはこの内蔵ランタイムで実行されます。GPU 版は段階的に追加されます。" },
      rtRecommend: { zh: "本机推荐变体", en: "Recommended variant", ja: "推奨バリアント" },
      rtPackLabel_cpu: { zh: "CPU 运行时（模型转换基座 + CPU 训练）", en: "CPU runtime (model conversion base + CPU training)", ja: "CPU ランタイム（モデル変換基盤 + CPU トレーニング）" },
    };
    return map[key]?.[lang] ?? map[key]?.en ?? key;
  };

  /** Catalog labels live in Rust (single catalog source) but as data, not copy —
   *  translate per variant here, falling back to the backend label for variants
   *  this build doesn't know yet. */
  const packLabel = (c: RuntimeCatalogItem) => {
    const key = `rtPackLabel_${c.variant.replace(/-/g, "_")}`;
    const v = L(key);
    return v === key ? c.label : v;
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
                {(hw.gpus?.length
                  ? hw.gpus
                  : hw.gpu_name.split(", ").map((name) => ({ name, vendor: "" }))
                ).map((g, i) => (
                  <span key={i} className="settings-value" style={{ maxWidth: "100%", display: "flex", gap: 6, alignItems: "center" }}>
                    {g.name}
                    {g.vendor && g.vendor !== "other" && (
                      <span className="settings-badge ok" style={{ textTransform: "uppercase" }}>{g.vendor}</span>
                    )}
                  </span>
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

        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("rtTitle")}</h3>
          {rt && (
            <>
              <div className="settings-field" style={{ flexDirection: "column", alignItems: "flex-start", gap: 2 }}>
                <label>{L("rtRoot")}</label>
                <span className="settings-value" style={{ maxWidth: "100%", wordBreak: "break-all", whiteSpace: "normal", fontSize: 11 }}>{rt.root || "…"}</span>
              </div>
              {!rt.root_ascii_ok && <p className="settings-error">{L("rtAsciiWarn")}</p>}

              {rt.packs.length > 0 && (
                <div className="settings-hw-info">
                  {rt.packs.map((p) => (
                    <div key={p.id} className="settings-pack">
                      <div className="settings-row">
                        <span className="settings-label">{p.variant}</span>
                        <span className={`settings-badge ${p.envtest?.overall === "pass" ? "ok" : "no"}`}>
                          {p.envtest ? (p.envtest.overall === "pass" ? L("rtTestPass") : L("rtTestFail")) : L("rtTestNone")}
                        </span>
                      </div>
                      <span className="settings-value" style={{ textAlign: "left", maxWidth: "100%" }}>
                        {p.id} · torch {p.torch || "?"} · py {p.python || "?"} · {fmtGB(p.disk_bytes)}
                      </span>
                      <div className="settings-pack-actions">
                        <button
                          className="settings-mini-btn"
                          disabled={rtBusy || envtesting !== null || deleting !== null}
                          onClick={() => handleRtEnvtest(p.id)}
                        >
                          {envtesting === p.id ? L("rtTesting") : L("rtTest")}
                        </button>
                        <button
                          className="settings-mini-btn danger"
                          disabled={rtBusy || envtesting !== null || deleting !== null}
                          onClick={() => handleRtDelete(p.id)}
                        >
                          {deleting === p.id ? L("rtDeleting") : L("rtDelete")}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {rt.catalog.filter((c) => !c.installed).map((c) => (
                <div key={c.id} className="settings-field">
                  <label>{packLabel(c)}{c.experimental ? `（${L("rtExperimental")}）` : ""}</label>
                  {c.downloadable ? (
                    <button
                      className="settings-download-btn"
                      disabled={rtBusy || envtesting !== null || deleting !== null}
                      onClick={() => handleRtDownload(c.id)}
                    >
                      {L("rtDownload")}（~{fmtGB(c.download_bytes)} / {L("rtRoot")} {fmtGB(c.disk_bytes)}）
                    </button>
                  ) : (
                    <p className="settings-note">{L("rtNotPublished")}</p>
                  )}
                </div>
              ))}

              <div className="settings-pack-actions">
                <button className="settings-mini-btn" disabled={rtBusy || envtesting !== null || deleting !== null} onClick={handleRtLocalInstall}>
                  {L("rtLocalInstall")}
                </button>
                {rtBusy && (
                  <button className="settings-mini-btn danger" onClick={() => { invoke("cancel_runtime_install").catch(() => {}); }}>
                    {L("rtCancel")}
                  </button>
                )}
              </div>

              {rtBusy && rtProgress && (
                <div className="settings-progress">
                  <div className="settings-progress-bar">
                    <div
                      className="settings-progress-fill"
                      style={{ width: `${Math.round(Math.min(1, Math.max(0, rtProgress.progress)) * 100)}%` }}
                    />
                  </div>
                  <span className="settings-progress-text">{rtProgress.message}</span>
                </div>
              )}
              {rtNotice && !rtBusy && (
                <p
                  className="settings-note"
                  style={rtNotice.includes("自检通过") || rtNotice.toLowerCase().includes("pass") ? { color: "#4ade80" } : undefined}
                >
                  {rtNotice}
                </p>
              )}
              {rtError && <p className="settings-error">{rtError}</p>}
              <p className="settings-note">
                {L("rtNote")}
                {hw?.recommended_variant ? ` ${L("rtRecommend")}: ${hw.recommended_variant}` : ""}
              </p>
            </>
          )}
        </section>
      </div>
    </aside>
  );
}
