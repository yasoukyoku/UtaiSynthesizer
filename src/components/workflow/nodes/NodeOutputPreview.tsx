import { useEffect, useRef, useState } from "react";
import { copyFile, exists, readFile } from "@tauri-apps/plugin-fs";
import { open } from "@tauri-apps/plugin-dialog";
import { useTranslation } from "react-i18next";
import { useWorkflowStore } from "../../../store/workflow";
import { useAppStore } from "../../../store/app";
import { preview } from "../../common/previewPlayer";
import { Scrubber } from "../../common/Scrubber";
import { t18 } from "../../../lib/models/msst-catalog";
import { exportOneAudioFileToFolder, laneExportErrorMessage } from "../../../lib/audio/exportLaneAudio";
import { useAmtStore } from "../../../store/amt";

/** Build a collision-free plain-file destination path: <dir>/<base>.<ext>,
 *  <base>_2.<ext>, <base>_3.<ext> … mirroring the project's export naming. */
async function uniqueFilePath(dir: string, base: string, ext: string): Promise<string> {
  let candidate = `${dir}/${base}.${ext}`;
  let n = 2;
  // eslint-disable-next-line no-await-in-loop
  while (await exists(candidate).catch(() => false)) {
    candidate = `${dir}/${base}_${n}.${ext}`;
    n += 1;
  }
  return candidate;
}

/** Phase 5-1: spectrogram PNG artifact thumbnail — reads the cached PNG into a
 *  blob URL (auto-revoked on unmount / path change). */
function PngThumb({ path }: { path: string }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    let created: string | null = null;
    readFile(path)
      .then((bytes) => {
        if (!alive) return;
        created = URL.createObjectURL(new Blob([new Uint8Array(bytes)], { type: "image/png" }));
        setUrl(created);
      })
      .catch(() => {});
    return () => {
      alive = false;
      if (created) URL.revokeObjectURL(created);
    };
  }, [path]);
  if (!url) return null;
  return <img className="wf-preview-thumb" src={url} alt="" />;
}

/** S66 — per-node output audition (the v1-style listen-before-deposit, §user). One compact
 *  row per output port whose wav exists in `nodeOutputs` (every intermediate node's outputs
 *  are real files in the per-run cache dir; rehydrated maps can be SPARSE — holes render
 *  nothing). Playback rides the shared preview singleton (training-audition pattern):
 *  takeover stops any other consumer, unmount/re-run stops us. Interactive controls follow
 *  the ParamSlider RF rules (nodrag + pointer-down stopPropagation). */
export function NodeOutputPreview({
  statusSeg,
  nodeId,
  outputLabels,
}: {
  statusSeg: string | null;
  nodeId: string;
  outputLabels?: string[];
}) {
  const { i18n } = useTranslation();
  const outputs = useWorkflowStore((s) => (statusSeg ? s.nodeOutputs[statusSeg]?.[nodeId] : undefined));
  const [active, setActive] = useState<number | null>(null);
  const [phase, setPhase] = useState<"idle" | "loading" | "playing" | "paused">("idle");
  const [pos, setPos] = useState(0);
  const [dur, setDur] = useState(0);
  const token = useRef(0);
  const activePath = useRef<string | null>(null);

  const resetOwned = () => {
    if (activePath.current && preview.path === activePath.current) {
      preview.onEnd = null;
      preview.stop();
    }
    activePath.current = null;
    setActive(null);
    setPhase("idle");
    setPos(0);
    setDur(0);
  };

  // Unmount / re-run cleanup: a fresh run replaces the paths — stop the stale playback.
  const pathsKey = outputs?.join("|") ?? "";
  useEffect(() => {
    return () => resetOwned();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathsKey]);

  // Position ticker while playing (parent-driven Scrubber contract).
  useEffect(() => {
    if (phase !== "playing") return;
    let raf = 0;
    const tick = () => {
      setPos(preview.position);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [phase]);

  if (!outputs || outputs.every((p) => !p)) return null;

  const toggle = async (idx: number, path: string) => {
    if (active === idx && phase === "playing") {
      preview.pause();
      setPhase("paused");
      return;
    }
    if (active === idx && phase === "paused") {
      preview.resume();
      setPhase("playing");
      return;
    }
    const my = ++token.current;
    // takeover: stop whatever the singleton is doing (training/audition/another node row)
    preview.onEnd = null;
    preview.stop();
    setActive(idx);
    setPhase("loading");
    try {
      const bytes = await readFile(path);
      if (token.current !== my) return; // superseded gesture
      const buffer = await preview.decode(bytes);
      if (token.current !== my) return;
      void preview.play(path, buffer);
      activePath.current = path;
      preview.onEnd = () => {
        activePath.current = null;
        setActive(null);
        setPhase("idle");
        setPos(0);
      };
      setDur(preview.duration);
      setPos(0);
      setPhase("playing");
    } catch {
      if (token.current !== my) return;
      setActive(null);
      setPhase("idle");
      useAppStore
        .getState()
        .showToast(
          t18(
            {
              zh: "试听失败——输出文件可能已被清理，重新运行该节点即可",
              en: "Preview failed — the output file may have been cleaned up; re-run the node",
              ja: "試聴に失敗しました — 出力ファイルが削除された可能性があります。ノードを再実行してください",
            },
            i18n.language,
          ),
          "error",
        );
    }
  };

  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const ss = Math.floor(s % 60);
    return `${m}:${ss.toString().padStart(2, "0")}`;
  };

  // 每条试听行的「下载」按钮：把该节点的输出音频原样复制到本地自选文件夹。
  const download = async (path: string, label?: string) => {
    const out = await open({ directory: true });
    if (!out || typeof out !== "string") return;
    try {
      if (/\.(mid|png|json)$/i.test(path)) {
        // Plain-file outputs (MIDI / PNG artifact / JSON dump) are copied
        // verbatim with the existing _2/_3 collision-free naming convention.
        const ext = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
        const baseName = label ? `${nodeId}_${label}` : nodeId;
        const dest = await uniqueFilePath(out, baseName, ext);
        await copyFile(path, dest);
      } else {
        await exportOneAudioFileToFolder(
          { label: label ?? nodeId, sourcePath: path },
          out,
          label ? `${nodeId}_${label}` : nodeId,
        );
      }
      useAppStore
        .getState()
        .showToast(
          t18({ zh: "已下载", en: "Downloaded", ja: "ダウンロードしました" }, i18n.language),
          "success",
        );
    } catch (e) {
      useAppStore.getState().showToast(laneExportErrorMessage(e), "error");
    }
  };

  return (
    <div className="wf-preview nodrag" onPointerDown={(e) => e.stopPropagation()}>
      {outputs.map((path, i) => {
        if (!path) return null; // sparse rehydrated slot
        const isMidi = path.toLowerCase().endsWith(".mid");
        // Phase 5: report ports carry inline JSON strings (never file paths) —
        // render a copy-only row; also gives complianceCheck's report a real UI.
        const isJson = !isMidi && path.trimStart().startsWith("{");
        const isPng = !isMidi && !isJson && path.toLowerCase().endsWith(".png");
        const isActive = active === i;
        const glyph = isActive && phase === "playing" ? "❚❚" : isActive && phase === "loading" ? "◌" : "▶";
        if (isJson) {
          return (
            <div key={i} className="wf-preview-row">
              <span className="wf-preview-btn" style={{ opacity: 0.7 }}>{"{ }"}</span>
              {outputLabels?.[i] && (
                <span className="wf-preview-label" title={outputLabels[i]}>
                  {outputLabels[i]}
                </span>
              )}
              <span className="wf-preview-scrub" style={{ opacity: 0.6 }}>
                {t18({ zh: "JSON 报告", en: "JSON report", ja: "JSON レポート" }, i18n.language)}
              </span>
              <button
                className="wf-preview-dl"
                title={t18({ zh: "复制报告 JSON", en: "Copy report JSON", ja: "レポート JSON をコピー" }, i18n.language)}
                onClick={() => {
                  void navigator.clipboard
                    .writeText(path)
                    .then(() =>
                      useAppStore
                        .getState()
                        .showToast(
                          t18({ zh: "已复制", en: "Copied", ja: "コピーしました" }, i18n.language),
                          "success",
                        ),
                    )
                    .catch(() =>
                      useAppStore
                        .getState()
                        .showToast(
                          t18({ zh: "复制失败", en: "Copy failed", ja: "コピーに失敗しました" }, i18n.language),
                          "error",
                        ),
                    );
                }}
              >
                ⧉
              </button>
            </div>
          );
        }
        if (isPng) {
          return (
            <div key={i} className="wf-preview-row wf-preview-row-png">
              <PngThumb path={path} />
              {outputLabels?.[i] && (
                <span className="wf-preview-label" title={outputLabels[i]}>
                  {outputLabels[i]}
                </span>
              )}
              <button
                className="wf-preview-dl"
                title={t18({ zh: "下载此图片", en: "Download this image", ja: "この画像をダウンロード" }, i18n.language)}
                onClick={() => void download(path, outputLabels?.[i])}
              >
                ⬇
              </button>
            </div>
          );
        }
        if (isMidi) {
          return (
            <div key={i} className="wf-preview-row">
              <span className="wf-preview-btn" style={{ opacity: 0.7 }}>♪</span>
              {outputLabels?.[i] && (
                <span className="wf-preview-label" title={outputLabels[i]}>
                  {outputLabels[i]}
                </span>
              )}
              <span className="wf-preview-scrub" style={{ opacity: 0.6 }}>
                {t18({ zh: "MIDI 文件", en: "MIDI file", ja: "MIDI ファイル" }, i18n.language)}
              </span>
              <button
                className="wf-preview-dl"
                title={t18({ zh: "打开 MIDI 工作台", en: "Open MIDI workbench", ja: "MIDI ワークベンチを開く" }, i18n.language)}
                onClick={() => useAmtStore.getState().openWorkbench(nodeId)}
              >
                ⚙
              </button>
              <button
                className="wf-preview-dl"
                title={t18({ zh: "下载此 MIDI", en: "Download this MIDI", ja: "この MIDI をダウンロード" }, i18n.language)}
                onClick={() => void download(path, outputLabels?.[i])}
              >
                ⬇
              </button>
            </div>
          );
        }
        return (
          <div key={i} className="wf-preview-row">
            <button
              className={`wf-preview-btn ${isActive && phase !== "idle" ? "on" : ""}`}
              onClick={() => void toggle(i, path)}
            >
              {glyph}
            </button>
            {outputLabels?.[i] && (
              <span className="wf-preview-label" title={outputLabels[i]}>
                {outputLabels[i]}
              </span>
            )}
            <Scrubber
              className="wf-preview-scrub"
              value={isActive && dur > 0 ? Math.min(1, pos / dur) : 0}
              onSeek={(frac) => {
                if (isActive && dur > 0) {
                  preview.seek(frac);
                  setPos(preview.position);
                } else {
                  void toggle(i, path); // inactive row: the track acts as a play trigger
                }
              }}
            />
            <span className="wf-preview-time">{isActive && dur > 0 ? `${fmt(pos)}/${fmt(dur)}` : ""}</span>
            <button
              className="wf-preview-dl"
              title={t18({ zh: "下载此音频", en: "Download this audio", ja: "この音声をダウンロード" }, i18n.language)}
              onClick={() => void download(path, outputLabels?.[i])}
            >
              ⬇
            </button>
          </div>
        );
      })}
    </div>
  );
}
