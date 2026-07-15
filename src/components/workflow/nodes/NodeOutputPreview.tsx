import { useEffect, useRef, useState } from "react";
import { readFile } from "@tauri-apps/plugin-fs";
import { useTranslation } from "react-i18next";
import { useWorkflowStore } from "../../../store/workflow";
import { useAppStore } from "../../../store/app";
import { preview } from "../../common/previewPlayer";
import { Scrubber } from "../../common/Scrubber";
import { t18 } from "../../../lib/models/msst-catalog";

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

  return (
    <div className="wf-preview nodrag" onPointerDown={(e) => e.stopPropagation()}>
      {outputs.map((path, i) => {
        if (!path) return null; // sparse rehydrated slot
        const isActive = active === i;
        const glyph = isActive && phase === "playing" ? "❚❚" : isActive && phase === "loading" ? "◌" : "▶";
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
          </div>
        );
      })}
    </div>
  );
}
