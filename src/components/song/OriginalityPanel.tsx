// 规划 12.4：原创度自检面板 —— 纯前端指标 + 现有后端检测（analyze_segment_tempo）。
// 输出：BPM 差异、原创化清单勾选、综合参考分、可导出 originality-report.json 归档。
// 免责：只提供技术指标，不构成法律意见。
import { useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { useTranslation } from "react-i18next";
import "./OriginalityPanel.css";

interface AnalyzeResult {
  bpm: number;
  confidence: number;
}

export interface OriginalityChecklist {
  voiceReplaced: boolean;
  lyricsRewritten: boolean;
  arrangementChanged: boolean;
  soundSourcesNew: boolean;
  waveformRetained: boolean; // 路线 A = false（无原始波形保留）
}

const DEFAULT_CHECKLIST: OriginalityChecklist = {
  voiceReplaced: false,
  lyricsRewritten: false,
  arrangementChanged: false,
  soundSourcesNew: false,
  waveformRetained: false,
};

async function detectBpm(path: string): Promise<number | null> {
  try {
    const res = await invoke<AnalyzeResult>("analyze_segment_tempo", {
      path,
      windowStartMs: 0,
      windowEndMs: 600000,
      beatsPerBar: 4,
    });
    return res.bpm;
  } catch {
    return null;
  }
}

export function OriginalityPanel({
  sourcePath,
  resultPath,
  onClose,
}: {
  sourcePath: string | null;
  resultPath: string | null;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const [srcBpm, setSrcBpm] = useState<number | null>(null);
  const [outBpm, setOutBpm] = useState<number | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [checklist, setChecklist] = useState<OriginalityChecklist>(DEFAULT_CHECKLIST);

  useEffect(() => {
    if (!sourcePath && !resultPath) return;
    setAnalyzing(true);
    void (async () => {
      const [a, b] = await Promise.all([
        sourcePath ? detectBpm(sourcePath) : Promise.resolve(null),
        resultPath ? detectBpm(resultPath) : Promise.resolve(null),
      ]);
      setSrcBpm(a);
      setOutBpm(b);
      setAnalyzing(false);
    })();
  }, [sourcePath, resultPath]);

  /** 参考分：BPM 差异（0~20）+ 清单勾选项（4×15）- 原波形保留惩罚（20） */
  const score = useMemo(() => {
    let s = 0;
    if (srcBpm != null && outBpm != null && srcBpm > 0) {
      const diff = Math.min(20, Math.round((Math.abs(outBpm - srcBpm) / srcBpm) * 100));
      s += diff;
    }
    const { waveformRetained, ...flags } = checklist;
    s += Object.values(flags).filter(Boolean).length * 15;
    if (waveformRetained) s = Math.max(0, s - 20);
    return Math.min(100, s);
  }, [srcBpm, outBpm, checklist]);

  const items: Array<{ key: keyof OriginalityChecklist; label: string }> = [
    { key: "voiceReplaced", label: t("songOriginality.voiceReplaced") },
    { key: "lyricsRewritten", label: t("songOriginality.lyricsRewritten") },
    { key: "arrangementChanged", label: t("songOriginality.arrangementChanged") },
    { key: "soundSourcesNew", label: t("songOriginality.soundSourcesNew") },
    { key: "waveformRetained", label: t("songOriginality.waveformRetained") },
  ];

  const exportReport = async () => {
    const path = await save({
      title: t("songOriginality.export"),
      defaultPath: "originality-report.json",
      filters: [{ name: "JSON", extensions: ["json"] }],
    });
    if (!path) return;
    const report = {
      generatedAt: new Date().toISOString(),
      disclaimer: t("songOriginality.disclaimer"),
      bpm: { source: srcBpm, result: outBpm },
      checklist,
      referenceScore: score,
      paths: { source: sourcePath, result: resultPath },
    };
    const { writeTextFile } = await import("@tauri-apps/plugin-fs");
    await writeTextFile(path, JSON.stringify(report, null, 2));
  };

  return (
    <div className="confirm-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="confirm-dialog orig-dialog">
        <div className="confirm-title">{t("songOriginality.title")}</div>

        <div className="orig-bpm-row">
          <div className="orig-bpm-card">
            <span className="orig-bpm-label">{t("songOriginality.source")}</span>
            <span className="orig-bpm-value">{analyzing ? "…" : srcBpm != null ? `${srcBpm} BPM` : "—"}</span>
          </div>
          <div className="orig-bpm-card">
            <span className="orig-bpm-label">{t("songOriginality.result")}</span>
            <span className="orig-bpm-value">{analyzing ? "…" : outBpm != null ? `${outBpm} BPM` : "—"}</span>
          </div>
          {srcBpm != null && outBpm != null && srcBpm > 0 && (
            <div className="orig-bpm-card diff">
              <span className="orig-bpm-label">{t("songOriginality.bpmDiff")}</span>
              <span className="orig-bpm-value">
                {Math.round((Math.abs(outBpm - srcBpm) / srcBpm) * 100)}%
              </span>
            </div>
          )}
        </div>

        <div className="orig-checklist">
          {items.map((it) => (
            <label key={it.key} className="orig-check">
              <input
                type="checkbox"
                checked={checklist[it.key]}
                onChange={(e) => setChecklist((c) => ({ ...c, [it.key]: e.target.checked }))}
              />
              <span>{it.label}</span>
            </label>
          ))}
        </div>

        <div className="orig-score">
          <span>{t("songOriginality.score")}</span>
          <strong className={score >= 60 ? "good" : "low"}>{score}</strong>
          <span>/ 100</span>
        </div>

        <div className="orig-disclaimer">{t("songOriginality.disclaimer")}</div>

        <div className="orig-actions">
          <button type="button" className="sow-btn" onClick={onClose}>
            {t("common.close")}
          </button>
          <button type="button" className="sow-btn primary" onClick={() => void exportReport()}>
            {t("songOriginality.export")}
          </button>
        </div>
      </div>
    </div>
  );
}
