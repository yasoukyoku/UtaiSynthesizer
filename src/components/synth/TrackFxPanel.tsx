// TrackFxPanel — S12 效果发送 / 总线 (Track Routing / Aux Bus / Sends).
// 每轨两个发送:整轨输出并行叠进全局混响 / 延迟总线(见 effectsBus.connectTrackOutput)。
// 本面板同时暴露全局总线参数(返回量 / 延迟时间 / 反馈),持久化在 localStorage
// (utai.fx*) 并经 setFxBusConfig 推进 effectsBus 单例 — 播放与导出的共同快照。

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useProjectStore } from "../../store/project";
import { loadSetting, saveSetting } from "../../lib/settings";
import {
  DEFAULT_FX_BUS,
  getFxBusConfig,
  setFxBusConfig,
} from "../../lib/audio/effectsBus";
import { ParamSlider } from "../workflow/nodes/ParamSlider";
import "./TempoStretchPanel.css";

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}
function sec2(v: number): string {
  return `${v.toFixed(2)}s`;
}

/** Anchor rect of the opening button — enables flip-above when the panel
 *  would otherwise run off the bottom of the window (bottom tracks). */
export interface FxAnchorRect {
  left: number;
  top: number;
  bottom: number;
}

export function TrackFxPanel({ anchor, track, onClose }: {
  anchor: FxAnchorRect;
  track: import("../../types/project").Track;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const [cfg, setCfg] = useState(() => getFxBusConfig());
  const send = (which: "reverbSend" | "delaySend", v: number) =>
    useProjectStore.getState().setTrackFxSend(track.id, which, v);

  const applyCfg = (patch: Partial<import("../../lib/audio/effectsBus").FxBusConfig>) => {
    const next = { ...cfg, ...patch };
    setFxBusConfig(next);
    setCfg(next);
    saveSetting("utai.fxReverbWet", next.reverbWet);
    saveSetting("utai.fxDelayWet", next.delayWet);
    saveSetting("utai.fxDelayTime", next.delayTimeSec);
    saveSetting("utai.fxDelayFeedback", next.delayFeedback);
  };

  // Hydrate the bus singleton from persisted settings on mount (safe: idempotent).
  useEffect(() => {
    setFxBusConfig({
      reverbWet: loadSetting("utai.fxReverbWet", DEFAULT_FX_BUS.reverbWet),
      delayWet: loadSetting("utai.fxDelayWet", DEFAULT_FX_BUS.delayWet),
      delayTimeSec: loadSetting("utai.fxDelayTime", DEFAULT_FX_BUS.delayTimeSec),
      delayFeedback: loadSetting("utai.fxDelayFeedback", DEFAULT_FX_BUS.delayFeedback),
    });
    // Sync the local sliders with the hydrated (clamped) singleton — otherwise the panel shows the
    // DEFAULT cfg while playback uses the persisted one, and a single drag would overwrite the other
    // three settings back to defaults (audit).
    setCfg({ ...getFxBusConfig() });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pw = 260;
  const px = Math.max(4, Math.min(anchor.left, window.innerWidth - pw - 8));
  const panelRef = useRef<HTMLDivElement | null>(null);
  const [py, setPy] = useState(() => anchor.bottom + 2);

  // Measure the REAL height after mount and keep the FULL panel on screen:
  // prefer below the button, flip above it when that would clip, and as a last
  // resort clamp to the viewport (the CSS max-height makes it scroll inside).
  useLayoutEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const h = el.offsetHeight;
    const vh = window.innerHeight;
    let top = anchor.bottom + 2;
    if (top + h > vh - 6) {
      const above = anchor.top - h - 2;
      top = above >= 4 ? above : Math.max(4, vh - h - 6);
    }
    setPy(top);
  }, [anchor.top, anchor.bottom]);

  return (
    <>
      <div className="stretch-backdrop" onMouseDown={onClose} onContextMenu={(e) => { e.preventDefault(); onClose(); }} />
      <div
        ref={panelRef}
        className="stretch-panel"
        style={{ left: px, top: py, width: pw }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="stretch-title" title={`${t("fx.panelTitle")} · ${track.name}`}>{t("fx.panelTitle")} · {track.name}</div>

        <div className="fx-section-label">{t("fx.trackSend")}</div>
        <ParamSlider
          label={t("fx.reverb")}
          title={t("fx.reverbTitle")}
          min={0} max={1} step={0.01}
          value={track.reverbSend ?? 0}
          onChange={(v) => send("reverbSend", v)}
          format={pct}
        />
        <ParamSlider
          label={t("fx.delay")}
          title={t("fx.delayTitle")}
          min={0} max={1} step={0.01}
          value={track.delaySend ?? 0}
          onChange={(v) => send("delaySend", v)}
          format={pct}
        />

        <div className="fx-section-label">{t("fx.busTitle")}</div>
        <ParamSlider
          label={t("fx.reverbWet")}
          min={0} max={1} step={0.01}
          value={cfg.reverbWet}
          onChange={(v) => applyCfg({ reverbWet: v })}
          format={pct}
        />
        <ParamSlider
          label={t("fx.delayWet")}
          min={0} max={1} step={0.01}
          value={cfg.delayWet}
          onChange={(v) => applyCfg({ delayWet: v })}
          format={pct}
        />
        <ParamSlider
          label={t("fx.delayTime")}
          min={0.05} max={2} step={0.01}
          value={cfg.delayTimeSec}
          onChange={(v) => applyCfg({ delayTimeSec: v })}
          format={sec2}
        />
        <ParamSlider
          label={t("fx.delayFeedback")}
          min={0} max={0.95} step={0.01}
          value={cfg.delayFeedback}
          onChange={(v) => applyCfg({ delayFeedback: v })}
          format={pct}
        />
      </div>
    </>
  );
}