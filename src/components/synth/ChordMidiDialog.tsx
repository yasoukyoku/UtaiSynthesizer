import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useProjectStore } from "../../store/project";
import { runChordMidi } from "../../lib/arrangement/autoArrange";
import { CHORD_MIDI_STYLES, type ChordMidiStyle } from "../../lib/arrangement/chordMidi";
import "./ArrangeDialog.css";

/** 调性选项：auto + 12 大调 + 12 小调（升号记法，音乐记号无需翻译）。 */
const KEY_OPTIONS: readonly string[] = [
  "auto",
  "C", "Cm", "C#", "C#m", "D", "Dm", "D#", "D#m", "E", "Em", "F", "Fm",
  "F#", "F#m", "G", "Gm", "G#", "G#m", "A", "Am", "A#", "A#m", "B", "Bm",
];

/**
 * Muno 阶段5「生成和弦 MIDI（配和声）」弹窗 —— 选和弦风格/每小节和弦数/调性 →
 * 生成 1 条和弦和声轨 + 顶部和弦行显示。目标 = 打开它的那条旋律轨。
 * 复用 ArrangeDialog 的 .arrange-* 样式（同一族弹窗，保持极简一致）。
 */
export function ChordMidiDialog({ trackId, onClose }: { trackId: string; onClose: () => void }) {
  const { t } = useTranslation();
  const trackName = useProjectStore((s) => s.tracks.find((tr) => tr.id === trackId)?.name) ?? "";
  const [style, setStyle] = useState<ChordMidiStyle>("POP_STANDARD");
  const [chordsPerBar, setChordsPerBar] = useState<1 | 2>(1);
  const [key, setKey] = useState("auto");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onClose]);

  const run = async () => {
    setBusy(true);
    try {
      await runChordMidi(trackId, { style, chordsPerBar, key });
      onClose();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="arrange-overlay" onClick={busy ? undefined : onClose}>
      <div className="arrange-panel" onClick={(e) => e.stopPropagation()}>
        <div className="arrange-head">
          <span className="arrange-title">🎹 {t("chordMidi.title")}</span>
          <button className="arrange-close" disabled={busy} onClick={onClose}>✕</button>
        </div>
        <div className="arrange-body">
          <div className="arrange-target" title={trackName}>
            {t("chordMidi.target")}: <strong>{trackName || "—"}</strong>
          </div>
          <label className="arrange-field">
            <span>{t("chordMidi.style")}</span>
            <select value={style} disabled={busy} onChange={(e) => setStyle(e.target.value as ChordMidiStyle)}>
              {CHORD_MIDI_STYLES.map((s) => (
                <option key={s} value={s}>{t(`chordMidi.styles.${s}`)}</option>
              ))}
            </select>
          </label>
          <label className="arrange-field">
            <span>{t("chordMidi.chordsPerBar")}</span>
            <select value={chordsPerBar} disabled={busy} onChange={(e) => setChordsPerBar(Number(e.target.value) === 2 ? 2 : 1)}>
              <option value={1}>{t("chordMidi.perBar1")}</option>
              <option value={2}>{t("chordMidi.perBar2")}</option>
            </select>
          </label>
          <label className="arrange-field">
            <span>{t("chordMidi.key")}</span>
            <select value={key} disabled={busy} onChange={(e) => setKey(e.target.value)}>
              {KEY_OPTIONS.map((k) => (
                <option key={k} value={k}>{k === "auto" ? t("chordMidi.keyAuto") : k}</option>
              ))}
            </select>
          </label>
          <div className="arrange-hint">
            {t("chordMidi.hint")}
          </div>
        </div>
        <div className="arrange-foot">
          <button className="arrange-btn" disabled={busy} onClick={onClose}>{t("chordMidi.cancel")}</button>
          <button className="arrange-btn primary" disabled={busy} onClick={() => void run()}>
            {busy ? t("chordMidi.running") : t("chordMidi.run")}
          </button>
        </div>
      </div>
    </div>
  );
}
