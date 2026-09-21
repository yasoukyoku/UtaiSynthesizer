import { useState } from "react";
import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { useVoiceModelStore } from "../../store/voice-models";
import type { VoiceModelEntry } from "../../store/voice-models";

interface Props { onClose: () => void; }

/**
 * Quick-start wizard: paste lyrics + pick voice -> create vocal track.
 * Melody notes still go in Piano Roll; G2P + auto-render fire after save.
 */
export function LyricToVocalWizard({ onClose }: Props) {
  const showToast = useAppStore((s) => s.showToast);
  const addTrack = useProjectStore((s) => s.addTrack);
  const setTempo = useProjectStore((s) => s.setTempo);
  const setTimeSignature = useProjectStore((s) => s.setTimeSignature);
  const voiceModels = useVoiceModelStore((s) => {
    const all: VoiceModelEntry[] = [];
    for (const k of Object.keys(s.models) as (keyof typeof s.models)[]) {
      all.push(...s.models[k]);
    }
    return all;
  });

  const [lyrics, setLyrics] = useState("");
  const [bpm, setBpm] = useState(100);
  const [voiceModel, setVoiceModel] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (!lyrics.trim()) { showToast("Please paste some lyrics", "error"); return; }
    setBusy(true);
    try {
      setTempo(bpm);
      setTimeSignature(4, 4);

      const modelName = voiceModel || voiceModels[0]?.name;
      if (!modelName) {
        showToast("No AI voices installed - go to Models menu", "error");
        onClose();
        return;
      }

      addTrack({
        type: "vocal",
        name: "AI Vocal (" + modelName + ")",
        singerId: modelName,
      } as any);

      showToast("Track created! Open Piano Roll (P) to write melody, then RENDER.", "success");
      onClose();
    } catch (e: unknown) {
      showToast(String(e ?? "Unknown error"), "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="wizard-backdrop" onClick={onClose}>
      <div className="wizard-card" onClick={(e) => e.stopPropagation()}>
        <h2>Lyrics to AI Vocal Track</h2>
        <p className="hint">Quick start: paste lyrics, pick voice, we create a vocal track. Melody notes go in Piano Roll.</p>

        <div className="field">
          <label>Lyrics (one line per sentence)</label>
          <textarea
            rows={5}
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            placeholder="Paste song lyrics here..."
          />
        </div>

        <div className="field-row">
          <div className="field">
            <label>BPM (speed)</label>
            <input type="number" value={bpm} min={40} max={200} onChange={(e) => setBpm(+e.target.value)} />
          </div>
          <div className="field" style={{ flex: 2 }}>
            <label>AI Voice</label>
            <select value={voiceModel} onChange={(e) => setVoiceModel(e.target.value)}>
              <option value="">-- Auto-pick first --</option>
              {voiceModels.map((v) => (
                <option key={v.name} value={v.name}>{v.name}</option>
              ))}
            </select>
            {voiceModels.length === 0 && (
              <p className="hint">No voices yet. Download from Models menu.</p>
            )}
          </div>
        </div>

        <div className="wizard-actions">
          <button onClick={onClose}>Cancel</button>
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? "Creating..." : "Create Vocal Track"}
          </button>
        </div>
      </div>
    </div>
  );
}
