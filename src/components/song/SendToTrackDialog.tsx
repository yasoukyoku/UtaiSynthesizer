/**
 * 发送到轨道对话框 - 产物驱动（规划 8.1/8.2）
 * 可用性由实际产物决定：audio_path / stems / midi_path|abc_path
 * 无分轨产物时提供"⚡ 现场分轨后发送"兜底（ACE 高质量 / demucs 快速）
 * P1-12：落点选项（工程末尾/当前播放位置/指定轨道后）+ MIDI 双按钮（直接落轨 / 工作台）
 */

import { useState } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import type { SongHistoryEntry } from "./SongStudioDialog";
import { deriveCapabilities } from "../../lib/song/types";
import { ACE_TRACK_CLASSES } from "../../lib/models/song-tasks";
import i18n from "../../i18n";
import "./SendToTrackDialog.css";

export type SendFormat = "single" | "multi" | "midi" | "midiTrack" | "liveExtract";

/** 规划 8.2 落点：工程末尾（默认）/ 当前播放位置 / 指定轨道后 */
export type SendDestination =
  | { kind: "end" }
  | { kind: "playhead" }
  | { kind: "afterTrack"; trackId: string };

export interface SendToTrackDialogProps {
  song: SongHistoryEntry;
  onConfirm: (format: SendFormat, extractEngine: "ace" | "demucs", dest: SendDestination) => void;
  onClose: () => void;
}

export function SendToTrackDialog({ song, onConfirm, onClose }: SendToTrackDialogProps) {
  const [selected, setSelected] = useState<SendFormat>("single");
  const [extractEngine, setExtractEngine] = useState<"ace" | "demucs">("demucs");
  // 规划 8.2：落点选项（默认工程末尾；"指定轨道后"默认当前激活轨）
  const [destKind, setDestKind] = useState<"end" | "playhead" | "afterTrack">("end");
  const activeTrackId = useAppStore((s) => s.activeTrackId);
  const [afterTrackId, setAfterTrackId] = useState<string | null>(activeTrackId ?? null);
  const tracks = useProjectStore((s) => s.tracks);

  const dest: SendDestination =
    destKind === "afterTrack" && afterTrackId
      ? { kind: "afterTrack", trackId: afterTrackId }
      : destKind === "playhead"
        ? { kind: "playhead" }
        : { kind: "end" };

  const caps = song.capabilities ?? deriveCapabilities({ outputs: song.outputs });
  const hasAudio = song.outputs.some((o) => o.audio_path);
  const hasMidi = caps.hasMidi || caps.hasAbc;
  const stemKinds = Object.keys(song.outputs.find((o) => o.stems)?.stems ?? {});

  type Card = {
    id: SendFormat;
    icon: string;
    label: string;
    description: string;
    supported: boolean;
    badge?: string;
  };
  const allCards: Card[] = [
    {
      id: "single",
      icon: "🎵",
      label: "单轨音频",
      description: "完整混音的单个音频文件",
      supported: hasAudio,
      badge: hasAudio ? undefined : "无音频产物",
    },
    {
      id: "multi",
      icon: "🎚️",
      label: "多轨音频",
      description: stemKinds.length
        ? `已含分轨：${stemKinds.join(" / ")}`
        : "分离的人声、伴奏、鼓等音轨",
      supported: caps.hasStems,
      badge: caps.hasStems ? undefined : "无分轨产物",
    },
    {
      id: "liveExtract",
      icon: "⚡",
      label: i18n.t("songResult.stemsOnTheFly"),
      description: "对主音频即时分轨，每个 stem 各建一条轨道",
      supported: hasAudio && !caps.hasStems,
      badge: caps.hasStems ? "已有分轨，可直接选多轨" : undefined,
    },
    {
      id: "midiTrack",
      icon: "🎹",
      label: "MIDI",
      description: caps.hasMidi
        ? "直接落为可编辑的 MIDI 音符轨"
        : caps.hasAbc
          ? "仅有 ABC 乐谱，落轨前自动转换为 MIDI"
          : i18n.t("songResult.noMidiHint"),
      supported: hasMidi,
      badge: hasMidi ? undefined : "不含 MIDI 产物",
    },
  ];
  // 已有分轨时不再展示"现场分轨"卡（8.1：多轨可选即不需要兜底）
  const cards = allCards.filter((c) => c.id !== "liveExtract" || !caps.hasStems);
  const current = cards.find((c) => c.id === selected) ?? cards[0]!;

  const fmt = (sec: number) =>
    `${Math.floor(sec / 60)}:${String(Math.round(sec % 60)).padStart(2, "0")}`;

  // 预览清单（规划 8.2）：确认前一目了然将创建哪些轨道
  const preview: { name: string; type: string; note?: string }[] = (() => {
    const label = song.label || "Untitled";
    const dur = `${fmt(song.settings.audio_duration)} · WAV`;
    switch (current.id) {
      case "single":
        return [{ name: label, type: "音频轨", note: dur }];
      case "multi":
        return stemKinds.map((k) => ({ name: `${label} - ${k}`, type: "音频轨", note: dur }));
      case "liveExtract":
        return (extractEngine === "demucs"
          ? ["vocals", "drums", "bass", "other"]
          : [...ACE_TRACK_CLASSES]
        ).map((k) => ({ name: `${label} - ${k}`, type: "音频轨", note: "分轨完成后创建" }));
      case "midi":
      case "midiTrack":
        return [
          {
            name: `${label} (MIDI)`,
            type: "MIDI 音符轨",
            note: caps.hasMidi ? undefined : "ABC→MIDI 自动转换",
          },
        ];
    }
  })();

  const destOptions: { id: typeof destKind; label: string }[] = [
    { id: "end", label: "工程末尾" },
    { id: "playhead", label: "当前播放位置" },
    { id: "afterTrack", label: "指定轨道后" },
  ];

  return (
    <div className="st-overlay" onClick={onClose}>
      <div className="st-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="st-header">
          <div className="st-title">
            <span className="st-title-icon">📤</span>
            {i18n.t("songResult.sendToTrack")}
          </div>
          <button className="st-close" onClick={onClose}>✕</button>
        </div>

        <div className="st-body">
          {/* 歌曲信息 */}
          <div className="st-song-info">
            <div className="st-song-icon">🎵</div>
            <div className="st-song-details">
              <div className="st-song-title">{song.label || "Untitled"}</div>
              <div className="st-song-meta">
                {song.modelFamily.toUpperCase()} · {song.settings.audio_duration}s · {song.settings.bpm} BPM
              </div>
            </div>
          </div>

          {/* 格式选择（产物驱动可用性） */}
          <div className="st-formats">
            <div className="st-formats-label">选择发送内容</div>
            <div className="st-formats-list">
              {cards.map((card) => (
                <div
                  key={card.id}
                  className={`st-format-card ${!card.supported ? "disabled" : ""} ${
                    current.id === card.id ? "selected" : ""
                  }`}
                  title={card.supported ? undefined : card.badge}
                  onClick={() => card.supported && setSelected(card.id)}
                >
                  <div className="st-format-icon">{card.icon}</div>
                  <div className="st-format-info">
                    <div className="st-format-label">
                      {card.label}
                      {!card.supported && card.badge && (
                        <span className="st-format-badge">{card.badge}</span>
                      )}
                    </div>
                    <div className="st-format-desc">{card.description}</div>
                  </div>
                  {card.supported && (
                    <div className="st-format-radio">
                      {current.id === card.id ? "●" : "○"}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* 现场分轨引擎选择 */}
          {current.id === "liveExtract" && (
            <div className="st-engine-row">
              <button
                className={`st-engine-chip ${extractEngine === "demucs" ? "on" : ""}`}
                onClick={() => setExtractEngine("demucs")}
              >
                ⚡ demucs 快速四轨
              </button>
              <button
                className={`st-engine-chip ${extractEngine === "ace" ? "on" : ""}`}
                onClick={() => setExtractEngine("ace")}
              >
                🎚 ACE 高质量全轨
              </button>
            </div>
          )}

          {/* 规划 8.2：落点选项 */}
          <div className="st-dest">
            <div className="st-formats-label">落点</div>
            <div className="st-dest-row">
              {destOptions.map((o) => (
                <button
                  key={o.id}
                  className={`st-engine-chip ${destKind === o.id ? "on" : ""}`}
                  onClick={() => setDestKind(o.id)}
                >
                  {o.label}
                </button>
              ))}
            </div>
            {destKind === "afterTrack" && (
              <select
                className="st-dest-select"
                value={afterTrackId ?? ""}
                onChange={(e) => setAfterTrackId(e.target.value)}
              >
                {tracks.length === 0 && <option value="">（工程中暂无轨道）</option>}
                {tracks.map((t) => (
                  <option key={t.id} value={t.id}>{t.name}</option>
                ))}
              </select>
            )}
          </div>

          {/* 预览清单（规划 8.2） */}
          <div className="st-preview">
            <div className="st-preview-title">将创建 {preview.length} 条轨道</div>
            {preview.map((p, i) => (
              <div key={i} className="st-preview-row">
                <span className="st-preview-name">🎵 {p.name}</span>
                <span className="st-preview-type">{p.type}</span>
                {p.note && <span className="st-preview-note">{p.note}</span>}
              </div>
            ))}
          </div>
        </div>

        <div className="st-footer">
          <button className="st-btn st-btn-cancel" onClick={onClose}>
            取消
          </button>
          {/* 规划 8.2：MIDI 双按钮——直接落轨为主，工作台为次级链接 */}
          {current.id === "midiTrack" && (
            <button
              className="st-btn st-btn-secondary"
              onClick={() => onConfirm("midi", extractEngine, dest)}
            >
              在 MIDI 工作台查看/编辑
            </button>
          )}
          <button
            className="st-btn st-btn-confirm"
            onClick={() => onConfirm(current.id === "midiTrack" ? "midiTrack" : current.id, extractEngine, dest)}
            disabled={!current.supported}
          >
            确认发送
          </button>
        </div>
      </div>
    </div>
  );
}
