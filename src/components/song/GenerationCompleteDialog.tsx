import { useEffect } from "react";
import { SONG_MODEL_CATALOG } from "../../lib/models/song-catalog";
import "./GenerationCompleteDialog.css";

export interface GenerationCompleteDialogProps {
  songName: string;
  modelId: string;
  processingTime?: number;
  onClose: () => void;
}

export function GenerationCompleteDialog({
  songName,
  modelId,
  processingTime,
  onClose,
}: GenerationCompleteDialogProps) {
  const model = SONG_MODEL_CATALOG.find((m) => m.id === modelId);
  const modelLabel = model?.label?.zh || modelId;

  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  const formatTime = (seconds?: number): string => {
    if (!seconds) return "未知";
    if (seconds < 60) return `${seconds.toFixed(1)}秒`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}分${secs}秒`;
  };

  return (
    <div className="gcd-overlay" onClick={onClose}>
      <div className="gcd-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="gcd-icon">
          <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
            <circle cx="32" cy="32" r="28" fill="#4CAF50" fillOpacity="0.1" />
            <circle cx="32" cy="32" r="24" fill="#4CAF50" fillOpacity="0.2" />
            <path
              d="M20 32L28 40L44 24"
              stroke="#4CAF50"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        <h2 className="gcd-title">🎉 生成完成</h2>

        <div className="gcd-content">
          <div className="gcd-info-row">
            <span className="gcd-label">歌曲名称</span>
            <span className="gcd-value">{songName}</span>
          </div>

          <div className="gcd-info-row">
            <span className="gcd-label">使用模型</span>
            <span className="gcd-value">{modelLabel}</span>
          </div>

          <div className="gcd-info-row">
            <span className="gcd-label">生成时间</span>
            <span className="gcd-value gcd-time">{formatTime(processingTime)}</span>
          </div>
        </div>

        <button className="gcd-button" onClick={onClose}>
          好的
        </button>
      </div>
    </div>
  );
}
