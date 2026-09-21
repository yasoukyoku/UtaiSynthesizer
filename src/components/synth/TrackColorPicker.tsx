import React from "react";
import { RANDOM_TRACK_COLORS } from "../../lib/trackColors";
import "./TrackColorPicker.css";

interface TrackColorPickerProps {
  currentColor: string;
  onColorChange: (color: string) => void;
  onClose: () => void;
  position: { x: number; y: number };
}

export const TrackColorPicker: React.FC<TrackColorPickerProps> = ({
  currentColor,
  onColorChange,
  onClose,
  position,
}) => {
  return (
    <>
      <div className="track-color-picker-backdrop" onClick={onClose} />
      <div
        className="track-color-picker"
        style={{
          left: `${position.x}px`,
          top: `${position.y}px`,
        }}
      >
        <div className="track-color-picker-grid">
          {RANDOM_TRACK_COLORS.map((color) => (
            <button
              key={color}
              className="track-color-option"
              style={{ backgroundColor: color }}
              onClick={() => {
                onColorChange(color);
                onClose();
              }}
              title={color}
            >
              {currentColor === color && (
                <svg width="16" height="16" viewBox="0 0 16 16">
                  <path
                    d="M13 4L6 11L3 8"
                    stroke="white"
                    strokeWidth="2"
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </button>
          ))}
        </div>
      </div>
    </>
  );
};
