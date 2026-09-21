import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { RANDOM_TRACK_COLORS } from "../../lib/trackColors";
import "./TrackColorPicker.css";
export const TrackColorPicker = ({ currentColor, onColorChange, onClose, position, }) => {
    return (_jsxs(_Fragment, { children: [_jsx("div", { className: "track-color-picker-backdrop", onClick: onClose }), _jsx("div", { className: "track-color-picker", style: {
                    left: `${position.x}px`,
                    top: `${position.y}px`,
                }, children: _jsx("div", { className: "track-color-picker-grid", children: RANDOM_TRACK_COLORS.map((color) => (_jsx("button", { className: "track-color-option", style: { backgroundColor: color }, onClick: () => {
                            onColorChange(color);
                            onClose();
                        }, title: color, children: currentColor === color && (_jsx("svg", { width: "16", height: "16", viewBox: "0 0 16 16", children: _jsx("path", { d: "M13 4L6 11L3 8", stroke: "white", strokeWidth: "2", fill: "none", strokeLinecap: "round", strokeLinejoin: "round" }) })) }, color))) }) })] }));
};
