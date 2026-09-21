import { jsx as _jsx } from "react/jsx-runtime";
import { NodeShell } from "./NodeShell";
export function AudioInputNode(_props) {
    return (_jsx(NodeShell, { label: "Audio In", icon: "[IN]", color: "#60a5fa", inputs: 0, outputs: 1, children: _jsx("span", { style: { fontSize: "10px", color: "var(--text-muted)" }, children: "Segment audio source" }) }));
}
