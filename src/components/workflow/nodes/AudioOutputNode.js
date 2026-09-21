import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { DEFAULT_OUTPUT_GROUP, OUTPUT_NODE_COLOR } from "../../../lib/constants";
import { useProjectStore } from "../../../store/project";
import { useAppStore } from "../../../store/app";
import { useWorkflowStore } from "../../../store/workflow";
import { collectGroupNames } from "../../../lib/workflow/engine";
import { open } from "@tauri-apps/plugin-dialog";
import { exportOneAudioFileToFolder, laneExportErrorMessage } from "../../../lib/audio/exportLaneAudio";
/** Sentinel <option> value for "create a new group" — never a legal group name (starts with a NUL). */
const NEW_GROUP = "\u0000__new__";
/** Strip characters that can't appear in a Windows filename. */
function sanitize(v) {
    return v.replace(/[<>:"/\\|?*\u0000]/g, "_").trim();
}
/**
 * 输出节点下载的默认文件名：顺着输入边找到供给节点——
 *   rvc/sovits（声音转换）→ 用所选模型名(params.voiceName)；
 *   separation（人声/伴奏分离）→ 用该端口对应的 stem 名(params.stemLabels[fromPort]，如「人声」「伴奏」)。
 * 多个源用 "_" 连接；没有可用名则退回轨道组名。
 */
function defaultOutputFileName(workflow, outputNodeId, fallback) {
    const parts = [];
    for (const c of workflow?.connections ?? []) {
        if (c.toNode !== outputNodeId)
            continue;
        const n = workflow?.nodes.find((x) => x.id === c.fromNode);
        if (!n)
            continue;
        let nm;
        if (n.nodeType === "rvc" || n.nodeType === "sovits")
            nm = n.params?.voiceName;
        else if (n.nodeType === "msstSeparation")
            nm = n.params?.stemLabels?.[c.fromPort];
        if (typeof nm === "string" && nm.trim())
            parts.push(nm.trim());
    }
    const base = parts.length > 0 ? parts.join("_") : fallback || DEFAULT_OUTPUT_GROUP;
    return sanitize(base) || DEFAULT_OUTPUT_GROUP;
}
export function AudioOutputNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const label = params.laneLabel ?? DEFAULT_OUTPUT_GROUP;
    const tracks = useProjectStore((s) => s.tracks);
    const groups = useMemo(() => collectGroupNames(tracks, [label]), [tracks, label]);
    const showToast = useAppStore((s) => s.showToast);
    const [dl, setDl] = useState(null);
    const promptNewGroup = async () => {
        const name = await useAppStore.getState().showConfirm({
            title: t("workflow.newGroupTitle"),
            body: "",
            buttons: [
                { id: "ok", label: t("common.confirm"), kind: "primary" },
                { id: "cancel", label: t("common.cancel") },
            ],
            input: {
                placeholder: t("workflow.newGroupPlaceholder"),
                // " · " is the label/stem separator and the \u0000 row-key separator — a group name
                // containing either would corrupt label parsing / row identity.
                invalid: (v) => (v.includes(" · ") || v.includes("\u0000") ? t("workflow.groupNameInvalid") : null),
            },
        });
        if (name)
            updateParams({ laneLabel: name });
    };
    // 本输出节点已放入轨道的音频文件(运行工作流到达此节点后产生)。一个节点可能输出多条
    // stems —— 全部收集去重。渲染链路(split-mid-render)下用 linkedSource 的 segment 解析,
    // 与 NodeShell 的 statusSeg 同口径。
    const segmentId = useAppStore((s) => s.workflowSegmentId);
    const linkedSource = useWorkflowStore((s) => (segmentId ? s.renderLinks[segmentId] : undefined));
    const statusSeg = linkedSource ?? segmentId;
    const { workflow, outputPaths } = useMemo(() => {
        let workflow;
        const out = [];
        const seen = new Set();
        if (statusSeg) {
            for (const trk of useProjectStore.getState().tracks) {
                for (const sg of trk.segments) {
                    if (sg.id !== statusSeg)
                        continue;
                    workflow = sg.workflow;
                    for (const po of sg.processedOutputs ?? []) {
                        if (po.outputNodeId === props.id && po.audioPath && !seen.has(po.audioPath.toLowerCase())) {
                            seen.add(po.audioPath.toLowerCase());
                            out.push(po.audioPath);
                        }
                    }
                }
            }
        }
        return { workflow, outputPaths: out };
    }, [statusSeg, props.id, tracks]);
    // 默认下载文件名，随上游节点/模型变化实时更新(打开下载表单时取一次)。
    const defaultName = useMemo(() => defaultOutputFileName(workflow, props.id, label), [workflow, props.id, label]);
    // 第一步：先选文件夹，随即进入命名表单。
    const download = async () => {
        if (outputPaths.length === 0) {
            showToast(t("workflow.outputDownloadNothing"), "info");
            return;
        }
        const out = await open({ directory: true, title: t("workflow.outputDownload") });
        if (!out || typeof out !== "string")
            return;
        setDl({ folder: out, name: defaultName });
    };
    // 第二步：按用户确认的文件名把每个结果复制进所选文件夹(原样复制,不重编码)。
    const saveDl = async () => {
        if (!dl)
            return;
        const base = sanitize(dl.name) || DEFAULT_OUTPUT_GROUP;
        try {
            const dests = [];
            for (let i = 0; i < outputPaths.length; i++) {
                const item = { label: dl.name, sourcePath: outputPaths[i] };
                const name = i === 0 ? base : `${base}_${i + 1}`;
                dests.push(await exportOneAudioFileToFolder(item, dl.folder, name));
            }
            setDl(null);
            showToast(`${t("tracks.exportDone")} ${dests.length} ${t("tracks.exportCopied")} ${dl.folder}`, "success");
        }
        catch (e) {
            showToast(laneExportErrorMessage(e), "error");
        }
    };
    // No run button: Output nodes deposit AUTOMATICALLY — connecting a rendered edge puts the audio on the
    // track, disconnecting/deleting removes it (see the reconciler in WorkflowEditor). The status badge
    // (spinner while depositing / OK when on the track) is the only feedback.
    return (_jsxs(NodeShell, { label: t("workflow.nodeOutput"), icon: "[>]", color: OUTPUT_NODE_COLOR, inputs: 1, outputs: 0, nodeId: props.id, noRunButton: true, children: [_jsx("label", { children: t("workflow.lane") }), _jsxs("select", { value: label, onClick: (e) => e.stopPropagation(), onChange: (e) => {
                    const v = e.target.value;
                    if (v === NEW_GROUP)
                        void promptNewGroup(); // controlled value stays put unless confirmed
                    else
                        updateParams({ laneLabel: v });
                }, children: [groups.map((g) => (_jsx("option", { value: g, children: g }, g))), _jsx("option", { value: NEW_GROUP, children: t("workflow.newGroup") })] }), dl ? (
            // 命名表单：先选文件夹，这里输入文件名，再保存/取消。
            _jsxs("div", { className: "wf-download-form nodrag", onPointerDown: (e) => e.stopPropagation(), onClick: (e) => e.stopPropagation(), children: [_jsx("label", { children: t("workflow.outputName") }), _jsx("input", { value: dl.name, autoFocus: true, placeholder: defaultName, onChange: (e) => setDl({ ...dl, name: e.target.value }), onKeyDown: (e) => {
                            if (e.key === "Enter" && dl.name.trim())
                                e.currentTarget.blur();
                        } }), _jsxs("div", { className: "wf-download-actions", children: [_jsx("button", { className: "wf-download-save", disabled: !dl.name.trim(), onClick: () => void saveDl(), children: t("workflow.outputSave") }), _jsx("button", { className: "wf-download-cancel", onClick: () => setDl(null), children: t("workflow.outputCancel") })] })] })) : (_jsx("div", { className: "wf-node-download-row nodrag", onPointerDown: (e) => e.stopPropagation(), onClick: (e) => e.stopPropagation(), children: _jsxs("button", { className: "wf-download-btn", disabled: outputPaths.length === 0, title: outputPaths.length === 0 ? t("workflow.outputDownloadNothing") : undefined, onClick: () => void download(), children: ["\u2B07 ", t("workflow.outputDownload")] }) }))] }));
}
