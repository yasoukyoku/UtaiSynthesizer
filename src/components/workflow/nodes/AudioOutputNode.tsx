import { useMemo } from "react";
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { DEFAULT_OUTPUT_GROUP, OUTPUT_NODE_COLOR } from "../../../lib/constants";
import { useProjectStore } from "../../../store/project";
import { useAppStore } from "../../../store/app";
import { collectGroupNames } from "../../../lib/workflow/engine";

/** Sentinel <option> value for "create a new group" — never a legal group name (starts with a NUL). */
const NEW_GROUP = "\u0000__new__";

export function AudioOutputNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const label = (params.laneLabel as string) ?? DEFAULT_OUTPUT_GROUP;
  const tracks = useProjectStore((s) => s.tracks);
  const groups = useMemo(() => collectGroupNames(tracks, [label]), [tracks, label]);

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
    if (name) updateParams({ laneLabel: name });
  };

  // No run button: Output nodes deposit AUTOMATICALLY — connecting a rendered edge puts the audio on the
  // track, disconnecting/deleting removes it (see the reconciler in WorkflowEditor). The status badge
  // (spinner while depositing / OK when on the track) is the only feedback.
  return (
    <NodeShell
      label={t("workflow.nodeOutput")}
      icon="[>]"
      color={OUTPUT_NODE_COLOR}
      inputs={1}
      outputs={0}
      nodeId={props.id}
      noRunButton
    >
      <label>{t("workflow.lane")}</label>
      {/* stopPropagation: without it the click that merely OPENS the dropdown bubbles to ReactFlow's
          node wrapper and fires the Output-node-click → select-group bridge (wiping the timeline
          multi-selection + expanding the track). Same exclusion as NodeShell's run button. */}
      <select
        value={label}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => {
          const v = e.target.value;
          if (v === NEW_GROUP) void promptNewGroup(); // controlled value stays put unless confirmed
          else updateParams({ laneLabel: v });
        }}
      >
        {groups.map((g) => (
          <option key={g} value={g}>{g}</option>
        ))}
        <option value={NEW_GROUP}>{t("workflow.newGroup")}</option>
      </select>
    </NodeShell>
  );
}
