import { useEffect } from "react";
import { useAppStore } from "./store/app";
import { Titlebar } from "./components/common/Titlebar";
import { DawView } from "./components/synth/DawView";
import { TrainingPanel } from "./components/training/TrainingPanel";
import { WorkflowEditor } from "./components/workflow/WorkflowEditor";
import { useTrainingStore } from "./store/training";
import "./App.css";

export function App() {
  const { trainingPanelOpen, workflowSegmentId, closeWorkflow } = useAppStore();
  const { fetchStatus } = useTrainingStore();

  useEffect(() => {
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  return (
    <div className="app-shell">
      <Titlebar />
      <div className="app-content">
        <DawView />
        {trainingPanelOpen && <TrainingPanel />}
        {workflowSegmentId && (
          <WorkflowEditor segmentId={workflowSegmentId} onClose={closeWorkflow} />
        )}
      </div>
    </div>
  );
}
