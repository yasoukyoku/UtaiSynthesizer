import { useEffect } from "react";
import { useAppStore } from "./store/app";
import { Titlebar } from "./components/common/Titlebar";
import { DawView } from "./components/synth/DawView";
import { TrainingPanel } from "./components/training/TrainingPanel";
import { WorkflowEditor } from "./components/workflow/WorkflowEditor";
import { MsstModelManager } from "./components/models/MsstModelManager";
import { LogViewer } from "./components/common/LogViewer";
import { Settings } from "./components/common/Settings";
import { useTrainingStore } from "./store/training";
import { ToastContainer } from "./components/common/Toast";
import "./App.css";

export function App() {
  const { trainingPanelOpen, modelManagerOpen, toggleModelManager, logViewerOpen, toggleLogViewer, settingsOpen, toggleSettings, workflowSegmentId, closeWorkflow } = useAppStore();
  const { fetchStatus } = useTrainingStore();

  useEffect(() => {
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  useEffect(() => {
    const block = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === "C") {
        e.preventDefault();
      }
    };
    document.addEventListener("keydown", block);
    return () => document.removeEventListener("keydown", block);
  }, []);

  return (
    <div className="app-shell">
      <Titlebar />
      <div className="app-content">
        <DawView />
        {trainingPanelOpen && <TrainingPanel />}
        {workflowSegmentId && (
          <WorkflowEditor segmentId={workflowSegmentId} onClose={closeWorkflow} />
        )}
        {logViewerOpen && <LogViewer onClose={toggleLogViewer} />}
        {settingsOpen && <Settings onClose={toggleSettings} />}
        {modelManagerOpen && <MsstModelManager onClose={toggleModelManager} />}
      </div>
      <ToastContainer />
    </div>
  );
}
