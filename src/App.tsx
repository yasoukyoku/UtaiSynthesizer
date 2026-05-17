import { useEffect } from "react";
import { useAppStore } from "./store/app";
import { Titlebar } from "./components/common/Titlebar";
import { DawView } from "./components/synth/DawView";
import { TrainingPanel } from "./components/training/TrainingPanel";
import { useTrainingStore } from "./store/training";

export function App() {
  const { trainingPanelOpen } = useAppStore();
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
      </div>
    </div>
  );
}
