import { useState } from "react";
import { Toolbar } from "./Toolbar";
import { TrackList } from "./TrackList";
import { Arrangement } from "./Arrangement";
import "./DawView.css";

export function DawView() {
  const [trackListWidth] = useState(200);

  return (
    <div className="daw-view">
      <Toolbar />
      <div className="daw-workspace">
        <TrackList width={trackListWidth} />
        <div className="daw-separator" />
        <Arrangement />
      </div>
    </div>
  );
}
