export interface Track {
  id: string;
  name: string;
  trackType: "vocal" | "audio" | "instrument";
  segments: Segment[];
  volumeDb: number;
  pan: number;
  muted: boolean;
  solo: boolean;
  voiceModel?: string;
}

export interface Segment {
  id: string;
  startTick: number;
  durationTicks: number;
  content: SegmentContent;
  workflow?: Workflow;
}

export type SegmentContent =
  | { type: "notes"; notes: Note[] }
  | { type: "audioClip"; sourcePath: string; offsetMs: number };

export interface Note {
  id: string;
  tick: number;
  duration: number;
  pitch: number;
  lyric: string;
  phoneme?: string;
  velocity: number;
}

export interface Workflow {
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
}

export interface WorkflowNode {
  id: string;
  nodeType: WorkflowNodeType;
  position: { x: number; y: number };
  params: Record<string, unknown>;
}

export type WorkflowNodeType =
  | "input"
  | "output"
  | "rvc"
  | "sovits"
  | "pitchShift"
  | "formantShift"
  | "audioEnhance"
  | "msstSeparation"
  | "split";

export interface WorkflowConnection {
  fromNode: string;
  fromPort: number;
  toNode: string;
  toPort: number;
}
