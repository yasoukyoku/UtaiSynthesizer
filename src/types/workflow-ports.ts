export type PortType = 
  | 'audio'
  | 'midi'
  | 'chords'
  | 'lyrics'
  | 'report'
  | 'image'
  | 'video'
  | 'any';

export interface PortSchema {
  id: string;
  type: PortType;
  label: string;
  optional?: boolean;
  multiple?: boolean;
}

export interface NodeSchema {
  type: string;
  inputs: PortSchema[];
  outputs: PortSchema[];
}

export const PORT_COLORS: Record<PortType, string> = {
  audio: '#3b82f6',
  midi: '#10b981',
  chords: '#f59e0b',
  lyrics: '#8b5cf6',
  report: '#6b7280',
  image: '#ec4899',
  video: '#f43f5e',
  any: '#64748b'
};

export const PORT_COMPATIBILITY: Record<PortType, PortType[]> = {
  audio: ['audio', 'any'],
  midi: ['midi', 'any'],
  chords: ['chords', 'any'],
  lyrics: ['lyrics', 'any'],
  report: ['report', 'any'],
  image: ['image', 'any'],
  video: ['video', 'any'],
  any: ['audio', 'midi', 'chords', 'lyrics', 'report', 'image', 'video', 'any']
};
