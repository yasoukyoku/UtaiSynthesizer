import { invoke } from '@tauri-apps/api/core';

export type AMTQuality = 'fast' | 'standard' | 'best' | 'guitar' | 'piano';

export interface AMTOptions {
  quality: AMTQuality;
  minNoteDuration?: number;
  hopLength?: number;
  multiTrack?: boolean;
  midiThreshold?: number;
  onsetThreshold?: number;
  frameThreshold?: number;
}

export interface MidiNote {
  pitch: number;
  startTime: number;
  duration: number;
  velocity: number;
  track?: number;
}

export interface AmtResult {
  notes: MidiNote[];
  tempo?: number;
  timeSignature?: [number, number];
  trackCount?: number;
}

export const AMT_QUALITY_PRESETS: Record<AMTQuality, {
  minNoteDuration: number;
  hopLength: number;
  multiTrack: boolean;
  midiThreshold: number;
  onsetThreshold: number;
  frameThreshold: number;
  description: string;
}> = {
  fast: {
    minNoteDuration: 0.125,
    hopLength: 512,
    multiTrack: false,
    midiThreshold: 0.5,
    onsetThreshold: 0.5,
    frameThreshold: 0.3,
    description: '快速转换，适合预览，最短音符 1/8 拍'
  },
  standard: {
    minNoteDuration: 0.0625,
    hopLength: 256,
    multiTrack: false,
    midiThreshold: 0.4,
    onsetThreshold: 0.5,
    frameThreshold: 0.3,
    description: '标准质量，平衡速度和精度，最短音符 1/16 拍'
  },
  best: {
    minNoteDuration: 0.03125,
    hopLength: 128,
    multiTrack: true,
    midiThreshold: 0.3,
    onsetThreshold: 0.4,
    frameThreshold: 0.2,
    description: '最佳质量，多轨分离，最短音符 1/32 拍'
  },
  guitar: {
    minNoteDuration: 0.0625,
    hopLength: 256,
    multiTrack: true,
    midiThreshold: 0.35,
    onsetThreshold: 0.45,
    frameThreshold: 0.25,
    description: '吉他专用，针对弦乐器特性优化'
  },
  piano: {
    minNoteDuration: 0.0625,
    hopLength: 256,
    multiTrack: true,
    midiThreshold: 0.3,
    onsetThreshold: 0.4,
    frameThreshold: 0.2,
    description: '钢琴专用，针对键盘乐器特性优化'
  }
};

export async function audioToMidi(
  inputPath: string,
  options: AMTOptions
): Promise<AmtResult> {
  const preset = AMT_QUALITY_PRESETS[options.quality];
  
  const result = await invoke<AmtResult>('audio_to_midi', {
    input: inputPath,
    minNoteDuration: options.minNoteDuration ?? preset.minNoteDuration,
    hopLength: options.hopLength ?? preset.hopLength,
    multiTrack: options.multiTrack ?? preset.multiTrack,
    midiThreshold: options.midiThreshold ?? preset.midiThreshold,
    onsetThreshold: options.onsetThreshold ?? preset.onsetThreshold,
    frameThreshold: options.frameThreshold ?? preset.frameThreshold
  });

  return result;
}

export async function handleAudioToMidi(
  audioPath: string,
  quality: AMTQuality,
  onProgress?: (progress: number, stage: string) => void
): Promise<AmtResult> {
  onProgress?.(0, '准备转换...');
  
  try {
    onProgress?.(20, '加载音频文件...');
    
    const result = await audioToMidi(audioPath, { quality });
    
    onProgress?.(100, `转换完成，检测到 ${result.notes.length} 个音符`);
    
    return result;
  } catch (error) {
    onProgress?.(-1, `转换失败: ${error}`);
    throw error;
  }
}

export function midiNotesToTrackData(notes: MidiNote[]): {
  notes: Array<{ pitch: number; time: number; duration: number; velocity: number }>;
} {
  return {
    notes: notes.map(note => ({
      pitch: note.pitch,
      time: note.startTime,
      duration: note.duration,
      velocity: note.velocity
    }))
  };
}
