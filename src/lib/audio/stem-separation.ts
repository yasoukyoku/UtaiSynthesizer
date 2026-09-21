import { invoke } from '@tauri-apps/api/core';

export type StemQuality = 'fast' | 'standard' | 'pro' | 'best';

export interface StemSeparationOptions {
  quality: StemQuality;
  outputDir: string;
  model?: string;
  shifts?: number;
  overlap?: number;
}

export interface StemSeparationResult {
  vocals?: string;
  instrumental?: string;
  drums?: string;
  bass?: string;
  other?: string;
  guitar?: string;
  piano?: string;
}

export const STEM_QUALITY_PRESETS: Record<StemQuality, {
  model: string;
  shifts: number;
  overlap: number;
  description: string;
}> = {
  fast: {
    model: 'htdemucs',
    shifts: 0,
    overlap: 0.25,
    description: '快速分离，适合预览，处理速度最快'
  },
  standard: {
    model: 'htdemucs',
    shifts: 1,
    overlap: 0.25,
    description: '标准质量，平衡速度和效果'
  },
  pro: {
    model: 'htdemucs_ft',
    shifts: 3,
    overlap: 0.5,
    description: '专业质量，使用微调模型，效果更好'
  },
  best: {
    model: 'htdemucs_6s',
    shifts: 5,
    overlap: 0.75,
    description: '最佳质量，6 stem 分离，处理时间较长'
  }
};

export async function separateStems(
  inputPath: string,
  options: StemSeparationOptions
): Promise<StemSeparationResult> {
  const preset = STEM_QUALITY_PRESETS[options.quality];
  
  const result = await invoke<Record<string, string>>('audio_separate_stems', {
    input: inputPath,
    outputDir: options.outputDir,
    model: options.model || preset.model,
    shifts: options.shifts ?? preset.shifts,
    overlap: options.overlap ?? preset.overlap
  });

  return result as StemSeparationResult;
}

export async function handleStemSeparation(
  trackId: string,
  audioPath: string,
  quality: StemQuality,
  onProgress?: (progress: number, stage: string) => void
): Promise<StemSeparationResult> {
  const outputDir = `${trackId}_stems_${quality}_${Date.now()}`;
  
  onProgress?.(0, '准备分离...');
  
  try {
    const result = await separateStems(audioPath, {
      quality,
      outputDir
    });
    
    onProgress?.(100, '分离完成');
    
    return result;
  } catch (error) {
    onProgress?.(-1, `分离失败: ${error}`);
    throw error;
  }
}
