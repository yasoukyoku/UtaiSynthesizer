import type { Workflow } from "../../types/project";

export interface ExampleWorkflow {
  id: string;
  name: string;
  nameEn: string;
  description: string;
  descriptionEn: string;
  category: "basic" | "advanced" | "effect" | "analysis";
  difficulty: "beginner" | "intermediate" | "advanced";
  workflow: Workflow;
  thumbnail?: string;
}

export const exampleWorkflows: ExampleWorkflow[] = [
  {
    id: "basic-voice-conversion",
    name: "基础变声流程",
    nameEn: "Basic Voice Conversion",
    description: "最简单的 RVC 变声流程：音频输入 → RVC 变声 → 输出。适合新手快速上手。",
    descriptionEn: "Simple RVC voice conversion: Audio Input → RVC → Output. Perfect for beginners.",
    category: "basic",
    difficulty: "beginner",
    workflow: {
      nodes: [
        {
          id: "input-1",
          nodeType: "input",
          position: { x: 50, y: 150 },
          params: {},
        },
        {
          id: "rvc-1",
          nodeType: "rvc",
          position: { x: 300, y: 150 },
          params: {
            f0_shift: 0,
            formant: 0,
            index_ratio: 0.75,
            protect: 0.33,
            noise_scale: 0.4,
            rms_mix_rate: 0.25,
            l2_normalize: false,
            gpu_extract: true,
          },
        },
        {
          id: "output-1",
          nodeType: "output",
          position: { x: 550, y: 150 },
          params: {},
        },
      ],
      connections: [
        { fromNode: "input-1", fromPort: 0, toNode: "rvc-1", toPort: 0 },
        { fromNode: "rvc-1", fromPort: 0, toNode: "output-1", toPort: 0 },
      ],
    },
  },
  {
    id: "separation-and-rvc",
    name: "分离+变声流程",
    nameEn: "Separation + Voice Conversion",
    description: "先分离人声和伴奏，然后对人声进行变声处理，最后与伴奏合并。",
    descriptionEn: "Separate vocals from accompaniment, convert voice, then merge back.",
    category: "basic",
    difficulty: "intermediate",
    workflow: {
      nodes: [
        {
          id: "input-1",
          nodeType: "input",
          position: { x: 50, y: 200 },
          params: {},
        },
        {
          id: "sep-1",
          nodeType: "msstSeparation",
          position: { x: 250, y: 200 },
          params: {
            category: "vocal_separation",
            overlap: 0.25,
            normalize: true,
          },
        },
        {
          id: "rvc-1",
          nodeType: "rvc",
          position: { x: 500, y: 150 },
          params: {
            f0_shift: 0,
            formant: 0,
            index_ratio: 0.75,
            protect: 0.33,
          },
        },
        {
          id: "merge-1",
          nodeType: "merge",
          position: { x: 750, y: 200 },
          params: {},
        },
        {
          id: "output-1",
          nodeType: "output",
          position: { x: 950, y: 200 },
          params: {},
        },
      ],
      connections: [
        { fromNode: "input-1", fromPort: 0, toNode: "sep-1", toPort: 0 },
        { fromNode: "sep-1", fromPort: 0, toNode: "rvc-1", toPort: 0 },
        { fromNode: "rvc-1", fromPort: 0, toNode: "merge-1", toPort: 0 },
        { fromNode: "sep-1", fromPort: 1, toNode: "merge-1", toPort: 1 },
        { fromNode: "merge-1", fromPort: 0, toNode: "output-1", toPort: 0 },
      ],
    },
  },
  {
    id: "pitch-shift-effect",
    name: "音高调整效果",
    nameEn: "Pitch Shift Effect",
    description: "使用变调节点调整音频的音高，可用于音乐制作或效果处理。",
    descriptionEn: "Adjust audio pitch using transpose node for music production or effects.",
    category: "effect",
    difficulty: "beginner",
    workflow: {
      nodes: [
        {
          id: "input-1",
          nodeType: "input",
          position: { x: 50, y: 150 },
          params: {},
        },
        {
          id: "transpose-1",
          nodeType: "transpose",
          position: { x: 300, y: 150 },
          params: {
            semitones: 2,
            formant: 0,
          },
        },
        {
          id: "output-1",
          nodeType: "output",
          position: { x: 550, y: 150 },
          params: {},
        },
      ],
      connections: [
        { fromNode: "input-1", fromPort: 0, toNode: "transpose-1", toPort: 0 },
        { fromNode: "transpose-1", fromPort: 0, toNode: "output-1", toPort: 0 },
      ],
    },
  },
  {
    id: "audio-analysis",
    name: "音频分析流程",
    nameEn: "Audio Analysis",
    description: "对音频进行多维度分析：频谱图、基频曲线、音色指标等。",
    descriptionEn: "Multi-dimensional audio analysis: spectrogram, F0 curve, timbre metrics.",
    category: "analysis",
    difficulty: "intermediate",
    workflow: {
      nodes: [
        {
          id: "input-1",
          nodeType: "input",
          position: { x: 50, y: 250 },
          params: {},
        },
        {
          id: "spec-1",
          nodeType: "spectrogram",
          position: { x: 300, y: 150 },
          params: {},
        },
        {
          id: "f0-1",
          nodeType: "f0Curve",
          position: { x: 300, y: 250 },
          params: {},
        },
        {
          id: "timbre-1",
          nodeType: "timbreMetrics",
          position: { x: 300, y: 350 },
          params: {},
        },
      ],
      connections: [
        { fromNode: "input-1", fromPort: 0, toNode: "spec-1", toPort: 0 },
        { fromNode: "input-1", fromPort: 0, toNode: "f0-1", toPort: 0 },
        { fromNode: "input-1", fromPort: 0, toNode: "timbre-1", toPort: 0 },
      ],
    },
  },
  {
    id: "mastering-transparent",
    name: "透明母带处理",
    nameEn: "Transparent Mastering",
    description: "高保真母带处理流程，追求最小的信号损伤（>130dB SNR）。",
    descriptionEn: "Hi-fi mastering with minimal signal damage (>130dB SNR).",
    category: "advanced",
    difficulty: "advanced",
    workflow: {
      nodes: [
        {
          id: "input-1",
          nodeType: "input",
          position: { x: 50, y: 150 },
          params: {},
        },
        {
          id: "dc-1",
          nodeType: "dcRemove",
          position: { x: 250, y: 150 },
          params: {},
        },
        {
          id: "lufs-1",
          nodeType: "lufsNormalize",
          position: { x: 450, y: 150 },
          params: {
            target_lufs: -14.0,
          },
        },
        {
          id: "dither-1",
          nodeType: "dither",
          position: { x: 650, y: 150 },
          params: {
            depth: 16,
          },
        },
        {
          id: "output-1",
          nodeType: "output",
          position: { x: 850, y: 150 },
          params: {},
        },
      ],
      connections: [
        { fromNode: "input-1", fromPort: 0, toNode: "dc-1", toPort: 0 },
        { fromNode: "dc-1", fromPort: 0, toNode: "lufs-1", toPort: 0 },
        { fromNode: "lufs-1", fromPort: 0, toNode: "dither-1", toPort: 0 },
        { fromNode: "dither-1", fromPort: 0, toNode: "output-1", toPort: 0 },
      ],
    },
  },
  {
    id: "complete-song-production",
    name: "完整歌曲制作",
    nameEn: "Complete Song Production",
    description: "从音频分离到变声、效果处理，再到最终母带输出的完整流程。",
    descriptionEn: "Complete workflow from separation to voice conversion, effects, and mastering.",
    category: "advanced",
    difficulty: "advanced",
    workflow: {
      nodes: [
        {
          id: "input-1",
          nodeType: "input",
          position: { x: 50, y: 300 },
          params: {},
        },
        {
          id: "sep-2",
          nodeType: "msstSeparation",
          position: { x: 250, y: 300 },
          params: {
            category: "vocal_separation",
            overlap: 0.25,
          },
        },
        {
          id: "rvc-1",
          nodeType: "rvc",
          position: { x: 500, y: 250 },
          params: {
            f0_shift: 0,
            index_ratio: 0.75,
            protect: 0.33,
          },
        },
        {
          id: "eq-1",
          nodeType: "busEq",
          position: { x: 750, y: 250 },
          params: {},
        },
        {
          id: "merge-1",
          nodeType: "merge",
          position: { x: 1000, y: 300 },
          params: {},
        },
        {
          id: "lufs-1",
          nodeType: "lufsNormalize",
          position: { x: 1250, y: 300 },
          params: {
            target_lufs: -14.0,
          },
        },
        {
          id: "output-1",
          nodeType: "output",
          position: { x: 1500, y: 300 },
          params: {},
        },
      ],
      connections: [
        { fromNode: "input-1", fromPort: 0, toNode: "sep-1", toPort: 0 },
        { fromNode: "sep-1", fromPort: 0, toNode: "rvc-1", toPort: 0 },
        { fromNode: "rvc-1", fromPort: 0, toNode: "eq-1", toPort: 0 },
        { fromNode: "eq-1", fromPort: 0, toNode: "merge-1", toPort: 0 },
        { fromNode: "sep-1", fromPort: 1, toNode: "merge-1", toPort: 1 },
        { fromNode: "merge-1", fromPort: 0, toNode: "lufs-1", toPort: 0 },
        { fromNode: "lufs-1", fromPort: 0, toNode: "output-1", toPort: 0 },
      ],
    },
  },
];

export function getExamplesByCategory(category: ExampleWorkflow["category"]): ExampleWorkflow[] {
  return exampleWorkflows.filter((ex) => ex.category === category);
}

export function getExamplesByDifficulty(difficulty: ExampleWorkflow["difficulty"]): ExampleWorkflow[] {
  return exampleWorkflows.filter((ex) => ex.difficulty === difficulty);
}

export function getExampleById(id: string): ExampleWorkflow | undefined {
  return exampleWorkflows.find((ex) => ex.id === id);
}