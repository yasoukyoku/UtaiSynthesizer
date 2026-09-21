/**
 * 参数教程弹窗 - 简单易懂的参数说明
 */

import { type SongModelFamily } from "../../lib/models/song-catalog";
import "./ParamsTutorialDialog.css";

export interface ParamsTutorialDialogProps {
  modelFamily: SongModelFamily;
  onClose: () => void;
}

interface ParamInfo {
  name: string;
  icon: string;
  description: string;
  tips: string;
}

const YUE2_PARAMS: ParamInfo[] = [
  {
    name: "节奏速度 (BPM)",
    icon: "🎼",
    description: "控制歌曲的快慢节奏，数值越大歌曲越快",
    tips: "流行歌曲通常在 100-130 之间，快歌可以到 140-180",
  },
  {
    name: "时长",
    icon: "⏱️",
    description: "生成歌曲的总长度，单位是秒",
    tips: "一般歌曲 3-5 分钟（180-300秒），试听可以用 30-60 秒",
  },
  {
    name: "语言",
    icon: "🌍",
    description: "选择歌词的语言类型",
    tips: "YuE-2 支持中文和英文，选对语言效果更好",
  },
  {
    name: "人声性别",
    icon: "🎤",
    description: "选择演唱的声音类型：男声、女声或自动",
    tips: "自动模式会根据歌词内容智能选择最合适的声音",
  },
  {
    name: "声音类型",
    icon: "🎵",
    description: "选择声音的音域：女高音、男高音、女中音等",
    tips: "女高音适合高音部分多的歌，男中音适合抒情歌曲",
  },
  {
    name: "多轨道",
    icon: "🎹",
    description: "是否分别输出人声和伴奏两个独立音轨",
    tips: "勾选后可以单独调整人声和伴奏的音量，方便后期混音",
  },
];

const ACESTEP_PARAMS: ParamInfo[] = [
  {
    name: "节奏速度 (BPM)",
    icon: "🎼",
    description: "控制音乐的快慢节奏",
    tips: "舒缓音乐 60-90，流行音乐 100-130，电子舞曲 120-140",
  },
  {
    name: "时长",
    icon: "⏱️",
    description: "生成音乐的总长度",
    tips: "ACE-Step 适合生成 30-180 秒的音乐片段",
  },
  {
    name: "推理步数",
    icon: "🔄",
    description: "生成音乐时的计算次数，越多质量越高但速度越慢",
    tips: "快速预览用 20-30 步，高质量输出用 50-100 步",
  },
  {
    name: "引导强度 (CFG)",
    icon: "🎯",
    description: "控制生成的音乐与你的描述的贴合程度",
    tips: "数值越大越贴合描述，但太大可能失真。推荐 5-10",
  },
  {
    name: "音高偏移",
    icon: "🎹",
    description: "调整整体音高，正数升高，负数降低",
    tips: "0 是标准音高，±3 是常用范围，超过 ±12 会很明显",
  },
  {
    name: "随机种子",
    icon: "🎲",
    description: "控制生成的随机性，相同种子会生成相似结果",
    tips: "-1 表示每次都随机，固定数字可以复现结果",
  },
  {
    name: "MIDI",
    icon: "🎹",
    description: "是否同时生成 MIDI 文件（乐器音符数据）",
    tips: "勾选后可以在音乐软件中编辑音符和编曲",
  },
  {
    name: "多轨道",
    icon: "🎚️",
    description: "分别输出人声、伴奏、鼓、贝斯等独立音轨",
    tips: "适合需要精细混音的专业制作",
  },
];

const HEARTMULA_PARAMS: ParamInfo[] = [
  {
    name: "节奏速度 (BPM)",
    icon: "🎼",
    description: "控制歌曲的快慢节奏",
    tips: "抒情歌 70-90，流行歌 100-130，动感歌曲 130-150",
  },
  {
    name: "时长",
    icon: "⏱️",
    description: "生成歌曲的总长度",
    tips: "HeartMuLa 适合生成 60-240 秒的完整歌曲段落",
  },
  {
    name: "语言",
    icon: "🌍",
    description: "选择歌词语言",
    tips: "支持中文、英文、日语、韩语、西班牙语多种语言",
  },
  {
    name: "推理步数",
    icon: "🔄",
    description: "生成时的迭代次数，影响质量和速度",
    tips: "默认 50 步平衡质量和速度，100 步获得最佳质量",
  },
  {
    name: "引导强度 (CFG)",
    icon: "🎯",
    description: "控制生成结果与描述的匹配度",
    tips: "推荐值 7-12，数值太小可能偏离主题，太大可能不自然",
  },
  {
    name: "随机种子",
    icon: "🎲",
    description: "控制随机性，用于复现特定结果",
    tips: "-1 每次随机生成新风格，固定数字可以微调同一风格",
  },
];

const PARAM_GUIDES: Record<SongModelFamily, ParamInfo[]> = {
  yue2: YUE2_PARAMS,
  acestep: ACESTEP_PARAMS,
  heartmula: HEARTMULA_PARAMS,
};

const MODEL_INTROS: Record<SongModelFamily, { title: string; intro: string }> = {
  yue2: {
    title: "YuE-2 · 歌词成曲模型",
    intro: "适合根据歌词生成带人声演唱的完整歌曲，支持中英文歌词，可以输出人声和伴奏分轨。",
  },
  acestep: {
    title: "ACE-Step · 文本成曲模型",
    intro: "适合根据文字描述生成纯音乐，支持 20+ 语言，可以导出 MIDI 和多轨音频，适合音乐制作。",
  },
  heartmula: {
    title: "HeartMuLa · 歌词成曲模型",
    intro: "适合生成多语言带人声的歌曲，支持中英日韩西班牙语，音质细腻，适合抒情和流行风格。",
  },
};

export function ParamsTutorialDialog({ modelFamily, onClose }: ParamsTutorialDialogProps) {
  const params = PARAM_GUIDES[modelFamily];
  const modelInfo = MODEL_INTROS[modelFamily];

  return (
    <div className="pt-overlay" onClick={onClose}>
      <div className="pt-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="pt-header">
          <div className="pt-title">
            <span className="pt-title-icon">📖</span>
            参数教程
          </div>
          <button className="pt-close" onClick={onClose}>✕</button>
        </div>

        <div className="pt-body">
          {/* 模型介绍 */}
          <div className="pt-model-intro">
            <div className="pt-model-intro-title">{modelInfo.title}</div>
            <div className="pt-model-intro-text">{modelInfo.intro}</div>
          </div>

          {/* 参数列表 */}
          <div className="pt-params-list">
            {params.map((param, idx) => (
              <div key={idx} className="pt-param-card">
                <div className="pt-param-header">
                  <span className="pt-param-icon">{param.icon}</span>
                  <span className="pt-param-name">{param.name}</span>
                </div>
                <div className="pt-param-desc">{param.description}</div>
                <div className="pt-param-tips">
                  <span className="pt-tips-icon">💡</span>
                  {param.tips}
                </div>
              </div>
            ))}
          </div>

          {/* 底部提示 */}
          <div className="pt-footer-tip">
            <span className="pt-footer-tip-icon">✨</span>
            <span>建议先用默认参数试试效果，再根据需要微调参数</span>
          </div>
        </div>
      </div>
    </div>
  );
}
