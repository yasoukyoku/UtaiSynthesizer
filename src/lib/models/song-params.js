// YuE-2 支持的参数
const YUE2_PARAMS = [
    {
        key: "cfg_scale",
        label: { zh: "CFG强度", en: "CFG Scale" },
        type: "number",
        default: 3.0,
        min: 0,
        max: 20,
        step: 0.5,
        description: { zh: "提示词引导强度", en: "Prompt guidance strength" },
        recommended: "3.0",
        example: {
            zh: "调高（5~7）会让歌曲更严格贴合提示词，适合指定明确风格；调低（1~2）更自由发挥。",
            en: "Higher (5–7) follows the prompt more strictly; lower (1–2) gives freer results.",
        },
    },
    {
        key: "cot",
        label: { zh: "推理模式", en: "CoT Mode" },
        type: "select",
        default: "off",
        options: [
            { label: "关闭 (Off)", value: "off" },
            { label: "旋律 (Melody)", value: "melody" },
            { label: "完整 (Full)", value: "full" },
        ],
        description: { zh: "Chain-of-Thought推理模式", en: "Chain-of-Thought reasoning mode" },
        recommended: "旋律 (Melody)",
        example: {
            zh: "需要完整歌曲结构（主歌/副歌）时选「完整」，只想要一段动听旋律时选「旋律」。",
            en: "Choose Full for complete song structure, Melody for a single melodic idea.",
        },
    },
    {
        key: "temperature",
        label: { zh: "采样温度", en: "Temperature" },
        type: "number",
        default: 1.0,
        min: 0,
        max: 5,
        step: 0.1,
        description: { zh: "生成随机性", en: "Generation randomness" },
        recommended: "1.0",
        example: {
            zh: "0.7~1.0 稳妥好听；1.5 以上更天马行空，可能产生意外惊喜或杂音。",
            en: "0.7–1.0 is safe; above 1.5 becomes more experimental.",
        },
    },
    {
        key: "top_p",
        label: { zh: "Top-P", en: "Top-P" },
        type: "number",
        default: 0.95,
        min: 0,
        max: 1,
        step: 0.05,
        description: { zh: "核采样概率", en: "Nucleus sampling probability" },
        recommended: "0.95",
        example: {
            zh: "保持 0.9~0.98 即可；过低会让旋律单调，过高容易跑调。",
            en: "Keep 0.9–0.98; too low sounds monotonous, too high goes off-key.",
        },
    },
    {
        key: "top_k",
        label: { zh: "Top-K", en: "Top-K" },
        type: "number",
        default: 50,
        min: 1,
        max: 200,
        step: 1,
        description: { zh: "保留前K个token", en: "Keep top K tokens" },
        recommended: "50",
        example: {
            zh: "中文歌曲建议 50；英文或器乐可试 80~120 更丰富。",
            en: "Use 50 for Chinese vocals; try 80–120 for English or instrumental.",
        },
    },
    {
        key: "repetition_penalty",
        label: { zh: "重复惩罚", en: "Repetition Penalty" },
        type: "number",
        default: 1.0,
        min: 0.5,
        max: 2.0,
        step: 0.1,
        description: { zh: "重复内容惩罚", en: "Penalize repetition" },
        recommended: "1.0",
        example: {
            zh: "若副歌重复过多显得单调，调到 1.1~1.3 可增加变化。",
            en: "If the chorus repeats too much, raise to 1.1–1.3 for variation.",
        },
    },
    {
        key: "ode_steps",
        label: { zh: "ODE步数", en: "ODE Steps" },
        type: "number",
        default: 32,
        min: 8,
        max: 100,
        step: 1,
        description: { zh: "ODE求解步数", en: "ODE solver steps" },
        recommended: "32",
        example: {
            zh: "32 步音质与速度平衡；追求极致音质可用 50~64，但生成更慢。",
            en: "32 balances quality and speed; 50–64 for maximum quality (slower).",
        },
    },
    {
        key: "ode_method",
        label: { zh: "ODE方法", en: "ODE Method" },
        type: "select",
        default: "midpoint",
        options: [
            { label: "中点法 (Midpoint)", value: "midpoint" },
            { label: "欧拉法 (Euler)", value: "euler" },
        ],
        description: { zh: "ODE求解方法", en: "ODE solving method" },
        recommended: "中点法 (Midpoint)",
        example: {
            zh: "默认中点法音质更好；欧拉法速度更快，适合快速试听。",
            en: "Midpoint gives better quality; Euler is faster for quick previews.",
        },
    },
];
// ACE-Step 支持的参数
const ACESTEP_PARAMS = [
    {
        key: "num_inference_steps",
        label: { zh: "推理步数", en: "Inference Steps" },
        type: "number",
        default: 60,
        min: 20,
        max: 200,
        step: 5,
        description: { zh: "扩散模型推理步数", en: "Diffusion inference steps" },
        recommended: "60",
        example: {
            zh: "60 步细节充足；快速草稿可用 30 步，成品建议 80~100 步。",
            en: "60 for good detail; 30 for drafts, 80–100 for final masters.",
        },
    },
    {
        key: "guidance_scale",
        label: { zh: "引导强度", en: "Guidance Scale" },
        type: "number",
        default: 15.0,
        min: 1,
        max: 30,
        step: 0.5,
        description: { zh: "提示词引导强度", en: "Prompt guidance strength" },
        recommended: "15.0",
        example: {
            zh: "15 是通用推荐；想更贴提示词调到 18~22，想更自然调到 10~13。",
            en: "15 is the general default; 18–22 to follow prompts closely, 10–13 for naturalness.",
        },
    },
    {
        key: "scheduler_type",
        label: { zh: "调度器", en: "Scheduler" },
        type: "select",
        default: "euler",
        options: [
            { label: "Euler", value: "euler" },
            { label: "DDPM", value: "ddpm" },
            { label: "DDIM", value: "ddim" },
            { label: "PNDM", value: "pndm" },
        ],
        description: { zh: "噪声调度器类型", en: "Noise scheduler type" },
        recommended: "Euler",
        example: {
            zh: "Euler 速度快、效果稳；想尝试不同听感可换 DDIM 或 PNDM。",
            en: "Euler is fast and stable; try DDIM/PNDM for different character.",
        },
    },
    {
        key: "cfg_type",
        label: { zh: "CFG类型", en: "CFG Type" },
        type: "select",
        default: "apg",
        options: [
            { label: "APG", value: "apg" },
            { label: "Standard", value: "standard" },
        ],
        description: { zh: "无分类器引导类型", en: "Classifier-free guidance type" },
        recommended: "APG",
        example: {
            zh: "APG 针对音频优化、更少爆音；Standard 为传统实现，兼容旧流程。",
            en: "APG is audio-optimized with fewer artifacts; Standard is the classic mode.",
        },
    },
    {
        key: "omega_scale",
        label: { zh: "Omega缩放", en: "Omega Scale" },
        type: "number",
        default: 10.0,
        min: 0,
        max: 20,
        step: 1,
        description: { zh: "APG Omega参数", en: "APG Omega parameter" },
        recommended: "10.0",
        example: {
            zh: "仅在 CFG 类型为 APG 时生效；默认 10 无需调整。",
            en: "Only affects APG mode; the default 10 usually needs no change.",
        },
    },
    {
        key: "guidance_interval",
        label: { zh: "引导区间", en: "Guidance Interval" },
        type: "number",
        default: 0.5,
        min: 0,
        max: 1,
        step: 0.1,
        description: { zh: "引导作用时间段", en: "Guidance active period" },
        recommended: "0.5",
        example: {
            zh: "0.5 表示后半程引导；想让整体更贴提示词可设为 1.0（全程引导）。",
            en: "0.5 guides the latter half; set 1.0 to guide the whole process.",
        },
    },
    {
        key: "min_guidance_scale",
        label: { zh: "最小引导", en: "Min Guidance" },
        type: "number",
        default: 3.0,
        min: 0,
        max: 10,
        step: 0.5,
        description: { zh: "最小引导强度", en: "Minimum guidance strength" },
        recommended: "3.0",
        example: {
            zh: "引导强度的下限，配合「引导区间」使用；默认 3.0 即可。",
            en: "Lower bound paired with Guidance Interval; default 3.0 is fine.",
        },
    },
    {
        key: "use_erg_tag",
        label: { zh: "ERG标签", en: "ERG Tag" },
        type: "boolean",
        default: true,
        description: { zh: "启用ERG标签增强", en: "Enable ERG tag enhancement" },
        recommended: "开启",
        example: {
            zh: "开启后模型更准确理解风格标签（如 city pop、lo-fi）。",
            en: "Improves understanding of style tags (e.g. city pop, lo-fi).",
        },
    },
    {
        key: "use_erg_lyric",
        label: { zh: "ERG歌词", en: "ERG Lyric" },
        type: "boolean",
        default: true,
        description: { zh: "启用ERG歌词增强", en: "Enable ERG lyric enhancement" },
        recommended: "开启",
        example: {
            zh: "有歌词的歌曲建议开启，人声咬字与段落对应更准。",
            en: "Recommended for songs with lyrics for better vocal alignment.",
        },
    },
    {
        key: "use_erg_diffusion",
        label: { zh: "ERG扩散", en: "ERG Diffusion" },
        type: "boolean",
        default: true,
        description: { zh: "启用ERG扩散增强", en: "Enable ERG diffusion enhancement" },
        recommended: "开启",
        example: {
            zh: "默认开启以获得更饱满的编曲层次；素材紧张时可关闭提速。",
            en: "On by default for richer arrangement; turn off to speed up.",
        },
    },
    {
        key: "task",
        label: { zh: "任务类型", en: "Task Type" },
        type: "select",
        default: "text2music",
        options: [
            { label: "文本生成音乐 (Text2Music)", value: "text2music" },
            { label: "重新生成 (Retake)", value: "retake" },
            { label: "局部重绘 (Repaint)", value: "repaint" },
            { label: "编辑 (Edit)", value: "edit" },
            { label: "延长 (Extend)", value: "extend" },
        ],
        description: { zh: "生成任务类型", en: "Generation task type" },
        recommended: "文本生成音乐",
        example: {
            zh: "从零创作选 Text2Music；已有歌曲想改造选 Repaint 或 Edit。",
            en: "Choose Text2Music for new songs; Repaint/Edit to modify existing ones.",
        },
    },
];
// HeartMuLa 支持的参数
const HEARTMULA_PARAMS = [
    {
        key: "cfg_scale",
        label: { zh: "CFG强度", en: "CFG Scale" },
        type: "number",
        default: 1.5,
        min: 1.0,
        max: 5.0,
        step: 0.1,
        description: { zh: "提示词引导强度", en: "Prompt guidance strength" },
        recommended: "1.5",
        example: {
            zh: "1.5 平衡自然与贴合；想更贴近提示词调到 2.0~2.5。",
            en: "1.5 balances naturalness and adherence; 2.0–2.5 to follow prompts closer.",
        },
    },
    {
        key: "temperature",
        label: { zh: "采样温度", en: "Temperature" },
        type: "number",
        default: 1.0,
        min: 0.1,
        max: 2.0,
        step: 0.1,
        description: { zh: "生成随机性", en: "Generation randomness" },
        recommended: "1.0",
        example: {
            zh: "0.8~1.1 最稳定；超过 1.5 变化大但可能不连贯。",
            en: "0.8–1.1 is most stable; above 1.5 varies widely and may be incoherent.",
        },
    },
    {
        key: "topk",
        label: { zh: "Top-K", en: "Top-K" },
        type: "number",
        default: 50,
        min: 10,
        max: 200,
        step: 10,
        description: { zh: "保留前K个token", en: "Keep top K tokens" },
        recommended: "50",
        example: {
            zh: "50 是推荐值；想要更新颖的旋律可试 80~100。",
            en: "50 is recommended; try 80–100 for more novel melodies.",
        },
    },
];
// 模型参数映射
export const MODEL_PARAMS_MAP = {
    yue2: YUE2_PARAMS,
    acestep: ACESTEP_PARAMS,
    heartmula: HEARTMULA_PARAMS,
};
// 根据模型family获取参数配置
export function getModelParams(family) {
    return MODEL_PARAMS_MAP[family] || [];
}
// 获取参数默认值对象
export function getDefaultParams(family) {
    const params = getModelParams(family);
    const defaults = {};
    params.forEach(p => {
        defaults[p.key] = p.default;
    });
    return defaults;
}
