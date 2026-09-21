// 统一任务运行器（规划 4.3）：所有模式（写歌/翻唱/重绘/补全/分轨/叠加/乐谱/伴奏/快速分轨）
// 的 UI 都只调用 runSongTask，由这里负责：
//   1. 按任务定义组装 SongGenRequest（task 路由、src_audio_path、专属字段、want_* 开关）
//   2. 模型就绪校验（stems=demucs 任务除外，它不需要生成模型）
//   3. 进度事件订阅与归一化（song-generation-progress → {stage, message, percent}）
//   4. extract 多选轨种时顺序提交（后端单任务锁），合并为标准 stems 产物
import { listen } from "@tauri-apps/api/event";
import { listSongModels, songGenerate, } from "../backendSong";
import { ACE_TASK_OF, SONG_TASKS } from "../models/song-tasks";
import { getSongInstallStatus, songModelById } from "../models/song-catalog";
// ── 内部工具 ───────────────────────────────────────────────────────────────
const STAGE_WEIGHTS = {
    init: 5,
    initialize: 5,
    plan: 10,
    semantic: 40,
    synthesize: 70,
    decode: 90,
    save: 95,
    midi: 97,
    stems: 98,
    generate: 70,
    done: 100,
    complete: 100,
};
function sanitizeName(s) {
    return s.replace(/[\\/:*?"<>|]/g, "_").trim();
}
function defaultLabel(task) {
    return `${task}_${Date.now().toString(36)}`;
}
async function ensureModelReady(modelId) {
    const catalog = songModelById(modelId);
    if (!catalog)
        return; // 未知模型（如未来扩展）交给后端校验
    const files = await listSongModels();
    const st = getSongInstallStatus(files, catalog);
    if (!st.installed) {
        throw new Error(`模型「${catalog.id}」尚未下载完整（${st.ready}/${st.required}），请先到「资源管理 → 生成歌曲」下载。`);
    }
}
/** 订阅后端进度事件并归一化为 0~100 百分比 */
async function bindProgress(cb, 
/** extract 批量时的整体包装：innerPct(0~100) → overallPct */
wrap) {
    const unlisten = await listen("song-generation-progress", (e) => {
        const { stage, current, total, stem_label } = e.payload;
        let pct = STAGE_WEIGHTS[stage] ?? 0;
        if (stage === "generate" && total > 0 && current <= total) {
            pct = Math.round((current / total) * 100);
        }
        if (wrap)
            pct = wrap(pct);
        cb({ stage, message: stem_label || "", percent: Math.max(0, Math.min(100, pct)) });
    });
    return unlisten;
}
function reqBase(p, modelId, songName) {
    const req = {
        model: modelId,
        song_name: songName,
        lyrics: p.lyrics ?? "",
        prompt: p.prompt ?? "",
        audio_duration: p.durationSec ?? 0,
        format: "wav",
        output_dir: "",
    };
    if (typeof p.seed === "number" && p.seed >= 0)
        req.seed = p.seed;
    if (p.wantStems)
        req.want_stems = true;
    if (p.wantMidi)
        req.want_midi = true;
    if (p.wantLrc)
        req.want_lrc = true;
    return req;
}
function firstAudio(outputs) {
    return outputs.find((o) => o.audio_path)?.audio_path;
}
// ── 主入口 ─────────────────────────────────────────────────────────────────
export async function runSongTask(task, payload, opts = {}) {
    const def = SONG_TASKS[task];
    const label = sanitizeName(payload.songName || "") || defaultLabel(task);
    const modelId = payload.model ?? def.models[0] ?? "acestep-v1.5";
    // demucs 快速分轨不需要生成模型（sidecar 在模型校验前处理 stems_only）
    if (task !== "stems")
        await ensureModelReady(modelId);
    const report = opts.onProgress ?? (() => { });
    // ── 快速四分轨（demucs，stems_only 轻量任务）──
    if (task === "stems") {
        if (!payload.srcAudioPath)
            throw new Error("请先选择源音频");
        const unlisten = await bindProgress(report);
        try {
            const outputs = await songGenerate({
                ...reqBase(payload, modelId, label),
                model: "acestep-v1.5", // 仅作占位：sidecar 在路由前直接处理 stems_only，不加载模型
                task: "stems_only",
                src_audio_path: payload.srcAudioPath,
            });
            const stems = outputs.find((o) => o.stems && Object.keys(o.stems).length > 0)?.stems;
            if (!stems)
                throw new Error("分轨完成但没有返回任何分轨文件");
            return {
                task, modelId: "demucs", outputs: [{ label, stems }],
                durationSec: payload.srcDurationSec ?? 0, label,
            };
        }
        finally {
            unlisten();
        }
    }
    // ── 分轨（ACE 原生 extract，逐轨顺序提交）──
    if (task === "extract") {
        if (!payload.srcAudioPath)
            throw new Error("请先选择源音频");
        const classes = (payload.trackClasses ?? []).filter(Boolean);
        if (classes.length === 0)
            throw new Error("请至少选择一个要分离的轨种");
        let done = 0;
        const unlisten = await bindProgress(report, (inner) => Math.round(((done + inner / 100) / classes.length) * 100));
        const stems = {};
        try {
            for (const cls of classes) {
                report({ stage: "extract", message: `分离 ${cls}（${done + 1}/${classes.length}）`, percent: 0 });
                const outputs = await songGenerate({
                    ...reqBase(payload, modelId, `${label}_${cls}`),
                    task: ACE_TASK_OF.extract,
                    src_audio_path: payload.srcAudioPath,
                    track_name: cls,
                });
                const audio = firstAudio(outputs);
                if (!audio)
                    throw new Error(`分离 ${cls} 失败：未返回音频文件`);
                stems[cls] = audio;
                done += 1;
            }
            return {
                task, modelId, outputs: [{ label, stems }],
                durationSec: payload.srcDurationSec ?? 0, label,
            };
        }
        finally {
            unlisten();
        }
    }
    // ── 乐谱（YuE2：生成式 ABC + 可选 MIDI；支持 abc 续写）──
    if (task === "sheet") {
        const unlisten = await bindProgress(report);
        try {
            const outputs = await songGenerate({
                ...reqBase(payload, "yue2-3b", label),
                task: "text2music",
                lyrics: "[Instrumental]",
                ...(payload.abc?.trim() ? { abc: payload.abc.trim() } : {}),
            });
            if (!firstAudio(outputs) && !outputs.some((o) => o.abc_path)) {
                throw new Error("生成完成但没有返回乐谱或音频");
            }
            return { task, modelId: "yue2-3b", outputs, durationSec: 0, label };
        }
        finally {
            unlisten();
        }
    }
    // ── 其余任务：generate / instrumental / cover / repaint / complete / lego ──
    const isAceTask = task in ACE_TASK_OF;
    const aceFamily = modelId.startsWith("acestep");
    const req = reqBase(payload, modelId, label);
    if (isAceTask) {
        // cover / repaint / complete / lego → ACE direct-conditioning 任务
        if (!payload.srcAudioPath)
            throw new Error("请先选择源音频");
        req.task = ACE_TASK_OF[task] ?? "text2music";
        req.src_audio_path = payload.srcAudioPath;
        if (task === "lego" && payload.trackName)
            req.track_name = payload.trackName;
        if (task === "complete" && payload.trackClasses?.length) {
            req.complete_track_classes = payload.trackClasses;
        }
        if (task === "repaint") {
            req.repaint_start = payload.repaintStart ?? 0;
            req.repaint_end = payload.repaintEnd ?? 0;
        }
        if (task === "cover" && typeof payload.coverStrength === "number") {
            req.audio_cover_strength = payload.coverStrength;
        }
    }
    else if (task === "instrumental" && aceFamily) {
        req.lyrics = ""; // ACE 侧空歌词自动按 [Instrumental] 处理
    }
    else if (task === "instrumental") {
        req.lyrics = "[Instrumental]";
    }
    else if (aceFamily) {
        req.task = "text2music";
    }
    else {
        req.cot = payload.extra?.cot ?? "full"; // YuE2 默认走完整 CoT
    }
    if (payload.extra)
        Object.assign(req, payload.extra);
    const unlisten = await bindProgress(report);
    try {
        const outputs = await songGenerate(req);
        if (!firstAudio(outputs))
            throw new Error("生成完成但没有返回音频文件");
        return {
            task, modelId, outputs,
            durationSec: payload.srcDurationSec ?? 0,
            label,
        };
    }
    finally {
        unlisten();
    }
}
