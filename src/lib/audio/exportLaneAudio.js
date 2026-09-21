import { invoke } from "@tauri-apps/api/core";
/** All audio pieces that back one whole track (in arrangement order). */
export function trackExportItems(track) {
    const items = [];
    const pushPath = (label, p, seen) => {
        if (!p)
            return;
        const key = p.toLowerCase();
        if (seen.has(key))
            return; // same source backing two segments → one copy
        seen.add(key);
        items.push({ label, sourcePath: p });
    };
    for (const seg of track.segments) {
        const seen = new Set();
        const c = seg.content;
        if (c?.type === "audioClip" && c.sourcePath) {
            pushPath(track.name || "audio", c.sourcePath, seen);
            // Sub-lane stems (分离/克隆 baked into this audio clip) each live in their own file.
            for (const po of seg.processedOutputs ?? [])
                pushPath(po.laneLabel || "lane", po.audioPath, seen);
        }
        else if (c?.type === "notes") {
            // ② vocal / 钢琴卷帘 baked stems (if any) — raw note tracks without a bake have no file.
            for (const po of seg.processedOutputs ?? [])
                pushPath(po.laneLabel || "vocal", po.audioPath, seen);
        }
    }
    return items;
}
/** The single audio piece that backs one sub-lane row (ProcessedOutput) — if any. */
export function laneExportItem(po, fallbackLabel) {
    if (!po?.audioPath)
        return null;
    return { label: po.laneLabel || fallbackLabel, sourcePath: po.audioPath };
}
/** Copy ONE audio file to a user-picked destination (a full-file save dialog).
 * @returns the destination path actually written (for the success toast). */
export async function exportOneAudioFile(item, destPath) {
    await invoke("copy_audio_file_to", { source: item.sourcePath, dest: destPath });
    return destPath;
}
/** Copy several audio files into a user-picked FOLDER (keeps each original filename). */
export async function exportAudioFilesToFolder(items, folder) {
    const out = [];
    const taken = new Set();
    for (const it of items) {
        const base = sanitizeBase(it.sourcePath);
        const extPart = ext(it.sourcePath);
        let name = `${base}${extPart}`;
        let n = 2;
        // sanitizeBase already strips the extension, so the collision suffix goes on the bare stem.
        while (taken.has(name.toLowerCase()))
            name = `${base}_${n++}${extPart}`;
        taken.add(name.toLowerCase());
        const dest = `${folder.replace(/[\\/]+$/, "")}\\${name}`;
        await invoke("copy_audio_file_to", { source: it.sourcePath, dest });
        out.push(dest);
    }
    return out;
}
/** Copy ONE file into a user-picked FOLDER, with an optional display label in the name. */
export async function exportOneAudioFileToFolder(item, folder, label) {
    const extPart = ext(item.sourcePath);
    const stem = sanitizeBase(item.sourcePath);
    // Strip an extension the caller already put in the label ("song.wav" → "song"), so the saved
    // file is "song.wav", not "song.wav.wav".
    const labelPart = label ? sanitize(label).replace(/\.[^.\\/]+$/, "") : "";
    const name = labelPart ? `${labelPart}${extPart}` : `${stem}${extPart}`;
    const dest = `${folder.replace(/[\\/]+$/, "")}\\${name}`;
    await invoke("copy_audio_file_to", { source: item.sourcePath, dest });
    return dest;
}
function ext(p) {
    const m = /(\.[^.\\/]+)$/.exec(p);
    return m ? m[1].toLowerCase() : "";
}
function sanitizeBase(p) {
    const base = p.replace(/^.*[\\/]/, "").replace(/\.[^.\\/]+$/, "");
    return sanitize(base) || "audio";
}
function sanitize(s) {
    return s.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").replace(/\s+/g, " ").trim().slice(0, 120);
}
/** Shared error mapping — keep the surface consistent with exportErrorMessage in ExportAudioDialog. */
export function laneExportErrorMessage(e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("ENOENT") || /No such file|not found|不存在|找不到/i.test(msg)) {
        return "导出失败：源音频文件不存在或已被清理（缓存可能被清空）。请重新渲染/运行一次工作流后再导出。";
    }
    if (msg.includes("EACCES") || /denied|权限/i.test(msg)) {
        return "导出失败：没有写入权限，请换一个目录。";
    }
    if (msg.includes("copy_audio_file_to")) {
        return `导出失败：${msg}`;
    }
    return `导出失败：${msg}`;
}
/** A track/lane "has something to export" — for enabling/disabling menu items. */
export function hasExportableAudio(track) {
    return trackExportItems(track).length > 0;
}
// ─── 轨道另存为 · 整个项目的所有轨道一起保存到同一个文件夹 ─────────────────────────────
// File→轨道另存为:把工程里每条轨道的可导出音频(导入的源文件 / 分离 stems / 克隆 bake wav)
// 全部复制进用户选定的一个文件夹。命名带上轨道名,避免不同轨道同源文件名互相混淆。
/** 整个工程所有轨道的可导出音频(带轨道名前缀,便于命名区分)。 */
export function projectExportItems(tracks) {
    const items = [];
    for (const trk of tracks) {
        for (const it of trackExportItems(trk)) {
            // label 形如 "<轨道名>__<内容名>";命名函数拆开用。
            items.push({ label: `${trk.name || "track"}__${it.label}`, sourcePath: it.sourcePath });
        }
    }
    return items;
}
/** 整个工程"有没有任何可导出的音频" —— 用于菜单启用/禁用。 */
export function anyProjectExportable(tracks) {
    return projectExportItems(tracks).length > 0;
}
/** 把所有轨道音频复制进 `folder`,每个文件命名为 <轨道名>_<内容名>.<ext>(重名自动加 _2/_3…),
 *  返回实际写入的路径列表。源文件永不改动;复用 Rust 的 fs::copy,不做解码/重编码。 */
export async function exportProjectTracksToFolder(tracks, folder) {
    const out = [];
    const taken = new Set();
    for (const it of projectExportItems(tracks)) {
        const [trackPartRaw, ...restParts] = it.label.split("__");
        const trackPart = sanitize(trackPartRaw ?? "") || "track";
        const stemPart = sanitize(restParts.join("_")) || sanitizeBase(it.sourcePath) || "audio";
        const extPart = ext(it.sourcePath);
        let name = `${trackPart}_${stemPart}${extPart}`;
        let n = 2;
        while (taken.has(name.toLowerCase()))
            name = `${trackPart}_${stemPart}_${n++}${extPart}`;
        taken.add(name.toLowerCase());
        const dest = `${folder.replace(/[\\/]+$/, "")}\\${name}`;
        await invoke("copy_audio_file_to", { source: it.sourcePath, dest });
        out.push(dest);
    }
    return out;
}
// ─── 片段另存为 · 把单个片段的音频(原始导入 + 各结果子轨)复制进选定的文件夹 ─────────────
// 编曲区片段右键 → 另存为:把该片段(含分离 stems / 克隆 bake 等结果子轨)的所有可导出音频
// 复制进用户选定的一个文件夹,命名带内容标签(重名自动加 _2/_3)。源文件永不改动。
/** 单个片段的可导出音频(原始音频 + processedOutputs 各结果子轨)。 */
export function segmentExportItems(seg) {
    const items = [];
    const seen = new Set();
    const pushPath = (label, p) => {
        if (!p)
            return;
        const key = p.toLowerCase();
        if (seen.has(key))
            return; // same source backing two outputs → one copy
        seen.add(key);
        items.push({ label, sourcePath: p });
    };
    const c = seg.content;
    if (c?.type === "audioClip" && c.sourcePath) {
        pushPath("audio", c.sourcePath);
        for (const po of seg.processedOutputs ?? [])
            pushPath(po.laneLabel || "lane", po.audioPath);
    }
    else {
        for (const po of seg.processedOutputs ?? []) {
            pushPath(po.laneLabel || (c?.type === "notes" ? "vocal" : "audio"), po.audioPath);
        }
    }
    return items;
}
/** 单个片段"有没有任何可导出的音频" —— 用于菜单启用/禁用。 */
export function anySegmentExportable(seg) {
    return segmentExportItems(seg).length > 0;
}
/** 把单个片段的所有音频复制进 `folder`,命名 <标签>.<ext>(重名自动加 _2/_3…),返回实际写入路径。 */
export async function exportSegmentToFolder(seg, folder) {
    const out = [];
    const taken = new Set();
    for (const it of segmentExportItems(seg)) {
        const extPart = ext(it.sourcePath);
        const stem = sanitizeBase(it.sourcePath);
        const label = sanitize(it.label) || stem;
        let name = `${label}${extPart}`;
        let n = 2;
        while (taken.has(name.toLowerCase()))
            name = `${label}_${n++}${extPart}`;
        taken.add(name.toLowerCase());
        const dest = `${folder.replace(/[\\/]+$/, "")}\\${name}`;
        await invoke("copy_audio_file_to", { source: it.sourcePath, dest });
        out.push(dest);
    }
    return out;
}
