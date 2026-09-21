// ── capabilities 推导（唯一依据是 outputs 实际内容，不认模型静态标签）────────
export function deriveCapabilities(entry) {
    let hasStems = false, hasMidi = false, hasLrc = false, hasAbc = false;
    const stemKinds = new Set();
    for (const o of entry.outputs ?? []) {
        if (o.audio_path) { /* 主音频单独不算能力；stems 才算 */ }
        if (o.midi_path)
            hasMidi = true;
        if (o.lrc_path)
            hasLrc = true;
        if (o.abc_path)
            hasAbc = true;
        for (const kind of Object.keys(o.stems ?? {})) {
            hasStems = true;
            stemKinds.add(kind);
        }
    }
    return { hasStems, hasMidi, hasLrc, hasAbc, stemKinds: [...stemKinds] };
}
// ── 旧历史 JSON 迁移（不允许旧文件打不开）─────────────────────────────────
/** 解析 + 迁移：task 缺省 → "generate"，capabilities 现算（覆盖脏值）。返回 null 表示 JSON 非法。 */
export function migrateHistoryEntries(raw) {
    let parsed;
    try {
        parsed = JSON.parse(raw);
    }
    catch {
        return null;
    }
    if (!Array.isArray(parsed))
        return null;
    return parsed.map((e) => ({
        ...e,
        task: e.task ?? "generate",
        capabilities: deriveCapabilities(e),
    }));
}
