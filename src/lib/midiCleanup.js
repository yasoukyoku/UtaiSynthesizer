/**
 * MIDI 音符「一键修复」核心（smart cleanup v3）——从 AmtResultPanel 抽出的共享纯逻辑，
 * 供转换结果工作台与 DAW 音符编辑器（VocalEditor）两个入口复用。
 *
 * v3 在 v2 基础上向「人工修谱」效果逼近：
 *   1) 超短伪音（<40ms）                                 → 删除
 *   2) 同音重叠 / 双触发                                  → 合并
 *   3) 和声幻觉（同起点 +12/+19/+24、力度≤基准60%，仅旋律轨）→ 删除
 *   4) 力度离群值：极端弱音抬升 + 局部中位数轮廓压缩        → 修正
 *   5) 调外 + 弱 + 短 的音符                             → 删除
 *   6) GM 音域外的音高                                    → 删除
 *   7) 空小节补漏：前两小节重复（v2）或自相似周期检测（v3：周期 1/2/4/8 小节）→ 平铺补齐
 *   8) 拖尾音符截断：AMT 常留悬挂长音（旋律轨收束为单声部、鼓/乐器封顶）   → 截断
 *   9) 轻量节拍量化：起点贴 1/32 网格（容差 = 1/20 拍）    → 吸附
 *
 * 确定性、可解释；调用方负责克隆/提交（各自记为一个 undo 步）。
 */
export const emptyCleanStats = () => ({
    removedShort: 0, mergedOverlap: 0, removedGhost: 0, removedKey: 0,
    fixedVelocity: 0, removedRange: 0, filledNotes: 0, trimmedHang: 0, snapped: 0, touched: [],
});
/** stats 汇总条数（用于「无变化」判定）。 */
export const cleanChangedTotal = (s) => s.removedShort + s.removedGhost + s.removedKey + s.removedRange +
    s.mergedOverlap + s.fixedVelocity + s.filledNotes + s.trimmedHang + s.snapped;
// ── Krumhansl–Kessler 调性检测（时长加权音级直方图） ─────────────────────────
const KK_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KK_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
function pearson(a, b) {
    const n = a.length;
    if (n === 0 || b.length !== n)
        return 0;
    let ma = 0;
    let mb = 0;
    for (let i = 0; i < n; i++) {
        ma += a[i] ?? 0;
        mb += b[i] ?? 0;
    }
    ma /= n;
    mb /= n;
    let num = 0;
    let da = 0;
    let db = 0;
    for (let i = 0; i < n; i++) {
        const x = (a[i] ?? 0) - ma;
        const y = (b[i] ?? 0) - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    return da > 0 && db > 0 ? num / Math.sqrt(da * db) : 0;
}
/** 最佳匹配音级集合（音符太少时返回 null，避免误判）。 */
export function detectKeyPitchClasses(all) {
    const hist = new Array(12).fill(0);
    let count = 0;
    for (const s of all) {
        if (s.channel === 9)
            continue; // 鼓不携带调性信息
        for (const n of s.notes) {
            const pc = ((n.pitch % 12) + 12) % 12;
            hist[pc] = (hist[pc] ?? 0) + Math.max(1, n.duration);
            count++;
        }
    }
    if (count < 24)
        return null;
    let bestR = -2;
    let bestScale = null;
    for (let root = 0; root < 12; root++) {
        const rotated = hist.map((_, i) => hist[(i + root) % 12] ?? 0);
        const rM = pearson(rotated, KK_MAJOR_PROFILE);
        if (rM > bestR) {
            bestR = rM;
            bestScale = [0, 2, 4, 5, 7, 9, 11].map((p) => (p + root) % 12);
        }
        const rN = pearson(rotated, KK_MINOR_PROFILE);
        if (rN > bestR) {
            bestR = rN;
            bestScale = [0, 2, 3, 5, 7, 8, 10].map((p) => (p + root) % 12);
        }
    }
    return bestScale ? new Set(bestScale) : null;
}
/** 音高集合 Jaccard × 音符数比值 —— 两小节音符表的相似度。 */
function barSimilarity(a, b) {
    if (a.length === 0 || b.length === 0)
        return 0;
    const pa = new Set(a.map((n) => n.pitch));
    const pb = new Set(b.map((n) => n.pitch));
    let inter = 0;
    for (const p of pa)
        if (pb.has(p))
            inter++;
    const union = pa.size + pb.size - inter;
    const countRatio = Math.min(a.length, b.length) / Math.max(a.length, b.length);
    return (inter / union) * countRatio;
}
/**
 * 就地清理一组 stem（调用方先克隆）。返回统计；不修改 stem 以外字段。
 */
export function smartCleanStems(stems, opts) {
    const stats = emptyCleanStats();
    const safePpq = opts.ppq > 0 ? opts.ppq : 480;
    const safeBpm = opts.bpm > 0 ? opts.bpm : 120;
    const msToTicks = (ms) => Math.max(1, Math.round((ms / 1000) * (safeBpm / 60) * safePpq));
    const minTicks = msToTicks(40);
    const gapTicks = msToTicks(30);
    const makeId = opts.makeId ?? (() => crypto.randomUUID());
    const keySet = detectKeyPitchClasses(stems); // 清理前直方图 → 全局调性
    const durationTicks = opts.durationTicks;
    const barTicks = Math.max(1, safePpq * 4); // 4/4 小节
    stems.forEach((s) => {
        const isDrum = s.channel === 9;
        const [lo, hi] = isDrum ? [27, 87] : [21, 108];
        // Pass 1+6（按音高分桶）：删伪音/域外音，再按时序合并同音重叠/同度重复。
        const byPitch = new Map();
        for (const n of s.notes) {
            if (n.pitch < lo || n.pitch > hi) {
                stats.removedRange++;
                continue;
            }
            if (n.duration < minTicks) {
                stats.removedShort++;
                continue;
            }
            const list = byPitch.get(n.pitch) ?? [];
            list.push(n);
            byPitch.set(n.pitch, list);
        }
        let kept = [];
        for (const list of byPitch.values()) {
            list.sort((a, b) => a.tick - b.tick);
            const out = [];
            for (const n of list) {
                const last = out[out.length - 1];
                // 重叠 或 <30ms 的同音间隙听起来就是一个音 —— 合并。
                if (last && n.tick < last.tick + last.duration + gapTicks) {
                    const end = Math.min(durationTicks, Math.max(last.tick + last.duration, n.tick + n.duration));
                    last.duration = Math.max(1, end - last.tick);
                    last.velocity = Math.max(last.velocity ?? 100, n.velocity ?? 100);
                    stats.mergedOverlap++;
                    if (last.id)
                        stats.touched.push(last.id);
                    continue;
                }
                out.push(n);
            }
            kept.push(...out);
        }
        kept.sort((a, b) => a.tick - b.tick || a.pitch - b.pitch);
        // Pass 3（仅旋律）：和声幻觉 —— AMT 模型常把基音在 +12/+19/+24 半音处重复一遍。
        // 真实的八度叠奏力度相当；同起点且力度 ≤60% 的上方针音 = 幻觉 → 删。
        if (!isDrum) {
            const dead = new Set();
            for (let i = 0; i < kept.length; i++) {
                const a = kept[i];
                if (!a || dead.has(a))
                    continue;
                for (let j = i + 1; j < kept.length; j++) {
                    const c = kept[j];
                    if (!c)
                        continue;
                    if (c.tick - a.tick >= gapTicks)
                        break; // 已排序 —— 超出起点窗口
                    if (dead.has(c))
                        continue;
                    const d = c.pitch - a.pitch;
                    if (d === 12 || d === 19 || d === 24) {
                        if ((c.velocity ?? 100) <= 0.6 * (a.velocity ?? 100)) {
                            dead.add(c);
                            stats.removedGhost++;
                        }
                    }
                    else if (d === -12 || d === -19 || d === -24) {
                        // 同窗口反向 —— 锚点自身才是幻觉。
                        if ((a.velocity ?? 100) <= 0.6 * (c.velocity ?? 100)) {
                            dead.add(a);
                            stats.removedGhost++;
                            break; // a 已删 —— 停止以它配对
                        }
                    }
                }
            }
            if (dead.size > 0)
                kept = kept.filter((n) => !dead.has(n));
        }
        // Pass 4：力度离群值 —— 两层处理：
        //  a) 病理性弱音（<中位数 45%）抬到中位数（v2 行为，保留节奏）；
        //  b) 局部轮廓压缩（v3）：与前后 ±2 音的局部中位数偏差 >30% 的力度
        //     向局部中位数靠一半 —— 消除 AMT 力度的逐音抖动，还原自然的强弱走向。
        const vels = kept.map((n) => n.velocity ?? 100).sort((a, b) => a - b);
        let median = 100;
        if (vels.length > 0) {
            median = vels[Math.floor(vels.length / 2)] ?? 100;
            const floor = Math.max(24, Math.round(median * 0.45));
            for (const n of kept) {
                if ((n.velocity ?? 100) < floor) {
                    n.velocity = median;
                    stats.fixedVelocity++;
                    if (n.id)
                        stats.touched.push(n.id);
                }
            }
        }
        if (kept.length >= 5) {
            const seq = [...kept].sort((a, b) => a.tick - b.tick || a.pitch - b.pitch);
            const localMedian = (i) => {
                const w = seq.slice(Math.max(0, i - 2), Math.min(seq.length, i + 3))
                    .map((n) => n.velocity ?? 100)
                    .sort((a, b) => a - b);
                return w[Math.floor(w.length / 2)] ?? 100;
            };
            for (let i = 0; i < seq.length; i++) {
                const n = seq[i];
                const m = localMedian(i);
                const v = n.velocity ?? 100;
                if (Math.abs(v - m) > Math.max(8, m * 0.3)) {
                    n.velocity = Math.round((v + m) / 2); // 靠一半：保留音乐性起伏，只压离群抖动
                    stats.fixedVelocity++;
                    if (n.id)
                        stats.touched.push(n.id);
                }
            }
        }
        // Pass 5（仅旋律）：调外弱音 —— 半音经过音/蓝调音是长且强的；
        // 调外 + 弱（≤中位数一半）+ 短（≤200ms）= 和声幻觉。
        if (!isDrum && keySet && vels.length > 0) {
            const keyFloor = Math.round(median * 0.5);
            kept = kept.filter((n) => {
                if (keySet.has(((n.pitch % 12) + 12) % 12))
                    return true;
                if ((n.velocity ?? 100) <= keyFloor && n.duration <= msToTicks(200)) {
                    stats.removedKey++;
                    return false;
                }
                return true;
            });
        }
        kept.sort((a, b) => a.tick - b.tick || a.pitch - b.pitch);
        // Pass 8（v3）：拖尾截断 —— AMT 模型常把音符尾部拖得很长（悬挂音），
        // 换音源渲染时表现为“糊成一团”。三层处理：
        //  a) 封顶：非鼓轨不超过 2 小节、鼓轨不超过半小节（真实鼓不会延音整小节）；
        //  b) 旋律轨（重合率 ≤15%）收束为单声部：音符尾部截到下一音起点；
        //  c) 全轨兜底：任何音符不超过其起点 + 4 小节（防极端悬挂）。
        if (kept.length > 0 && durationTicks > 0) {
            const cap = isDrum ? Math.floor(barTicks / 2) : barTicks * 2;
            for (const n of kept) {
                if (n.duration > cap) {
                    n.duration = cap;
                    stats.trimmedHang++;
                    if (n.id)
                        stats.touched.push(n.id);
                }
            }
            if (!isDrum) {
                let overlapCount = 0;
                for (let i = 0; i + 1 < kept.length; i++) {
                    const cur = kept[i];
                    const next = kept[i + 1];
                    if (next.tick < cur.tick + cur.duration)
                        overlapCount++;
                }
                if (kept.length >= 4 && overlapCount / kept.length <= 0.15) {
                    for (let i = 0; i + 1 < kept.length; i++) {
                        const cur = kept[i];
                        const next = kept[i + 1];
                        const span = next.tick - cur.tick;
                        if (span >= minTicks && cur.duration > span) {
                            cur.duration = span;
                            stats.trimmedHang++;
                            if (cur.id)
                                stats.touched.push(cur.id);
                        }
                    }
                }
            }
            kept.sort((a, b) => a.tick - b.tick || a.pitch - b.pitch);
        }
        s.notes = kept;
    });
    // Pass 7：补漏小节。v2 规则（前两小节近乎相同 = 漏检证明）之上，v3 增加
    // 自相似周期检测：对周期 d ∈ {1,2,4,8} 计算全曲「bar[i] ≈ bar[i-d]」的平均
    // 相似度，用最强周期回填空档 —— 副歌/主歌的周期性重复不再要求紧邻小节相似。
    if (durationTicks > 0) {
        const totalBars = Math.ceil(durationTicks / barTicks);
        if (totalBars > 0) {
            const buckets = stems.map((s) => {
                const bars = Array.from({ length: totalBars }, () => []);
                for (const n of s.notes) {
                    const b = Math.min(totalBars - 1, Math.floor(n.tick / barTicks));
                    const list = bars[b];
                    list?.push(n);
                }
                return bars;
            });
            let firstBar = totalBars;
            let lastBar = -1;
            for (const bars of buckets) {
                bars.forEach((list, b) => {
                    if (list.length > 0) {
                        if (b < firstBar)
                            firstBar = b;
                        if (b > lastBar)
                            lastBar = b;
                    }
                });
            }
            if (lastBar >= firstBar) {
                const singleStem = stems.length === 1;
                stems.forEach((s, si) => {
                    const bars = buckets[si] ?? [];
                    // 跨轨证据：其他轨在该小节有声（单一轨跳过 —— 周期性门槛独自承担证明）。
                    const otherBar = new Array(totalBars).fill(false);
                    if (!singleStem) {
                        for (let b = firstBar; b <= lastBar; b++) {
                            otherBar[b] = buckets.some((ob, oj) => oj !== si && ((ob?.[b]?.length ?? 0) > 0));
                        }
                    }
                    // v3：自相似周期强度 R(d) = mean barSimilarity(bars[i], bars[i-d])。
                    const periodScore = (d) => {
                        let sum = 0;
                        let n = 0;
                        for (let i = firstBar + d; i <= lastBar; i++) {
                            const a = bars[i];
                            const b = bars[i - d];
                            if (a && b && a.length > 0 && b.length > 0) {
                                sum += barSimilarity(a, b);
                                n++;
                            }
                        }
                        return n >= 2 ? sum / n : 0;
                    };
                    const bestPeriod = (() => {
                        let bd = 0;
                        let br = 0;
                        for (const d of [1, 2, 4, 8]) {
                            const r = periodScore(d);
                            if (r > br) {
                                br = r;
                                bd = d;
                            }
                        }
                        return { d: bd, r: br };
                    })();
                    let b = firstBar;
                    while (b <= lastBar) {
                        if ((bars[b]?.length ?? 0) > 0 || (!singleStem && !otherBar[b])) {
                            b++;
                            continue;
                        }
                        let end = b;
                        while (end + 1 <= lastBar &&
                            (bars[end + 1]?.length ?? 0) === 0 &&
                            (singleStem || otherBar[end + 1] === true)) {
                            end++;
                        }
                        const runLen = end - b + 1;
                        const srcIdx = b - 1;
                        if (runLen <= 8) {
                            let filled = false;
                            // 规则一（v2）：前两小节近乎相同 → 平铺前一小节。
                            if (srcIdx >= 2) {
                                const src = bars[srcIdx] ?? [];
                                const srcPrev = bars[srcIdx - 1] ?? [];
                                const sim = barSimilarity(src, srcPrev);
                                const need = runLen > 1 ? 0.9 : singleStem ? 0.85 : 0.75;
                                if (src.length > 0 && sim >= need) {
                                    for (let k = 0; k < runLen; k++) {
                                        const offset = (b + k - srcIdx) * barTicks;
                                        for (const n of src) {
                                            const newId = makeId();
                                            s.notes.push({
                                                ...n,
                                                id: newId,
                                                tick: Math.max(0, Math.min(durationTicks - 1, n.tick + offset)),
                                            });
                                            stats.touched.push(newId);
                                            stats.filledNotes++;
                                        }
                                    }
                                    filled = true;
                                }
                            }
                            // 规则二（v3）：自相似周期回填 —— 副歌整段重复但紧邻小节不相似时，
                            // 用全曲最强周期 d 的对应小节补（证据门槛更高：0.82 / 单轨 0.88）。
                            if (!filled && bestPeriod.d > 0) {
                                const need = singleStem ? 0.88 : 0.82;
                                if (bestPeriod.r >= need) {
                                    for (let k = 0; k < runLen; k++) {
                                        const srcIdx2 = b + k - bestPeriod.d;
                                        if (srcIdx2 < firstBar)
                                            continue;
                                        const src = bars[srcIdx2] ?? [];
                                        if (src.length === 0)
                                            continue;
                                        const offset = (b + k - srcIdx2) * barTicks;
                                        for (const n of src) {
                                            const newId = makeId();
                                            s.notes.push({
                                                ...n,
                                                id: newId,
                                                tick: Math.max(0, Math.min(durationTicks - 1, n.tick + offset)),
                                            });
                                            stats.touched.push(newId);
                                            stats.filledNotes++;
                                        }
                                    }
                                }
                            }
                        }
                        b = end + 1;
                    }
                    s.notes.sort((x, y) => x.tick - y.tick || x.pitch - y.pitch);
                });
            }
        }
    }
    // Pass 9（v3）：轻量节拍量化 —— AMT 起点常在网格 ±几十 tick 漂移，换音源
    // 渲染后节奏“不稳”。起点距 1/32 网格 ≤ 1/20 拍（480ppq = 24 tick）时吸附。
    // 只动“已经几乎在网格上”的音符 —— 真正的切分/摇摆不受影响。
    const gridTicks = Math.max(1, Math.round(safePpq / 8));
    const snapTol = Math.max(4, Math.round(safePpq / 20));
    stems.forEach((s) => {
        for (const n of s.notes) {
            const r = ((n.tick % gridTicks) + gridTicks) % gridTicks;
            const nearest = r < gridTicks / 2 ? n.tick - r : n.tick + (gridTicks - r);
            if (nearest >= 0 && nearest !== n.tick && Math.abs(nearest - n.tick) <= snapTol) {
                n.tick = nearest;
                stats.snapped++;
                if (n.id)
                    stats.touched.push(n.id);
            }
        }
        s.notes.sort((x, y) => x.tick - y.tick || x.pitch - y.pitch);
    });
    return stats;
}
