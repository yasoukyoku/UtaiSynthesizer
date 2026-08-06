# -*- coding: utf-8 -*-
"""S112 —— 把**所有还没判**的德语滑音键枚举成一份【以后能一条一条找回来】的清单。

用户 2026-08-06 的要求(原话要点):「这些个真没有任何办法的词一定要仔细记好……
而且大概率这个玩意被反馈上来也要等很久……这几个在 wiktionary 上都没有就说明它们本身
也就很偏僻,所以可能得等很久之后才能收到回复,**到时候千万不要找不着/乱了就行**」。

⇒ 每一条都要带四样东西,少一样就等于没记:
   ① 键 ② **今天唱成什么**(实际切法)③ **另一种可能是什么**(翻过来的切法)
   ④ **要改它得动哪个旋钮**(精确到常量名)
   —— 这样收到「XX 这个词唱错了」的反馈时,能直接 grep 到这一行,而不是回头重做全量分析。
"""

from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
import json
import os
import re
import sys
from collections import defaultdict

os.environ["C24_G2P_REV"] = ""
sys.stdout.reconfigure(encoding="utf-8")

import c24_lib as L      # noqa: E402
import s111_lib as S     # noqa: E402
import ruler             # noqa: E402
import wikt              # noqa: E402
import enwikt            # noqa: E402

DE = json.load(open(_HERE / "wikt_cache.json", encoding="utf-8"))
EN = json.load(open(_HERE / "enwikt_cache.json", encoding="utf-8"))
d = S.de()
V = "aeiouyäöüáàâéèêíìîóòôúùû"
T = list(L.DE_LITERAL_J_SPELLING) + list(L.DE_ROMANCE_GN_SPELLING)
PHONE_ALIAS = {"tʰ": "t", "kʰ": "k", "pʰ": "p", "ts": "t͡s", "ɡ": "ɡ"}


def lic_count(key):
    ch, n, i = list(key), 0, 0
    while i + 3 < len(ch):
        if ch[i] in V and ch[i + 1] == "l":
            j = i + 2
            if ch[j] == "l":
                j += 1
            if j + 1 < len(ch) and ch[j] == "i" and ch[j + 1] in V:
                n += 1
                i = j + 1
                continue
        i += 1
    return n


def glide(key, ph):
    out = set()
    if key not in L.DE_CJ_ONSET_KEYS and any(p == ph for p in d.pronunciations(key)):
        nuc = [i for i, p in enumerate(ph) if L.is_vowel(p)]
        by_c = defaultdict(list)
        for a, b in zip(nuc, nuc[1:]):
            if b >= a + 3 and ph[b - 1] == "j":
                by_c[ph[b - 2]].append(b - 1)
        for c, pos in by_c.items():
            if sum(key.count(bg) for p, bg in T if p == c) >= len(pos):
                out |= set(pos)
    sites = [j for j, c in S.sites(key, ph) if c == "l"]
    if sites and not ((not any(s in key for s in L.DE_LJ_CODA_SPELLINGS))
                      and lic_count(key) >= len(sites)):
        out |= set(sites)
    return sorted(out)


def syl(key, extra_block=()):
    ph = d.prim[key]
    any_, st = d.seams(key, ph)
    g = sorted(set(glide(key, ph)) | set(extra_block))
    return " | ".join(" ".join(ph[s:e]) for s, e in d.syllabify(ph, seams=(any_, st, g)))


def syl_without(key, drop):
    ph = d.prim[key]
    any_, st = d.seams(key, ph)
    g = sorted(set(glide(key, ph)) - set(drop))
    return " | ".join(" ".join(ph[s:e]) for s, e in d.syllabify(ph, seams=(any_, st, g)))


def verdict(key, cons):
    want = PHONE_ALIAS.get(cons, cons)
    for t in S.wikt_titles(key):
        for src, lst in (("de", wikt.ipas(DE.get(t))), ("en", enwikt.ipas(EN.get(t), raw=True))):
            if not lst:
                continue
            v, sp, per = ruler.source_verdict(lst, want)
            if v:
                return v
    return None


groups = {}

# ── ① §C24b 的代价:授权谓词把它们放成 onset,而尺子沉默 ────────────────────────────────────
pop = json.load(open(_HERE / "p4_population.json", encoding="utf-8"))
res8 = json.load(open(_HERE / "p8_result.json", encoding="utf-8"))
still = sorted(set(res8.get("None/—", [])) | set(res8.get("ONSET/弱", [])))
rows = []
for k in still:
    ph = d.prim[k]
    sites = [j for j, c in S.sites(k, ph) if c == "l"]
    rows.append((k, syl(k), syl(k, extra_block=sites)))
groups["A. §C24b 授权放行、尺子沉默(今天=ONSET)"] = (
    rows, "旋钮 = `DE_LJ_CODA_SPELLINGS` 加这个键(照 `gilliam` 的形状),它就翻回 CODA")

# ── ② ⟨ny⟩:`n j` 的第三个子族,两个方向都零证据 ────────────────────────────────────────────
rows = []
for k, ph in d.prim.items():
    if "ny" not in k or "gn" in k:
        continue
    sites = [j for j, c in S.sites(k, ph) if c == "n"]
    if not sites or verdict(k, "n"):
        continue
    rows.append((k, syl(k), syl(k, extra_block=sites)))
groups["B. ⟨ny⟩ —— `n j` 的第三个子族,零证据(今天=ONSET)"] = (
    sorted(rows), "旋钮 = 给 `DE_ROMANCE_GN_SPELLING` 加一行 `(\"n\",\"ny\")`(照 ⟨gn⟩ 的形状)")

# ── ③ ⟨lli⟩ 前导元音子句拦下的 ───────────────────────────────────────────────────────────────
rows = []
for k in pop["leadV"]:
    ph = d.prim[k]
    sites = [j for j, c in S.sites(k, ph) if c == "l"]
    rows.append((k, syl(k), syl_without(k, sites)))
groups["C. ⟨lli⟩ 授权的前导元音子句拦下的(今天=CODA)"] = (
    rows, "旋钮 = `de_lj_license_count` 的前导元音字母要求;⚠ 拿掉它 `liouville` 会拿词首授权词尾")

# ── ④ 上游行本身坏掉的(⇒ §C22/§G10,**不许在 G2P 侧代偿**)────────────────────────────────
BROKEN = {
    "designs": "德语说 [dɪˈzaɪ̯ns],根本没有 /j/;上游写成 `d ɪ z aj n j uː s`",
    "industriedesigns": "同上",
    "metallionen": "Metall+Ionen,而 de.tsv 的 `ionen` 存成 `j oː n ə n`(少一个音节,§C22⑤)",
    "sowjet": "词首 ⟨S⟩ 在元音前德语是 /z/,而我们写 `s`;⟨wj⟩ 边界也在唇音之前(§C24c 明写弃权)",
}
rows = [(k, syl(k) if k in d.prim else "<不是键>", why) for k, why in BROKEN.items()]
groups["D. ⛔上游行本身坏 —— 不许在 G2P 侧代偿,归 §C22/§G10 重训桶"] = (
    rows, "⚠ 第三列是**病因**不是替代切法:这些词两种切法都错,改 onset/coda 都救不了")

out = {}
print("=" * 108)
for name, (rows, knob) in groups.items():
    print(f"\n★ {name}  [{len(rows)} 个]\n   旋钮:{knob}")
    for r in rows:
        print(f"   {r[0]:26s} 今天: {r[1]}")
        if len(r) > 2 and r[2] != r[1]:
            print(f"   {'':26s} 另一种: {r[2]}")
    out[name] = {"knob": knob, "rows": [list(r) for r in rows]}

# ── ⑤ 血径之外、仍然沉默的 /l j/ 键(只报数与再生成方法)────────────────────────────────────
n_ext = 0
for k, ph in d.prim.items():
    sites = [j for j, c in S.sites(k, ph) if c == "l"]
    if sites and not verdict(k, "l"):
        n_ext += 1
print(f"\n★ 另:全词典有 /l j/ 位点而尺子沉默的键共 {n_ext} 个(含上面 A 组)。"
      f"再生成 = 本脚本;不逐个列,因为它们绝大多数**今天与 S110 之前一致**、没有被任何一刀动过。")
out["_meta"] = {"silent_lj_keys_total": n_ext,
                "regen": "py -3.10 scripts/g2p_rulers/de/open_costs_manifest.py(仓库根下跑)"}
json.dump(out, open(_HERE / "open_costs_manifest.json",
                    "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n[落盘] open_costs_manifest.json")
