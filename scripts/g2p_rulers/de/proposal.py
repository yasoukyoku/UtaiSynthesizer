# -*- coding: utf-8 -*-
"""S111 —— **完整方案的模型实测**。先在这里量准,再写 Rust。

三刀(各自独立,可以分开取舍):
  ① §C24c-fortis:表按【清化痕迹】配对 —— 加 ('p','bj') ('t','dj')
  ② §C24d-lenis :**删掉六行浊阻塞音**('z','sj') ('d','dj') ('b','bj') ('ɡ','gj') ('v','wj') ('v','vj')
                  —— 德语音节末只能是清音 ⇒ 浊音在 /j/ 前**不可能**是 coda,那几行断言了不可能的事
  ③ §C24b      :`l j` 进 KEEP,但**只在拼写显示 ⟨(l)li⟩+元音时放行**(fail-closed:
                  未知拼写保持今天的 coda),外加 william 家族的 curated 例外

⚠ 每一刀都要回答同一个问题:**被它碰到的每个词,改之前是对是错**(S110 血训)。
"""
import json
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
import s111_lib as S  # noqa: E402
import measure as M  # noqa: E402

d = S.de()
VOWELS = "aeiouyäöüáàâéèêíìîóòôúùû"
# ⟨li⟩/⟨lli⟩ + 元音 = ⟨i⟩ 派生的滑音。
# ★ 前面**必须**再有一个元音字母:位点按定义是【元音间】的,所以授权模式也得是。
#   第一版没要求,`liouville` 就拿**词首**的 ⟨lio⟩ 去授权了**词尾**的那个 /l j/ 位点 ——
#   词级谓词错配的活样本(S88:把 A 当 B 的代理必须写下为什么,而这里代理不成立)。
RE_LICENSE = re.compile(r"[" + VOWELS + r"]ll?i[" + VOWELS + "]")
LJ_EXCEPT = ("willia",)                              # curated:英语 ⟨illia⟩ 名,de.wikt = /ˈvɪl.jam/

ADD_FORTIS = [("p", "bj"), ("t", "dj")]
DROP_LENIS = [("z", "sj"), ("d", "dj"), ("b", "bj"), ("ɡ", "gj"), ("v", "wj"), ("v", "vj")]


def table(add=(), drop=()):
    t = [r for r in S.DE_LITERAL_J_SPELLING if r not in drop]
    return t + list(add)


def literal_j(key, phones, pairs):
    """`de_literal_j_blocks` 的等价物,吃任意一张表。"""
    if key not in d.prim or "j" not in phones:
        return []
    if not any(p == phones for p in d.pronunciations(key)):
        return []
    nuc = [i for i, p in enumerate(phones) if S.is_vowel(p)]
    by_c = defaultdict(list)
    for a, b in zip(nuc, nuc[1:]):
        if b >= a + 3 and phones[b - 1] == "j":
            by_c[phones[b - 2]].append(b - 1)
    out = []
    for c, pos in by_c.items():
        lit = sum(key.count(bg) for p, bg in pairs if p == c)
        if lit >= len(pos):
            out += pos
    return sorted(out)


def lj_blocks(key, phones):
    """§C24b:/l j/ 位点里**不许**当 onset 的那些。fail-closed —— 默认全拦,只在拼写授权时放行。"""
    sites = [j for j, c in S.sites(key, phones) if c == "l"]
    if not sites:
        return []
    if any(x in key for x in LJ_EXCEPT):
        return sites
    n_lic = len(RE_LICENSE.findall(key))
    if n_lic >= len(sites):
        return []                 # 全部放行
    return sites                  # 0 个 or 不够 ⇒ 全拦(= 保持今天),不猜


def cut(key, phones, mode):
    any_, st = d.seams(key, phones)
    if mode == "head":
        return d.syllabify(phones, seams=(any_, st, literal_j(key, phones, S.DE_LITERAL_J_SPELLING)))
    pairs = table(add=ADD_FORTIS if "c" in mode else (), drop=DROP_LENIS if "d" in mode else ())
    g = literal_j(key, phones, pairs)
    if "b" in mode:
        g = sorted(set(g) | set(lj_blocks(key, phones)))
        return d.syllabify(phones, seams=(any_, st, g), extra_onsets=["l j"])
    return d.syllabify(phones, seams=(any_, st, g))


def audit(mode, title):
    print(f"\n{'=' * 106}\n{title}\n{'=' * 106}")
    fams = defaultdict(Counter)
    ex = defaultdict(lambda: defaultdict(list))
    for k, ph in d.prim.items():
        if "j" not in ph:
            continue
        a, b = cut(k, ph, "head"), cut(k, ph, mode)
        if a == b:
            continue
        for j, c in S.sites(k, ph):
            sa = "ONSET" if S.site_in_onset(a, j) else "CODA"
            sb = "ONSET" if S.site_in_onset(b, j) else "CODA"
            if sa == sb:
                continue
            fam = f"/{c}/ {sa}→{sb}"
            v, src, _sp, ipa = M.ask(k, c)
            key2 = "✅改对" if v == sb else ("⛔改坏" if v == sa else src)
            fams[fam][key2] += 1
            if len(ex[fam][key2]) < 5:
                ex[fam][key2].append(f"{k} {ipa}")
    g = b_ = 0
    for fam, c in sorted(fams.items(), key=lambda x: -sum(x[1].values())):
        g += c["✅改对"]
        b_ += c["⛔改坏"]
        print(f"   {fam:22s} 共{sum(c.values()):5d}  ✅{c['✅改对']:4d}  ⛔{c['⛔改坏']:4d}  "
              f"沉默{c['沉默']:4d}  无页面{c['无页面']:4d}")
        for bucket in ("⛔改坏", "✅改对"):
            for line in ex[fam][bucket][:5]:
                print(f"        {bucket}  {line}")
    print(f"   ---- 合计 ✅{g} : ⛔{b_}")
    return g, b_


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cdb"
    if "c" in which:
        audit("c", "① §C24c-fortis:加 ('p','bj') ('t','dj')")
    if "d" in which:
        audit("d", "② §C24d-lenis:删掉六行浊阻塞音守卫")
    if "b" in which:
        audit("b", "③ §C24b:`l j` 进 KEEP + 只在 ⟨(l)li⟩+V 时放行 + william 例外")
    if len(which) > 1:
        audit(which, f"★ 三刀合起来({which})")
