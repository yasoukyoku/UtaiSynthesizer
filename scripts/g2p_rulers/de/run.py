# -*- coding: utf-8 -*-
"""S111 —— 三个人群的实测驱动。用法: py run.py A|B|C"""

from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
import s111_lib as S  # noqa: E402
import measure as M  # noqa: E402
from targets import C24C_ROWS  # noqa: E402

d = S.de()
VOWELS = "aeiouyäöüáàâéèêíìîóòôúùû"


def side(bounds, j):
    return "ONSET" if S.site_in_onset(bounds, j) else "CODA"


def cuts(key):
    """(pre-S110, HEAD, +C24c, +lj) 四种切法。"""
    ph = d.prim[key]
    any_, st = d.seams(key, ph)
    g = S.literal_j_with(d, key, ph)
    gc = S.literal_j_with(d, key, ph, extra_pairs=C24C_ROWS)
    return {
        "pre": d.syllabify(ph, seams=(any_, st, ()), drop_onsets=["n j"]),
        "head": d.syllabify(ph, seams=(any_, st, g)),
        "c24c": d.syllabify(ph, seams=(any_, st, gc)),
        "lj": d.syllabify(ph, seams=(any_, st, g), extra_onsets=["l j"]),
    }


def fam_spelling(key, cons):
    """§C24b 用:S110 那个**按拼写机械分族**的分类器。现在它自己也要被尺子检验。"""
    if cons != "l":
        return "非 l"
    import re
    if re.search(r"ll?i[" + VOWELS + "]", key):
        return "⟨lli⟩+V"
    if re.search(r"ill[" + VOWELS.replace("i", "") + "]", key) or key.endswith("ill"):
        return "⟨ill⟩罗曼"
    if re.search(r"ll?y", key):
        return "⟨ly⟩"
    return "其它-l"


def fam_c24c(key, cons):
    for _p, bg in C24C_ROWS:
        if bg in key and _p == cons:
            return f"⟨{bg}⟩"
    return "其它"


def fam_a(key, cons):
    bgs = [b for b in S.bigrams_in(key)]
    lit = [b for p, b in S.DE_LITERAL_J_SPELLING if p == cons and b in key]
    if lit:
        return f"字面⟨{lit[0]}⟩"
    return f"/{cons}/ 无字面"


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "A").upper()
    pops = json.load(open(_HERE / "populations.json", encoding="utf-8"))

    if which == "A":
        keys = open(_HERE / "s110_changed_keys.txt", encoding="utf-8").read().split()
        items = []
        for k in keys:
            c = cuts(k)
            sites = [(j, cn) for j, cn in S.sites(k, d.prim[k]) if side(c["pre"], j) != side(c["head"], j)]
            if sites:
                items.append((k, sites))
        M.report("A", "★ 回头审 S110 自己的落地(pre-S110 → HEAD,431 词型)—— 它改对了吗?",
                 items, lambda k, j: side(cuts(k)["pre"], j), lambda k, j: side(cuts(k)["head"], j), fam_a)

    elif which == "B":
        items = [(k, [tuple(x) for x in v]) for k, v in pops["B"].items()]
        M.report("B", "★ §C24c —— 补上「字母 lenis / 音素 fortis」那几行(HEAD → +C24c,85 词型)",
                 items, lambda k, j: side(cuts(k)["head"], j), lambda k, j: side(cuts(k)["c24c"], j), fam_c24c)

    elif which == "C":
        items = [(k, [tuple(x) for x in v]) for k, v in pops["C"].items()]
        M.report("C", "★ §C24b —— 把 `l j` 放进 onset 集(HEAD → +lj,285 词型)",
                 items, lambda k, j: side(cuts(k)["head"], j), lambda k, j: side(cuts(k)["lj"], j), fam_spelling)


if __name__ == "__main__":
    main()
