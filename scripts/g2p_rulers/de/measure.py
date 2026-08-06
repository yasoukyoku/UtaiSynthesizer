# -*- coding: utf-8 -*-
"""S111 —— 拿**过了对照组的**尺子去量三个人群。

⚠ 三条纪律,写在最前面:
  ① **净方向的分母是「改之前是对是错」**,不是「改之后修了多少」(S110 血训:`l j` 就栽在这)。
     所以每个词都要同时记 `今天切成什么` 与 `尺子说该切成什么`。
  ② **沉默 ≠ 支持**。尺子没给答案的词一律进 `无判决`,不许并进任何一侧。
  ③ **家族传递是推论,不是观测** —— 分开报,并标明。
"""

from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
import json
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
import s111_lib as S  # noqa: E402
import wikt  # noqa: E402
import enwikt  # noqa: E402
import ruler  # noqa: E402

DE_CACHE = json.load(open(_HERE / "wikt_cache.json", encoding="utf-8"))
EN_CACHE = json.load(open(_HERE / "enwikt_cache.json", encoding="utf-8"))

# de.tsv 音素 → 维基可能写的符号(送气/塞擦的写法差异)
PHONE_ALIAS = {"tʰ": "t", "kʰ": "k", "pʰ": "p", "ts": "t͡s", "ɡ": "ɡ"}


def ask(key, cons):
    """尺子对这个词的这个辅音说什么。返回 (判决, 源, 是否有变体离散, **做出判决的那一条转写**)。

    ⚠ 第一版返回的是整个 IPA 列表,而例子行只印 `ipa[:1]` ⇒ 印出来的往往**不是**定案的那条
      (`adjutant` 印的是 `at.juˈtant` 却报 ONSET,看着像尺子发疯,其实定案的是列表里的第二条)。
      **汇报层骗人**是 S110 刚栽过一次的形状,当场修掉。
    """
    want = PHONE_ALIAS.get(cons, cons)
    for t in S.wikt_titles(key):
        for src, lst in (("de", wikt.ipas(DE_CACHE.get(t))),
                         ("en", enwikt.ipas(EN_CACHE.get(t), raw=True))):
            if not lst:
                continue
            v, spread, per = ruler.source_verdict(lst, want)
            if v:
                decided = next((s for s, vv, _ in per if vv == v), lst[0])
                return v, src, spread, [decided]
    # 有页面但尺子沉默 / 根本没页面 —— 分开记
    has_page = any(wikt.ipas(DE_CACHE.get(t)) or enwikt.ipas(EN_CACHE.get(t), raw=True)
                   for t in S.wikt_titles(key))
    return (None, "沉默" if has_page else "无页面", False, [])


def report(name, title, items, cur_side_fn, new_side_fn, fam_fn):
    """items: [(key, [(j_idx, cons)])]。cur/new_side_fn(key, j) -> 'ONSET'|'CODA'。"""
    print(f"\n{'=' * 108}\n{title}\n{'=' * 108}")
    per_fam = defaultdict(lambda: Counter())
    examples = defaultdict(lambda: defaultdict(list))
    for key, sites in items:
        for j_idx, cons in sites:
            fam = fam_fn(key, cons)
            cur = cur_side_fn(key, j_idx)
            new = new_side_fn(key, j_idx)
            v, src, spread, ipa = ask(key, cons)
            if v is None:
                per_fam[fam][src] += 1
                continue
            if cur == new:
                per_fam[fam]["未移动"] += 1
                continue
            if v == new:
                per_fam[fam]["✅改对了"] += 1
                bucket = "✅改对了"
            elif v == cur:
                per_fam[fam]["⛔改坏了"] += 1
                bucket = "⛔改坏了"
            else:
                per_fam[fam]["?"] += 1
                bucket = "?"
            if spread:
                per_fam[fam]["(其中有变体离散)"] += 1
            if len(examples[fam][bucket]) < 6:
                flag = " ⚠变体离散(同源两条转写打架,这个判决是软的)" if spread else ""
                examples[fam][bucket].append(f"{key}[{cons}] {cur}→{new} 尺子={v} {ipa[:1]}{flag}")
    tot = Counter()
    # ⚠★ S112 修:`离散` 这一列此前**算了但从来不打印** —— 而 `ruler.source_verdict` 的 docstring
    #   明写「离散度单独报出来,**不许悄悄吞掉**」。汇报层把它吞了 5 个 session。
    #   代价是具体的:§C24-res 重算后唯一剩下的那条 ⛔(`adjektiv`)**正是**一个离散例
    #   —— en.wikt 两条转写 `ˈa.djɛkˌtiːf`(ONSET)与 `ˈat.jɛk-`(CODA)自相矛盾,
    #   而「第一条开口者胜」把它读成了一个干净的 ONSET。不打印这一列,就看不出那个判决是软的。
    #   (⇒ 与 S106/S107/S109/S110 那四次「汇报层自己就是核对器」同族,这是第五次。)
    print(f"{'族':16s}{'✅改对':>7s}{'⛔改坏':>7s}{'沉默':>6s}{'无页面':>7s}{'未移动':>7s}{'离散':>6s}   净方向")
    for fam, c in sorted(per_fam.items(), key=lambda x: -sum(x[1].values())):
        good, bad = c["✅改对了"], c["⛔改坏了"]
        net = "—" if not (good or bad) else (f"{good / bad:.2f}:1" if bad else f"{good}:0 ✅")
        if bad and good < bad:
            net += "  ⛔净负"
        print(f"{fam:16s}{good:7d}{bad:7d}{c['沉默']:6d}{c['无页面']:7d}{c['未移动']:7d}"
              f"{c['(其中有变体离散)']:6d}   {net}")
        tot.update(c)
    good, bad = tot["✅改对了"], tot["⛔改坏了"]
    print(f"{'合计':16s}{good:7d}{bad:7d}{tot['沉默']:6d}{tot['无页面']:7d}{tot['未移动']:7d}"
          f"{tot['(其中有变体离散)']:6d}")
    for fam in sorted(examples):
        for bucket in ("⛔改坏了", "✅改对了"):
            for line in examples[fam][bucket]:
                print(f"    {fam:14s} {bucket}  {line}")
    return per_fam
