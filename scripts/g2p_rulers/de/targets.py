# -*- coding: utf-8 -*-
"""S111 —— 三个目标人群的枚举 + 维基覆盖率**先测**(go/no-go)。

⚠ 顺序是有意的:**先量覆盖率再谈结论**。如果尺子对某个人群结构性沉默,
   那是用户定的**第②类**(覆盖率事实,不是判决)⇒ 只能缩使用范围,不能假装有答案。

三个人群:
  A = **S110 守卫的血径**(`DE_LITERAL_J_SPELLING` 今天拦下的词型)—— 回头审上一轮
  B = **§C24c**(字母 lenis / 音素 fortis,今天切成 onset)
  C = **§C24b**(把 `l j` 放进 onset 集会改变切法的词型)
"""
import json
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
import s111_lib as S  # noqa: E402
import wikt  # noqa: E402
import enwikt  # noqa: E402

# §C24c 要补的行(**候选**,还没判);⟨s⟩ 不在里面(无分辨力),⟨v⟩ 不在里面(⟨v⟩ 本来就读 /f/)
C24C_ROWS = [("p", "bj"), ("t", "dj"), ("f", "wj"), ("k", "cj"), ("k", "xj")]


def populations():
    d = S.de()
    A, B, C = {}, {}, {}
    for k, ph in d.prim.items():
        ss = S.sites(k, ph)
        if not ss:
            continue
        base = S.cut_of(d, k, ph)
        blocked = set(S.literal_j_with(d, k, ph))

        # A:守卫**真正改变了切法**的词型。
        # ⚠ 第一版写成「有位点被拦下」= 397,而 S110 记的是 116 —— 差别在于:
        #   一个位点可以被拦下却不改变任何东西(那个簇本来就不是 onset)。
        #   「被拦」≠「被改」,这与本轮普查的口径修正是同一条。
        if blocked:
            any_, st = d.seams(k, ph)
            unguarded = d.syllabify(ph, seams=(any_, st, ()))
            if unguarded != base:
                A[k] = [(j, c) for j, c in ss if S.site_in_onset(unguarded, j) and not S.site_in_onset(base, j)]

        # B:补上 C24c 那几行之后会移动的词型
        b2 = S.cut_of(d, k, ph, extra_pairs=C24C_ROWS)
        if b2 != base:
            moved = [(j, c) for j, c in ss if S.site_in_onset(base, j) and not S.site_in_onset(b2, j)]
            if moved:
                B[k] = moved

        # C:把 `l j` 放进 onset 集之后会移动的词型
        any_, st = d.seams(k, ph)
        g = S.literal_j_with(d, k, ph)
        before = d.syllabify(ph, seams=(any_, st, g))
        after = d.syllabify(ph, seams=(any_, st, g), extra_onsets=["l j"])
        if before != after:
            moved = [(j, c) for j, c in ss if c == "l" and not S.site_in_onset(before, j) and S.site_in_onset(after, j)]
            if moved:
                C[k] = moved
    return A, B, C


def main():
    A, B, C = populations()
    print(f"A(S110 守卫血径)      = {len(A)} 个词型")
    print(f"B(§C24c 会移动的)     = {len(B)} 个词型")
    print(f"C(§C24b 会移动的)     = {len(C)} 个词型")

    allkeys = sorted(set(A) | set(B) | set(C))
    titles = []
    for k in allkeys:
        titles += S.wikt_titles(k)
    titles = list(dict.fromkeys(titles))
    print(f"\n要取的维基标题 = {len(titles)}(两个源各一遍)")
    wikt.fetch(titles)
    enwikt.fetch(titles)

    # 覆盖率:一个键只要**任一**标题候选在任一源上有德语 IPA,就算有覆盖
    de_cache = json.load(open("wikt_cache.json", encoding="utf-8"))
    en_cache = json.load(open("enwikt_cache.json", encoding="utf-8"))

    def covered(k):
        for t in S.wikt_titles(k):
            if wikt.ipas(de_cache.get(t)):
                return "de"
            if enwikt.ipas(en_cache.get(t), raw=True):
                return "en"
        return None

    rows = {}
    for name, pop in (("A", A), ("B", B), ("C", C)):
        cnt = Counter()
        for k in pop:
            c = covered(k)
            cnt[c or "无"] += 1
            rows.setdefault(k, {})[name] = True
        tot = sum(cnt.values())
        hit = tot - cnt["无"]
        print(f"\n人群 {name}: {tot} 词型 —— 有 IPA 覆盖 {hit} ({hit / tot:.0%})   {dict(cnt)}")

    json.dump({"A": {k: v for k, v in A.items()}, "B": B, "C": C},
              open("populations.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[落盘] populations.json")


if __name__ == "__main__":
    main()
