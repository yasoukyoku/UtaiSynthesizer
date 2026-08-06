# -*- coding: utf-8 -*-
"""S111 共享底座 —— 在 S110 的 `c24_lib` 之上只加两件事:

  ① **位点级**的枚举(哪个 /j/、前面是哪个辅音、**今天实际切在哪**)
  ② de.tsv 键 ↔ de.wiktionary 标题的桥

⛔ 不复制 `c24_lib` 的任何一行 —— 它**解析**生产常量(`DE_ONSET_KEEP` /
   `DE_LITERAL_J_SPELLING` / `single_onset_forbidden` / 七个接缝常量),
   所以我这一轮改了 g2p.rs 的表,它会自己跟上。抄一份 = S106/S110 两次踩过的漂移。

★ 「没被守卫拦下」**不等于**「切错了」—— `ŋ j` 这种压根不在 onset 集里的簇,
  最大 onset 自己就切在 /j/ 前。所以每个位点必须问的是【今天 C 落在 onset 还是 coda】,
  这是 S111 普查相对 S110 的口径修正。
"""
import sys
from pathlib import Path

import c24_lib as L  # noqa: E402

LETTERS = "abcdefghijklmnopqrstuvwxyzäöüß"

# 「字母 lenis / 音素 fortis」= 德语音节末清化(Auslautverhärtung)的书面痕迹。
# ★前提已在 S111 开工时用正负对照实测过:元音间 lenis 全保留(haben=b · reise=z · loewe=v),
#  音节末 lenis 全清化(lieb=p · tag=k · halb=p)⇒ 清化是**位置性**的,不是全本约定。
# ⛔ ⟨v⟩ 不在表里:德语 ⟨v⟩ 本来就读 /f/(nerven = n ɛ ʁ f ə n),那不是清化。
# ⛔ ⟨s⟩→/s/ 也不在:德语 ⟨s⟩ 在多数位置本来就是 /s/,无分辨力。
LENIS_TO_FORTIS = {"b": "p", "d": "t", "g": "k", "w": "f"}


def de():
    return L.de()


def sites(key, phones):
    """全部【元音间】C+/j/ 位点 → [(j_idx, 前面的辅音音素)]。与生产扫描口径逐行相同。"""
    if "j" not in phones:
        return []
    nuc = [i for i, p in enumerate(phones) if L.is_vowel(p)]
    out = []
    for a, b in zip(nuc, nuc[1:]):
        if b >= a + 3 and phones[b - 1] == "j":
            out.append((b - 1, phones[b - 2]))
    return out


def cut_of(d, key, phones, extra_glide=(), extra_pairs=()):
    """今天(或加上假想守卫之后)的切点集合。返回 set(每个音节的起点)。

    `extra_pairs` = 往 `DE_LITERAL_J_SPELLING` 里临时加的 (音素, 字母二合) 行,
    用来量「补上这一族会移动多少词型」——**不改生产源码就能量**。
    """
    any_, st = d.seams(key, phones)
    g = literal_j_with(d, key, phones, extra_pairs)
    g = sorted(set(g) | set(extra_glide))
    return d.syllabify(phones, seams=(any_, st, g))


def literal_j_with(d, key, phones, extra_pairs=()):
    """`de_literal_j_blocks` 的逐行等价物,但允许临时扩表。"""
    if key not in d.prim or "j" not in phones:
        return []
    if not any(p == phones for p in d.pronunciations(key)):
        return []
    # ★ S112:生产端 `de_literal_j_blocks` 在这里还有一道**例外键**早退,工装此前漏了它
    #   ⇒ 在 HEAD 面上会把 `orjol`/`reykjavik` 继续算成被拦(它们其实早就被放开了)。
    #   与上面那条同理:钉住旧 rev 时该常量不存在、解析返回空集 ⇒ 对历史口径 no-op。
    if key in L.DE_CJ_ONSET_KEYS:
        return []
    # ★ S112:生产端从此有**两张**表喂同一个计数器(`DE_LITERAL_J_SPELLING` 的字面 ⟨j⟩
    #   + `DE_ROMANCE_GN_SPELLING` 的罗曼 ⟨gn⟩)。不 chain 第二张,这个工装就会在 HEAD 面上
    #   悄悄少算一族 —— 而它在**钉住的旧 rev 上是 no-op**(那个版本没有这个常量,解析返回 []),
    #   所以加上去不会动任何历史口径。
    pairs = list(L.DE_LITERAL_J_SPELLING) + list(L.DE_ROMANCE_GN_SPELLING) + list(extra_pairs)
    nuc = [i for i, p in enumerate(phones) if L.is_vowel(p)]
    by_c = {}
    for a, b in zip(nuc, nuc[1:]):
        if b >= a + 3 and phones[b - 1] == "j":
            by_c.setdefault(phones[b - 2], []).append(b - 1)
    out = []
    for c, pos in by_c.items():
        lit = sum(key.count(bg) for p, bg in pairs if p == c)
        if lit >= len(pos):
            out += pos
    return sorted(out)


def site_in_onset(syl_bounds, j_idx):
    """C+/j/ 里的 C(在 j_idx-1)与 /j/ 是否同属一个音节 ⇒ True = C 在 onset。"""
    for s, e in syl_bounds:
        if s <= j_idx < e:
            return s <= j_idx - 1
    raise AssertionError("j_idx 不在任何音节里")


def bigrams_in(key):
    return sorted({x + "j" for x in LETTERS if x + "j" in key})


def wikt_titles(key):
    """de.tsv 键 → de.wiktionary 该试的标题。德语名词首字母大写,所以两个都要试。"""
    out = [key]
    if key[:1].isalpha():
        out.append(key[0].upper() + key[1:])
    # de.tsv 把 ⟨ä ö ü ß⟩ 保留,但也有 `grosswildjagd` 这种 ss 写法 ⇒ 再试一个 ß 还原
    if "ss" in key:
        alt = key.replace("ss", "ß")
        out.append(alt[0].upper() + alt[1:] if alt[:1].isalpha() else alt)
    return list(dict.fromkeys(out))


# 直接透出,免得下游再 import 两层
is_vowel = L.is_vowel
DE_LITERAL_J_SPELLING = L.DE_LITERAL_J_SPELLING
DE_KEEP = L.DE_KEEP
