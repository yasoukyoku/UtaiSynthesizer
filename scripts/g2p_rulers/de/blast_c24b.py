# -*- coding: utf-8 -*-
"""S111 —— §C24 全轮(S110 §C24 + S111 §C24c/d/b + S112 §C24e)的累计血径,独立于 Rust 算一遍。

pre = 无 `n j`、无 `l j`、无任何 glide_block   →   now = HEAD(字面守卫 + 例外键 + ⟨li⟩ 授权 + ⟨gn⟩ 守卫)

⚠★ S112 修:此前 `ONSET_KEYS` 与 `CODA_SPELL` 是**手抄**的字面量(`{"orjol","reykjavik"}` /
   `("willia",)`),而这个文件被 `g2p.rs` 的闸点名当作「独立复算」。对抗审查抓到:HEAD 一改这两个
   常量,它就会在**声称量 HEAD** 的同时量一张不存在的表,并制造一次假的「两个实现不一致」——
   最省事的收场是把 Rust 那个数改掉,于是第二把尺子就没了。现在三个常量全部**解析自 g2p.rs**
   (`c24_lib` 的 `_opt_str_array` / `_opt_pairs`),表一改这里自动跟上。
"""
import re, sys
sys.stdout.reconfigure(encoding="utf-8")
import c24_lib as L
import s111_lib as S, proposal as P

d = S.de()
T = list(S.DE_LITERAL_J_SPELLING) + list(L.DE_ROMANCE_GN_SPELLING)   # 两张表喂同一个计数器
ONSET_KEYS = L.DE_CJ_ONSET_KEYS
CODA_SPELL = L.DE_LJ_CODA_SPELLINGS
VOW = "aeiouyäöüáàâéèêíìîóòôúùû"
RE_LIC = re.compile(r"[" + VOW + r"]ll?i[" + VOW + "]")
print(f"解析自 g2p.rs:DE_CJ_ONSET_KEYS={sorted(ONSET_KEYS)} · DE_LJ_CODA_SPELLINGS={list(CODA_SPELL)}"
      f" · DE_ROMANCE_GN_SPELLING={L.DE_ROMANCE_GN_SPELLING}")

def glide_block(k, ph):
    g = [] if k in ONSET_KEYS else P.literal_j(k, ph, T)
    sites = [j for j, c in S.sites(k, ph) if c == "l"]
    if sites:
        lic = (not any(s in k for s in CODA_SPELL)) and len(RE_LIC.findall(k)) >= len(sites)
        if not lic:
            g = sorted(set(g) | set(sites))
    return sorted(set(g))

moved = bad_n = bad_s = 0
fams = {}
for k, ph in d.prim.items():
    any_, st = d.seams(k, ph)
    pre = d.syllabify(ph, seams=(any_, st, ()), drop_onsets=["n j", "l j"])
    now = d.syllabify(ph, seams=(any_, st, glide_block(k, ph)))
    if len(pre) != len(now): bad_n += 1
    if [p for s,e in pre for p in ph[s:e]] != [p for s,e in now for p in ph[s:e]]: bad_s += 1
    if pre != now:
        moved += 1
        for j, c in S.sites(k, ph):
            a = "ON" if S.site_in_onset(pre, j) else "CO"
            b = "ON" if S.site_in_onset(now, j) else "CO"
            if a != b: fams[f"/{c}/ {a}->{b}"] = fams.get(f"/{c}/ {a}->{b}", 0) + 1
print(f"累计血径 moved = {moved}   音节数变={bad_n}  音素序列变={bad_s}")
for f, n in sorted(fams.items(), key=lambda x: -x[1]): print(f"   {f:18s} {n}")
