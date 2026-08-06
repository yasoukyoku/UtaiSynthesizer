# -*- coding: utf-8 -*-
"""S111 —— 那把尺子本身,以及**它的对照组**。

⭐ 用户 2026-08-05 给 §C24c 定的前置(原话要点):
   「那个**尺子本身**你到时候也再确认一下**它是不是好的** —— 毕竟这几次都有个很重要的经验就是,
     **查问题之前先问是不是再问为什么**」
   而且结果要分成三类,**不许一律「判死就扔」**:
     ① 与不容争辩的答案**相反** ⇒ 证伪 ⇒ 扔
     ② 对这一格**结构性沉默**   ⇒ 覆盖率事实,不是判决 ⇒ **缩使用范围,不扔**
     ③ 口径/实现/取的面选错了   ⇒ 修了再用,**修完必须重新过一遍对照组**

★★ 预注册:下面 `CONTROL` 里每一条的 `want` 是**在跑之前**写死的,依据写在 `why` 里。
   ⛔ 不许看到结果再回来改 want —— 那就是拿尺子去校准对照组,方向反了。

────────────────────────────────────────────────────────────────────────────────
本文件同时评四把尺子(它们不是一把):
  S1  de.wikt 重音相邻     ˈ/ˌ 紧挨 /j/ ⇒ coda;紧挨 C ⇒ onset;否则沉默。**双向**
  S2  滑音写法 i̯ vs j      写成 i̯ ⇒ onset(非成节元音不能单独起音节)。**单向**,j 不给判决
  S3  en.wikt 显式音节点   `.` 落在 C 与 j 之间 ⇒ coda;落在 C 之前 ⇒ onset。**双向**
  S4  de.wikt Worttrennung 正字法断词。★**已被证伪 —— 但只在一半上**,负结果写在这里
      (⚠ S112 修:这一行原来写「见文件末的负结果」,而**文件末根本没有那段**。
        一个指向不存在的证据的指针,比没有指针更糟 —— 下一个人会以为验过了。
        实际的负结果在 `verify_s4_worttrennung.py`,结论抄在这里,不再靠指针):
        · 当**音节数**尺子 ⇒ **证伪**:`Mil·li·on` 断成 3 节而 IPA 是 2 节(`mɪˈli̯oːn`),
          `So·w·jet` 更是把一个字母单独断出来 —— 正字法断词服务的是换行,不是音节。
        · 当**语素边界**证据 ⇒ **好用**:`Ob·jekt` `Ad·jek·tiv` `Ad·junkt` 与词法一致,
          §C24c 就是靠它(加上 ob-/sub- 词法与重音位置)立起来的。
        ⇒ 所以 S4 **不进主判决**,只在讨论「这里是不是语素边界」时当旁证。
      ⚠ 而且它有方向性:Worttrennung 结构上只会支持 CODA(它断的就是边界),
        所以它在 `orjol` 这种真 onset 上必然给错答案 —— 不能拿它当双向信号。
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import wikt  # noqa: E402
import enwikt  # noqa: E402

STRESS = "\u02c8\u02cc"      # ˈ ˌ
DOT = "."
TIE = "\u0361"               # ͡ (t͡s)
LENGTH = "\u02d0\u02d1"      # ː ˑ
GLIDE_I = "i\u032f"          # i̯


# ─── 尺子本体 ─────────────────────────────────────────────────────────────────────────────

def _is_diacritic(ch):
    return unicodedata.combining(ch) != 0 or ch in LENGTH


def glide_sites(ipa):
    """IPA 串里全部「辅音 + 滑音」位点。

    返回 [(glide_kind, consonant, separator)]:
      glide_kind = 'i̯' | 'j'
      consonant  = 滑音前面那个音段(含连结符,如 't͡s');没有辅音 ⇒ None
      separator  = 'stress' | 'dot' | ''   —— 辅音与滑音**之间**有什么
    外加 before = 辅音**前面**紧挨的是不是 stress/dot。
    """
    out = []
    i = 0
    n = len(ipa)
    while i < n:
        kind = None
        if ipa.startswith(GLIDE_I, i):
            kind, glen = "i̯", 2
        elif ipa[i] == "j":
            kind, glen = "j", 1
        if kind is None:
            i += 1
            continue
        # 往回找:先吃掉 stress/dot,记下来
        k = i - 1
        sep = ""
        while k >= 0 and (ipa[k] in STRESS or ipa[k] == DOT):
            sep = "stress" if ipa[k] in STRESS else (sep or "dot")
            k -= 1
        # 再吃掉附加符号,取一个音段(含 tie bar 连成的塞擦音)
        seg_end = k
        while k >= 0 and _is_diacritic(ipa[k]):
            k -= 1
        if k < 0:
            out.append((kind, None, sep, ""))
            i += glen
            continue
        cons_start = k
        k -= 1
        # tie bar:t͡s 里 s 前面是 U+0361,再前面是 t
        while k >= 1 and ipa[k] == TIE:
            k -= 1
            while k >= 0 and _is_diacritic(ipa[k]):
                k -= 1
            cons_start = k
            k -= 1
        cons = ipa[cons_start : seg_end + 1]
        # 辅音前面紧挨的
        before = ""
        m = cons_start - 1
        while m >= 0 and (ipa[m] in STRESS or ipa[m] == DOT):
            before = "stress" if ipa[m] in STRESS else (before or "dot")
            m -= 1
        out.append((kind, cons, sep, before))
        i += glen
    return out


VOWEL_CHARS = "aeiouyɛɔɪʊœøyæɐəɑʌɒːɜ"

# ⛔⛔ S111 对抗审查抓出的**致命循环**,修在这里,别再退回去:
# 选位点时按【清浊】匹配辅音,而 §C24c/d 的被测假设恰恰是「这个音素的清浊决定 onset/coda」
# ⇒ **选点变量 = 被测变量**,凡「我们写浊、源写清」的词,反面证据会被静默变成「沉默」。
# 实测代价:⟨dj⟩+/d/ 那 10 个词的外部转写**写清音 16 条、写浊音 3 条**,我的过滤器只看见那 3 条,
# 于是报出「3 证实 : 0 反驳」;`Adjunkt` 两源都写 `ˌatˈjʊnkt`(重音正落在 /t/ 与 /j/ 之间 =
# 主信号 S1 直说 CODA)却被记成「沉默」。
# ⇒ **位点匹配必须清浊中立**(按发音部位),清浊留给被测假设自己去说。
DEVOICE = {"d": "t", "b": "p", "ɡ": "k", "g": "k", "v": "f", "z": "s", "ʒ": "ʃ", "d͡ʒ": "t͡ʃ"}


def _norm_cons(c):
    """辅音归一化:去送气/长音标记、去连结符、**清浊中立**。"""
    c = c.replace(TIE, "").rstrip("ʰː")
    return DEVOICE.get(c, c)


def verdict_from_ipa(ipa, want_cons=None):
    """一条 IPA → 这条 IPA 对「C 在 onset 还是 coda」说了什么。

    返回 dict(signal -> 'ONSET'|'CODA'|None)。`want_cons` 只用来在**多位点**时挑对位点,
    且匹配是**清浊中立**的(见上面的 ⛔)。
    """
    # ⚠ 滑音前面是元音的位点不是「辅音+滑音」(`buˈjɔ̃ː` 的 /u/、`ɔʁiˈjak` 的 /i/)⇒ 剔掉,
    #   否则它们会冒充位点、把真位点挤掉(审查发现 #2)。
    sites = [s for s in glide_sites(ipa)
             if s[1] is not None and s[1][0] not in VOWEL_CHARS]
    if want_cons and len(sites) > 1:
        base = _norm_cons(want_cons)
        cand = [s for s in sites if _norm_cons(s[1]) == base]
        if len(cand) != 1:
            cand = [s for s in sites if _norm_cons(s[1])[:1] == base[:1]]
        sites = cand
    if len(sites) != 1:
        return {"S1": None, "S2": None, "S3": None, "n_sites": len(sites)}
    kind, cons, sep, before = sites[0]
    s1 = s3 = None
    if sep == "stress":
        s1 = "CODA"
    elif before == "stress":
        s1 = "ONSET"
    if sep == "dot":
        s3 = "CODA"
    elif before == "dot":
        s3 = "ONSET"
    s2 = "ONSET" if kind == "i̯" else None
    return {"S1": s1, "S2": s2, "S3": s3, "n_sites": 1, "cons": cons}


def merge(verdicts, sig):
    """⛔⛔ **死代码,零调用点(S112 核过:全目录 grep `merge(` 只有这一处定义)。**

    留着**只是为了那条负结果的正文**,不是为了被调用 —— 删掉它,负结果就只剩下一句
    没有实现可对照的传说(S102 血训:只有编号没有正文的条目 = 定时炸弹)。
    ⇒ 谁要复活它,先读完下面这段,再重新过一遍对照组。

    ⛔ 第一版的合议规则,**已被对照组判为第③类(口径错)**,留着当负结果,不许再用。

    它把一个词的**全部**变体在某个信号上求并集。听起来稳,实际上错在:
    维基列的多条 IPA 是**不同的读法**,不是同一个读法的多次转写 ——
      `Linie` = ˈliːni̯ə(滑音,/n/ 在 onset)/ ˈliː.ni.ə(三音节)/ **ˈliːn.jə(/n/ 真在 coda)**
    而**点号只出现在次要变体上** ⇒ 求并集必然让次要变体顶掉主读音,
    于是 S3 在 `Linie`/`Spanien` 上给出与不容争辩答案相反的 CODA。
    ★ 这两条**不是尺子在说谎,是我在问错问题**:我要问的是「主读音怎么切」,
      问出来的却是「有没有任何一种读法这么切」。
    """
    vals = {v[sig] for v in verdicts if v.get(sig)}
    if not vals:
        return None
    if len(vals) > 1:
        return "CONFLICT"
    return vals.pop()


def transcription_verdict(ipa, cons):
    """**一条**转写说了什么。三个信号在同一条转写内部不许打架(打架 ⇒ 'CONFLICT')。"""
    v = verdict_from_ipa(ipa, cons)
    vals = {v[s] for s in ("S1", "S2", "S3") if v.get(s)}
    if not vals:
        return None, v
    if len(vals) > 1:
        return "CONFLICT", v
    return vals.pop(), v


def source_verdict(ipa_list, cons):
    """一个源的判决 = **第一条开口的转写**;其余变体只算「变体离散度」,不参与判决。

    ★ 为什么取第一条:两个维基都把主读音排在最前(后面是次要变体/复数/外语式读法)。
    ★ 为什么其余变体只记不判:见 `merge` 的负结果 —— 它们是**别的读法**,
      拿它们否定主读音等于换了个问题。**离散度单独报出来,不许悄悄吞掉。**
    """
    first, spread, per = None, set(), []
    for s in ipa_list:
        v, detail = transcription_verdict(s, cons)
        per.append((s, v, detail))
        if v and v != "CONFLICT":
            spread.add(v)
            if first is None:
                first = v
    return first, (len(spread) > 1), per


# ─── 预注册的对照组 ───────────────────────────────────────────────────────────────────────
# want: 期望答案 | why: 凭什么说它「不容争辩」 | cons: 该看哪个辅音(词里可能有多个 C+j)
CONTROL = [
    # ── ① 不容争辩的 CODA:两个**自由语素**拼起来的德语复合词,边界按定义就是音节边界
    ("Schuljahr",   "CODA",  "Schule+Jahr,两个自由语素", "l"),
    ("Halbjahr",    "CODA",  "halb+Jahr", "p"),
    ("Lehrjahr",    "CODA",  "Lehre+Jahr", "ʁ"),
    ("Vorjahr",     "CODA",  "vor+Jahr", "ʁ"),
    ("Kirchenjahr", "CODA",  "Kirche+Jahr", "n"),
    ("Lichtjahr",   "CODA",  "Licht+Jahr", "t"),
    ("Schaltjahr",  "CODA",  "schalten+Jahr", "t"),
    ("Nebenjob",    "CODA",  "neben+Job", "n"),
    ("Feldjäger",   "CODA",  "Feld+Jäger", "t"),
    ("Fachjargon",  "CODA",  "Fach+Jargon", "x"),
    ("Abfangjäger", "CODA",  "abfangen+Jäger", "ŋ"),
    ("Sonnenjahr",  "CODA",  "Sonne+Jahr", "n"),
    # ── ② 不容争辩的 ONSET:语素内部由 ⟨i⟩ 派生的滑音,那里没有任何语素边界
    ("Million",     "ONSET", "语素内部 ⟨lli⟩+元音,无语素边界", "l"),
    ("Milliarde",   "ONSET", "同上", "l"),
    ("Billion",     "ONSET", "同上", "l"),
    ("Nation",      "ONSET", "⟨-tion⟩ 后缀,/ts/+滑音同属一个音节", "ts"),
    ("Union",       "ONSET", "语素内部 ⟨ni⟩+元音", "n"),
    ("Familie",     "ONSET", "同上", "l"),
    ("Linie",       "ONSET", "同上", "n"),
    ("Religion",    "ONSET", "语素内部 ⟨gi⟩+元音", "ɡ"),
    ("Spanien",     "ONSET", "同上", "n"),
    ("Italien",     "ONSET", "同上", "l"),
    ("Aktion",      "ONSET", "⟨-tion⟩", "ts"),
    ("Position",    "ONSET", "⟨-tion⟩", "ts"),
    # ── ③ 预期【结构性沉默】:重音落在别处,尺子对这一格没有分辨力(= 第②类,不是证伪)
    ("Medaille",    "SILENT", "重音在 /d/ 前,离簇两个音段", "l"),
    ("Pavillon",    "SILENT", "重音在词首", "l"),
    ("Anja",        "SILENT", "重音在词首", "n"),
    ("Katja",       "SILENT", "重音在词首", "t"),
]

# ⚠ 下面这些**不是对照组**,是「拿尺子去问」的目标(答案本来就不确定)。
#   用户点名要一起喂的「纯音译名」放在这里 —— 它们的正确答案没人知道,
#   所以它们只能测【覆盖率】,不能测【正确性】。把它们混进对照组就是自证。
PROBE = ["Objekt", "Subjekt", "Adjektiv", "Konjunktur", "Injektion", "Sowjet",
         "Sonja", "Tanja", "Nadja", "Ilja", "Skopje", "Reykjavik", "Banjo",
         "brillant", "Bataillon", "Vanille", "Emaille", "Taille", "Patrouille"]


def collect(word, cons):
    """两个源各出一个判决(第一条开口的转写),再报两源是否打架。"""
    de_wt = wikt.fetch([word], verbose=False)[word]
    de_ipas = wikt.ipas(de_wt)
    en_ipas = enwikt.ipas(word)
    de_v, de_spread, de_per = source_verdict(de_ipas, cons)
    en_v, en_spread, en_per = source_verdict(en_ipas, cons)
    both = {v for v in (de_v, en_v) if v}
    # 主判决:de 优先(它是德语母语维基);de 沉默才用 en,并标明来源
    final = de_v or en_v
    # 逐信号的覆盖统计,仍然只看**第一条开口的转写**
    sig = {"S1": None, "S2": None, "S3": None}
    for lst in (de_per, en_per):
        for _s, v, detail in lst:
            for k in sig:
                if sig[k] is None and detail.get(k):
                    sig[k] = detail[k]
    return {
        "de_ipa": de_ipas, "en_ipa": en_ipas,
        "de": de_v, "en": en_v, "final": final,
        "src": "de" if de_v else ("en" if en_v else None),
        "spread": de_spread or en_spread,
        "cross_conflict": len(both) > 1,
        **sig,
        "trenn": wikt.worttrennung(de_wt),
    }


def main():
    words = [w for w, _, _, _ in CONTROL]
    wikt.fetch(words + PROBE)
    enwikt.fetch(words + PROBE)

    print("=" * 112)
    print("★ 对照组(第二版口径:每个源只信【第一条开口的转写】)—— 期望答案是预注册的,没动过")
    print("=" * 112)
    print(f"{'词':14s}{'期望':8s}{'判决':9s}{'源':4s}{'离散':5s}{'两源打架':9s}  S1/S2/S3")
    tally = [0, 0, 0]          # 一致 / 相反 / 沉默
    sigtally = {"S1": [0, 0, 0], "S2": [0, 0, 0], "S3": [0, 0, 0]}
    rows, spread_hits, cross_hits = [], [], []
    for w, want, why, cons in CONTROL:
        r = collect(w, cons)
        got = r["final"]
        if want == "SILENT":
            mark = "(预期沉默)" if got is None else f"⚠出声={got}"
        elif got is None:
            tally[2] += 1
            mark = "沉默"
        elif got == want:
            tally[0] += 1
            mark = "✅"
        else:
            tally[1] += 1
            mark = f"⛔{got}"
        for s in ("S1", "S2", "S3"):
            v = r[s]
            if want == "SILENT":
                continue
            sigtally[s][2 if v is None else (0 if v == want else 1)] += 1
        if r["spread"]:
            spread_hits.append(w)
        if r["cross_conflict"]:
            cross_hits.append(w)
        sigs = "/".join(str(r[s] or "-")[:5] for s in ("S1", "S2", "S3"))
        print(f"{w:14s}{want:8s}{mark:9s}{str(r['src']):4s}{'是' if r['spread'] else '  ':5s}"
              f"{'⚠是' if r['cross_conflict'] else '  ':9s}  {sigs}")
        rows.append({"word": w, "want": want, "why": why, "cons": cons, **r})

    print("\n" + "=" * 112)
    print(f"★ 主判决:与不容争辩答案 **一致 {tally[0]} / 相反 {tally[1]} / 沉默 {tally[2]}**"
          f"   (对照组 24 条,另 4 条预期沉默)")
    print(f"  变体离散(同词有两种读法): {spread_hits}")
    print(f"  两源打架:                  {cross_hits or '无'}")
    # ⚠★★ S112 修表头口径 —— 它此前写的是「(只看第一条开口的转写)」,而 `collect()` 填 `sig`
    #   时是**扫过两个源的全部转写、取第一条对该信号出声的**(见 collect() 里那个双层循环)。
    #   两个口径不同,而**这张表恰恰是判断「哪个信号可信」的地方**,写错就会把结论引到反面。
    #   实测(S112 自己复算,不是照抄别人):
    #     · 主判决口径(每个源只信第一条开口的转写)  S3 = 0 一致 / 0 相反 / 24 沉默
    #       ⇒ **S3 在对照组上从没定过案**,「它没出错」不等于「它被考过」。
    #     · 本表这个更松的口径                         S3 = 0 一致 / **2 相反** / 22 沉默
    #       (`Linie` ˈliːn.jə · `Spanien` ˈʃpaːn.jən —— 两条都是 en.wikt 的**次要变体**,
    #        主判决正确地没理它们。)S1 = 17 / 0 / 7,S2 = 12 / 0 / 12,在**两个口径下都不出错**。
    #   ⇒ 该带走的一句话:**S3 只有在别的信号沉默时才会定案,而它唯一开过口的两次都是错的。**
    #     凡某条结论只由 S3 独家支撑,必须在结论旁边写明这一点。
    print(f"\n{'信号':6s}{'一致':>8s}{'相反':>8s}{'沉默':>8s}"
          f"   ⚠口径 = 扫两个源的**全部**转写、取第一条对该信号出声的(与上面的主判决**不同**)")
    for s in ("S1", "S2", "S3"):
        a, b, c = sigtally[s]
        print(f"{s:6s}{a:8d}{b:8d}{c:8d}")

    print("\n" + "=" * 112)
    print("★ 探针(**不是对照组**:正确答案没人知道,只测覆盖率与两源一致性)")
    print(f"{'词':14s}{'判决':9s}{'源':4s}{'离散':5s}  de.IPA")
    for w in PROBE:
        r = collect(w, None)
        print(f"{w:14s}{str(r['final']):9s}{str(r['src']):4s}{'是' if r['spread'] else '  ':5s}  {r['de_ipa']}")
        rows.append({"word": w, "want": "?", "why": "probe", "cons": None, **r})

    Path("control_result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n[落盘] control_result.json")


if __name__ == "__main__":
    main()
