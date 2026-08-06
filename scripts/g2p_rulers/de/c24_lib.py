# -*- coding: utf-8 -*-
"""S110 §C24 —— 德语 C+j 的共享底座。

⚠⚠ **为什么不能直接用 `s106_c3_seam\\c3_lib.py`(本轮开工第一件查出来的事)**:
   那份底座的 onset 集构造是 **S105 语义**(`observed ∩ KEEP`,单辅音「谁词首出现过就算谁」),
   而 **S108 §C21 把它整个换掉了**:
     ① 单辅音改成 **STATED** —— 词典里用到的每个辅音都可以起音节,**除非** `single_onset_forbidden`
        点名(de: 只有 `ŋ`);
     ② 四张 KEEP 表变 **权威表** —— 不再与 observed 求交,表里有就是有。
   ⇒ 拿 c3_lib 量今天的德语,**量的是一个已经不存在的系统**。这正是它自己 docstring 里
   「工装如果手抄生产的表,下一次生产改了表、工装照旧,分析就在描述一个不存在的系统」那条
   警告的**另一种触发方式**:表没手抄,**语义**过期了。

本模块只做一件事:**按 HEAD 的 `WordDict::from_tsv` + `syllabify` 逐行复算**,并把
`compound_seams`(S106,HEAD 未改)从 `c3_build.py` 借过来 —— 借的是**纯函数**,不是那份 onset 集。

★ **它不是权威**:凡本模块算出来的量,必须先过 `c24_crosscheck_rust.py`(拿真 Rust 引擎的
  `g2p_probe` 输出逐词对拍)。「先证明自己在重现生产,再谈分歧」——S109 §C23 血训。
"""
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[3]   # scripts/g2p_rulers/de → 仓库根
DICTS = REPO / "data" / "dictionaries"
G2P = REPO / "src-tauri" / "src" / "inference" / "g2p.rs"
TABLES = REPO / "src-tauri" / "src" / "inference" / "g2p_tables.rs"


# ★★ S112 新增:**解析哪一个面**从此是参数,不是常量 ────────────────────────────────────────
# 本模块有意「解析生产常量、零手抄」,S110 靠它躲过了手抄漂移。但 S111 栽在它的**镜像**上
# (工具坑 B1i):我改完 g2p.rs 之后再拿它量「改前 vs 改后」,**两臂读到的都是改后的表**,
# 于是算出 1029/1029、报「这次改动什么也没影响」,而 Rust 的闸同时在报 1046→1029。
# ⇒ 规矩:凡工装的输入包含我这一轮正在改的东西,**至少一臂必须来自版本控制**。
#   `C24_G2P_REV=<commit>` 未设时,行为与此前逐字节相同(默认仍读工作树)。
def _src(p: Path) -> str:
    rev = os.environ.get("C24_G2P_REV", "").strip()
    if not rev:
        return p.read_text(encoding="utf-8")
    rel = p.relative_to(REPO).as_posix()
    out = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=REPO,
                         capture_output=True, check=True)
    # ⚠ 响亮:钉住的面与工作树不同是**结论的一部分**,不许静默(报到 stderr,免得污染报表 stdout)
    print(f"[c24_lib] 解析的面 = git {rev}:{rel}(不是工作树)", file=sys.stderr)
    return out.stdout.decode("utf-8")


def _assert_dict_matches_rev():
    """★★ S112,对抗审查抓出来的那一半:`C24_G2P_REV` **钉不住词典**。

    `data/` 整个在 .gitignore 里(`git ls-files data/dictionaries/` = 0 个文件)⇒ de.tsv
    永远来自工作树。今天无害(S110/S111/S112 都是词典字节零改动),但队列里迟早会重生成词典
    (§C22/§G10 那个桶),那之后照配方「钉到某个 rev 重算」拿到的是
    **【旧代码面 + 新词典】这个从没存在过的系统**,而横幅还理直气壮地说「钉在 <rev>」——
    比 S111 原来那个坑更难发现,因为多了一句让人放心的话。

    ⇒ 判据用的是**已经在版本控制里**的东西:S109 §H7 立的 `src-tauri/dictionaries.sha256`。
      不一致就**报错退出**,不许继续算。这同时是那份基准第一次被分析工装消费。
    """
    rev = os.environ.get("C24_G2P_REV", "").strip()
    if not rev:
        return
    import hashlib
    want = None
    man = subprocess.run(["git", "show", f"{rev}:src-tauri/dictionaries.sha256"], cwd=REPO,
                         capture_output=True)
    if man.returncode != 0:
        print(f"[c24_lib] ⚠ {rev} 里没有 dictionaries.sha256(它是 S109 才立的)—— "
              f"词典面**无法核对**,这一轮的结论只对代码面成立", file=sys.stderr)
        return
    for ln in man.stdout.decode("utf-8").splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[1] == "de.tsv":
            want = parts[0]
    got = hashlib.sha256((DICTS / "de.tsv").read_bytes()).hexdigest()
    if want and want != got:
        raise RuntimeError(
            f"⛔ 工作树的 de.tsv 与 {rev} 记的不是同一份词典\n"
            f"   {rev} 记 {want}\n   工作树   {got}\n"
            f"   钉住代码面而词典面是新的 = 在量一个从没存在过的系统。先想清楚要问哪个面。")
    print(f"[c24_lib] 词典面已核对:de.tsv sha256 与 {rev} 一致({got[:12]}…)", file=sys.stderr)

# ─── 生产常量:一律解析,零手抄 ────────────────────────────────────────────────────────────────


def _rust_str_array(src: str, name: str) -> list:
    """`const NAME: &[&str] = &[ ... ];` 里的全部字面量(先剥注释 —— 注释里也有 "…")。"""
    m = re.search(r"const\s+" + re.escape(name) + r"\s*:\s*&\[&str\]\s*=\s*&\[", src)
    if not m:
        raise RuntimeError(f"没找到 {name} —— g2p.rs 的形状变了,工装必须同步")
    i, depth, body = m.end(), 1, []
    while depth:
        ch = src[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                break
        body.append(ch)
        i += 1
    body = re.sub(r"//[^\n]*", "", "".join(body))
    return re.findall(r'"((?:[^"\\]|\\.)*)"', body)


def _single_forbidden_de(src: str) -> set:
    """从 `single_onset_forbidden` 里取 de 那一臂。**不硬编 {'ŋ'}** —— 那一行改了工装要跟着红。"""
    m = re.search(r"fn single_onset_forbidden\(lang: Lang, phone: &str\) -> bool \{(.*?)\n\}", src, re.S)
    if not m:
        raise RuntimeError("没找到 single_onset_forbidden")
    body = m.group(1)
    arm = re.search(r"Lang::De[^=]*=>\s*(.+?),\s*\n", body)
    if not arm:
        raise RuntimeError(f"single_onset_forbidden 的 De 臂形状变了:\n{body}")
    txt = arm.group(1)
    # 今天的形状是 `phone == "ŋ"`(与 Fr/Es 共用一臂)。任何别的形状必须由人来读。
    lits = re.findall(r'phone == "([^"]+)"', txt)
    if not lits or "ends_with" in txt or "||" in txt.replace('phone == "', ""):
        pass  # 允许多个 `phone == "x" || phone == "y"`,但 ends_with 之类必须报错
    if "ends_with" in txt:
        raise RuntimeError(f"De 臂出现了 ends_with,工装看不懂,请人工同步:{txt}")
    if not lits:
        raise RuntimeError(f"De 臂里没解析出任何字面量:{txt}")
    return set(lits)


G2P_SRC = _src(G2P)
TABLES_SRC = _src(TABLES)

DE_VOWELS = set(_rust_str_array(TABLES_SRC, "MFA_VOWELS_DE"))
DE_KEEP = set(_rust_str_array(G2P_SRC, "DE_ONSET_KEEP"))
DE_SINGLE_FORBIDDEN = _single_forbidden_de(G2P_SRC)


def _pairs(src, name):
    """`const NAME: &[(&str, &str)] = &[ ("a","b"), … ];` —— 同样解析,零手抄。"""
    m = re.search(r"const\s+" + re.escape(name) + r"\s*:\s*&\[\(&str,\s*&str\)\]\s*=\s*&\[(.*?)\n\];", src, re.S)
    if not m:
        raise RuntimeError(f"没找到 {name}")
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', body)


DE_LITERAL_J_SPELLING = _pairs(G2P_SRC, "DE_LITERAL_J_SPELLING")


# ★★ S112 新增:这三个常量此前**没有 parser**,于是下游工装(`blast_c24b.py`)手抄了它们。
#   对抗审查抓到:那份脚本的 docstring 声称「now = HEAD」,而 HEAD 一改表它就在量一张不存在的表
#   —— 正是本文件开头引的那条警告的复发,只是这次漂的是**没被解析的那几个常量**。
#   ⚠ 它们不是所有版本都存在(`DE_CJ_ONSET_KEYS`/`DE_LJ_CODA_SPELLINGS` 是 S111 才有的,
#     `DE_ROMANCE_GN_SPELLING` 是 S112),所以解析必须**容许缺席**:钉到旧 rev 时返回空,
#     那正是那个版本的真实状态,不是错误。
def _opt_str_array(src, name):
    try:
        return _rust_str_array(src, name)
    except RuntimeError:
        return []


def _opt_pairs(src, name):
    try:
        return _pairs(src, name)
    except RuntimeError:
        return []


DE_CJ_ONSET_KEYS = set(_opt_str_array(G2P_SRC, "DE_CJ_ONSET_KEYS"))
DE_LJ_CODA_SPELLINGS = tuple(_opt_str_array(G2P_SRC, "DE_LJ_CODA_SPELLINGS"))
DE_ROMANCE_GN_SPELLING = _opt_pairs(G2P_SRC, "DE_ROMANCE_GN_SPELLING")

# ─── 接缝(S106,HEAD 未改)—— 借 c3_build 的纯函数,不借它的 onset 集 ────────────────────────
SEAM_HEAD_CHARS = 2
SEAM_HEAD_PHONES = 1
SEAM_STRICT_HEAD_PHONES = 2
SEAM_TAIL_CHARS = 2
SEAM_TAIL_PHONES = 2
SEAM_STRICT_TAIL_CHARS = 4
SEAM_STRICT_TAIL_PHONES = 3
SYLL = {"n̩": ["ə", "n"], "m̩": ["ə", "m"], "l̩": ["ə", "l"]}
BOUND_DE = {"ver": ["f", "ɛ", "ɐ"], "ge": ["ɡ", "ə"], "rück": ["ʁ", "ʏ", "k"]}


def _assert_seam_consts_match_head():
    """★ 上面七个数是手抄的 ⇒ 必须当场对拍源码,否则又是一份会静默漂移的副本。"""
    want = {
        "SEAM_HEAD_CHARS": SEAM_HEAD_CHARS,
        "SEAM_HEAD_PHONES": SEAM_HEAD_PHONES,
        "SEAM_STRICT_HEAD_PHONES": SEAM_STRICT_HEAD_PHONES,
        "SEAM_TAIL_CHARS": SEAM_TAIL_CHARS,
        "SEAM_TAIL_PHONES": SEAM_TAIL_PHONES,
        "SEAM_STRICT_TAIL_CHARS": SEAM_STRICT_TAIL_CHARS,
        "SEAM_STRICT_TAIL_PHONES": SEAM_STRICT_TAIL_PHONES,
    }
    bad = []
    for name, v in want.items():
        m = re.search(r"const\s+" + name + r"\s*:\s*usize\s*=\s*(\d+)\s*;", G2P_SRC)
        if not m:
            bad.append(f"{name}: 源码里找不到")
        elif int(m.group(1)) != v:
            bad.append(f"{name}: 源码 {m.group(1)} != 工装 {v}")
    # 绑定前缀表同样对拍
    m = re.search(r"const DE_SEAM_BOUND_PREFIXES:.*?=\s*(.*?);\s*\n", G2P_SRC, re.S)
    if not m:
        bad.append("DE_SEAM_BOUND_PREFIXES: 找不到")
    else:
        pairs = re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', re.sub(r"//[^\n]*", "", m.group(1)))
        got = {p: ph.split() for p, ph in pairs}
        if got != BOUND_DE:
            bad.append(f"DE_SEAM_BOUND_PREFIXES: 源码 {got} != 工装 {BOUND_DE}")
    if bad:
        raise RuntimeError("接缝常量与 HEAD 不一致:\n  " + "\n  ".join(bad))


def destress(toks):
    return [t[:-1] if t[-1:] in "012" else t for t in toks]


def syllabic_expand(toks):
    out = []
    for t in toks:
        out += SYLL.get(t, [t])
    return out


def is_vowel(ph):
    return ph in DE_VOWELS


# ─── 词典 ────────────────────────────────────────────────────────────────────────────────────────


class DeDict:
    """HEAD 的 `WordDict::from_tsv(Lang::De)` 逐行等价物。"""

    def __init__(self):
        self.prim = {}
        self.alts = defaultdict(list)
        self.votes = Counter()
        consonants = set()
        for ln in (DICTS / "de.tsv").read_text(encoding="utf-8").splitlines():
            w, _, ph = ln.partition("\t")
            ph = ph.strip()
            if not w or not ph:
                continue
            key = w.lower()
            toks = ph.split()
            if key not in self.prim:
                self.prim[key] = toks
            elif toks != self.prim[key] and toks not in self.alts[key]:
                self.alts[key].append(toks)
            for t in toks:
                if not is_vowel(t):
                    consonants.add(t)
            vi = next((i for i, p in enumerate(toks) if is_vowel(p)), None)
            if vi is not None:
                self.votes[" ".join(toks[:vi])] += 1
        # ① 单辅音:STATED(S108)
        self.onsets = {""}
        for c in consonants:
            if c not in DE_SINGLE_FORBIDDEN:
                self.onsets.add(c)
        # ② 多辅音簇:KEEP 表**权威**(S108),不与 observed 求交
        self.onsets |= DE_KEEP
        self.consonants = consonants

    def pronunciations(self, key):
        if key in self.prim:
            yield self.prim[key]
        for a in self.alts.get(key, ()):
            yield a

    # ── S106 compound_seams 的逐行等价物 ─────────────────────────────────────────────────────
    def seams(self, key, phones):
        any_, strict = [], []
        if len(phones) < 4:
            return any_, strict
        if key not in self.prim:
            return any_, strict
        chars = list(key)
        want = destress(phones)
        for c in range(SEAM_HEAD_CHARS, max(SEAM_HEAD_CHARS, len(chars) - (SEAM_TAIL_CHARS - 1))):
            head, tail = "".join(chars[:c]), "".join(chars[c:])
            if tail not in self.prim:
                continue
            bound = BOUND_DE.get(head)
            if head not in self.prim and bound is None:
                continue
            tail_chars = len(chars) - c
            heads = [(p, False) for p in self.pronunciations(head)]
            if bound is not None:
                heads.append((bound, True))
            for hp, is_bound in heads:
                n = len(hp)
                if n < SEAM_HEAD_PHONES or n + SEAM_TAIL_PHONES > len(phones):
                    continue
                if destress(hp) != want[:n]:
                    continue
                rest = syllabic_expand(want[n:])
                if not any(syllabic_expand(destress(tp)) == rest for tp in self.pronunciations(tail)):
                    continue
                if n not in any_:
                    any_.append(n)
                    if (
                        not is_bound
                        and n >= SEAM_STRICT_HEAD_PHONES
                        and tail_chars >= SEAM_STRICT_TAIL_CHARS
                        and len(want) - n >= SEAM_STRICT_TAIL_PHONES
                        and chars[c].lower() not in ("i", "e")
                        and any(is_vowel(p) and p not in ("ə", "ɐ") for p in want[:n])
                    ):
                        strict.append(n)
                break
        return sorted(any_), sorted(strict)

    # ── S110 §C24 的 `de_literal_j_blocks` 逐行等价物 ────────────────────────────────────
    def literal_j(self, key, phones):
        if key not in self.prim or "j" not in phones:
            return []
        if not any(p == phones for p in self.pronunciations(key)):
            return []          # 与生产同一条免疫:这些音素不是词典发的
        nuc = [i for i, p in enumerate(phones) if is_vowel(p)]
        by_c = defaultdict(list)
        for a, b in zip(nuc, nuc[1:]):
            if b >= a + 3 and phones[b - 1] == "j":
                by_c[phones[b - 2]].append(b - 1)
        out = []
        for c, pos in by_c.items():
            lit = sum(key.count(bg) for p, bg in DE_LITERAL_J_SPELLING if p == c)
            if lit >= len(pos):
                out += pos
            # 0 < lit < len(pos) ⇒ 弃权
        return sorted(out)

    # ── HEAD 的 `syllabify` ────────────────────────────────────────────────────────────────
    def syllabify(self, phones, seams=((), (), ()), extra_onsets=(), drop_onsets=()):
        if len(seams) == 2:
            seams = (seams[0], seams[1], ())
        any_, strict, glide = seams
        onsets = (self.onsets | set(extra_onsets)) - set(drop_onsets)
        nuc = [i for i, p in enumerate(phones) if is_vowel(p)]
        if not nuc:
            return [(0, len(phones))]
        bounds = [0]
        for a, b in zip(nuc, nuc[1:]):
            cl = phones[a + 1 : b]
            cand = [s for s in any_ if a + 1 <= s <= b]
            lo = 0
            if len(cand) == 1 and a + 1 < cand[0] < b and cand[0] in strict:
                lo = cand[0] - (a + 1)
            for g in glide:                      # S110:与接缝取 max,且不受弃权影响
                if a + 1 <= g < b:
                    lo = max(lo, g - (a + 1))
            cut = len(cl)
            for s in range(lo, len(cl) + 1):
                if " ".join(cl[s:]) in onsets:
                    cut = s
                    break
            bounds.append(a + 1 + cut)
        bounds.append(len(phones))
        return list(zip(bounds, bounds[1:]))

    def syl_str(self, phones, **kw):
        return " | ".join(" ".join(phones[s:e]) for s, e in self.syllabify(phones, **kw))


_D = None


def de():
    global _D
    if _D is None:
        _assert_seam_consts_match_head()
        _assert_dict_matches_rev()      # S112:钉了代码面就必须核词典面,否则那个 rev 是半个谎
        _D = DeDict()
    return _D


if __name__ == "__main__":
    d = de()
    print(f"de.tsv keys={len(d.prim)}  alts={sum(len(v) for v in d.alts.values())}")
    print(f"consonants={len(d.consonants)}  forbidden-as-single={sorted(DE_SINGLE_FORBIDDEN)}")
    print(f"onsets={len(d.onsets)}  (KEEP={len(DE_KEEP)})")
    print(f"'n j' in onsets = {'n j' in d.onsets}   'l j' in onsets = {'l j' in d.onsets}")
    for w in ("richtlinie", "million", "schuljahr", "zehnjähriger", "konjunktur", "sowjetunion"):
        ph = d.prim.get(w)
        if ph is None:
            print(f"  {w:16s} <not a key>")
            continue
        s = d.seams(w, ph)
        print(f"  {w:16s} {d.syl_str(ph, seams=s):40s}  seams={s}")
