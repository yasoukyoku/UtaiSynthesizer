# -*- coding: utf-8 -*-
"""S111 —— de.wiktionary 的**机械**取数层(§C24b/§C24c 那把尺子的地基)。

⚠ 本模块的存在理由 = 用户 2026-08-05 给 §C24c 加的前置:
   「那个**尺子本身**你到时候也再确认一下**它是不是好的** —— 查问题之前先问是不是再问为什么」。

★ 设计上只有一条硬规矩:**这一层里不许有任何判断**。
  取 wikitext → 落盘缓存 → 正则抽 `{{Lautschrift|…}}` / `{{Worttrennung}}`,**全是机械的**。
  凡「这个词该怎么切」的判断一律在 `ruler.py`,而且要能被对照组打红。
  ⛔ 绝不让 agent/LLM 去读页面再把结论转述给我 —— 那正是 S98 自证陷阱的形状
     (「我的解释永远自洽,所以自洽不携带任何信息」)。缓存落盘 = 事后任何人都能复查原文。

★ 只取**德语**那一节。de.wiktionary 一个页面上可以同时有德/法/英词条
  (`brillant` 就是),混进来会静默污染整把尺子。
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
CACHE = HERE / "wikt_cache.json"
API = "https://de.wiktionary.org/w/api.php"
UA = "UtaiSynth-dict-research/1.0 (offline singing-synth G2P research; contact via github)"
BATCH = 50


# ─── 取数(带盘上缓存,幂等)────────────────────────────────────────────────────────────────

def _load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(c):
    CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=0), encoding="utf-8")


def fetch(titles, verbose=True):
    """titles → {title: wikitext | None}。None = 页面不存在(**也缓存**,免得反复打网)。"""
    cache = _load_cache()
    todo = [t for t in dict.fromkeys(titles) if t not in cache]
    if verbose and todo:
        print(f"[wikt] 需要取 {len(todo)} 个标题(缓存里已有 {len(cache)})", flush=True)
    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        q = {
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
            "formatversion": "2",
        }
        q["maxlag"] = "5"
        req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": UA})
        # ⚠ 第一版 0.4s 一发、退避 2-8s ⇒ 打到 1520 个标题时被 429 掐了。
        #   对方明说了要等多久就等多久,别自作主张。
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if attempt == 5:
                    raise
                wait = int(e.headers.get("Retry-After", 0) or 0) or (10 * (attempt + 1))
                print(f"[wikt] {e.code},等 {wait}s 再试({attempt + 1}/5)", flush=True)
                time.sleep(wait)
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"[wikt] 重试 {attempt + 1}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        # 归一化映射(API 会把 `Ss` 之类改写);缺失页面也要记 None
        norm = {n["from"]: n["to"] for n in data.get("query", {}).get("normalized", [])}
        got = {}
        for p in data.get("query", {}).get("pages", []):
            if p.get("missing"):
                got[p["title"]] = None
            else:
                got[p["title"]] = p["revisions"][0]["slots"]["main"]["content"]
        for t in chunk:
            key = norm.get(t, t)
            cache[t] = got.get(key, None)
        if verbose:
            print(f"[wikt]   {i + len(chunk)}/{len(todo)}", flush=True)
        _save_cache(cache)
        time.sleep(1.5)
    return {t: cache.get(t) for t in titles}


# ─── 抽取(纯正则,零判断)────────────────────────────────────────────────────────────────

# 实际形状 = `== Million ({{Sprache|Deutsch}}) ==`(⚠ 第一版漏了那个右括号,12/12 全判「无德语节」
# —— 正是 S89「自己写的核对器会瞎」;是先看原文才发现的,不是靠想)。只认二级标题。
_LANG_HEAD = re.compile(r"^==[^=](.*?)==\s*$", re.M)
_SPRACHE = re.compile(r"\{\{Sprache\|([^}|]+)\}\}")


def german_section(wikitext):
    """只留 `{{Sprache|Deutsch}}` 那一节。没有德语节 ⇒ None。"""
    if not wikitext:
        return None
    heads = list(_LANG_HEAD.finditer(wikitext))
    if not heads:
        return None
    out = []
    for i, m in enumerate(heads):
        sp = _SPRACHE.search(m.group(1))
        if sp is None or sp.group(1).strip() != "Deutsch":
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(wikitext)
        out.append(wikitext[m.end() : end])
    return "\n".join(out) if out else None


_LAUT = re.compile(r"\{\{Lautschrift\|([^}|]*)\}\}")
_TRENN = re.compile(r"\{\{Worttrennung\}\}\s*\n(.*?)(?:\n\s*\n|\n\{\{)", re.S)


def ipas(wikitext):
    """德语节里的全部 IPA 串(**含**复数/变体行,顺序保留,去重)。"""
    sec = german_section(wikitext)
    if not sec:
        return []
    seen, out = set(), []
    for m in _LAUT.finditer(sec):
        s = m.group(1).strip()
        # 空壳 `{{Lautschrift|}}`、以及听力样本里的文件名不会命中(那是 {{Audio}})
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def worttrennung(wikitext):
    """正字法音节划分(`Mil·li·on`)。★留着当【对照尺子】,不是主尺 —— 见 ruler.py 的负结果。"""
    sec = german_section(wikitext)
    if not sec:
        return []
    m = _TRENN.search(sec)
    if not m:
        return []
    body = re.sub(r"\{\{[^}]*\}\}", "", m.group(1))
    body = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", body)
    out = []
    for part in re.split(r"[,;]", body.replace(":", " ")):
        part = part.strip()
        if "·" in part:
            out.append(part)
    return out


# ─── 标题候选:de.tsv 的键是小写,而德语名词页面首字母大写 ──────────────────────────────

def title_candidates(key):
    """一个 de.tsv 键 → 该试的页面标题(顺序 = 优先级)。"""
    cands = [key]
    if key[:1].isalpha():
        cands.append(key[0].upper() + key[1:])
    return list(dict.fromkeys(cands))


if __name__ == "__main__":
    ts = sys.argv[1:] or ["Million", "brillant", "Schuljahr", "Medaille"]
    res = fetch(ts)
    for t, wt in res.items():
        print(f"\n=== {t} ===")
        if wt is None:
            print("   <页面不存在>")
            continue
        if german_section(wt) is None:
            print("   <无德语节>")
            continue
        print("   IPA :", ipas(wt))
        print("   Trenn:", worttrennung(wt))
