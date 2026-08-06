# -*- coding: utf-8 -*-
"""S111 —— **第二个源**:en.wiktionary 的德语条目。

为什么要第二个源(而不是只用 de.wiktionary):
  ① 它写**显式音节点**(`/ˈɔp.jɛkt/`),那是对「C 在 onset 还是 coda」的**直接**回答,
     比重音位置更强;
  ② 它列**更多变体**(`Million` 另有 `[mɪlˈjoːn]` = coda)⇒ 能暴露「同一个词其实有两读」,
     而单看 de.wikt 会把它读成一个干净的判决。**这正是尺子该被对照组抓出来的性质。**
  ⚠ 两个源并非完全独立(维基之间会互抄),所以「两源一致」**不能**当成两条独立证据 ——
     只能当成「至少没有互相矛盾」。这一条写在这里,免得下游把它读成加倍的置信度。
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
CACHE = HERE / "enwikt_cache.json"
API = "https://en.wiktionary.org/w/api.php"
UA = "UtaiSynth-dict-research/1.0 (offline singing-synth G2P research; contact via github)"
BATCH = 50


def _load():
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def fetch(titles, verbose=True):
    cache = _load()
    todo = [t for t in dict.fromkeys(titles) if t not in cache]
    if verbose and todo:
        print(f"[enwikt] 需要取 {len(todo)}(缓存 {len(cache)})", flush=True)
    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        q = {
            "action": "query", "titles": "|".join(chunk), "prop": "revisions",
            "rvprop": "content", "rvslots": "main", "format": "json", "formatversion": "2",
        }
        q["maxlag"] = "5"
        req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": UA})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if attempt == 5:
                    raise
                wait = int(e.headers.get("Retry-After", 0) or 0) or (10 * (attempt + 1))
                print(f"[enwikt] {e.code},等 {wait}s({attempt + 1}/5)", flush=True)
                time.sleep(wait)
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"[enwikt] 重试 {attempt + 1}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        norm = {n["from"]: n["to"] for n in data.get("query", {}).get("normalized", [])}
        got = {}
        for p in data.get("query", {}).get("pages", []):
            got[p["title"]] = None if p.get("missing") else p["revisions"][0]["slots"]["main"]["content"]
        for t in chunk:
            cache[t] = got.get(norm.get(t, t), None)
        if verbose:
            print(f"[enwikt]   {i + len(chunk)}/{len(todo)}", flush=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        time.sleep(1.5)
    return {t: cache.get(t) for t in titles}


_HEAD2 = re.compile(r"^==([^=].*?)==\s*$", re.M)
_IPA_TPL = re.compile(r"\{\{IPA\|de\|([^}]*)\}\}")


def german_section(wikitext):
    if not wikitext:
        return None
    heads = list(_HEAD2.finditer(wikitext))
    for i, m in enumerate(heads):
        if m.group(1).strip() == "German":
            end = heads[i + 1].start() if i + 1 < len(heads) else len(wikitext)
            return wikitext[m.end() : end]
    return None


def ipas(word_or_wikitext, raw=False):
    """en.wikt 德语节里的全部 IPA。参数可以是词(走缓存)或 wikitext。"""
    wt = word_or_wikitext if raw else _load().get(word_or_wikitext)
    sec = german_section(wt)
    if not sec:
        return []
    out, seen = [], set()
    for m in _IPA_TPL.finditer(sec):
        for part in m.group(1).split("|"):
            part = part.strip()
            if "=" in part.split("/")[0].split("[")[0]:
                continue                      # qual3=… 之类的具名参数
            if len(part) > 2 and (part[0] == "/" or part[0] == "["):
                s = part.strip("/[]")
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


if __name__ == "__main__":
    ts = sys.argv[1:] or ["Million", "Objekt", "Sowjet", "Medaille"]
    fetch(ts)
    for t in ts:
        print(f"{t:14s} {ipas(t)}")
