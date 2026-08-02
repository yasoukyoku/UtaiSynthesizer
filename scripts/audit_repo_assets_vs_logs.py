# -*- coding: utf-8 -*-
r"""S98 —— 「S96 炸仓到底还有什么我没恢复」的证据驱动审计。

我此前的恢复方法是「把我注意到缺了的补回来」,它对「我没注意到的」结构上盲 —— CUDA runtime
(runtime/ort/cuda 4 个文件 290MB + runtime/cuda 16 个文件 2.1GB)就是这么漏了整整两个 session。

正确的问法不是「我觉得还缺什么」,而是:事故之前 app 与构建实际打开过哪些仓库内路径?
那份记录客观存在 —— %LOCALAPPDATA%\UtaiSynthesizer\logs\ 下每一条带仓库前缀的路径。
本脚本把它们全抽出来、去重、逐条查现在还在不在,并按事故前后分桶。只读。
"""
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

LOGS = Path(os.environ["LOCALAPPDATA"]) / "UtaiSynthesizer" / "logs"
ACCIDENT = "2026-08-01"          # S96 的 git worktree remove --force

pat = re.compile(r"[A-Za-z]:[\\/]MyDev[\\/]Utai_v2-dev[\\/][^\s\"',;:()]+")
seen = defaultdict(lambda: {"first": None, "last": None, "n": 0})

logs = sorted(LOGS.glob("utai.log*"))
for f in logs:
    day = f.name.replace("utai.log.", "").replace("utai.log", "current")
    try:
        txt = f.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  !! {f.name}: {e}")
        continue
    for m in pat.finditer(txt):
        p = m.group(0).rstrip(".,)")
        rec = seen[p]
        rec["n"] += 1
        if rec["first"] is None:
            rec["first"] = day
        rec["last"] = day

print(f"日志 {len(logs)} 份,抽到 {len(seen)} 条不同的仓库内路径\n")

missing_pre, missing_post, present = [], [], []
for p, rec in seen.items():
    low = p.lower().replace("/", "\\")
    if "\\target\\" in low:          # 构建产物,不是资产
        continue
    exists = Path(p).exists()
    row = (rec["last"] or "?", rec["n"], p)
    if exists:
        present.append(row)
    elif row[0] < ACCIDENT:
        missing_pre.append(row)
    else:
        missing_post.append(row)

print("=" * 100)
print(f"⛔ 事故前用过、现在【不存在】:{len(missing_pre)} 条  <<< 这就是「我没意识到的」")
for last, n, p in sorted(missing_pre):
    print(f"   最后见于 {last}  x{n:<5} {p}")
print()
print(f"⚠ 事故当天或之后仍被引用、现在不存在:{len(missing_post)} 条")
for last, n, p in sorted(missing_post)[:30]:
    print(f"   最后见于 {last}  x{n:<5} {p}")
print()
print(f"✅ 现在存在:{len(present)} 条")
