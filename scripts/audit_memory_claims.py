# -*- coding: utf-8 -*-
"""交接自检 —— **每次收工写完 memory 之后必跑一次**(S99 立,用户 2026-08-03 逼出来的)。

S99 查出 S97 的交接把一个泳道基线目录的身份**记反了**(post_s97b 被写成「003f120 之前」,
实际是它之后),躺了一个 session;而它没被发现的原因是:**写的时候没验证,只是照目录名推断的**。
同一个动作在 S98 也犯过(把脚本注释当 provenance)。这个脚本把「引用的东西存不存在」变成机械检查。

⚠ 它查不出【记反了】那一类语义错(post_s97b 那条路径存在、哈希也存在)。语义那半只有一条规矩:
**凡是描述「我没亲手做过的既有事物」,句子里必须带上验证方法,否则标【未验证】。**

原始问题:「以前的交接是不是也一直在胡乱做,只是没被发现?」

不靠印象回答。把 memory/ 里**机械可核**的两类事实全量扫一遍:
  ① commit 哈希(7-40 位 hex)—— 在两个仓里是否真的存在
  ② 绝对路径(D:\... / C:\...)—— 现在是否真的存在
这两类查不出「语义是否记反了」(post_s97b 那种),但能查出「引用的东西根本不存在」。
诚实边界写在报告末尾。
"""
import re
import subprocess
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

MEM = Path(r"C:\Users\admin\.claude\projects\D--MyDev-Utai-v2-dev\memory")
REPOS = [Path(r"D:\MyDev\Utai_v2-dev"), Path(r"D:\MyDev\Much-Better-S2H")]

# ⚠ 第一版这条正则把 frontmatter 里 `originSessionId` 的 UUID 片段当成了 commit,报出 325 个
# 「找不到的哈希」= 纯假警报。判据本身会骗人(S89 血训),所以:①剥掉 frontmatter
# ②哈希两侧不许紧邻 `-`(UUID 分隔符)。
HASH_RE = re.compile(r"(?<![0-9a-zA-Z\-])`?([0-9a-f]{7,40})`?(?![0-9a-zA-Z\-])")
FM_RE = re.compile(r"\A---\n.*?\n---\n", re.S)
PATH_RE = re.compile(r"[A-Za-z]:\\\\?[^\s`（）()「」,、;:!?\"'\]]+")

# 全 hex 但明显不是 commit 的(纯数字、常见十进制)先过滤
def looks_like_hash(h: str) -> bool:
    return not h.isdigit() and any(c.isalpha() for c in h)


cache: dict[str, bool] = {}
def commit_exists(h: str) -> bool:
    if h in cache:
        return cache[h]
    ok = False
    for r in REPOS:
        p = subprocess.run(["git", "-C", str(r), "cat-file", "-e", h + "^{commit}"],
                           capture_output=True)
        if p.returncode == 0:
            ok = True
            break
    cache[h] = ok
    return ok


tot_h = tot_hbad = tot_p = tot_pbad = 0
rows = []
for f in sorted(MEM.glob("*.md")):
    txt = FM_RE.sub("", f.read_text(encoding="utf-8", errors="replace"))
    hashes = {h for h in HASH_RE.findall(txt) if looks_like_hash(h)}
    bad_h = sorted(h for h in hashes if not commit_exists(h))
    paths = set()
    for m in PATH_RE.findall(txt):
        m = m.rstrip(".,;:)]}」』**")
        if len(m) > 6 and "\\" in m:
            paths.add(m)
    bad_p = sorted(p for p in paths if not Path(p).exists())
    tot_h += len(hashes); tot_hbad += len(bad_h)
    tot_p += len(paths);  tot_pbad += len(bad_p)
    if bad_h or bad_p:
        rows.append((f.name, len(hashes), bad_h, len(paths), bad_p))

print(f"扫描 {len(list(MEM.glob('*.md')))} 个记忆文件")
print(f"commit 哈希:{tot_h} 个引用 / {tot_hbad} 个在两个仓里都找不到")
print(f"绝对路径  :{tot_p} 条引用 / {tot_pbad} 条现在不存在")
print()
for name, nh, bh, np_, bp in rows:
    print(f"── {name}")
    if bh:
        print(f"     ⛔ 找不到的 commit({len(bh)}/{nh}): {', '.join(bh[:8])}{' …' if len(bh) > 8 else ''}")
    if bp:
        print(f"     ⚠ 不存在的路径({len(bp)}/{np_}):")
        for p in bp[:6]:
            print(f"        {p}")
        if len(bp) > 6:
            print(f"        … 共 {len(bp)} 条")
print()
print("⚠ 这个审计的边界(说清楚,别当成『全都没问题』):")
print("   · 只能查【引用的东西存不存在】,查不出【记反了】——post_s97b 那条是语义错,它路径存在、哈希存在。")
print("   · 路径不存在 ≠ 记错:scratchpad / 临时产物 / 已按计划清理的目录本来就会消失。")
