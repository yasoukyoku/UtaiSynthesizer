# -*- coding: utf-8 -*-
"""PreToolUse 钩子:碰到登记过的区域时,把「动这块之前必读什么」**打到脸上**。

⛔ 为什么需要它(用户 2026-08-11 原话):「**钩子存在但是不读 这个问题你也想想办法吧...
   每次出这个问题的后果都挺灾难**」。

已经发生过两次:
  · S120 —— 没读 MEMORY.md 里明写的「动训练侧前必读 S75+S76」,用户提醒后才补读;
  · S135 —— 拿「S134 已经把那 47 条销过账」**顶替**了「去读以前动完训练侧之后的核验经验」。
    第二次更隐蔽:钩子不是被忘了,是被一份**看起来已经覆盖了**的产物挡住了。

**记忆层解决不了它。** 失效点不在「我不知道有这条规矩」,而在于:
  ⑴ 钩子是一行文本,和另外一万九千字符躺在同一份索引里;
  ⑵ 触发时刻(动手改文件)与阅读时刻(开工读索引)隔着整整一场会话;
  ⑶ 没有任何东西在动手那一刻拦一下。
⇒ 唯一可靠的形状是**让工具链在那一刻打给我**,而不是指望我想起来。

行为:
  · 只在 Edit / Write / NotebookEdit 之前跑,读 stdin 的 tool_input.file_path;
  · 命中 scripts/memory_hooks.json 里某个区的前缀 ⇒ 把该区的必读清单与硬规矩注入上下文;
  · ⛔ **每个 session × 每个区只打一次** —— 打多了必然被调成噪音,然后被无视。
    这条不是为了省字,是为了让它**保持有效**。
  · 一律 fail-open:钩子自己出任何问题都不许挡住工具调用(它是提醒,不是闸)。
    但**出问题时会在 systemMessage 里说出来**,不静默死掉。

自检:`python scripts/memory_hook_guard.py --selftest`
"""
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONF = os.path.join(HERE, "memory_hooks.json")
STATE_ROOT = os.path.join(tempfile.gettempdir(), "claude_memory_hook_fired")


def _emit(context=None, note=None):
    """fail-open:永远退出 0,永远不挡工具调用。"""
    out = {}
    if context:
        out["hookSpecificOutput"] = {
            "hookEventName": "PreToolUse",
            "additionalContext": context,
        }
        out["systemMessage"] = note or "记忆钩子:这块有必读项"
    elif note:
        out["systemMessage"] = note
    if out:
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.exit(0)


def _rel(path):
    """把绝对路径变成仓库相对、正斜杠、小写的形式,好做前缀匹配。"""
    try:
        p = os.path.abspath(path)
        if os.path.commonprefix([p.lower(), REPO.lower()]) == REPO.lower():
            p = os.path.relpath(p, REPO)
    except Exception:                                   # noqa: BLE001
        p = path or ""
    return p.replace("\\", "/").lstrip("./").lower()


def match_area(rel, areas):
    for a in areas:
        for m in a.get("match", []):
            if rel.startswith(m.lower()):
                return a
    return None


def render(area, memory_dir):
    lines = []
    lines.append("⛔ 记忆钩子 [%s] —— %s" % (area["id"], area.get("headline", "")))
    lines.append("")
    lines.append("**动这块之前必读**(路径是绝对的,直接 Read):")
    for f in area.get("must_read", []):
        p = os.path.join(memory_dir, f)
        mark = "" if os.path.isfile(p) else "   ⚠ 这份不在了,去 MEMORY.md 找它搬去哪了"
        lines.append("  · %s%s" % (p, mark))
    rules = area.get("hard_rules", [])
    if rules:
        lines.append("")
        lines.append("**这一区已经买过血的教训**:")
        for r in rules:
            lines.append("  %s" % r)
    lines.append("")
    lines.append("⛔ 这条提醒**每个 session 每个区只出现一次**。"
                 "别把它读成『看过就算』—— 它要求的是**打开、逐条对着今天的代码回答**,"
                 "而不是『我记得那篇说过什么』。"
                 "(S135 血训:拿『某一场已经销过账』顶替了『去读核验经验』——"
                 "**销账是结论,经验是做法,不能互相顶替**。)")
    return "\n".join(lines)


def main():
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as e:                              # noqa: BLE001
        _emit(note="记忆钩子读不动 stdin(%s)—— 已放行,但它今天没在保护你" % e)

    tool = payload.get("tool_name", "")
    if tool not in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        _emit()

    fp = (payload.get("tool_input") or {}).get("file_path") or ""
    if not fp:
        _emit()

    try:
        conf = json.loads(io.open(CONF, encoding="utf-8").read())
    except Exception as e:                              # noqa: BLE001
        _emit(note="记忆钩子读不动 %s(%s)—— 已放行,但它今天没在保护你" % (CONF, e))

    area = match_area(_rel(fp), conf.get("areas", []))
    if area is None:
        _emit()

    # 每个 session × 每个区只打一次
    sid = str(payload.get("session_id") or "nosession")
    safe = "".join(c for c in sid if c.isalnum() or c in "-_")[:64] or "nosession"
    d = os.path.join(STATE_ROOT, safe)
    marker = os.path.join(d, area["id"] + ".fired")
    try:
        if os.path.isfile(marker):
            _emit()
        os.makedirs(d, exist_ok=True)
        with open(marker, "w") as f:
            f.write(fp)
    except Exception:                                   # noqa: BLE001
        pass    # 写不了标记就每次都打 —— 宁可吵,不可漏

    _emit(context=render(area, conf.get("memory_dir", "")),
          note="⛔ 记忆钩子 [%s]:动这块之前有必读项" % area["id"])


def _selftest():
    """⛔ 一条从没被执行过的分支就是一条空判据 —— 这里逐条真的触发。"""
    conf = json.loads(io.open(CONF, encoding="utf-8").read())
    areas = conf["areas"]
    fails = []

    cases = [
        ("converter/verify/training/gate0_compare.py", "gate"),
        (os.path.join(REPO, "converter", "verify", "training", "x.py"), "gate"),
        ("training/utai_train/rvc/train.py", "training-py"),
        ("src-tauri/src/training/trun.rs", "training-rs"),
        ("src/lib/training/foo.ts", "training-ui"),
        ("scripts/g2p_rulers/de/x.py", "dictline"),
        ("scripts/release.ps1", "release"),
        ("src/components/PianoRoll.tsx", None),
        ("README.md", None),
    ]
    for path, want in cases:
        got = match_area(_rel(path), areas)
        gid = got["id"] if got else None
        if gid != want:
            fails.append("%s -> %s(应为 %s)" % (path, gid, want))
        else:
            print("  ok   %-46s -> %s" % (path, gid))

    # 每一份登记的必读文件都必须真的在(名单能静默变空 = 没有名单)
    md = conf["memory_dir"]
    for a in areas:
        for f in a["must_read"]:
            p = os.path.join(md, f)
            if not os.path.isfile(p):
                fails.append("[%s] 登记的必读文件不在:%s" % (a["id"], p))
    print("  ok   全部登记的必读文件都在场" if not any("必读文件不在" in x for x in fails)
          else "  FAIL 有登记的必读文件不在场")

    # 渲染不许炸
    for a in areas:
        try:
            render(a, md)
        except Exception as e:                          # noqa: BLE001
            fails.append("[%s] render 炸了:%s" % (a["id"], e))

    print()
    if fails:
        for f in fails:
            print("  FAIL %s" % f)
        print("memory_hook_guard 自检: FAILED(%d)" % len(fails))
        return 1
    print("memory_hook_guard 自检: ALL OK(%d 个区)" % len(areas))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
