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

⛔⛔ **S138 实测:在此之前这个钩子在本机【从来 fire 不出来】。**
   本机控制台/管道的默认编码是 **cp932**,而 `_emit` 写的是
   `json.dumps(..., ensure_ascii=False)`,内容全是中文与 ⛔/★ 这类符号
   ⇒ 每一次命中都是 `UnicodeEncodeError` + **退出码 1**,而它承诺的是 fail-open 退 0。
   ⇒ **这套专门为「钩子存在但不读」造的机制,自己是哑的**,而且哑得没有任何症状
      (PreToolUse 的非零退出是非阻塞的,工具照常跑,清单就是不出现)。
   ⇒ 修法两层,缺一不可:
     ① 进程一启动就把 stdout/stderr 钉成 UTF-8(下面第一件事);
     ② `_emit` 自己**不许有任何一条能抛异常的路径** —— 编码失败就退回
        `ensure_ascii=True`,再失败就只发一句纯 ASCII 的 systemMessage,
        最后无论如何 `os._exit(0)`。
   ⇒ 自检里配了**阴性对照**:把 stdout 强行换成 cp932 再 emit 一次,必须仍然成功。
"""
import io
import json
import os
import re
import sys
import tempfile
import time

# ① ⛔ 必须是模块里的第一件事。本机控制台是 cp932,而这个钩子的全部产出都是中文。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:  # noqa: BLE001  py<3.7 / 已被替换的流
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONF = os.path.join(HERE, "memory_hooks.json")
STATE_ROOT = os.path.join(tempfile.gettempdir(), "claude_memory_hook_fired")


def _emit(context=None, note=None, deny=None):
    """fail-open:永远退出 0,永远不挡工具调用。

    ⛔ S138:这个函数**此前有一条能抛异常的路径**,而它是本模块唯一的出口 ——
       cp932 的 stdout 写不出中文 ⇒ UnicodeEncodeError ⇒ 退出码 1 ⇒
       「fail-open」这句承诺不成立,而且是**静默**不成立(工具照常跑,清单不出现)。
       ⇒ 现在写出去这一步有三级退路,而且**最后一定 `os._exit(0)`**。

    ⚠ `deny=` 是**唯一**一条会挡住工具的路(S157 的在途渲染闸)。
       「fail-open」这个词在这里仍然成立:出错的时候走的是 `context`/`note` 那两条,
       只有**确认有一个活着的在途标记**时才会 deny。
    """
    out = {}
    if deny:
        out["hookSpecificOutput"] = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": deny,
        }
        out["systemMessage"] = note or "⛔ 有渲染在途 —— 这一笔编辑会毁掉臂之间的二进制身份"
        _write_json(out)
        sys.stdout.flush()
        os._exit(0)
    if context:
        out["hookSpecificOutput"] = {
            "hookEventName": "PreToolUse",
            "additionalContext": context,
        }
        out["systemMessage"] = note or "记忆钩子:这块有必读项"
    elif note:
        out["systemMessage"] = note
    if out:
        _write_json(out)
    # ⛔ os._exit 而不是 sys.exit:后者靠异常展开,而解释器退出时若还要 flush 一个
    #    编码不兼容的缓冲区,同样会以非零收场。
    sys.stdout.flush()
    os._exit(0)


def _write_json(out):
    """三级退路,任何一级都不许把异常放出去。"""
    try:
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        return
    except Exception:  # noqa: BLE001
        pass
    try:  # ② 退成 \uXXXX 转义 —— 纯 ASCII,任何编码都写得出
        sys.stdout.write(json.dumps(out, ensure_ascii=True))
        return
    except Exception:  # noqa: BLE001
        pass
    try:  # ③ 连那样都不行:至少让它说一句话,别静默死掉
        sys.stdout.write(
            '{"systemMessage":"memory hook: could not encode its own output '
            '(see scripts/memory_hook_guard.py)"}')
    except Exception:  # noqa: BLE001
        pass


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


# ── S157:在途渲染闸 ──────────────────────────────────────────────────────────
#
# ⛔ 与上面那七个「必读区」**不是一回事**,三处都不同:
#   ⑴ 它**每次都打**(必读区每个 session 每区只打一次);
#   ⑵ 它是 **deny**,不是提醒 —— 它挡的不是「你可能没读过某篇」,而是
#      「你正在毁掉一个几小时才能重来一次、而且已经跑掉一半的东西」;
#   ⑶ 它也管 **Bash**(bypass 模式下我大部分改动是 sed/heredoc 走的,
#      只挡 Edit/Write 等于挡了个寂寞 —— 那正是「一条闸在它最常被绕过的路上失明」)。
#
# 标记由 `scripts/render_guard.py begin` 写、`end` 撤;配置在 `memory_hooks.json`
# 的 `render_in_flight` 段。

#: Bash 命令里出现这些 = 有写意图。⚠ 故意**不**收裸的 `>` / `>>`(整天在往日志重定向),
#: 只收重定向到受保护前缀的那种。
_WRITEISH = [
    r"\bsed\s+-i\b", r"\btee\b", r"\bcp\b", r"\bmv\b", r"\brm\b", r"\bpatch\b",
    r"\btouch\b", r"\btruncate\b", r"\bdd\b", r"\bninja\b", r"\bcargo\s+fix\b",
    r"\bopen\s*\(", r"\.write\s*\(", r"\bwritelines\b", r"\bwrite_text\b",
    r"Set-Content", r"Out-File", r"Add-Content", r"New-Item",
    r">>?\s*[\"']?(?:\./)?(?:src-tauri|src)/",
]

#: 在途时**无条件**拒的 Bash 命令(不用提到任何受保护文件也一样危险)。
_ALWAYS_DENY = r"\bgit\s+(checkout|restore|apply|stash|reset|clean|revert|merge|rebase|pull|cherry-pick|switch)\b"

_PATH_TOKEN = re.compile(r"[A-Za-z0-9_./\\:-]+")


def flight_conf(conf):
    return conf.get("render_in_flight") or {}


def flight_dir(conf):
    d = flight_conf(conf).get("dir") or "../TESTING/RENDER_IN_FLIGHT"
    return os.path.normpath(os.path.join(REPO, d))


def live_markers(conf):
    """[(路径, dict, 年龄小时)] —— 陈标记也返回,由调用方决定读法。"""
    out = []
    try:
        names = sorted(os.listdir(flight_dir(conf)))
    except OSError:
        return out
    for n in names:
        if not n.endswith(".json"):
            continue
        p = os.path.join(flight_dir(conf), n)
        try:
            m = json.loads(io.open(p, encoding="utf-8").read())
        except Exception:                               # noqa: BLE001
            m = {"label": "<读不动>"}
        try:
            age = (time.time() - os.path.getmtime(p)) / 3600.0
        except OSError:
            age = 0.0
        out.append((p, m, age))
    return out


def is_guarded(rel, conf):
    """这个仓库相对路径改了会不会让 cargo 重编。`rel` 已经过 `_rel()`(小写、正斜杠)。"""
    for g in flight_conf(conf).get("guarded", []):
        pre = (g.get("prefix") or "").lower()
        if not pre or not rel.startswith(pre):
            continue
        sufs = [s.lower() for s in (g.get("suffixes") or [])]
        if not sufs or any(rel.endswith(s) for s in sufs):
            return True
    return False


def flight_hits(tool, payload, conf):
    """在途时这一笔工具调用碰到了什么。返回 (原因串, [受保护路径])(没碰到 ⇒ (None, []))。"""
    if tool == "Bash":
        cmd = (payload.get("tool_input") or {}).get("command") or ""
        if re.search(_ALWAYS_DENY, cmd):
            return ("这条命令会动工作树/HEAD", [])
        hits = []
        for t in _PATH_TOKEN.findall(cmd):
            r = _rel(t.strip("\"'"))
            if is_guarded(r, conf) and r not in hits:
                hits.append(r)
        if not hits:
            return (None, [])
        for pat in _WRITEISH:
            if re.search(pat, cmd):
                return ("这条命令看起来要写 " + ", ".join(hits[:3]), hits)
        return (None, hits)     # 只是读它 —— 放行
    fp = (payload.get("tool_input") or {}).get("file_path") or ""
    r = _rel(fp)
    return (("要改 " + r, [r]) if is_guarded(r, conf) else (None, []))


def ear_conf(conf):
    return conf.get("ear_judgement") or {}


def wav_seconds(path):
    """读 WAV 头算秒数。⛔ 读不动就返回 None —— 这条闸永远不许因为读不动一个文件而挡人。"""
    try:
        import wave
        with wave.open(path, "rb") as w:
            r = w.getframerate()
            return (w.getnframes() / float(r)) if r else None
    except Exception:                                   # noqa: BLE001
        return None


def render_ear(conf, why, extra=None):
    e = ear_conf(conf)
    lines = ["⛔⛔ 耳判交付闸 —— " + why, ""]
    lines.append(e.get("headline", ""))
    lines.append("")
    for r in e.get("hard_rules", []):
        lines.append("  " + r)
    if extra:
        lines.append("")
        lines += extra
    lines += [
        "",
        "**要做的是**:把**整曲**那两条臂直接交出去(它们已经渲好了,就在 `UTAI_MG_OUTDIR` 里),",
        "   `cp` 成看得懂的名字放进 `TESTING\\s162_耳判\\<轮次>\\`,然后 SendUserFile 整曲。",
        "⛔ 别把这条读成噪音:这条规矩**已经被记过两次而我仍然犯了** ——",
        "   失效点不是不知道,是动手那一刻没有东西拦我。",
    ]
    return "\n".join(lines)


def render_flight(reason, live, conf):
    stale = float(flight_conf(conf).get("stale_hours") or 12)
    lines = ["⛔⛔ 有整曲渲染在途,而 " + reason + " —— 改它会让 cargo 重编,",
             "   于是**这几条臂就不再是同一个二进制**,已经跑掉的部分全部作废。",
             "   (S156 血训 #6:那次是「在等后台渲染的时候顺手加了一行注释」。",
             "    ⭐ 失效点不是不知道规矩,是后台任务让人忘了它还在跑。)",
             ""]
    for p, m, age in live:
        n = len(m.get("stamps") or [])
        tail = "   ⚠ 已经 %.1f h,超过 %.0f h 的陈标记线 —— 很可能是上一次渲染没收干净" % (age, stale) \
            if age > stale else ""
        lines.append("   在途:label=%s pid=%s 起于 %s 已跑 %.2f h stamps=%d%s"
                     % (m.get("label"), m.get("pid"), m.get("started"), age, n, tail))
    lines += [
        "",
        "**要么等它跑完,要么先确认它已经死了再撤标记**:",
        "   python scripts/render_guard.py status",
        "   python scripts/render_guard.py end --marker \"<上面那个路径>\"   # 正常收工(会顺便量二进制身份)",
        "   rm \"<上面那个路径>\"                                            # 确认渲染已死时才用",
        "",
        "⛔ 别把这条读成噪音:它只在有活标记时出现,而 `end` 一跑它立刻消失。",
    ]
    return "\n".join(lines)


def split_must_read(entry):
    """`must_read` 一条 = `<文件名>` 或 `<文件名> —— <为什么要读它>`。

    ⛔ S160 查出来的真缺陷:S159 起往这里加了带说明的条目,而两处消费点都拿**整串**当文件名
    ⇒ 自检报「登记的必读文件不在」,而**渲染出来的钩子会对那两份最重要的交接文档打
    「⚠ 这份不在了,去 MEMORY.md 找它搬去哪了」**。这块的存在意义就是在动手那一刻把名单打出来 ——
    名单指着不存在的文件,比没有名单更坏。⇒ 说明留着(它有用),解析这一侧认它。
    ⚠ 分隔符只认全角破折号 `——`,前后各一个空格;文件名里不会有它。
    """
    if isinstance(entry, str) and " —— " in entry:
        name, why = entry.split(" —— ", 1)
        return name.strip(), why.strip()
    return (entry or "").strip(), ""


def render(area, memory_dir):
    lines = []
    lines.append("⛔ 记忆钩子 [%s] —— %s" % (area["id"], area.get("headline", "")))
    lines.append("")
    lines.append("**动这块之前必读**(路径是绝对的,直接 Read):")
    for f in area.get("must_read", []):
        name, why = split_must_read(f)
        p = os.path.join(memory_dir, name)
        mark = "" if os.path.isfile(p) else "   ⚠ 这份不在了,去 MEMORY.md 找它搬去哪了"
        lines.append("  · %s%s%s" % (p, ("  —— " + why) if why else "", mark))
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
    # ⚠ Bash 只走「在途渲染闸」那一段:那七个必读区是按 file_path 登记的,
    #   而在途闸恰恰必须管 Bash(bypass 模式下大部分改动是 sed / heredoc 走的)。
    if tool not in ("Edit", "Write", "NotebookEdit", "MultiEdit", "Bash", "SendUserFile"):
        _emit()

    try:
        conf = json.loads(io.open(CONF, encoding="utf-8").read())
    except Exception as e:                              # noqa: BLE001
        _emit(note="记忆钩子读不动 %s(%s)—— 已放行,但它今天没在保护你" % (CONF, e))

    # ⛔⛔ 在途渲染闸(S157)—— 排在必读区之前,而且它是唯一一条会 deny 的路。
    live = live_markers(conf)
    if live:
        reason, _hits = flight_hits(tool, payload, conf)
        if reason:
            _emit(deny=render_flight(reason, live, conf),
                  note="⛔⛔ 有整曲渲染在途:%s —— 已拒。撤标记的命令在正文里" % reason)

    if tool == "Bash":
        cmd = (payload.get("tool_input") or {}).get("command") or ""
        # ⛔ 用**动作位正则**,不是裸子串:第一版拿子串匹配,当场拦住了我
        #    「往记忆文件里写这条规矩本身」的那条命令 —— 那几个词在散文里纯属正常。
        #    一条整天误报的闸会被调成噪音然后被无视(本模块 `_how` 里就写着这句)。
        for pat in ear_conf(conf).get("deny_regex", []):
            try:
                if re.search(pat, cmd):
                    _emit(deny=render_ear(conf, "这条命令在**切片段**(命中 `%s`)" % pat),
                          note="⛔⛔ 耳判交付闸:别切片段,给整曲 —— 已拒")
            except re.error:
                pass    # 正则写坏了不许把工具挡死 —— fail-open
        _emit()

    if tool == "SendUserFile":
        files = (payload.get("tool_input") or {}).get("files") or []
        lo = float(ear_conf(conf).get("min_seconds") or 60)
        short = []
        for f in files:
            if not str(f).lower().endswith((".wav", ".flac", ".mp3")):
                continue
            d = wav_seconds(str(f))
            if d is not None and d < lo:
                short.append("   · %.1f s  %s" % (d, f))
        if short:
            _emit(context=render_ear(conf, "要交的音频里有**短于 %.0f 秒**的" % lo,
                                     ["**这几条**:"] + short),
                  note="⛔ 耳判交付闸:交的是片段,不是整曲")
        _emit()

    fp = (payload.get("tool_input") or {}).get("file_path") or ""
    if not fp:
        _emit()

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
            p = os.path.join(md, split_must_read(f)[0])
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

    # ⛔⛔ S138 的阴性对照:**把 stdout 强行换成本机真实的 cp932,emit 必须仍然成功。**
    #    这条不是形式 —— 在加它之前,这个钩子在本机每一次命中都是 UnicodeEncodeError
    #    + 退出码 1,而症状是「什么都没发生」。⇒ 一条没被证明会在敌对编码下活下来的
    #    fail-open 承诺,就是一条空承诺。
    for enc in ("cp932", "ascii"):
        buf = io.BytesIO()
        try:
            fake = io.TextIOWrapper(buf, encoding=enc, errors="strict")
            real, sys.stdout = sys.stdout, fake
            try:
                _write_json({"systemMessage": "⛔ 记忆钩子 [gate]:必读 · 星★ · 箭⇒"})
                fake.flush()
            finally:
                sys.stdout = real
            payload = buf.getvalue().decode(enc)
            obj = json.loads(payload)
            assert "systemMessage" in obj, "emit 出来的不是合法 JSON"
            print("  ok   stdout=%-6s 下 emit 仍然成功(%d 字节)" % (enc, len(payload)))
        except Exception as e:                          # noqa: BLE001
            fails.append("stdout=%s 下 emit 失败:%s: %s" % (enc, type(e).__name__, e))

    # ⛔⛔ S157 在途渲染闸 —— **整条走子进程**,因为这条闸唯一的失效方式是
    #    「JSON 的形状不对 ⇒ 宿主读不出 deny ⇒ 什么都没发生」,而那在进程内测不到。
    #    ⇒ 这里真的把钩子当钩子跑一遍(stdin 喂 payload,stdout 收 JSON)。
    import subprocess

    ME = os.path.abspath(__file__)
    fdir = flight_dir(conf)
    marker = os.path.join(fdir, "selftest_%d.json" % os.getpid())
    had_dir = os.path.isdir(fdir)

    def hook(payload):
        p = subprocess.run([sys.executable, ME], input=json.dumps(payload),
                           capture_output=True, text=True, encoding="utf-8")
        if p.returncode != 0:
            return {"__rc": p.returncode, "__err": p.stderr[-300:]}
        try:
            return json.loads(p.stdout) if p.stdout.strip() else {}
        except Exception as e:                          # noqa: BLE001
            return {"__badjson": str(e), "__raw": p.stdout[:300]}

    def denied(out):
        return ((out.get("hookSpecificOutput") or {}).get("permissionDecision")) == "deny"

    def ed(path):
        return {"tool_name": "Edit", "session_id": "selftest", "tool_input": {"file_path": path}}

    def sh(cmd):
        return {"tool_name": "Bash", "session_id": "selftest", "tool_input": {"command": cmd}}

    GUARDED = "src-tauri/src/inference/vocal_range.rs"
    try:
        # ⑴ 阴性对照:**没有**在途标记时,同一笔编辑必须放行。
        #    (缺这条,「命中就 deny」可能只是「永远 deny」。)
        if denied(hook(ed(GUARDED))):
            fails.append("⛔ 没有在途标记时也 deny 了 —— 这条闸恒真,等于把编辑关死")
        else:
            print("  ok   在途闸:没有标记时放行")

        os.makedirs(fdir, exist_ok=True)
        io.open(marker, "w", encoding="utf-8").write(json.dumps(
            {"label": "selftest", "pid": os.getpid(), "started": "selftest", "stamps": []}))

        cases = [
            (ed(GUARDED), True, "Edit 受保护的 .rs"),
            (ed(os.path.join(REPO, "src-tauri", "Cargo.toml")), True, "Edit 绝对路径的 Cargo.toml"),
            (ed("src/lib/vocalNotes.ts"), True, "Edit 被 include_str! 编进去的 .ts"),
            (ed("README.md"), False, "Edit 一个编不进二进制的文件"),
            (ed("scripts/render_guard.py"), False, "Edit scripts 下的 python"),
            (sh("grep -n add_bell src-tauri/crates/utai-dsp/src/psola.rs"), False, "只读地 grep"),
            (sh("sed -i 's/a/b/' " + GUARDED), True, "sed -i 改它"),
            (sh("cat > %s <<'EOF'\nx\nEOF" % GUARDED), True, "重定向覆盖它"),
            (sh("git checkout -- ."), True, "动工作树(没提任何受保护路径)"),
            (sh("python x.py > /tmp/out.txt"), False, "重定向到无关文件"),
            (sh("cargo test -p utai --lib"), False, "跑测试本身不该被拦"),
        ]
        for payload, want, why in cases:
            out = hook(payload)
            if "__rc" in out or "__badjson" in out:
                fails.append("在途闸子进程异常(%s):%s" % (why, out))
                continue
            got = denied(out)
            if got != want:
                fails.append("⛔ 在途闸 %s —— 期望 deny=%s 实际 %s" % (why, want, got))
            else:
                print("  ok   在途闸:%-34s deny=%s" % (why, got))

        # ⑵ 变异:把 guarded 前缀改成一个不存在的目录 ⇒ 承重那条必须变绿(= 判据不空)
        mutated = json.loads(io.open(CONF, encoding="utf-8").read())
        mutated["render_in_flight"]["guarded"] = [{"prefix": "no/such/dir/", "suffixes": [".rs"]}]
        bak = io.open(CONF, encoding="utf-8").read()
        try:
            io.open(CONF, "w", encoding="utf-8").write(json.dumps(mutated, ensure_ascii=False))
            if denied(hook(ed(GUARDED))):
                fails.append("⛔⛔ 把 guarded 名单换成一个假目录之后照样 deny —— "
                             "说明 deny 不是名单给的,这条闸是空的")
            else:
                print("  ok   在途闸变异:名单换假目录 ⇒ 不再 deny(判据不空)")
        finally:
            io.open(CONF, "w", encoding="utf-8").write(bak)

        # ⑶ deny 的正文必须真的写着怎么撤标记(否则第二天一定被人 rm -rf 整个目录)
        body = (hook(ed(GUARDED)).get("hookSpecificOutput") or {}).get("permissionDecisionReason", "")
        if "render_guard.py" not in body or "selftest" not in body:
            fails.append("⛔ deny 正文里既要有撤标记的命令、也要点名是哪个标记")
        else:
            print("  ok   在途闸:deny 正文带撤标记命令 + 点名标记")
    finally:
        try:
            os.remove(marker)
        except OSError:
            pass
        if not had_dir:
            try:
                os.rmdir(fdir)
            except OSError:
                pass

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
