# -*- coding: utf-8 -*-
"""S157 —— 长时渲染「在途」标记 + **二进制身份**的一条测量。

⛔ 为什么存在(S156 血训 #6,而 S155 刚记过同一条):
   多条臂的整曲渲染排在后台跑,我在等它的时候顺手给 `psola.rs` 加了一行注释 ——
   **注释也会让 cargo 重编,那就破坏了「N 条臂同一个二进制」这个契约**。
   ⭐ 失效点不是「我不知道规矩」,而是**后台任务让我忘了它还在跑**。
   ⇒ 药只能下在动手那一刻:① 在途时**拦住**编辑(`memory_hook_guard.py` 读本模块写的标记);
     ② 渲染结束时**量一次**,而不是论证一次。

⛔⛔ 这里的第二件事比第一件重要。S156 那次事后是这样收场的:
   「HEAD 没变 + 树干净」——**而那个检查看不见「改了又改回去」**:
   dirty 计数回到 0、HEAD 一个字没动,可 cargo 已经重编过一遍了。
   ⇒ 本模块的判据**不看源码、只看产物**:`target/debug/deps/utai*` 的
   (文件名, 字节数, mtime_ns) 全集。重编必然新增或改写其中一项。
   ⭐ 这条是**观察那个事件本身**(重链接),不是观察它的代用品(树脏不脏)。

用法(shell)::

    # ⛔ `$GUARD` **必须是绝对路径**:渲染脚本几乎都会 `cd src-tauri`(workspace 在那儿),
    #    而 trap 是在【退出时的 cwd】里跑的。S157 第一次用它就踩了:相对路径 ⇒ EXIT 时
    #    python 报 "can't open file" ⇒ **标记没撤、身份没量**,而脚本自己打的是「ALL DONE」。
    #    ⇒ trap 里还要带一条 `|| echo`,否则 end 的非零退出会被 shell 吞掉。
    GUARD=/d/MyDev/Utai_v2-dev/scripts/render_guard.py
    M=$(python "$GUARD" begin --label s157) || exit 5
    trap 'python "$GUARD" end --marker "$M" || echo "⛔⛔ 身份没通过,读数不可比"' EXIT
    python "$GUARD" stamp --marker "$M" --tag pre
    ... 渲第一条臂 ...
    python scripts/render_guard.py stamp --marker "$M" --tag armA --log "$L/render_armA.log"
    ... 渲第二条臂 ...
    python scripts/render_guard.py stamp --marker "$M" --tag armB --log "$L/render_armB.log"
    # end 退 0 = 全部 stamp 的指纹逐字节相同 且 没有一条臂的日志里出现 Compiling

自检:``python scripts/render_guard.py --selftest``(⛔ 含变异,见 `_selftest`)
"""
import fnmatch
import glob
import hashlib
import io
import json
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:  # noqa: BLE001
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

#: 在途标记目录。⚠ 与 `memory_hook_guard.py` 的 `_flight_dir` **必须是同一个**,
#: 两边都从 `scripts/memory_hooks.json` 的 `render_in_flight` 段读,别各写各的。
CONF = os.path.join(HERE, "memory_hooks.json")

#: 产物指纹扫的是这些 glob(相对仓库根)。`cargo test -p utai --lib` 产出 `utai_lib-*.exe`,
#: 两个 crate 的 `--lib` 测试产出 `utai_dsp-*.exe` / `utai_stretch-*.exe`;`utai.exe` 是主程序。
#: ⚠ 故意包含**全部**(不只最新那个):重编会新增一个新哈希名的文件,
#: 只盯「最新那个」会在两次重编产出同名同尺寸时漏掉。
#: ⛔ 只扫**可执行产物**,不扫 `deps/*.o`(本机 17027 个,一次指纹要 5.4 s,而且会被
#:    完全无关的 crate 抖动)—— 任何一次重编都必然重链接出可执行文件,这一层已经够。
#:    ⚠ 自检里那三条阳性对照就是钉这句话的:新增文件 / 只改 mtime 都必须读出变化。
BIN_GLOBS = [
    "src-tauri/target/debug/deps/utai*.exe",
    "src-tauri/target/debug/deps/utai*.dll",
    "src-tauri/target/debug/utai.exe",
]

def _conf():
    try:
        return json.loads(io.open(CONF, encoding="utf-8").read())
    except Exception:  # noqa: BLE001
        return {}


def flight_dir(conf=None):
    """在途标记目录的绝对路径。默认 `<仓库上级>/TESTING/RENDER_IN_FLIGHT`。"""
    c = (conf if conf is not None else _conf()).get("render_in_flight") or {}
    d = c.get("dir") or "../TESTING/RENDER_IN_FLIGHT"
    return os.path.normpath(os.path.join(REPO, d))


def stale_hours(conf=None):
    c = (conf if conf is not None else _conf()).get("render_in_flight") or {}
    try:
        return float(c.get("stale_hours", 12))
    except Exception:  # noqa: BLE001
        return 12.0


def fingerprint(repo=REPO):
    """产物指纹 = 排序后的 (相对路径, 字节数, mtime_ns) 全集的 sha256 + 条目数。

    ⛔ 不读文件内容:一个 300 MB 的 debug 二进制读全量太贵,而**重链接必然动 mtime**,
       这条已经足够灵敏。⚠ 反过来它**过于**灵敏(任何人跑一次 cargo 都会动它)——
       那正是这条闸要抓的:渲染期间**任何**重编都破坏臂之间的可比性。
    """
    # ⚠ 用 scandir 而不是 glob:`deps/` 本机有 17 k 个文件,每个 glob 都要把整个目录
    #    重列一遍(实测 3 个 glob = 7.1 s,一次渲染要 stamp 六七回)。scandir 的
    #    DirEntry 在 Windows 上自带 stat,一次列举就够。
    bydir = {}
    for g in BIN_GLOBS:
        d, pat = os.path.split(os.path.join(repo, g.replace("/", os.sep)))
        bydir.setdefault(d, []).append(pat)
    rows = []
    for d, pats in bydir.items():
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                if not any(fnmatch.fnmatch(e.name, pat) for pat in pats):
                    continue
                try:
                    if not e.is_file():
                        continue
                    st = e.stat()
                except OSError:
                    continue
                rel = os.path.relpath(os.path.join(d, e.name), repo).replace("\\", "/")
                rows.append((rel, st.st_size, st.st_mtime_ns))
    rows.sort()
    h = hashlib.sha256()
    for r in rows:
        h.update(("%s|%d|%d\n" % r).encode("utf-8"))
    return {"sha256": h.hexdigest(), "n": len(rows)}


def scan_log(path):
    """一条臂的日志里有没有重编的痕迹。返回命中的行(去重,最多 5 条)。

    ⛔ 只认 `Compiling <crate>`。`Fresh` 不算(那正是「没重编」的证据),
       而 cargo 只在**真的重建**时打 Compiling。S156 就是靠这一行分开
       「arm C 没重编」与「arm D 重编了 2 次」的。
    """
    hits = []
    try:
        with io.open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("Compiling "):
                    if s not in hits:
                        hits.append(s)
                    if len(hits) >= 5:
                        break
    except Exception as e:  # noqa: BLE001
        return ["<日志读不动: %s>" % e]
    return hits


def live_markers(conf=None):
    """返回 [(标记路径, 内容 dict, 年龄小时)] —— **不**过滤陈标记,由调用方决定怎么读。"""
    d = flight_dir(conf)
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            m = json.loads(io.open(p, encoding="utf-8").read())
        except Exception:  # noqa: BLE001
            m = {"label": "<读不动>"}
        try:
            age = (time.time() - os.path.getmtime(p)) / 3600.0
        except OSError:
            age = 0.0
        out.append((p, m, age))
    return out


# ── 子命令 ────────────────────────────────────────────────────────────────────
def cmd_begin(args):
    d = flight_dir()
    os.makedirs(d, exist_ok=True)
    label = args.get("--label") or "render"
    safe = "".join(c for c in label if c.isalnum() or c in "-_") or "render"
    path = os.path.join(d, "%s_%d.json" % (safe, os.getpid()))
    m = {
        "label": label,
        "pid": os.getpid(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_epoch": time.time(),
        "cwd": os.getcwd(),
        "argv": sys.argv[1:],
        "stamps": [],
    }
    io.open(path, "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False, indent=1))
    sys.stderr.write("render_guard: 在途标记已写 %s\n" % path)
    sys.stdout.write(path)
    return 0


def cmd_stamp(args):
    path = args.get("--marker")
    if not path or not os.path.isfile(path):
        sys.stderr.write("render_guard: stamp 找不到标记 %r —— 不许静默继续\n" % path)
        return 5
    m = json.loads(io.open(path, encoding="utf-8").read())
    fp = fingerprint()
    row = {
        "tag": args.get("--tag") or "?",
        "at": time.strftime("%H:%M:%S"),
        "fp": fp,
        "compiled": scan_log(args["--log"]) if args.get("--log") else [],
    }
    m.setdefault("stamps", []).append(row)
    io.open(path, "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False, indent=1))
    sys.stderr.write(
        "render_guard: stamp %-10s fp=%s n=%d%s\n"
        % (row["tag"], fp["sha256"][:12], fp["n"],
           ("  ⛔ 这条臂里重编了:%s" % row["compiled"]) if row["compiled"] else "")
    )
    return 0


def cmd_end(args):
    path = args.get("--marker")
    rc = 0
    if not path or not os.path.isfile(path):
        sys.stderr.write("render_guard: end 找不到标记 %r\n" % path)
        return 5
    m = json.loads(io.open(path, encoding="utf-8").read())
    stamps = m.get("stamps") or []
    if len(stamps) < 2:
        sys.stderr.write(
            "render_guard: ⛔ 只有 %d 个 stamp —— 二进制身份【没有被量过】,"
            "别把这次收工读成『臂可比』\n" % len(stamps)
        )
        rc = 6
    fps = sorted({s["fp"]["sha256"] for s in stamps})
    if len(fps) > 1:
        sys.stderr.write("render_guard: ⛔⛔ 二进制身份被破坏 —— %d 个不同指纹:\n" % len(fps))
        for s in stamps:
            sys.stderr.write("    %-10s %s n=%d\n" % (s["tag"], s["fp"]["sha256"][:12], s["fp"]["n"]))
        rc = 7
    bad = [s for s in stamps if s.get("compiled")]
    if bad:
        sys.stderr.write("render_guard: ⛔⛔ 有臂在渲染中重编:\n")
        for s in bad:
            sys.stderr.write("    %-10s %s\n" % (s["tag"], s["compiled"]))
        rc = 7
    if rc == 0:
        sys.stderr.write(
            "render_guard: ✅ 二进制身份成立 —— %d 个 stamp 指纹全同 (%s),"
            "没有一条臂重编\n" % (len(stamps), fps[0][:12])
        )
    # ⛔ 标记**无论如何**都要撤掉:留一个陈标记会把编辑闸永久钉住,
    #    那种闸第二天一定被人绕过去,然后就再也不是闸了。
    keep = path + ".done"
    try:
        if os.path.exists(keep):
            os.remove(keep)
        os.rename(path, keep)
    except OSError:
        try:
            os.remove(path)
        except OSError:
            pass
    return rc


def cmd_status(args):
    live = live_markers()
    if not live:
        print("render_guard: 没有在途渲染")
        return 0
    for p, m, age in live:
        print("在途 %-24s label=%s pid=%s 已跑 %.2f h  stamps=%d"
              % (os.path.basename(p), m.get("label"), m.get("pid"), age, len(m.get("stamps") or [])))
    return 0


def _selftest():
    """⛔ 一条从没被执行过的分支就是一条空判据 —— 这里把每条出口都真的触发一次,
    并且**当场变异**(S155/S156 规矩:凡新加一条闸当场变异一次)。
    """
    import shutil
    import tempfile

    fails = []
    tmp = tempfile.mkdtemp(prefix="render_guard_selftest_")
    try:
        # ── ⑴ fingerprint 对 mtime 敏感、对「同内容改名」也敏感 ─────────────────
        fake = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(fake, "src-tauri", "target", "debug", "deps"))
        b = os.path.join(fake, "src-tauri", "target", "debug", "deps", "utai_lib-aaaa.exe")
        io.open(b, "wb").write(b"\x00" * 16)
        f1 = fingerprint(fake)
        if f1["n"] != 1:
            fails.append("fingerprint 没数到那个二进制(n=%d)" % f1["n"])
        if fingerprint(fake)["sha256"] != f1["sha256"]:
            fails.append("fingerprint 不稳定 —— 没改任何东西却读出两个值")
        # 阳性对照 ⒜:重编 = 换了个哈希名的新文件
        b2 = os.path.join(fake, "src-tauri", "target", "debug", "deps", "utai_lib-bbbb.exe")
        io.open(b2, "wb").write(b"\x00" * 16)
        if fingerprint(fake)["sha256"] == f1["sha256"]:
            fails.append("⛔ 新增一个 utai_lib-*.exe 指纹没变 —— 这条闸是空的")
        os.remove(b2)
        # 阳性对照 ⒝:同名同尺寸、只有 mtime 变(= 原地重链接)
        os.utime(b, ns=(0, 12345678901234567))
        if fingerprint(fake)["sha256"] == f1["sha256"]:
            fails.append("⛔ 只改 mtime 指纹没变 —— 『改了又改回去』这一类看不见")
        # 阴性对照:仓库外的文件不该被数进来
        io.open(os.path.join(fake, "src-tauri", "target", "debug", "deps", "other.exe"),
                "wb").write(b"\x00")
        if fingerprint(fake)["n"] != 1:
            fails.append("⛔ 指纹把不叫 utai* 的产物也数进来了")
        print("  ok   fingerprint:稳定 + 三条对照")

        # ── ⑵ scan_log 只认真的重编行 ────────────────────────────────────────
        lg = os.path.join(tmp, "arm.log")
        io.open(lg, "w", encoding="utf-8").write(
            "warning: unused\nFresh utai v0.11.0\ntest result: ok. 706 passed\n")
        if scan_log(lg):
            fails.append("⛔ scan_log 把 Fresh 读成了重编 —— 会天天误报,然后被无视")
        io.open(lg, "a", encoding="utf-8").write("   Compiling utai v0.11.0 (D:\\MyDev)\n")
        if not scan_log(lg):
            fails.append("⛔ scan_log 看不见真的 Compiling 行 —— 这条闸是空的")
        print("  ok   scan_log:阴性(Fresh)+ 阳性(Compiling)")

        # ── ⑶ begin / stamp / end 的三条出口都真的走一遍 ──────────────────────
        d = os.path.join(tmp, "flight")
        os.makedirs(d)
        mk = os.path.join(d, "t.json")
        io.open(mk, "w", encoding="utf-8").write(json.dumps({"label": "t", "stamps": []}))

        def _stamp(tag, fp_sha, compiled=None):
            m = json.loads(io.open(mk, encoding="utf-8").read())
            m["stamps"].append({"tag": tag, "at": "00:00:00",
                                "fp": {"sha256": fp_sha, "n": 1},
                                "compiled": compiled or []})
            io.open(mk, "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False))

        _stamp("pre", "aa")
        if cmd_end({"--marker": mk}) != 6:
            fails.append("⛔ 只有一个 stamp 时 end 没有报『没被量过』")
        os.rename(mk + ".done", mk)
        _stamp("armA", "aa")
        if cmd_end({"--marker": mk}) != 0:
            fails.append("⛔ 两个相同指纹的 stamp,end 却不通过")
        os.rename(mk + ".done", mk)
        _stamp("armB", "bb")
        if cmd_end({"--marker": mk}) != 7:
            fails.append("⛔⛔ 指纹变了 end 竟然放行 —— 这是这个模块唯一的承重判据")
        os.rename(mk + ".done", mk)
        m = json.loads(io.open(mk, encoding="utf-8").read())
        for s in m["stamps"]:
            s["fp"]["sha256"] = "aa"
        io.open(mk, "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False))
        m = json.loads(io.open(mk, encoding="utf-8").read())
        m["stamps"][-1]["compiled"] = ["Compiling utai v0.11.0"]
        io.open(mk, "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False))
        if cmd_end({"--marker": mk}) != 7:
            fails.append("⛔ 日志里有 Compiling 而指纹碰巧相同时,end 放行了")
        # end 之后标记必须不在了(陈标记会把编辑闸永久钉死)
        if os.path.exists(mk):
            fails.append("⛔ end 之后在途标记还在 —— 会把编辑闸永久钉住")
        print("  ok   end:1 个 stamp/指纹全同/指纹不同/日志有 Compiling 四条出口 + 标记已撤")

        # ── ⑷ 找不到标记时**不许**静默成功 ──────────────────────────────────
        if cmd_stamp({"--marker": os.path.join(tmp, "nope.json")}) == 0:
            fails.append("⛔ stamp 在标记不存在时退了 0 —— 「跑不起来」会被读成「通过」")
        if cmd_end({"--marker": os.path.join(tmp, "nope.json")}) == 0:
            fails.append("⛔ end 在标记不存在时退了 0")
        print("  ok   标记缺失时 stamp/end 都是非零")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if fails:
        for f in fails:
            print("  FAIL %s" % f)
        print("render_guard 自检: FAILED(%d)" % len(fails))
        return 1
    print("render_guard 自检: ALL OK")
    return 0


def main(argv):
    if "--selftest" in argv:
        return _selftest()
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    args = {}
    i = 0
    while i < len(rest):
        a = rest[i]
        if not a.startswith("--"):
            i += 1
            continue
        if i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            args[a] = rest[i + 1]
            i += 2
        else:
            args[a] = "1"
            i += 1
    table = {"begin": cmd_begin, "stamp": cmd_stamp, "end": cmd_end, "status": cmd_status}
    if cmd not in table:
        sys.stderr.write("render_guard: 不认识的子命令 %r(有 %s)\n" % (cmd, "/".join(table)))
        return 2
    return table[cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
