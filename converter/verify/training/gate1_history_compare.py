# -*- coding: utf-8 -*-
"""gate1 的**第二条判据**:今天的我方侧产物 vs 07-07/07-17 与 08-11 的历史读数。

    training/.venv/Scripts/python.exe converter/verify/training/run_gate1_chain.py all
    (跑器会把它作为每条链的第 5 段 `5_history` 自动跑掉;单独手敲要给 --chain 与 GATE1_T0)

gate1 自己那五条 compare 问的是「**今天**我们的训练循环 == 上游吗」。
这一条问的是**另一个问题**:「自上一次跑 gate1 以来的那些 commit,数值上动过什么吗」。
两条一起,才把「今天对」和「一直没变过」分开。

────────────────────────────────────────────────────────────────────────────
S140:它的前身 `TESTING\\s134_f7\\compare_vs_history.py` 整条链上**没有一处能红**
────────────────────────────────────────────────────────────────────────────
⑴ ⛔⛔⛔ **新鲜度守卫是一条单调恒真的谓词**。它比的是「live 的 mtime 是不是晚于备份的
   mtime」,而备份是 `copy2` 拷的、mtime 冻在 2026-07-07;live 自 2026-08-11 那一跑起
   **永远更新** ⇒ 这条分支再也不可能进。实测:今天**不跑任何一条链**直接执行它,
   四档全 `BITWISE-SAME`、`EXIT=0`。⇒ 换成跑器钉的 `GATE1_T0`。
   ⚠ **别只把 `<=` 改成别的比较** —— 问题不在符号,在于「参照物的 mtime 是一个永远不动
     的常数」,任何基于两侧 mtime 的比较都会退化成这样。
⑵ ⛔⛔ **缺件走静默绿**。四条 CASES 全部 SKIP 时 `rc` 一次都不被写 ⇒ 干净退 0。
   实测三种改法(备份根改名 / live 根改名 / 单档文件名写错)**全部 exit 0**,
   而 exit 0 会被读成「那些 commit 什么都没动」。
   ⚠ 最毒的一层:备份脚本自己的 REFUSING 文案教人「要重做请先手动改名」——照做一次,
     这道判据当场全空且仍然绿。
⑶ ⛔⛔ **地板是与真值无关的常数**(`steps×5`),而真实分量数是 6/7/10/2 ⇒
   实测 rvc 删掉**任意一个** loss 键剩恰好 150 = 地板(六个键全部过线);
   **sovits_v2 可以整键消失 5 个(十个分量的一半)仍打 `BITWISE-SAME`**。
   而 `only_new/only_old` 算的是**步号**、不是键,且**只打印、不改 rc**。
⑷ ⛔⛔ **NaN 隐形**:`if d > worst[0]` 对 NaN 恒为 False ⇒ NaN **计入对数**(帮着过地板)、
   **永不移动 worst** ⇒ 打 `BITWISE-SAME` 退 0。而 `json.loads('NaN')` 返回 `float('nan')`
   且 `isinstance(nan,(int,float))` 为 True ⇒ 它连 `:65-66` 那道非数值过滤都穿得过去。
⑸ ⛔ **后写覆盖吃掉整条记录**:`out[step] = losses` ⇒ diff 链 step 24 被文件末尾那条
   `losses:{}` 整个盖掉,**最后一个训练步的 loss 与最后一个 validation 值一个都没被比过**,
   而输出里没有任何字提示。⇒ 这里改成 `setdefault(step,{}).update(losses)`,
   diff 的可比对数从 25 涨到 **27**,最后一步回到判据里。
⑹ ⛔ **`EMPTY-COMPARISON` 无条件覆写 `DIFFERS`**(`if` 而不是两条独立判据)⇒
   又红又空的时候转录**只说「空」**,而「这一轮的数字和七月不一样了」这条唯一携带信息的
   读数被吃掉,人会去查夹具而不是去查那些 commit。
⑺ ⛔ 三种含义压成同一个 exit 1(NOT-YET-RUN / EMPTY-COMPARISON / DIFFERS),而在跑器
   那张表里 1 被单列为「★ 只有这一种是被测的东西不对」;缺件那一种反而退 0。
⑻ ⛔ 零 `run()` 兜底、零 `_say` 退路、零 `--selftest`、`main()` 模块级裸调、全仓零调用者。

⛔ 口径收窄(S139 交接要求,S140 实测坐实):S134 那句「BITWISE-SAME」的准确含义是
   「**除 `eta_secs` 外整份 jsonl 逐字节相同**」—— 四条 jsonl 的原始 sha256 **全不同**
   (rvc 7888/7892 · sovits 4616/4617 · v2 5034/5035 · diff 8577/8577 但内容不同),
   照字面做文件级对拍会得一条**假红**。
⛔ 而且**存的是下面那段归一化代码,不是那四个 sha** —— 「去掉 eta_secs」有无数种字节实现,
   两种实现算出来的 sha 不一样。代码在仓库里、有 git 历史,那才是可复算的载体。

⛔ 这一条**不覆盖**的(诚实边界,别当它覆盖了):
   * 历史侧的**输入身份在物理上不可知** —— 七月那批基线产出时 `gate1_input.identity.json`
     这套机制还不存在(S139 才建)⇒ 一条 DRIFT 说不清是「代码变了」还是「夹具变了」。
     ⇒ 每一跑都 `note_uncovered` 记这一笔,默认落 exit 3 / PASS-WITH-GAPS。
   * 它只看**我方侧**。「参照物(上游代码)自己有没有被改过」是另一条线。
   * rvc 与 sovits 的**原版侧七月 TB 根本不存在**(`baseline_backup\\rvc_orig_logs` 只有
     config+filelist、`sovits_orig_logs` 目录整个不存在)⇒ 那两条链的历史证据只有我方 jsonl。

自检:`gate1_history_compare.py --selftest`
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402
import gate1_vocoder_compare as VC                              # noqa: E402  —— 只借它的 REQUIRED_TAGS

GATE = "GATE1 HISTORY"
LIVE = r"D:\MyDev\TESTING\utai-v2-testing"

# ── 两个历史基线,回答**两个不同的问题**。
#    ⛔ `expect_sha` 是「这份参照物还是不是同一份东西」(`gate0_guard.declare_frozen` 的契约),
#      与下面那段归一化**没有关系** —— 它算的是原始字节。参照被换掉/被还原/指到别处必须红。
BASELINES = [
    ("july", r"D:\MyDev\TESTING\s134_f7\baseline_backup\_loose",
     "c8dce91188c7cb3327024702c82eddd25bf2c140a4c1f49ffde69d0e86a69a32",
     "2026-07-07 / 07-17 的 gate1 我方侧读数(S134 的 baseline_backup)"),
    ("s134", r"D:\MyDev\TESTING\s135_f7\backup_pre_gate0\root_files",
     "86af29fd9c91239574a52a96e6cb5c90abcda06dc7ed70d7fde91d928b763921",
     "2026-08-11 S134 那一跑的我方侧读数(S135 的 backup_pre_gate0)"),
]

# ── 登记的真值(2026-08-12 逐条实测,不是从别处抄的)。
#    ⛔ 地板必须是**这条链的真值** —— 前身用的是 `steps×5` 那个与真值无关的常数,
#      而真实分量数是 6/7/10/2。`steps` 直接转引 `gate1_guard.EXPECT`,不抄第二份。
CASES = {
    "rvc": dict(kind="jsonl", file="gate1_ours_steps.jsonl",
                keys=["d_total", "fm", "g_total", "gen", "kl", "mel"], pairs=180),
    "sovits": dict(kind="jsonl", file="gate1_sovits_ours_steps.jsonl",
                   keys=["d_total", "fm", "g_total", "gen", "kl", "lf0", "mel"], pairs=112),
    "sovits_v2": dict(kind="jsonl", file="gate1_sovits_v2_ours_steps.jsonl",
                      keys=["adv", "d_total", "fm", "g_total", "kl", "lf0", "mel",
                            "mel_am", "mel_ddsp", "spec_ddsp"], pairs=140),
    # ⚠ diff 每步只有一个 `loss`,另加三个边界步上的 `val` ⇒ 24×1 + 3 = 27。
    #   前身把它算成 25(step 24 被末行 `losses:{}` 盖掉了),而地板正好是 24 ⇒ 余量 1 对。
    "diff": dict(kind="jsonl", file="gate1_diff_ours_steps.jsonl",
                 keys=["loss", "val"], pairs=27),
    # ⚠ 声码器**没有 jsonl 步流**(两侧都从 TB 取)—— 但「没有历史对拍」这句话只对 jsonl
    #   这一种载体成立:两侧七月 TB 都躺在 `baseline_backup\voc_gate\` 里。
    #   ⛔⛔ 而它单靠数值**分不清臂、也分不清轮次**:四份声码器 TB(七月两侧 / S134 两侧)
    #      那 143 个值**两两全等,含自比** ⇒ 只有配上「live 侧必须是本轮产物(require_fresh)」
    #      与「历史侧 declare_frozen + expect_sha」这两条,它才携带信息。两条缺一不可。
    "vocoder": dict(kind="tb", pairs=143,
                    live=os.path.join(r"D:\MyDev\TESTING\gate1_vocoder", "ours", "gate1_voc",
                                      "lightning_logs", "lastest"),
                    july=os.path.join(r"D:\MyDev\TESTING\s134_f7\baseline_backup", "voc_gate",
                                      "ours", "gate1_voc", "lightning_logs", "lastest"),
                    # ⚠ 这是**整个目录**的 dirhash(events + 那个 3 字节的 `hparams.yaml`),
                    #   不是 events 单件的。S140 第一次真跑当场踩到:我量 sha 时只算了 events,
                    #   而 `declare_frozen(..., [""])` 无后缀过滤 ⇒ 它算的是整目录 ⇒ 判 exit 3
                    #   「冻结参照的内容变了」。⛔ **闸是对的,错的是登记值** —— 已复核
                    #   events 单件 sha 仍逐位等于 74a31d18e12fe368…(参照物没变)。
                    #   ⇒ 改登记整目录口径,顺带把 hparams.yaml 也钉住(更强)。
                    july_sha="eda6a6f53f1d7b0d2fb4522e5d5f4e39c5a1665a1b74f7a5225ea1385928f81e"),
}

# ⛔ 历史对拍量的是**同实现跨时间**,不是跨实现 ⇒ 容差是 **0.0**,不是 1e-6。
#    前身那条 `SAME<=1e-6` 是一条**没人挣来的容差带**:实测四条链今天的 max_rel 全是
#    `0.000e+00`,而移植是逐位确定的 ⇒ 只要动了就该是非零,1e-6 那一带**只会吞掉真信号**,
#    不可能吞掉噪声。而且它打印出来像绿、退出码也是绿(`rc` 一个字节都不动)。
HISTORY_TOL = 0.0

DROP_KEYS = ("eta_secs",)

# 两个基线根里那四份 jsonl(冻结参照的 expect_sha 就是按这个名单算的)
JSONL_FILES = [c["file"] for c in CASES.values() if c["kind"] == "jsonl"]


# ─────────────────────────────────────────────────────── 归一化(**这就是被存下来的口径**)
def normalize_jsonl(path, drop=DROP_KEYS):
    """逐行 json 解析 → 去掉 `drop` 里的键 → `sort_keys` 紧凑重序列化 → 拼回字节。

    ⛔ 这段代码**就是**「除 eta_secs 外逐字节相同」这句话的定义。别去存四个 sha:
       「去掉 eta_secs」有无数种字节实现,两种实现算出来的 sha 不一样,而 sha 存进记忆
       之后没有任何东西能把它和产生它的那段实现绑在一起。
    """
    buf = io.StringIO()
    n_line = n_drop = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n_line += 1
            d = json.loads(s)
            for k in drop:
                if k in d:
                    d.pop(k)
                    n_drop += 1
            buf.write(json.dumps(d, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")))
            buf.write("\n")
    return buf.getvalue().encode("utf-8"), n_line, n_drop


def merged_steps(path):
    """{step: {key: value}} —— **合并**同一个 step 号的多条记录,不是后写覆盖。

    ⛔ 前身是 `out[step] = losses or {}`(无条件覆盖、且没有 need_key 过滤)⇒ diff 链
       step 24 的最后一行是收尾行 `losses:{}`,把 `{loss, val}` **整个盖掉** ⇒ 那一步贡献
       0 对,而输出里没有一个字提示。⇒ 改成合并;并把合并之后仍然为空的步(各条链末尾那条
       forced 收尾 step,rvc 30 / sovits 17 / sovits_v2 14)显式剔除并报出来。
    """
    out = {}
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        d = json.loads(s)
        if d.get("type") != "step":
            continue
        out.setdefault(d["step"], {}).update(d.get("losses") or {})
    empty = sorted(s for s, v in out.items() if not v)
    for s in empty:
        del out[s]
    return out, empty


# ─────────────────────────────────────────────────────── 逐条判据
def _require(path, what):
    if not os.path.isfile(path) and not os.path.isdir(path):
        raise G1.GateUnrunnable(
            "%s 不在:%s\n"
            "       ⛔ 缺件**不是**「这一轮什么都没变」—— 前身在这里是 `print` + `continue`、"
            "`rc` 一个字节都不动 ⇒ 实测把备份根改个名就能拿到一个干净的 exit 0。" % (what, path))


def compare_jsonl(chain, cfg, t0, failures):
    live_p = os.path.join(LIVE, cfg["file"])
    _require(live_p, "本轮我方侧产物")
    # ⛔ **这一条就是那条死掉的守卫的替代品**:产物必须是**本轮**算出来的。
    #    ⚠ `suffixes` 是 endswith ⇒ 传完整文件名(gate0_guard.collect:191)。
    G1.require_fresh("live/%s" % chain, LIVE, [""], t0, minimum=1,
                     suffixes=[cfg["file"]])
    live_norm, n_line, n_drop = normalize_jsonl(live_p)
    live_steps, live_empty = merged_steps(live_p)

    # ⛔ 真值转引 `gate1_guard.EXPECT`,**不抄第二份** —— 生产的 CASES 里没有 `steps` 这个键。
    #    (`CASES[...]["steps"]` 只是给合成夹具用的覆盖口子,生产路径永远走 EXPECT。)
    want_steps = cfg.get("steps", G1.EXPECT.get(chain, {}).get("steps"))
    if want_steps is None:
        raise G1.GateUnrunnable("EXPECT[%s] 没有登记 steps ⇒ 这条链说不出它该有几步" % chain)
    G1.require_exact_steps("live/%s(历史对拍)" % chain, chain, live_steps, want_steps)
    got_keys = sorted({k for v in live_steps.values() for k in v})
    if got_keys != sorted(cfg["keys"]):
        raise G1.GateUnrunnable(
            "live/%s 的分量名单与登记的不同:\n       实测 %s\n       登记 %s\n"
            "       ⇒ 一个 loss 分量被改名 / 不再上报 ⇒ **这一跑到底比了什么**变了。\n"
            "       ⚠ 前身对这件事完全瞎:它的地板是 `步数×5`,而真实分量数是 %d ——\n"
            "         实测 rvc 删掉任意一个键仍然恰好过线,sovits_v2 可以删掉五个。"
            % (chain, got_keys, sorted(cfg["keys"]), len(cfg["keys"])))
    n_live_pairs = sum(len(v) for v in live_steps.values())
    if n_live_pairs != cfg["pairs"]:
        raise G1.GateUnrunnable(
            "live/%s 的可比对数是 %d,登记的真值是 %d ⇒ 覆盖面变了" % (chain, n_live_pairs, cfg["pairs"]))
    G1._say("  [COVERAGE] live/%-10s %d 行(去掉 %d 个 %s)· %d 步 · %d 分量 · %d 对%s"
            % (chain, n_line, n_drop, "/".join(DROP_KEYS), len(live_steps),
               len(got_keys), n_live_pairs,
               ";末尾空 losses 的收尾步 %s 已剔除" % live_empty if live_empty else ""))

    for name, root, _sha, _why in BASELINES:
        base_p = os.path.join(root, cfg["file"])
        if not os.path.isfile(base_p):
            G1.note_uncovered("%s / 基线 %s 缺 %s" % (chain, name, cfg["file"]),
                              "这条链在这个基线里没有历史读数 ⇒ 这一档不构成一次对拍")
            G1._say("  [NO-BASELINE] %-10s %-5s %s" % (chain, name, base_p))
            continue
        base_norm, _l, _d = normalize_jsonl(base_p)
        base_steps, _e = merged_steps(base_p)

        same = live_norm == base_norm
        # ⛔ 两条**独立**判据,各自求值、各自打印 —— 前身用 `if/elif`,`EMPTY-COMPARISON`
        #    会无条件覆写 `DIFFERS`,于是又红又空时转录只说「空」。
        common_steps = sorted(set(live_steps) & set(base_steps))
        items = [(s, k, base_steps[s][k], live_steps[s][k])
                 for s in common_steps
                 for k in sorted(set(base_steps[s]) & set(live_steps[s]))]
        # ⚠ 键的**不对称**在前身里连打印都没有(only_new/only_old 算的是步号)
        odd = [(s, sorted(set(base_steps[s]) ^ set(live_steps[s])))
               for s in common_steps if set(base_steps[s]) != set(live_steps[s])]
        if odd:
            raise G1.GateUnrunnable(
                "%s / %s:有 %d 个 step 两侧的分量名单不同(例:%s)\n"
                "       ⇒ 取交集会让消失的那个分量静默退出比较。" % (chain, name, len(odd), odd[:3]))
        r = G1.compare_pairs("%s/%s" % (chain, name), items, HISTORY_TOL,
                             floor=1e-12, min_cmp=cfg["pairs"])
        verdict = "LOSS-TRACE-IDENTICAL" if (same and not r["failures"]) else (
            "NORMALIZED-DIFFERS" if not same else "VALUE-DRIFTED")
        G1._say("  [%s] %-10s vs %-5s  %d 对真比过  max_rel=%.3e @step %s %s  -> %s"
                % ("PASS" if not r["failures"] and same else "FAIL", chain, name,
                   r["n_cmp"], r["worst"], r["worst_step"], r["worst_tag"] or "-", verdict))
        if not same:
            # 归一化字节不等而数值又全等 ⇒ 差在**非 losses 的字段**(ckpt 路径 / summary / stage)
            failures.append(
                "%s/%s:归一化之后**整份 jsonl 仍然不逐字节相同**(口径 = 除 %s 外)"
                % (chain, name, "/".join(DROP_KEYS)))
        failures.extend(r["failures"])
    return 1


def compare_tb(chain, cfg, t0, failures):
    live, july = cfg["live"], cfg["july"]
    _require(live, "本轮我方侧 TB")
    _require(july, "七月的我方侧 TB")
    tags = list(VC.REQUIRED_TAGS)          # ⛔ 转引,不抄第二份
    a = G1.tb_scalars("live/%s" % chain, live, tags, t0)
    # ⛔ 冻结参照:`tb_scalars` 的 frozen 分支只判「够数、非空、恰好一个 events」——
    #    「还是同一份东西」要靠这一句。⛔ 没有它,把路径指到任何一份声码器 TB 上读数都一样
    #    (四份 TB 那 143 个值两两全等,**含自比**)。
    G1.declare_frozen("baseline/july(%s)" % chain, july, [""], minimum=1,
                      why="2026-07-07 的声码器我方侧 TB(S134 的 baseline_backup)",
                      expect_sha=cfg["july_sha"])
    b = G1.tb_scalars(
        "baseline/july(%s)" % chain, july, tags, t0,
        frozen_why="2026-07-07 的声码器我方侧 TB(S134 的 baseline_backup)")
    items = [(s, tag, b[tag][s], a[tag][s])
             for tag in tags for s in sorted(set(a[tag]) & set(b[tag]))]
    for tag in tags:
        if set(a[tag]) != set(b[tag]):
            raise G1.GateUnrunnable(
                "%s / %s:两侧步轴不同 live=%s july=%s"
                % (chain, tag, sorted(a[tag])[:6], sorted(b[tag])[:6]))
    r = G1.compare_pairs("%s/july" % chain, items, HISTORY_TOL, floor=1e-12,
                         symmetric=True, min_cmp=cfg["pairs"])
    G1._say("  [%s] %-10s vs july   %d 对真比过(登记 %d)  max_rel=%.3e @%s step %s"
            % ("PASS" if not r["failures"] else "FAIL", chain, r["n_cmp"], cfg["pairs"],
               r["worst"], r["worst_tag"], r["worst_step"]))
    failures.extend(r["failures"])
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gate1_history_compare", add_help=True)
    ap.add_argument("--chain", choices=sorted(CASES) + ["all"], default="all")
    ap.add_argument("--allow-uncovered", action="store_true")
    args, _rest = ap.parse_known_args(argv)

    t0 = G1.read_t0(GATE)
    chains = sorted(CASES) if args.chain == "all" else [args.chain]
    G1.header(GATE, args.chain, [("live 根", LIVE)] +
              [("基线 " + n, r) for n, r, _s, _w in BASELINES])
    # ⛔ 历史侧的输入身份**在物理上不可知**,而且**永远如此** —— 七月/08-11 那批基线产出时
    #    `gate1_input.identity.json` 这套机制还不存在(S139 才建)。
    # ⛔ 所以它是**登记式的致盲面**,不是 `note_uncovered`:后者会让**每一跑都要**
    #    `--allow-uncovered`,而 S139 刚为夹取致盲判过同一件事 ——
    #    「一个每次都要加的开关三天之内就变成肌肉记忆,等于没有」。
    #    ⇒ 照 `gate1_guard.check_clamped` 的先例:每跑打一行把致盲面**说出来**。
    G1._say("[NOTE] 历史侧的输入身份**不可知,且永远如此**:七月/08-11 那批基线产出时 "
            "`gate1_input.identity.json` 这套机制还不存在(S139 才建)\n"
            "       ⇒ 这一条如果红了,它说不清是「代码变了」还是「夹具变了」。\n"
            "       ⇒ 本轮的输入身份由各条链自己的 4_compare 打(见 [INPUT-ID] 行),"
            "两者要一起读。")

    # ⛔⛔ 冻结参照必须【够数 + 非空 + **还是同一份东西**】。前身对这一层完全没有守卫 ——
    #    而它的备份脚本自己的 REFUSING 文案就教人「要重做请先手动改名」,照做一次,
    #    那个基线根就没了,而前身会打几行 SKIP 然后**干净退 0**。
    #    ⇒ 每一跑都对**整个基线根的四份 jsonl** 逐字节对拍一次(而不是只看当前这条链那一份)。
    for name, root, expect_sha, why in BASELINES:
        G1.declare_frozen(
            "baseline/%s" % name, root, [""], minimum=len(JSONL_FILES), why=why,
            suffixes=JSONL_FILES, expect_sha=expect_sha)

    failures, done = [], 0
    for c in chains:
        cfg = CASES[c]
        done += (compare_jsonl if cfg["kind"] == "jsonl" else compare_tb)(c, cfg, t0, failures)

    # ⛔ 总账:跑了几档必须等于要跑的几档,并把这个数打出来。
    #    前身没有这一条 ⇒ 四档全 SKIP 时它打几行字然后 exit 0。
    if done != len(chains):
        raise G1.GateUnrunnable("只完成了 %d/%d 档 ⇒ 这一跑不构成一次判定" % (done, len(chains)))
    G1._say("[COVERAGE] 历史对拍完成 %d/%d 档" % (done, len(chains)))
    G1.finish(GATE, failures, allow_uncovered=args.allow_uncovered)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import gate1_history_negctl                              # noqa: E402
        sys.exit(gate1_history_negctl.main())
    G1.run(GATE, main)
