# -*- coding: utf-8 -*-
"""S134 (§F7 first pass) — bind every gate driver's call sites against utai_train's LIVE signatures.

WHY THIS EXISTS
---------------
§F2⒝ (S122/S125) split "the run directory" from "the pool" and gave five training entry points a
new `pool_dir` parameter. The production call sites were all updated. The seven call sites in
`converter/verify/training/` were not — and nothing noticed for four months, because:

  * the gate scripts are not imported by anything (no tsc/cargo/vitest reaches them),
  * `converter/verify/training/README.md:598` still claims "既有阶段函数签名零改动",
  * and the failure only shows up when somebody actually runs a gate — which is exactly the
    moment when a `TypeError` reads like "the training side is broken".

So: five chains' gate1 could not start AT ALL, and the fact was invisible from every green gate
we have. This checker is the missing tripwire.

WHAT IT CHECKS (and what it deliberately does NOT)
-------------------------------------------------
Pure `ast` on both sides — it never imports torch, so it is seconds, not minutes.
For every call in the gate scripts that resolves to a module-level `def` inside `training/utai_train`,
it emulates `inspect.Signature.bind` over the *shape* of the call: positional count, keyword names,
`*args`/`**kwargs` presence.

⛔ HONEST BOUNDARY, and it is a real one: arity is not binding. `pool_dir` sits in POSITION 3 of
`train()` / `_train_diff()` and position 3 of `_train()` — appending it at the end of a call makes
the arity correct and silently rebinds every later parameter (pool_dir=config, config=reporter, ...).
This checker CANNOT see that. It catches "the call no longer fits the signature", which is the
failure mode that actually happened; it does not catch "the call fits but means something else".
For that, the criterion has to be the gate's own output, not this script.

USAGE
    training/.venv/Scripts/python.exe converter/verify/training/gate_driver_arity.py
    ... --selftest      # negative control: inject a bad call, the checker must flag exactly it

Exit codes are DISTINCT on purpose (S129: a gate's red must be attributable):
    0 = all resolved call sites bind
    1 = at least one call site does not bind          (the thing this exists to catch)
    2 = the checker resolved too few call sites       (it is not looking at anything = no evidence)
    3 = --selftest failed                              (the checker cannot report a failure at all)
"""
import argparse
import ast
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))

# ⛔ S135 追加:**驱动生产函数的脚本不止仓内这一个目录**。这三个在仓库外(无 git、
# 不进任何自动闸),而它们恰恰是「gate_resume_state 的 D/V 组之所以携带信息」的唯一来源
# 与「M20 每轮收工手跑」的那一条。实测 S135:前两个今天已经漂了。
OUT_OF_REPO_DRIVERS = [
    r"D:\MyDev\TESTING\s118_f8a\smoke_diff_resume.py",    # §F8⒜ 端到端续训冒烟(S118, 20/20)
    r"D:\MyDev\TESTING\s119_vocoder\smoke_voc_resume.py",  # §F8⒝ 声码器端到端冒烟(S119, 16/16)
    r"D:\MyDev\TESTING\s129_f2b2g\legs_s129.py",           # M20 每轮收工手跑的行为腿
]
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TRAINING = os.path.join(REPO, "training")

# If the checker ever resolves fewer than this many call sites, something in the resolver broke and
# a clean "ALL BIND" would be a lie about coverage (S122: never let ALL PASS imply a coverage it
# does not have). The number is a floor, not a pin — it is fine for it to grow.
MIN_RESOLVED = 25


def module_path(dotted):
    p = os.path.join(TRAINING, *dotted.split(".")) + ".py"
    return p if os.path.isfile(p) else None


_def_cache = {}


def defs_of(dotted):
    """{name: ast.FunctionDef} for module-level defs of a utai_train module."""
    if dotted in _def_cache:
        return _def_cache[dotted]
    p = module_path(dotted)
    out = {}
    if p:
        tree = ast.parse(open(p, encoding="utf-8").read(), filename=p)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[node.name] = node
    _def_cache[dotted] = out
    return out


def signature_of(fn):
    a = fn.args
    pos = [x.arg for x in (a.posonlyargs + a.args)]
    n_defaults = len(a.defaults)
    required_pos = len(pos) - n_defaults
    kwonly = [x.arg for x in a.kwonlyargs]
    kwonly_required = {
        x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is None
    }
    return {
        "pos": pos,
        "required_pos": required_pos,
        "kwonly": kwonly,
        "kwonly_required": kwonly_required,
        "vararg": a.vararg is not None,
        "kwarg": a.kwarg is not None,
    }


def bind(sig, n_pos, kwnames, star_args, star_kwargs):
    """Return None if the call shape fits, else a human sentence saying why not."""
    if star_args or star_kwargs:
        return None  # unresolvable shape — say nothing rather than guess
    if not sig["vararg"] and n_pos > len(sig["pos"]):
        return "too many positional args: %d given, at most %d (%s)" % (
            n_pos, len(sig["pos"]), ", ".join(sig["pos"]))
    filled = set(sig["pos"][:n_pos])
    for k in kwnames:
        if k in filled:
            return "duplicate value for %r (given positionally AND by keyword)" % k
        if k in sig["pos"] or k in sig["kwonly"]:
            filled.add(k)
        elif not sig["kwarg"]:
            return "unknown keyword %r (params: %s)" % (
                k, ", ".join(sig["pos"] + sig["kwonly"]))
    missing = [p for p in sig["pos"][: sig["required_pos"]] if p not in filled]
    missing += [k for k in sorted(sig["kwonly_required"]) if k not in filled]
    if missing:
        return "missing %d required arg(s): %s  (signature: %s)" % (
            len(missing), ", ".join(missing), ", ".join(sig["pos"]))
    return None


def resolve_imports(tree):
    """local name -> ('func', module, name) | ('mod', module) for anything under utai_train."""
    funcs, mods = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("utai_train"):
            for al in node.names:
                local = al.asname or al.name
                sub = "%s.%s" % (node.module, al.name)
                if module_path(sub):
                    mods[local] = sub          # `from utai_train.sovits import diff_pipeline`
                elif module_path(node.module):
                    funcs[local] = (node.module, al.name)
        elif isinstance(node, ast.Import):
            for al in node.names:
                if al.name.startswith("utai_train") and module_path(al.name):
                    mods[al.asname or al.name] = al.name
    return funcs, mods


def scan(path, report):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    funcs, mods = resolve_imports(tree)
    resolved = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = None
        if isinstance(node.func, ast.Name) and node.func.id in funcs:
            target = funcs[node.func.id]
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            base = node.func.value.id
            if base in mods:
                target = (mods[base], node.func.attr)
        if not target:
            continue
        mod, name = target
        fn = defs_of(mod).get(name)
        if fn is None:
            continue  # a class, a re-export, or something we cannot see statically
        resolved += 1
        star_args = any(isinstance(a, ast.Starred) for a in node.args)
        n_pos = sum(0 if isinstance(a, ast.Starred) else 1 for a in node.args)
        kwnames = [k.arg for k in node.keywords if k.arg is not None]
        star_kwargs = any(k.arg is None for k in node.keywords)
        why = bind(signature_of(fn), n_pos, kwnames, star_args, star_kwargs)
        if why:
            report.append((path, node.lineno, "%s.%s" % (mod, name), why))
    return resolved


SELFTEST_SRC = '''
import sys
sys.path.insert(0, "training")
from utai_train.rvc.train import train
train(1, 2)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="negative control: feed the checker a call that CANNOT bind")
    args = ap.parse_args()

    if args.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "zz_selftest_driver.py")
            open(p, "w", encoding="utf-8").write(SELFTEST_SRC)
            bad = []
            n = scan(p, bad)
            ok = n == 1 and len(bad) == 1 and "missing" in bad[0][3]
            print("SELFTEST resolved=%d flagged=%d -> %s" % (n, len(bad), "PASS" if ok else "FAIL"))
            for b in bad:
                print("   flagged: %s:%d %s -- %s" % (os.path.basename(b[0]), b[1], b[2], b[3]))
            if not ok:
                print("VERDICT SELFTEST-FAILED — a checker that cannot report a failure carries no "
                      "information in its passes (S116).")
                sys.exit(3)

    scripts = sorted(
        os.path.join(HERE, f) for f in os.listdir(HERE)
        if f.endswith(".py") and f != os.path.basename(__file__)
    )

    # ⛔ S135:这道闸自己带着它要防的那个盲区 —— 它只扫仓内这一个目录,
    # 而**驱动生产函数的脚本有一部分在仓库外**(TESTING\ 下,无 git、不进任何闸)。
    # S134 立它时写的理由原文是「the gate scripts are not imported by anything
    # (no tsc/cargo/vitest reaches them)」—— 那条理由对仓外驱动**一字不差地成立**,
    # 而且今天已经真的漂了两个(见下)。
    # ⇒ 显式登记它们。⛔ 名单里的路径**必须存在**,否则当场判 exit 2 ——
    #    一份可以静默变空的名单等于没有名单。
    for extra in OUT_OF_REPO_DRIVERS:
        if not os.path.isfile(extra):
            print("VERDICT LIST-STALE — 登记的仓外驱动不在了:%s\n"
                  "      (名单一旦能静默变空,这道闸的覆盖面就是假的)" % extra)
            sys.exit(2)
        scripts.append(extra)
    scripts = sorted(scripts)

    bad, total = [], 0
    for s in scripts:
        n = scan(s, bad)
        total += n
        if n:
            print("  %-38s %2d call site(s) bound" % (os.path.basename(s), n))
    print("\nresolved call sites: %d across %d scripts" % (total, len(scripts)))

    if total < MIN_RESOLVED:
        print("VERDICT NOT-LOOKING — only %d call sites resolved (floor %d). The resolver is broken; "
              "a clean result here would be a lie about coverage." % (total, MIN_RESOLVED))
        sys.exit(2)
    if bad:
        print("\n%d call site(s) DO NOT BIND against today's signatures:" % len(bad))
        for path, line, who, why in bad:
            print("  %s:%d  ->  %s\n      %s" % (os.path.relpath(path, REPO), line, who, why))
        print("\nVERDICT DRIFTED")
        sys.exit(1)
    print("VERDICT ALL-BIND  (⛔ arity only — this cannot see an argument inserted at the wrong "
          "position; see the module docstring)")
    sys.exit(0)


main()
