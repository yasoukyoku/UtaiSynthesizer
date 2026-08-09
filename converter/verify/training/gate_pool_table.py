# -*- coding: utf-8 -*-
r"""§F2⒝ — drive the Rust pool DECISION TABLE against what python actually does.

`tpool::POOL_ENTRIES` decides which of a slot's top-level entries the layout migration moves into
`pools/<identity>/`. python decides where it WRITES them. Nothing connects the two, so a new
preprocessing product added on the python side would simply be left at the slot root by the
migration — visible only as a rebuild costing hours, months later, on someone else's machine.

This gate is that connection, in BOTH directions:

  (1) every directory python joins onto a POOL base is named in the Rust table for that family;
  (2) every entry the Rust table names is actually produced by python;
  (3) no pool product is ever joined onto a RUN base — the product lands where nothing reads it,
      and every run rebuilds its own copy of something the pool exists to share;
  (3') the MIRROR, added with §F2⒝ batch 2: no RUN product is ever joined onto the pool. Two runs
      sharing a slot would then overwrite each other's weights, sidecars and retrieval assets. This
      one reads `trun::RUN_ENTRIES`, which has PREFIX rows (`G_`, `model_`, `events.out.tfevents`)
      — a literal-only comparison would not see a single checkpoint;
  (4) the two `pool_id_for` implementations agree, because Rust names the FIRST pool of every
      migrated slot and python names every pool minted afterwards;
  (5) the SEMANTIC ratchet, and the reason (3)/(3') can be trusted at all: `open_pool` is handed a
      name bound from `cfg["workspace"]` (the slot) and the run products hang off a name bound from
      `cfg["run_dir"]`. Checks (0c)-(3') classify by VARIABLE NAME, so re-pointing an existing name
      at the other directory leaves all of them green while inverting what they measure. This is
      the only check that can see that, and it is why `open_pool(run_dir, …)` — which would mint an
      empty pool inside the run and re-run hours of preprocessing under one info line — cannot be
      introduced quietly.

⚠ Assertion order is从最具体到最兜底 (S108): a mutation aimed at (3) must not die inside (1).
Run it standalone; it prints a mutation-ready summary of what it read.

    <venv>\Scripts\python.exe converter\verify\training\gate_pool_table.py
"""
import ast
import hashlib
import json
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN = os.path.join(REPO, "training", "utai_train")
TPOOL_RS = os.path.join(REPO, "src-tauri", "src", "training", "tpool.rs")
TRUN_RS = os.path.join(REPO, "src-tauri", "src", "training", "trun.rs")

FAIL = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, ("  - %s" % detail) if detail else ""))
    if not cond:
        FAIL.append(name)


# ─────────────────────────── the Rust side, parsed from source ───────────────────────────

# e("rvc", "0_gt_wavs"),   /   e("*", FINGERPRINT),
ENTRY = re.compile(r'e\(\s*"([^"]+)"\s*,\s*(?:"([^"]+)"|([A-Z_]+))\s*\)')


def rust_table():
    """{family: set(names)} from `POOL_ENTRIES`, with `*` expanded over the family list.

    Parsed from the SOURCE rather than from a JSON the Rust side prints, so this gate keeps
    working (and keeps being the thing that must be updated) even when nobody remembers to re-run
    the exporter. The parser self-checks: it must find the block and a plausible number of rows,
    or it fails LOUDLY instead of silently comparing against an empty table.
    """
    src = open(TPOOL_RS, encoding="utf-8").read()
    m = re.search(r"pub const POOL_ENTRIES:\s*&\[PoolEntry\]\s*=\s*&\[(.*?)\n\];", src, re.S)
    if not m:
        raise SystemExit("gate parser: POOL_ENTRIES block not found in tpool.rs")
    body = m.group(1)
    # strip comments so a name mentioned in prose cannot be read as an entry (S116 §3)
    body = "\n".join(l.split("//")[0] for l in body.splitlines())
    rows = ENTRY.findall(body)
    if len(rows) < 10:
        raise SystemExit("gate parser: only %d POOL_ENTRIES rows parsed — parser is broken"
                         % len(rows))
    fam_m = re.search(r"pub const FAMILIES:\s*\[&str;\s*\d+\]\s*=\s*\[(.*?)\];",
                      open(os.path.join(REPO, "src-tauri", "src", "training", "tproject.rs"),
                           encoding="utf-8").read(), re.S)
    families = re.findall(r'"([^"]+)"', fam_m.group(1))
    fingerprint = re.search(r'pub const FINGERPRINT:\s*&str\s*=\s*"([^"]+)"', src).group(1)
    table = {f: set() for f in families}
    # ⚠ Collected rather than indexed straight into `table`: a family literal that is not a real
    # family used to raise KeyError here, i.e. the gate CRASHED instead of reporting. Check (0a)
    # below is the one that should speak, and it cannot if the parser dies first.
    bad_families = sorted({fam for fam, _, _ in rows} - set(families) - {"*"})
    for fam, lit, const in rows:
        name = lit if lit else {"FINGERPRINT": fingerprint}.get(const)
        if name is None:
            raise SystemExit("gate parser: unresolved const in POOL_ENTRIES: %r" % const)
        for f in (families if fam == "*" else [fam]):
            if f in table:
                table[f].add(name)
    return table, fingerprint, families, bad_families


# ─────────────────────────── the python side, parsed with ast ───────────────────────────

#: bases that ARE the pool (or a directory inside it). Everything else is the slot / a run dir.
#:
#: ⚠ `self.pool_dir` was listed here and reaches NOTHING — the attribute exists
#: (`rvc/preprocess.py`, `vocoder/harness.py`) but no `os.path.join` site uses it as a base. Gone,
#: for the reason below.
POOL_BASES = {"pool_dir"}
#: bases that are one RUN's directory. A pool product joined onto one of these is finding (3); a
#: run product joined onto the POOL is finding (3').
#:
#: ⚠ `ws`, `self.workspace`, `slot_dir` and `workspace` were listed here and the census below finds
#: they reach NOTHING — no join site in the tree uses any of them as a base. They are gone: a
#: classification nothing can reach cannot be wrong in a way anything notices, so it rots into a
#: false claim of coverage (the same lesson `run_id_is_usable` in `trun.rs` records about its own
#: deleted clauses). Nothing is lost, because check (0c) refuses any base it has not been told
#: about — so if one ever comes back it arrives as a FAILURE demanding classification.
#:
#: ⛔ And the SLOT is not defended by a name any more: after §F2⒝ batch 2 the slot only ever reaches
#: `open_pool`, never `os.path.join`. What guards it is check (5), which reads the BINDING.
RUN_BASES = {"exp_dir", "run_dir", "work_dir", "expdir", "self.expdir", "hps.model_dir"}
#: ★§F2⒝ batch 2 — EVERY base name the scan finds, whether or not this gate classifies it.
#:
#: The two sets above are what checks (1)-(3') act on, and a join onto anything else is simply
#: skipped. That is fine while the unclassified names are all leaf directories already derived
#: from a pool or a run — but it means the gate LOSES COVERAGE SILENTLY the moment python grows a
#: new base. Batch 3 introduced exactly that (`run_dir`), and the `len(joins) > 60` floor below
#: cannot notice: re-basing `os.path.join(exp_dir, "weights")` to `os.path.join(run_dir, "weights")`
#: leaves the site count unchanged while moving the site out of every check.
#:
#: So the census is declared. It is a source-vs-declaration ratchet, not a table compared with
#: itself: the left side is parsed out of the python tree by the AST scan below. Adding a base
#: turns check (0c) red, and whoever adds it has to say which of the two sets it belongs to (or
#: that it belongs to neither, here).
#:
#: ⚠ It ratchets on the NAME, not on the meaning. Batch 3 measured that: binding an existing name
#: to the other directory (`exp_dir = cfg["run_dir"]`) leaves this check green while every one of
#: its 22 join sites changes what it addresses. That is what check (5) is for; this one only
#: guarantees that a NEW name cannot slip past unclassified.
KNOWN_BASES = {
    # the two the checks act on
    "exp_dir", "run_dir", "work_dir", "expdir", "self.expdir", "hps.model_dir",
    "pool_dir",
    # leaf directories, each already derived from a pool or a run by the code that binds them
    "weights_dir", "flist_dir", "configs_dir", "cluster_dir", "meta_dir",
    "slices_dir", "slice_dir", "out_dir", "out_spk_dir", "tmp",
    "self.gt_wavs_dir", "self.wavs16k_dir",
}
#: marker for a name this gate could not resolve to a literal (a `%`-format or an f-string).
#: Plain ASCII on purpose — this repo has five separate incidents of a NUL byte written into a
#: source file by an escape that looked harmless (feedback_claude_tooling_pitfalls A3).
DYNAMIC = "<dynamic>"


def base_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return "%s.%s" % (node.value.id, node.attr)
    return None


def literals_of(node):
    """Every string literal the second argument of a join can evaluate to.

    ⚠ `ast.Constant` alone is not enough and the first version of this gate learned it the hard
    way: `extract_feature.py` writes
    `os.path.join(pool_dir, "3_feature256" if version == "v1" else "3_feature768")`, so BOTH of
    those declared entries looked like ghosts. A conditional is still a fixed, enumerable set of
    names; a `%`-format is not, and those are reported separately rather than guessed at.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value], []
    if isinstance(node, ast.IfExp):
        a, da = literals_of(node.body)
        b, db = literals_of(node.orelse)
        return a + b, da + db
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        lits, _ = literals_of(node.left)
        # "3_feature%s" % dim -> the literal head "3_feature" is all that can be compared
        return [], [s.split("%")[0] for s in lits]
    if isinstance(node, ast.JoinedStr):
        parts = [v.value for v in node.values
                 if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        return [], ["".join(parts)] if parts else []
    return [], []


def scan_joins(root):
    """[(relfile, lineno, base, literal)] for every `os.path.join(<base>, "<literal>", …)`.

    ⛔ States what it CANNOT see, rather than implying coverage it does not have: a path built by
    string formatting, by a variable, or by a helper that receives the name as an argument is
    invisible here and lands in the `dynamic` list instead (see [`literals_of`]).
    """
    out = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace("\\", "/")
            tree = ast.parse(open(p, encoding="utf-8").read(), filename=p)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if not (isinstance(f, ast.Attribute) and f.attr == "join"):
                    continue
                if len(node.args) < 2:
                    continue
                base = base_name(node.args[0])
                if not base:
                    continue
                lits, dynamic = literals_of(node.args[1])
                for v in lits:
                    out.append((rel, node.lineno, base, v))
                for v in dynamic:
                    out.append((rel, node.lineno, base, DYNAMIC + v))
    return out


def family_of(relfile):
    head = relfile.split("/")[0]
    if head in ("rvc", "sovits_v2", "vocoder"):
        return head
    if head == "sovits":
        return "sovits"  # sovits_diff shares the slot BY DESIGN
    return None


# ─────────────────────────── the RUN table (§F2⒝ batch 2) ───────────────────────────

RUN_EXACT = re.compile(r'RunEntry::Exact\("([^"]+)"\)')
RUN_PREFIX = re.compile(r'RunEntry::Prefix\("([^"]+)"\)')


def run_table():
    """`(exact_names, prefixes)` from `trun::RUN_ENTRIES`.

    ⛔ The two kinds are kept APART on purpose. Five of the rows are PREFIX rows (`G_`, `D_`,
    `model_`, `events.out.tfevents`, `aug_gate_report`), and every checkpoint this repo produces is
    matched by one of them — a comparison that only looked at exact names would report zero run
    products and pass without measuring anything.
    """
    src = open(TRUN_RS, encoding="utf-8").read()
    m = re.search(r"pub const RUN_ENTRIES:\s*&\[RunEntry\]\s*=\s*&\[(.*?)\n\];", src, re.S)
    if not m:
        raise SystemExit("gate parser: RUN_ENTRIES block not found in trun.rs")
    body = "\n".join(l.split("//")[0] for l in m.group(1).splitlines())
    exact, prefix = set(RUN_EXACT.findall(body)), set(RUN_PREFIX.findall(body))
    if len(exact) < 15 or len(prefix) < 3:
        raise SystemExit("gate parser: %d exact / %d prefix RUN_ENTRIES rows — parser is broken"
                         % (len(exact), len(prefix)))
    return exact, prefix


def is_run_product(name, exact, prefix):
    return name in exact or any(name.startswith(p) for p in prefix)


# ─────────────────────────── the BINDINGS (check 5) ───────────────────────────

def scan_bindings(root):
    """Per pipeline module: which local names are bound from which `cfg[...]` key, and what the
    first argument of every `open_pool(...)` call is.

    This is the only part of the gate that looks at MEANING rather than at a name. Everything else
    classifies by variable name, so re-binding an existing name to the other directory — the exact
    shape `exp_dir = cfg["run_dir"]` — leaves them all green while inverting what they measure.
    """
    out = {}
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in files:
            if not (fn == "pipeline.py" or fn == "diff_pipeline.py"):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace("\\", "/")
            tree = ast.parse(open(p, encoding="utf-8").read(), filename=p)
            binds, opens, checked = {}, [], []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    tgt = node.targets[0]
                    if not isinstance(tgt, ast.Name):
                        continue
                    v = node.value
                    # name = cfg["<key>"]
                    if (isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name)
                            and v.value.id == "cfg" and isinstance(v.slice, ast.Constant)):
                        binds[tgt.id] = v.slice.value
                    # name = checked_run_dir(cfg, <slot name>)
                    elif (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                            and v.func.id == "checked_run_dir"):
                        arg = v.args[1] if len(v.args) > 1 else None
                        checked.append((tgt.id, arg.id if isinstance(arg, ast.Name) else None))
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "open_pool" and node.args):
                    a = node.args[0]
                    opens.append((node.lineno, a.id if isinstance(a, ast.Name) else "<expr>"))
            out[rel] = {"binds": binds, "opens": opens, "checked": checked}
    return out


def main():
    table, fingerprint, families, bad_families = rust_table()
    joins = scan_joins(TRAIN)
    all_pool_names = set().union(*table.values())
    print("Rust table: %s" % json.dumps({k: sorted(v) for k, v in table.items()},
                                        ensure_ascii=False))
    print("python: %d os.path.join(<name>, \"<literal>\") sites" % len(joins))

    # ── (0) the parsers themselves, before anything they produce is believed ────────────────
    print("\n(0) the readers")
    # ⚠ This used to be `set(table) == set(families)` — and `table` is BUILT as `{f: set() for f in
    # families}`, so it compared a dictionary with the list it was made from and could never go
    # red. That is the shape this very file warns about at check (4) (S105: a table compared with
    # itself). What can go red: every family literal appearing in POOL_ENTRIES must be a real
    # family, and every family must actually own entries.
    check("every family named in POOL_ENTRIES is a real family", not bad_families,
          str(bad_families))
    check("every family owns pool entries",
          all(table[f] for f in families), str({f: len(table[f]) for f in families}))
    check("the python scan found a plausible number of join sites", len(joins) > 60, str(len(joins)))
    seen_bases = {b for _, _, b, _ in joins}
    check(
        "every join base is declared (a NEW one silently leaves checks 1-3 uncovered)",
        seen_bases == KNOWN_BASES,
        "unexpected: %s | vanished: %s"
        % (sorted(seen_bases - KNOWN_BASES), sorted(KNOWN_BASES - seen_bases)),
    )

    # ── (1) most specific: every POOL join is declared, per family ──────────────────────────
    print("\n(1) every directory python writes into the pool is DECLARED in the Rust table")
    undeclared = []
    dynamic = []
    for rel, line, base, lit in joins:
        if base not in POOL_BASES:
            continue
        fam = family_of(rel)
        if fam is None:
            continue
        if lit.startswith(DYNAMIC):
            dynamic.append((rel, line, fam, lit[len(DYNAMIC):]))
            continue
        if lit not in table[fam]:
            undeclared.append("%s:%d %s/%s" % (rel, line, fam, lit))
    check("no undeclared pool product", not undeclared, str(undeclared[:5]))
    # ⛔ Stated, not hidden: a name this gate cannot resolve is not covered by (1). Each one must
    # still be a PREFIX of something declared, which is the strongest thing available without
    # evaluating the expression — and printing them keeps the blind spot visible instead of
    # letting "ALL PASS" imply a coverage the parser does not have.
    unresolved = [
        "%s:%d %s/%s…" % (rel, line, fam, pat) for rel, line, fam, pat in dynamic
        if not any(n.startswith(pat) for n in table[fam])
    ]
    for rel, line, fam, pat in dynamic:
        print("     [note] dynamic name %s:%d %s/%s… (not covered by (1))" % (rel, line, fam, pat))
    check("every dynamically-built pool name at least PREFIXES a declared entry",
          not unresolved, str(unresolved[:5]))

    # ── (2) the reverse: every declared entry is really produced ────────────────────────────
    print("\n(2) every entry the Rust table declares is really produced by python")
    produced = {f: set() for f in families}
    for rel, _line, base, lit in joins:
        fam = family_of(rel)
        if fam and base in POOL_BASES and not lit.startswith(DYNAMIC):
            produced[fam].add(lit)
    # `sovits_v2` writes its products through helpers that live in the `sovits` package
    # (`slice_and_resample`, `_speaker_meta_dir`), so its own directory names are produced there.
    produced["sovits_v2"] |= produced["sovits"]
    missing = []
    for fam in families:
        for name in sorted(table[fam]):
            if name == fingerprint:
                continue  # written by utai_train/pool.py, not by a pipeline join
            if name not in produced[fam]:
                missing.append("%s/%s" % (fam, name))
    check("no declared entry is a ghost", not missing, str(missing))

    # ── (3) the regression this relocation can suffer, and it is silent ─────────────────────
    print("\n(3) no pool product is written into a RUN")
    stray = []
    for rel, line, base, lit in joins:
        if lit.startswith(DYNAMIC):
            # a formatted name joined onto a RUN base is just as bad — compare on the prefix
            pat = lit[len(DYNAMIC):]
            if base in RUN_BASES and pat and any(n.startswith(pat) for n in all_pool_names):
                stray.append("%s:%d %s(%s…)" % (rel, line, base, pat))
            continue
        if base in RUN_BASES and lit in all_pool_names and lit != fingerprint:
            stray.append("%s:%d %s(%s)" % (rel, line, base, lit))
    check("no pool product joined onto a run base", not stray, str(stray[:5]))

    # ── (3') the MIRROR, and the one per-run destroys data with ────────────────────────────
    print("\n(3') no RUN product is written into the POOL")
    # Before batch 2 this could not even be asked: the run WAS the slot, so every one of these
    # joins was correct by construction. Now two runs share one pool, and a run product written
    # there is not a stale copy — it is the other run's weights, sidecars or retrieval matrix,
    # overwritten with no error. The prefix arm is load-bearing: `G_2333333.pth` and
    # `model_ckpt_steps_*.ckpt` are only matched by it.
    run_exact, run_prefix = run_table()
    print("     RUN_ENTRIES: %d exact + %d prefix (%s)"
          % (len(run_exact), len(run_prefix), ",".join(sorted(run_prefix))))
    intruders = []
    for rel, line, base, lit in joins:
        if base not in POOL_BASES:
            continue
        if lit.startswith(DYNAMIC):
            pat = lit[len(DYNAMIC):]
            # a formatted name whose LITERAL HEAD already commits it to a run product
            if pat and (any(n.startswith(pat) for n in run_exact)
                        or any(pat.startswith(p) for p in run_prefix)):
                intruders.append("%s:%d %s(%s…)" % (rel, line, base, pat))
            continue
        if is_run_product(lit, run_exact, run_prefix):
            intruders.append("%s:%d %s(%s)" % (rel, line, base, lit))
    check("no run product joined onto the pool", not intruders, str(intruders[:5]))

    # ── (5) the SEMANTIC ratchet: what each name is BOUND to ────────────────────────────────
    print("\n(5) `open_pool` gets the slot, the run products get the run")
    binds = scan_bindings(TRAIN)
    print("     %d pipeline entry points" % len(binds))
    check("all five chains were parsed", len(binds) == 5, str(sorted(binds)))
    wrong_pool, missing_run = [], []
    for rel, info in sorted(binds.items()):
        slot_names = {n for n, key in info["binds"].items() if key == "workspace"}
        run_names = {n for n, arg in info["checked"] if arg in slot_names}
        if not run_names:
            missing_run.append("%s: no `run_dir = checked_run_dir(cfg, <slot>)`" % rel)
        for line, arg in info["opens"]:
            if arg not in slot_names:
                wrong_pool.append("%s:%d open_pool(%s) — bound from %r, not cfg['workspace']"
                                  % (rel, line, arg, info["binds"].get(arg)))
    # ⛔ THE one. `open_pool` resolves `<slot>/pools/*` and accepts a matching slot root as a pool;
    # handed a run directory it finds neither, mints an empty pool inside the run and re-runs every
    # preprocessing stage — hours, announced by one `logger.info`. Nothing else in this gate can
    # see it, because the mistake does not change a single joined literal.
    check("every open_pool call is handed a name bound from cfg['workspace']",
          not wrong_pool, str(wrong_pool))
    check("every chain resolves its run through checked_run_dir(cfg, <slot>)",
          not missing_run, str(missing_run))

    # ── (6) 「先练扩散」 must be a positive fact, not a missing file ──────────────────────────
    print("\n(6) the diff-first branch refuses to fire on an absent config.json alone")
    sys.path.insert(0, os.path.join(REPO, "training"))
    from utai_train.sovits.diff_pipeline import assert_diff_first  # noqa: E402

    def verdict(cfg):
        # ⚠ catches EVERY exception, not just RuntimeError: dropping the presence check turns the
        # refusal into a bare KeyError, and a probe that only caught RuntimeError would crash the
        # gate instead of reporting — a mutation measured that.
        try:
            assert_diff_first(cfg, "<run>")
            return "allowed"
        except Exception as e:  # noqa: BLE001
            return type(e).__name__ if not isinstance(e, RuntimeError) else str(e).split(":")[0]

    # A run config that never mentions the fact must NOT be read as diff-first — that is the shape
    # a hand-built gate cfg or a pre-batch-3 `run.json` has, and reading it as diff-first is exactly
    # the silent disabling this check exists to prevent.
    check("an unstated main-model fact is refused",
          verdict({}) == "DIFF_MAIN_MODEL_UNKNOWN", verdict({}))
    check("a slot WITH a main model refuses the placeholder",
          verdict({"slot_has_main_model": True}) == "DIFF_MAIN_CONFIG_MISSING",
          verdict({"slot_has_main_model": True}))
    check("a slot with no main model is genuinely diff-first",
          verdict({"slot_has_main_model": False}) == "allowed",
          verdict({"slot_has_main_model": False}))

    # ── (4) the two derivations of the pool NAME ────────────────────────────────────────────
    print("\n(4) Rust and python name the same pool for the same identity")
    sys.path.insert(0, os.path.join(REPO, "training"))
    from utai_train.pool import pool_id_for  # noqa: E402

    r = subprocess.run(
        ["cargo", "test", "--test", "tpool_migrate", "--",
         "--ignored", "--nocapture", "tpool_layout_constants"],
        cwd=os.path.join(REPO, "src-tauri"), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    line = [l for l in ((r.stdout or "") + "\n" + (r.stderr or "")).splitlines()
            if l.startswith("TPOOL_JSON ")]
    if not line:
        check("the Rust side published its constants", False, "rc=%d" % r.returncode)
    else:
        L = json.loads(line[0][len("TPOOL_JSON "):])
        check("pool_id_for agrees for every probe",
              all(pool_id_for(k) == v for k, v in L["pool_id_for"].items()),
              json.dumps(L["pool_id_for"], ensure_ascii=False))
        check("the container names agree",
              L["fingerprint"] == fingerprint
              and L["pools_dir"] == __import__("utai_train.pool", fromlist=["x"]).POOLS_DIR,
              "%s / %s" % (L["fingerprint"], L["pools_dir"]))
        # ★ the derivation is pinned to a LITERAL too, or (4) is the two sides agreeing to drift
        # together — the same shape as comparing a table with itself (S105).
        want = "p" + hashlib.sha256(b"abc123").hexdigest()[:12]
        check("…and both agree with an independently computed sha256",
              pool_id_for("abc123") == want == L["pool_id_for"]["abc123"], want)

    print("")
    print("RESULT: %s" % ("ALL PASS" if not FAIL else "FAIL (%d): %s" % (len(FAIL), FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
