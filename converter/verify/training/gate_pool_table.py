# -*- coding: utf-8 -*-
r"""§F2⒝ — drive the Rust pool DECISION TABLE against what python actually does.

`tpool::POOL_ENTRIES` decides which of a slot's top-level entries the layout migration moves into
`pools/<identity>/`. python decides where it WRITES them. Nothing connects the two, so a new
preprocessing product added on the python side would simply be left at the slot root by the
migration — visible only as a rebuild costing hours, months later, on someone else's machine.

This gate is that connection, in BOTH directions:

  (1) every directory python joins onto a POOL base is named in the Rust table for that family;
  (2) every entry the Rust table names is actually produced by python;
  (3) no pool product is ever joined onto the SLOT (`exp_dir`) — that is the exact regression the
      relocation could suffer, and it is silent (the product lands where nothing reads it);
  (4) the two `pool_id_for` implementations agree, because Rust names the FIRST pool of every
      migrated slot and python names every pool minted afterwards.

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
    for fam, lit, const in rows:
        name = lit if lit else {"FINGERPRINT": fingerprint}.get(const)
        if name is None:
            raise SystemExit("gate parser: unresolved const in POOL_ENTRIES: %r" % const)
        for f in (families if fam == "*" else [fam]):
            table[f].add(name)
    return table, fingerprint, families


# ─────────────────────────── the python side, parsed with ast ───────────────────────────

#: bases that ARE the pool (or a directory inside it). Everything else is the slot / a run dir.
POOL_BASES = {"pool_dir", "self.pool_dir"}
#: bases that are the SLOT. A pool product joined onto one of these is finding (3).
#:
#: ⚠ `ws` and `self.workspace` were listed here and the census below found they reach NOTHING —
#: no join site in the tree uses either name. They are gone: a classification nothing can reach
#: cannot be wrong in a way anything notices, so it rots into a false claim of coverage (the same
#: lesson `run_id_is_usable` in `trun.rs` records about its own deleted clauses). Nothing is lost,
#: because check (0c) refuses any base it has not been told about — so if `ws` ever comes back it
#: arrives as a FAILURE demanding classification, not as a silent gap.
SLOT_BASES = {"exp_dir", "slot_dir", "workspace", "work_dir"}
#: ★§F2⒝ batch 2 — EVERY base name the scan finds, whether or not this gate classifies it.
#:
#: The two sets above are what checks (1)-(3) act on, and a join onto anything else is simply
#: skipped. That is fine while the unclassified names are all leaf directories already derived
#: from a pool or a slot — but it means the gate LOSES COVERAGE SILENTLY the moment python grows a
#: new base. Batch 3 introduces exactly that (`run_dir`), and the `len(joins) > 60` floor below
#: cannot notice: re-basing `os.path.join(exp_dir, "weights")` to `os.path.join(run_dir, "weights")`
#: leaves the site count unchanged while moving the site out of every check.
#:
#: So the census is declared. It is a source-vs-declaration ratchet, not a table compared with
#: itself: the left side is parsed out of the python tree by the AST scan below. Adding a base
#: turns check (0c) red, and whoever adds it has to say which of the two sets it belongs to (or
#: that it belongs to neither, here).
KNOWN_BASES = {
    # the two the checks act on
    "exp_dir", "slot_dir", "workspace", "work_dir",
    "pool_dir",
    # leaf directories, each already derived from a pool or a slot by the code that binds them
    "weights_dir", "flist_dir", "configs_dir", "cluster_dir", "meta_dir",
    "slices_dir", "slice_dir", "out_dir", "out_spk_dir", "tmp",
    "self.gt_wavs_dir", "self.wavs16k_dir",
    # the shallow-diffusion expdir, which is a RUN directory (`<run>/diffusion`) — it is not a
    # slot and not a pool, and batch 3 is where it stops being derived from the slot
    "expdir", "self.expdir",
    # so-vits' own name for the run directory it writes checkpoints into
    "hps.model_dir",
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


def main():
    table, fingerprint, families = rust_table()
    joins = scan_joins(TRAIN)
    all_pool_names = set().union(*table.values())
    print("Rust table: %s" % json.dumps({k: sorted(v) for k, v in table.items()},
                                        ensure_ascii=False))
    print("python: %d os.path.join(<name>, \"<literal>\") sites" % len(joins))

    # ── (0) the parsers themselves, before anything they produce is believed ────────────────
    print("\n(0) the readers")
    check("the Rust table covers every family", set(table) == set(families), str(sorted(table)))
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
    print("\n(3) no pool product is written to the SLOT root")
    stray = []
    for rel, line, base, lit in joins:
        if lit.startswith(DYNAMIC):
            # a formatted name joined onto a SLOT base is just as bad — compare on the prefix
            pat = lit[len(DYNAMIC):]
            if base in SLOT_BASES and pat and any(n.startswith(pat) for n in all_pool_names):
                stray.append("%s:%d %s(%s…)" % (rel, line, base, pat))
            continue
        if base in SLOT_BASES and lit in all_pool_names and lit != fingerprint:
            stray.append("%s:%d %s(%s)" % (rel, line, base, lit))
    check("no pool product joined onto a slot base", not stray, str(stray[:5]))

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
