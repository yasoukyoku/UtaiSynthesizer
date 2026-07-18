"""ONNX fp32 -> fp16 weight conversion for MSST separation models.

THE single fp16 implementation (NO-DUP): imported by convert.py (--precision
fp16/both) and invoked directly as a CLI by the Rust-side post-conversion
command. Recipe numerically proven vs fp32 on all four separation
architectures (45 dB gate; S68c re-gate with the norm-stats protection):
BSRoformer 72.8/67.9 dB, MelBandRoformer 68.7/63.5 dB (S33 fused-attention
exports — the com.microsoft RotaryEmbedding caches store cos/sin VALUES,
fixing the old decomposed graphs' fp16 large-angle rotary degradation),
MDX23C 71.0/75.5 dB (S68c recipe byte-identical, gate inherited), and
HTDemucs 6-stem (52.9-56.8 dB on the four non-quiet stems; needs the step-0
fp32 stats island below — its path is byte-preserved by S68c).

The recipe (all steps are required — ORT load hard-fails otherwise):
0. Keep normalization-STATISTICS subgraphs fp32 via node_block_list:
   - htdemucs: the near-input global-stats island (_input_norm_block_list —
     spec stats overflow AND underflow fp16 on real CUDA kernels; CPU EP
     masks it). Kept on its own S31-verified path, byte-for-byte.
   - every other arch (S68c root fix): ALL statistic-division spines
     (_norm_stats_block_list). The roformers' F.normalize RMSNorm
     (ReduceL2→Clip→Expand→Div) is fp16-lethal on true-fp16 kernels: the
     Σx² accumulation overflows (50²·512 ≫ 65504) and a silent band
     underflows to an EXACT 0 norm whose clamp floor (1e-12) is itself 0
     in fp16 → 0/0 = NaN poisoning the whole stem (the shipped 0.5.0 RVC
     crash chain; driver-dependent, which is why some machines never saw
     it). CPU EP emulates fp16 in fp32 and hides all of it.
1. onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)
   — model IO stays fp32; internals + weights become fp16.
2. Retarget Cast nodes carried over from the torch export whose `to=FLOAT` is
   now stale -> `to=FLOAT16`. Only pre-existing casts are touched: converter-
   inserted boundary casts are new names with correct `to` attrs.
3. Clear graph.value_info — stale fp32 intermediate type annotations
   contradict the retyped tensors and fail ORT's load-time type check.

File-naming contract (Rust side depends on it): fp32 = `<stem>.onnx`,
fp16 = `<stem>.fp16.onnx`, BOTH share the ONE `<stem>.json` (never write a
`.fp16.json`).

CLI:
    python onnx_fp16.py <in.onnx> [out.onnx]

Default output is the `<stem>.fp16.onnx` sibling of the input. Contract for
the Rust caller: exit 0 + output file exists on success; message on stderr +
nonzero exit on failure.
"""

import os
import sys
from pathlib import Path

import onnx
from onnx import TensorProto
from onnxconverter_common import float16


def default_fp16_path(in_path) -> Path:
    """`<stem>.fp16.onnx` sibling of `<stem>.onnx` (the file-naming contract)."""
    in_path = Path(in_path)
    return in_path.with_name(in_path.stem + ".fp16.onnx")


def _is_htdemucs(model) -> bool:
    """THE htdemucs dispatch predicate (single source — three call sites): our own
    converter's IO naming contract for htdemucs exports (convert.py input_names)."""
    return {i.name for i in model.graph.input} == {"cac_spec", "mix"}


def _input_norm_block_list(model) -> list:
    """Nodes that must STAY fp32: the htdemucs input-normalization stats.

    HTDemucs normalizes inside the graph: (x - mean) / (1e-5 + std) with
    GLOBAL mean/std over the whole CaC spec. Real spec values span ~1e-5..400,
    so the squared terms span ~1e-11..1e5 — fp16 can represent NEITHER end
    (subnormal floor 6e-8, max 65504). On true fp16 kernels (CUDA EP) the
    variance collapses to 0 or inf and every stem output is NaN. The CPU EP
    hides this by emulating fp16 ops in fp32 — do NOT trust CPU-only checks.

    Detection is structural, not name-based: a normalization Div has a SMALL
    ancestor closure (<= 64 nodes) that touches a graph input and contains a
    ReduceMean. Mid-network norms (LayerNorm etc.) have the whole upstream
    graph as ancestors, so the closure bound excludes them. Blocked nodes are
    kept fp32 by convert_float_to_float16, which auto-inserts boundary casts;
    downstream consumers of mean/std (the output de-normalization) get fp16
    copies, which is safe — the stats VALUES fit fp16 fine, only their
    COMPUTATION doesn't.

    Gated on the htdemucs IO signature (cac_spec + mix, our own converter's
    naming contract) so every other arch keeps the exact conversion path its
    45 dB gate verified (a roformer's first-layer LayerNorm could otherwise
    match the structural rule).
    """
    if not _is_htdemucs(model):
        return []
    producer = {}
    for node in model.graph.node:
        for o in node.output:
            if o:
                producer[o] = node
    input_names = {i.name for i in model.graph.input}

    def closure_of(names):
        closure, queue, seen = [], list(names), set()
        while queue:
            name = queue.pop()
            if name in seen or name not in producer:
                continue
            seen.add(name)
            up = producer[name]
            closure.append(up)
            if len(closure) > 64:
                return None  # unbounded: not a near-input norm
            queue.extend(up.input)
        return closure

    blocked = {}
    for node in model.graph.node:
        if node.op_type != "Div" or len(node.input) < 2:
            continue
        # A norm Div's DENOMINATOR (eps + std) reduces over the data — a
        # ReduceMean ancestor. (A GELU div's denominator is a constant.)
        denom = closure_of([node.input[1]])
        if denom is None or not any(n.op_type == "ReduceMean" for n in denom):
            continue
        full = closure_of(node.input)
        if full is None:
            continue
        if any(i in input_names for n in full + [node] for i in n.input):
            for n in full + [node]:
                if n.name:
                    blocked[n.name] = True
    return list(blocked)


# Ops a normalization-statistics spine may consist of. The denominator walk only
# traverses THESE (bounded); anything else is the data tensor / weights and stops
# the walk — which is what keeps a mid-network norm's spine small even though its
# data input has the entire upstream graph behind it.
_STAT_SPINE_OPS = {
    "Sqrt", "Add", "ReduceMean", "ReduceSumSquare", "ReduceSum", "ReduceL2",
    "Pow", "Mul", "Sub", "Cast", "Constant", "Clip", "Max", "Abs", "Div",
    "Expand", "Unsqueeze", "Squeeze", "Reshape",
}
_STAT_SPINE_BOUND = 24  # census: real spines are 4-9 nodes (BS/MelBand, S68c)


def _norm_stats_block_list(model) -> list:
    """Nodes that must STAY fp32: every normalization-statistics spine (any arch).

    A node divides by a COMPUTED STATISTIC iff its denominator's bounded
    producer walk through _STAT_SPINE_OPS reaches a Reduce* op. Constant
    denominators (GELU's sqrt(2), attention 1/sqrt(d), mean-pool /N) never
    match, so everything the 45 dB gates verified outside norms converts
    exactly as before. Only the spine (denominator statistic computation) is
    blocked — the numerator data path stays fp16, and the auto-inserted
    boundary casts hand the Div fp32 operands whose VALUES all fit fp16 fine;
    it's the statistics' COMPUTATION that doesn't.

    S68c census of the shipped exports: BSRoformer 111× / MelBandRoformer
    132× F.normalize spines (ReduceL2→Clip→Expand→Div, the fp16-lethal
    pattern — see module docstring), MDX23C 0× (fused InstanceNormalization
    only ⇒ empty list ⇒ byte-identical conversion, gate inherited).
    """
    producer = {}
    for node in model.graph.node:
        for o in node.output:
            if o:
                producer[o] = node

    def spine_of(denom_name):
        spine, reduces, queue, seen = [], False, [denom_name], set()
        while queue and len(spine) <= _STAT_SPINE_BOUND:
            name = queue.pop()
            if name in seen or name not in producer:
                continue
            seen.add(name)
            n = producer[name]
            if n.op_type not in _STAT_SPINE_OPS:
                continue  # data tensor / weights — do not traverse past
            spine.append(n)
            if n.op_type.startswith("Reduce"):
                # The statistic ENDS at the reduction — everything upstream of it is the
                # DATA path. Review round: traversing through it escaped into the residual
                # backbone (x=attn(x)+x chains are all Add/Mul, inside _STAT_SPINE_OPS),
                # silently over-blocking at transformer inner-depth 2-6 and silently
                # DROPPING protection past depth 7 (spine > bound). Stop here instead.
                reduces = True
                continue
            queue.extend(n.input)
        if len(spine) > _STAT_SPINE_BOUND:
            if reduces:
                # A norm denominator we RECOGNIZED but cannot bound — shipping it fp16
                # would re-arm the 0/0=NaN class this recipe exists to kill. Refuse
                # loudly; the caller falls back to fp32 (safe), never to a silent hole.
                raise RuntimeError(
                    f"norm-stats spine exceeds bound ({_STAT_SPINE_BOUND}) at denominator "
                    f"'{denom_name}' — refusing fp16 rather than shipping an unprotected "
                    "normalization; convert this model as fp32"
                )
            return [], False
        return spine, reduces

    blocked = {}
    unnamed = 0
    for node in model.graph.node:
        if node.op_type != "Div" or len(node.input) < 2:
            continue
        spine, has_reduce = spine_of(node.input[1])
        if not has_reduce:
            continue
        for n in spine + [node]:
            # node_block_list works BY NAME — an unnamed spine node would silently
            # punch an fp16 hole in the island (third-party exports aren't as
            # thoroughly named as torch's). Name it rather than skip it.
            if not n.name:
                unnamed += 1
                n.name = f"utai_norm_stats_{unnamed}"
            blocked[n.name] = True
    if unnamed:
        print(f"named {unnamed} anonymous norm-stats node(s) so they can be blocked")
    return list(blocked)


def _fp32_block_list(model) -> list:
    """Arch dispatch for the fp32 island (see module docstring step 0).

    htdemucs keeps its EXACT S31-verified path (near-input stats island only —
    its mid-network norms passed the true-CUDA-kernel gate as fp16, so there is
    no evidence-backed reason to touch a byte of that conversion); every other
    arch gets the general statistics protection.
    """
    if _is_htdemucs(model):
        return _input_norm_block_list(model)
    return _norm_stats_block_list(model)


def convert_onnx_to_fp16(in_path, out_path) -> Path:
    """Convert a fp32 .onnx to fp16 weights/internals (IO kept fp32).

    Returns out_path. Prints input/output sizes and the Cast-retarget count.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    model = onnx.load(str(in_path))
    orig_cast_names = {n.name for n in model.graph.node if n.op_type == "Cast"}
    print(f"original graph has {len(orig_cast_names)} Cast nodes")

    block_list = _fp32_block_list(model)
    if block_list:
        print(f"keeping {len(block_list)} norm-stats nodes fp32")
    fp16_model = float16.convert_float_to_float16(
        model, keep_io_types=True, node_block_list=block_list or None)

    # ★S68c: collapse fp16 ROUND-TRIPS inside the kept-fp32 islands. For a blocked
    # node's fp32 output that feeds another blocked node, the library sometimes
    # routes the value through its original (now fp16) tensor name plus a
    # compensating input cast: fp32 → Cast(to=fp16) → Cast(to=fp32) → consumer.
    # That round-trip SQUASHES the value to fp16 en route — the F.normalize eps
    # (1e-12) became +0.0 and Clip(min=0) let a silent band divide 0/0 = NaN in
    # PLAIN fp32 math (found by bisection on the real graph; the minimal repro's
    # hand-built island was fine, so this is purely a wiring artifact). Rewire
    # the outer Cast's consumers straight to the fp32 source; the dangling cast
    # pair is left for ORT's dead-node elimination.
    #
    # htdemucs is EXEMPT: its S31-verified conversion (real-CUDA 52.9-56.8 dB
    # gate) must stay byte-for-byte — collapsing would alter a graph we cannot
    # re-gate here, and its eps (1e-5) never matters on real content (std of a
    # whole-song CaC spec is never ~0). Only the new norm-stats path collapses.
    blocked_set = set(block_list) if not _is_htdemucs(model) else set()
    fp32_products = {
        o for n in fp16_model.graph.node if n.name in blocked_set for o in n.output
    }
    cast_to = {}
    cast_in = {}
    for n in fp16_model.graph.node:
        # LIBRARY-inserted casts only (new names): a cast pair the ORIGINAL export put
        # there deliberately must never be collapsed — its quantization is semantics.
        if n.op_type == "Cast" and n.input and n.output and n.name not in orig_cast_names:
            to = next((a.i for a in n.attribute if a.name == "to"), None)
            cast_to[n.output[0]] = to
            cast_in[n.output[0]] = n.input[0]
    rewired = 0
    for n in fp16_model.graph.node:
        for k, name in enumerate(n.input):
            if cast_to.get(name) != TensorProto.FLOAT:
                continue  # input isn't a to-fp32 cast output
            inner = cast_in[name]
            if cast_to.get(inner) != TensorProto.FLOAT16:
                continue  # not a round-trip
            src = cast_in[inner]
            if src in fp32_products:
                n.input[k] = src
                rewired += 1
    # Prune the now-dangling cast pairs OURSELVES instead of trusting the runtime's
    # dead-node elimination: leaving them in hung the AMD 780M DML driver on the
    # SECOND run (DXGI 887A0006 device-hung; NVIDIA didn't care) — S68c real-GPU find.
    if rewired:
        consumed = {i for n in fp16_model.graph.node for i in n.input if i}
        graph_outs = {o.name for o in fp16_model.graph.output}
        pruned = 0
        while True:
            dead = [
                n for n in fp16_model.graph.node
                if n.op_type == "Cast" and all(o not in consumed and o not in graph_outs for o in n.output)
            ]
            if not dead:
                break
            for n in dead:
                fp16_model.graph.node.remove(n)
                pruned += 1
            consumed = {i for n in fp16_model.graph.node for i in n.input if i}
    else:
        pruned = 0
    print(f"collapsed {rewired} fp16 round-trip(s) inside fp32 islands; pruned {pruned} dead cast(s)")

    # Only casts carried over from the torch export have stale to=FLOAT;
    # converter-inserted casts (io boundary etc.) are new names with correct
    # `to` attrs. Exact rule: the node name existed before conversion.
    # Blocked (kept-fp32) casts must keep their FLOAT target.
    blocked_names = set(block_list)
    fixed = 0
    for node in fp16_model.graph.node:
        if node.op_type != "Cast" or node.name not in orig_cast_names:
            continue
        if node.name in blocked_names:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT16
                fixed += 1
    print(f"retargeted {fixed} Cast nodes FLOAT->FLOAT16")

    # Intermediate type annotations are optional in ONNX; ORT re-infers them.
    del fp16_model.graph.value_info[:]
    print("cleared graph.value_info")

    # Atomic write (review round): onnx.save opens the TARGET with 'wb' — a failed/killed
    # conversion would leave a healthy pre-existing fp16 truncated to garbage, and every
    # consumer (UI badge, separation variant pick) treats existence as health. tmp+replace
    # also closes the "separation starts mid-reconvert and reads a half-written file" window.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    onnx.save(fp16_model, str(tmp_path))
    os.replace(tmp_path, out_path)

    in_size = os.path.getsize(in_path)
    out_size = os.path.getsize(out_path)
    print(f"fp32: {in_path} ({in_size:,} bytes)")
    print(f"fp16: {out_path} ({out_size:,} bytes)")
    return out_path


def main():
    if len(sys.argv) not in (2, 3):
        print("usage: python onnx_fp16.py <in.onnx> [out.onnx]", file=sys.stderr)
        sys.exit(2)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) == 3 else default_fp16_path(in_path)

    if not in_path.exists():
        print(f"Error: input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    try:
        convert_onnx_to_fp16(in_path, out_path)
    except Exception as e:
        print(f"Error: fp16 conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not out_path.exists():
        print(f"Error: fp16 conversion produced no output file: {out_path}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
