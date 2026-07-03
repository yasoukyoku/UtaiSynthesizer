"""ONNX fp32 -> fp16 weight conversion for MSST separation models.

THE single fp16 implementation (NO-DUP): imported by convert.py (--precision
fp16/both) and invoked directly as a CLI by the Rust-side post-conversion
command. Recipe numerically proven vs fp32 on all four separation
architectures (45 dB gate): BSRoformer 53.5/59.2 dB, MelBandRoformer
58.4/53.7 dB, MDX23C, and HTDemucs 6-stem (52.9-56.8 dB on the four
non-quiet stems; needs the step-0 fp32 stats island below).

The recipe (all steps are required — ORT load hard-fails otherwise):
0. htdemucs only: keep the input-normalization stats subgraph fp32 via
   node_block_list (see _input_norm_block_list — global spec stats overflow
   AND underflow fp16 on real CUDA kernels; CPU EP masks it).
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
    if {i.name for i in model.graph.input} != {"cac_spec", "mix"}:
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


def convert_onnx_to_fp16(in_path, out_path) -> Path:
    """Convert a fp32 .onnx to fp16 weights/internals (IO kept fp32).

    Returns out_path. Prints input/output sizes and the Cast-retarget count.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    model = onnx.load(str(in_path))
    orig_cast_names = {n.name for n in model.graph.node if n.op_type == "Cast"}
    print(f"original graph has {len(orig_cast_names)} Cast nodes")

    block_list = _input_norm_block_list(model)
    if block_list:
        print(f"keeping {len(block_list)} input-norm stats nodes fp32: {block_list}")
    fp16_model = float16.convert_float_to_float16(
        model, keep_io_types=True, node_block_list=block_list or None)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(fp16_model, str(out_path))

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
