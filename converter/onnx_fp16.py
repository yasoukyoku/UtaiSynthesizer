"""ONNX fp32 -> fp16 weight conversion for MSST separation models.

THE single fp16 implementation (NO-DUP): imported by convert.py (--precision
fp16/both) and invoked directly as a CLI by the Rust-side post-conversion
command. Recipe numerically proven vs fp32 on both roformer architectures
(BSRoformer 53.5/59.2 dB, MelBandRoformer 58.4/53.7 dB — both pass the 45 dB
gate); mdx23c/htdemucs are NOT yet verified and are refused by convert.py.

The recipe (all three steps are required — ORT load hard-fails otherwise):
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


def convert_onnx_to_fp16(in_path, out_path) -> Path:
    """Convert a fp32 .onnx to fp16 weights/internals (IO kept fp32).

    Returns out_path. Prints input/output sizes and the Cast-retarget count.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    model = onnx.load(str(in_path))
    orig_cast_names = {n.name for n in model.graph.node if n.op_type == "Cast"}
    print(f"original graph has {len(orig_cast_names)} Cast nodes")

    fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)

    # Only casts carried over from the torch export have stale to=FLOAT;
    # converter-inserted casts (io boundary etc.) are new names with correct
    # `to` attrs. Exact rule: the node name existed before conversion.
    fixed = 0
    for node in fp16_model.graph.node:
        if node.op_type != "Cast" or node.name not in orig_cast_names:
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
