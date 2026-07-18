"""Static evidence gate: inspect Clip clamp floors inside norm spines of fp16 graphs.

Old recipe: the F.normalize eps (1e-12) rounds to +0.0 in fp16 -> the division
guard is DEAD (Clip(min=0)) => silent band -> 0/0 = NaN on true-fp16 kernels.
New recipe: spine kept fp32 -> the 1e-12 floor must SURVIVE.

Usage: python check_fp16_graph.py <model.onnx> ...
Prints, per model: Clip nodes count by (dtype of min, min value class).
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


def const_value(model, name, _depth=0, _fp16_transit=False):
    """Resolve a constant through Cast/Identity chains, TRACKING whether the value
    transits an fp16 tensor on the way (which squashes it at RUNTIME even if the
    source constant is fp32 — the S68c round-trip blind spot)."""
    if _depth > 4:
        return None, _fp16_transit
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init), _fp16_transit
    for n in model.graph.node:
        if n.output and n.output[0] == name:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        return numpy_helper.to_array(a.t), _fp16_transit
            if n.op_type in ("Cast", "Identity"):
                to = next((a.i for a in n.attribute if a.name == "to"), None)
                transit = _fp16_transit or (n.op_type == "Cast" and to == TensorProto.FLOAT16)
                return const_value(model, n.input[0], _depth + 1, transit)
            return None, _fp16_transit
    return None, _fp16_transit


for path in sys.argv[1:]:
    m = onnx.load(path, load_external_data=False)
    from collections import Counter
    stats = Counter()
    samples = {}
    for n in m.graph.node:
        if n.op_type != "Clip" or len(n.input) < 2 or not n.input[1]:
            continue
        v, transit = const_value(m, n.input[1])
        if v is None:
            stats[("min=dynamic", "?")] += 1
            continue
        dt = str(v.dtype)
        val = float(np.asarray(v).reshape(-1)[0])
        if transit and abs(val) < 6e-8:
            cls = "FP16-TRANSIT->ZERO(dead)"  # fp32 source but squashed en route
        elif val == 0.0:
            cls = "ZERO(dead-guard)"
        elif val < 1e-6:
            cls = "tiny(alive)" + ("+fp16transit!" if transit else "")
        else:
            cls = f"{val:g}"
        stats[(dt, cls)] += 1
        samples.setdefault((dt, cls), n.name)
    print(f"\n== {path}")
    for k, c in sorted(stats.items()):
        print(f"  {c:5d}  min dtype={k[0]:8s} class={k[1]}  e.g. {samples.get(k)}")
