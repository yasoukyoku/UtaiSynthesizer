"""Locate the FIRST non-finite intermediate tensor (topological order) for a given
input regime, by re-exposing sampled node outputs as graph outputs.

Usage: python nan_bisect.py <model.onnx> <device: cpu|dml0|dml1> <regime: zeros|tiny|randn3> [probe_count]
Iteratively refines: round 1 samples N probes across the node list, finds the first
bad probe, then re-probes densely between the previous good probe and the bad one.
"""
import sys

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

ort.set_default_logger_severity(3)

model_path, device, regime = sys.argv[1], sys.argv[2], sys.argv[3]
probe_n = int(sys.argv[4]) if len(sys.argv) > 4 else 120

base = onnx.load(model_path)
base.ir_version = min(base.ir_version, 10)
nodes = list(base.graph.node)
all_outs = [(i, o) for i, n in enumerate(nodes) for o in n.output if o]
print(f"{len(nodes)} nodes / {len(all_outs)} tensors")

providers = (["CPUExecutionProvider"] if device == "cpu"
             else [("DmlExecutionProvider", {"device_id": int(device[-1])})])


def feeds_for(sess):
    rng = np.random.RandomState(7)
    f = {}
    for i in sess.get_inputs():
        shape = []
        for d in i.shape:
            shape.append(d if isinstance(d, int) and d > 0 else (1 if not shape else 801))
        if regime == "zeros":
            f[i.name] = np.zeros(shape, np.float32)
        elif regime == "tiny":
            f[i.name] = (rng.randn(*shape) * 1e-5).astype(np.float32)
        else:
            f[i.name] = (rng.randn(*shape) * 3).astype(np.float32)
    return f


def probe(indices):
    m = onnx.ModelProto()
    m.CopyFrom(base)
    del m.graph.output[:]
    names = []
    for idx in indices:
        _, o = all_outs[idx]
        m.graph.output.append(helper.make_empty_tensor_value_info(o))
        names.append(o)
    onnx.save(m, model_path + ".probe.onnx")
    s = ort.InferenceSession(model_path + ".probe.onnx", providers=providers)
    outs = s.run(None, feeds_for(s))
    res = []
    for name, o in zip(names, outs):
        arr = np.asarray(o)
        bad = int((~np.isfinite(arr)).sum()) if arr.dtype.kind == "f" else 0
        res.append((name, bad, arr.dtype, arr.shape))
    del s
    return res


lo, hi = 0, len(all_outs) - 1
step_indices = sorted(set(np.linspace(lo, hi, probe_n, dtype=int).tolist()))
for round_no in range(6):
    res = probe(step_indices)
    first_bad = None
    prev_good = 0
    for k, (name, bad, dt, shp) in enumerate(res):
        if bad > 0:
            first_bad = k
            break
        prev_good = step_indices[k]
    if first_bad is None:
        print(f"round {round_no}: no non-finite probe among {len(step_indices)} probes — done (clean?)")
        break
    bad_idx = step_indices[first_bad]
    name, bad, dt, shp = res[first_bad]
    node_i, _ = all_outs[bad_idx]
    print(f"round {round_no}: first bad probe -> tensor '{name}' (node #{node_i} {nodes[node_i].op_type}, "
          f"dtype={dt}, shape={shp}, nonfinite={bad}); window=({prev_good},{bad_idx})")
    if bad_idx - prev_good <= 1:
        n = nodes[node_i]
        print("CULPRIT node:", n.op_type, "name:", n.name, "inputs:", list(n.input), "outputs:", list(n.output))
        break
    step_indices = sorted(set(np.linspace(prev_good, bad_idx, min(probe_n, bad_idx - prev_good + 1), dtype=int).tolist()))
