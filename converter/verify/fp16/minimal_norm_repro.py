"""Minimal repro: does an fp32-protected F.normalize island survive DML, and does the
pure-fp16 version NaN on zero/tiny input?

Graph (input x: fp32 [1, 8], keep_io_types style):
  branch A (pure fp16): xh=Cast16(x); n=ReduceL2(xh); c=Clip(n, min=fp16(1e-7));
                        e=Expand(c,[1,8]); outA=Cast32(xh/e)
  branch B (fp32 island): n=ReduceL2(x); c=Clip(n, min=1e-12); e=Expand(c,[1,8]);
                        outB=x/e   (all fp32)
Run zeros / 1e-5 / 1.0-scale inputs on CPU + DML dev0/dev1.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

ort.set_default_logger_severity(3)

f16 = TensorProto.FLOAT16
f32 = TensorProto.FLOAT

shape_init = numpy_helper.from_array(np.array([1, 8], np.int64), "shp")
min16 = numpy_helper.from_array(np.array(1e-7, np.float16), "min16")  # rounds to subnormal ~1.19e-7
min32 = numpy_helper.from_array(np.array(1e-12, np.float32), "min32")

nodes = [
    # branch A — pure fp16
    helper.make_node("Cast", ["x"], ["xh"], to=f16),
    helper.make_node("ReduceL2", ["xh"], ["nA"], axes=[1], keepdims=1),
    helper.make_node("Clip", ["nA", "min16"], ["cA"]),
    helper.make_node("Expand", ["cA", "shp"], ["eA"]),
    helper.make_node("Div", ["xh", "eA"], ["dA"]),
    helper.make_node("Cast", ["dA"], ["outA"], to=f32),
    # branch B — fp32 island
    helper.make_node("ReduceL2", ["x"], ["nB"], axes=[1], keepdims=1),
    helper.make_node("Clip", ["nB", "min32"], ["cB"]),
    helper.make_node("Expand", ["cB", "shp"], ["eB"]),
    helper.make_node("Div", ["x", "eB"], ["outB"]),
]
graph = helper.make_graph(
    nodes, "norm_repro",
    [helper.make_tensor_value_info("x", f32, [1, 8])],
    [helper.make_tensor_value_info("outA", f32, [1, 8]),
     helper.make_tensor_value_info("outB", f32, [1, 8])],
    initializer=[shape_init, min16, min32],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8  # installed onnxruntime-directml caps at IR 11; onnx 1.20 defaults to 13
onnx.checker.check_model(model)
path = sys.argv[1] if len(sys.argv) > 1 else "norm_repro.onnx"
onnx.save(model, path)

cases = {
    "zeros": np.zeros((1, 8), np.float32),
    "tiny 1e-5": np.full((1, 8), 1e-5, np.float32),
    "ones": np.ones((1, 8), np.float32),
}


def show(providers, tag):
    try:
        s = ort.InferenceSession(path, providers=providers)
    except Exception as e:
        print(f"[{tag}] session failed: {str(e)[:100]}")
        return
    for name, x in cases.items():
        a, b = s.run(None, {"x": x})
        print(f"[{tag}] {name:<10} A(fp16-norm)={a.ravel()[:3]}  B(fp32-island)={b.ravel()[:3]}")


show(["CPUExecutionProvider"], "CPU")
show([("DmlExecutionProvider", {"device_id": 0})], "DML0")
show([("DmlExecutionProvider", {"device_id": 1})], "DML1")
