# Replaces RVC's extract_feature_print.py (fairseq HuBERT/ContentVec) with the
# project's own ContentVec ONNX extractors — S35-exported and gate-verified
# bit-identical to the fairseq originals (converter/export_contentvec.py):
#   v1 -> contentvec_256l9.onnx  (layer 9 + final_proj, 256-dim — same as fairseq
#         output_layer=9 + model.final_proj)
#   v2 -> contentvec_768l12.onnx (layer 12 raw, 768-dim)
# This kills the fairseq dependency from the training venv AND guarantees the
# training feature space equals the inference feature space (one-ContentVec-space
# principle). Output layout is unchanged: 3_feature{256,768}/<stem>.npy, [T,dim]
# float32 at 50 fps, NaN-checked before save, skip-if-exists.
import logging
import os
import traceback

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ContentVec conv frontend needs at least 400 samples @16k (S35 aux contract)
MIN_SAMPLES = 400


def extract_features(exp_dir, version, contentvec_onnx, reporter, stop):
    import onnxruntime as ort

    inp_root = os.path.join(exp_dir, "1_16k_wavs")
    out_root = os.path.join(
        exp_dir, "3_feature256" if version == "v1" else "3_feature768"
    )
    os.makedirs(out_root, exist_ok=True)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        contentvec_onnx, so, providers=["CPUExecutionProvider"]
    )

    names = [n for n in sorted(os.listdir(inp_root)) if n.endswith(".wav")]
    failed = 0
    for n, name in enumerate(names):
        stop.check()
        reporter.stage("feature", done=n, total=len(names), message=name)
        try:
            # original: out name via name.replace("wav", "npy")
            out_path = os.path.join(out_root, name.replace("wav", "npy"))
            if os.path.exists(out_path):
                continue
            wav, sr = sf.read(os.path.join(inp_root, name))
            assert sr == 16000, "expected 16k wav, got %s" % sr
            if wav.ndim == 2:
                wav = wav.mean(-1)
            wav = wav.astype(np.float32)
            if len(wav) < MIN_SAMPLES:
                logger.warning("%s shorter than %s samples, skipped", name, MIN_SAMPLES)
                continue
            feats = sess.run(["features"], {"waveform": wav[None, :]})[0][0]
            if np.isnan(feats).sum() == 0:
                np.save(out_path, feats.astype(np.float32), allow_pickle=False)
            else:
                failed += 1
                logger.warning("%s contains nan feature, skipped", name)
        except Exception:
            failed += 1
            logger.error("feature failed for %s\n%s", name, traceback.format_exc())
    reporter.stage("feature", done=len(names), total=len(names))
    if names and failed == len(names):
        raise RuntimeError("所有切片的特征提取均失败（详见日志）")
