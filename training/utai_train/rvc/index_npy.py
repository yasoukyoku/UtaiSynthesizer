# Ported from RVC 20240604 infer-web.py train_index() lines 617-712, WITHOUT faiss.
# The app's runtime KNN (src-tauri inference/features.rs) does exact brute-force
# search over the raw [N,dim] f32 matrix — mathematically what faiss IndexFlatL2
# stores — so the IVF build step is dead weight; we save what upstream calls
# total_fea.npy and that IS the retrieval asset (<model>.npy next to the onnx).
# Preserved: sorted-name concat, row shuffle, >2e5 rows -> MiniBatchKMeans down to
# 10000 centers (batch_size 256*n_cpu, init random, compute_labels False; on kmeans
# failure fall back to the raw matrix). Deviation: seeded shuffle, float32 cast
# (faiss stored f32 anyway).
import logging
import os
import traceback

import numpy as np

from ..augment import is_aug_name

logger = logging.getLogger(__name__)


def build_index(exp_dir, version, seed, reporter, n_cpu=None):
    reporter.stage("index", message="构建检索特征库")
    feature_dir = os.path.join(
        exp_dir, "3_feature256" if version == "v1" else "3_feature768"
    )
    # S41: the retrieval asset is built from ORIGINAL slices only — PSOLA aug
    # copies are near-duplicate vectors that inflate the index for ~zero
    # retrieval benefit, and excluding them keeps total_fea.npy identical to a
    # non-augmented run (also immunizes against stale aug orphans; design B3)
    names = sorted(n for n in os.listdir(feature_dir) if not is_aug_name(n))
    if not names:
        raise RuntimeError("特征目录为空，无法构建检索库")
    npys = []
    for name in names:
        npys.append(np.load(os.path.join(feature_dir, name)))
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.RandomState(seed).shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        reporter.stage(
            "index", message="特征量 %s 行，KMeans 压缩到 10000 中心" % big_npy.shape[0]
        )
        try:
            from sklearn.cluster import MiniBatchKMeans

            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    batch_size=256 * (n_cpu or os.cpu_count() or 4),
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception:
            logger.error("kmeans failed, keeping raw matrix\n%s", traceback.format_exc())

    big_npy = np.ascontiguousarray(big_npy.astype(np.float32))
    out_path = os.path.join(exp_dir, "total_fea.npy")
    np.save(out_path, big_npy, allow_pickle=False)
    logger.info("index matrix saved: %s rows -> %s", big_npy.shape[0], out_path)
    reporter.stage("index", done=1, total=1)
    return out_path, int(big_npy.shape[0])
