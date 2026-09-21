# Retrieval / kmeans assets from the extracted ContentVec features. Runs right
# after extraction (before training) so an early stop still leaves a usable
# index — same by-construction policy as the RVC trainer.
#
# Retrieval (default): the app's runtime KNN (inference/features.rs) does exact
# brute-force search over a raw [N, dim] f32 matrix — mathematically what the
# upstream faiss "IVF,Flat" index stores and what its inference reads back via
# reconstruct_n. We skip faiss (RVC index_npy precedent) and emit the matrix
# directly under the file name contract of converter/export_cluster.py:
# <speaker_id>.index_vectors.npy. Preserved from upstream utils.train_index:
# sorted-name concat of *.wav.soft.pt ([1,dim,T] -> [T,dim]), row shuffle,
# >2e5 rows -> MiniBatchKMeans down to 10000 centers. Deviation: seeded shuffle.
#
# Kmeans (optional): upstream cluster/train_cluster.py output format, one dict
# per speaker {"n_features_in_", "_n_threads", "cluster_centers_"} saved as
# kmeans_10000.pt keyed by SPEAKER NAME. The import chain feeds this .pt to
# converter/export_cluster.py which owns the <safe_name>.centers.npy naming —
# no second copy of the sanitize rules here. Deviations: uses the upstream
# use_minibatch code path (MiniBatchKMeans batch_size=4096 max_iter=80; the
# upstream default full KMeans over 10000 clusters is intractable on real
# datasets), n_clusters clamped to the row count (upstream errors out when a
# small dataset has fewer rows than clusters).
import logging
import os
import traceback

import numpy as np
import torch

from ..augment import is_aug_name

logger = logging.getLogger(__name__)

N_CLUSTERS = 10000


def _load_features(spk_dir, stop):
    # S41: retrieval/kmeans assets are built from ORIGINAL slices only —
    # PSOLA aug copies are near-duplicate ContentVec vectors (formant-
    # preserving) that inflate the index ~x(1+copies) with ~zero retrieval
    # benefit, and excluding them keeps the inference-side assets identical
    # to a non-augmented run (design S41 B3, red-team F3/R3/A3)
    names = [
        n
        for n in sorted(os.listdir(spk_dir))
        if n.endswith(".wav.soft.pt") and not is_aug_name(n)
    ]
    if not names:
        raise RuntimeError("特征目录为空，无法构建检索/聚类库")
    mats = []
    for name in names:
        stop.check()
        phone = torch.load(os.path.join(spk_dir, name), map_location="cpu", weights_only=False)
        mats.append(phone[0].transpose(-1, -2).numpy())  # [T, dim]
    return np.concatenate(mats, 0)


def build_retrieval(exp_dir, spk_dir, seed, reporter, stop, n_cpu=None, spk_id=0):
    reporter.stage("index", message="构建检索特征库")
    big_npy = _load_features(spk_dir, stop)

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
                    n_clusters=N_CLUSTERS,
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
    cluster_dir = os.path.join(exp_dir, "cluster")
    os.makedirs(cluster_dir, exist_ok=True)
    # file name contract of export_cluster.py = "<speaker_id>.index_vectors.npy".
    # spk_id defaults to 0 (single-speaker; byte-identical to pre-①c + the gate);
    # ①c calls this once per speaker with its list-order id.
    out_path = os.path.join(cluster_dir, "%d.index_vectors.npy" % spk_id)
    np.save(out_path, big_npy, allow_pickle=False)
    logger.info("retrieval matrix saved: %s rows -> %s", big_npy.shape[0], out_path)
    reporter.stage("index", done=1, total=1)
    return out_path, int(big_npy.shape[0])


def _fit_kmeans(features):
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = min(N_CLUSTERS, features.shape[0])
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, verbose=False, batch_size=4096, max_iter=80
    ).fit(features)
    return kmeans, n_clusters


def build_kmeans(exp_dir, named_spk_dirs, reporter, stop):
    """upstream kmeans_10000.pt = one dict keyed by SPEAKER NAME (export_cluster.py
    owns <safe_name>.centers.npy naming). `named_spk_dirs` = [(name, spk_dir), ...];
    single-speaker passes one pair -> byte-identical to the pre-①c {name:{...}} dict.
    ①c accumulates every co-trained speaker into the one .pt."""
    reporter.stage("index", message="训练聚类中心 (kmeans)")
    ckpt = {}
    total = 0
    dim = 0
    for name, spk_dir in named_spk_dirs:
        features = _load_features(spk_dir, stop).astype(np.float32)
        stop.check()
        kmeans, n_clusters = _fit_kmeans(features)
        ckpt[name] = {
            "n_features_in_": kmeans.n_features_in_,
            "_n_threads": kmeans._n_threads,
            "cluster_centers_": kmeans.cluster_centers_,
        }
        total += n_clusters
        dim = features.shape[1]
    cluster_dir = os.path.join(exp_dir, "cluster")
    os.makedirs(cluster_dir, exist_ok=True)
    out_path = os.path.join(cluster_dir, "kmeans_%d.pt" % N_CLUSTERS)
    tmp = out_path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, out_path)
    logger.info("kmeans centers saved: %s speakers, %s x %s -> %s",
                len(ckpt), total, dim, out_path)
    reporter.stage("index", done=1, total=1)
    return out_path, int(total)
