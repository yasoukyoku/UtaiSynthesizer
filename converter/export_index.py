# -*- coding: utf-8 -*-
"""S167 (§F2⒟) — build the COMMUNITY faiss retrieval index from a run's total_fea.npy.

The math is upstream RVC 20240604 train_index()'s faiss half (infer-web.py 617-712), ported
verbatim. Our training pipeline's `index_npy.py` already performed the FIRST half exactly as
upstream does (sorted-name concat → seeded row shuffle → >2e5 rows → MiniBatchKMeans to 10000
centers) and saved the result as the run's `total_fea.npy` — that file is the input here, so the
IVF build sees exactly the matrix upstream's faiss.write_index would have been handed.

Runs under the CONVERTER role interpreter (faiss ships in converter/.venv and in every runtime
pack; the TRAINING dev venv deliberately has no faiss — index_npy.py is the "WITHOUT faiss" port).

Deliberate deviations, both stated:
  * the output path is caller-provided and ASCII — faiss's narrow fopen cannot take CJK paths on
    Windows (the S68f2 lesson, from the .index IMPORT side); the Rust caller renames the file to
    its final community name (`added_IVF{nlist}_Flat_nprobe_1_{name}_{version}.index`) afterwards;
  * nlist keeps upstream's min(16*sqrt(N), N//39) but is floored at 1 so a tiny pool still builds
    (upstream would crash on N < 39 — a fixture-scale dataset, not a real one).

Usage: python export_index.py <total_fea.npy> <out.index>
Prints one machine-readable line on success: EXPORT_INDEX_OK rows=<n> dim=<d> nlist=<k>
"""
import sys

import numpy as np


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: export_index.py <total_fea.npy> <out.index>", file=sys.stderr)
        return 2
    src, out = sys.argv[1], sys.argv[2]
    import faiss  # deferred: the error message below names the real problem, not an import trace

    big = np.load(src)
    if big.ndim != 2 or big.shape[0] == 0:
        print(f"EXPORT_INDEX_BAD_FEATURES: shape {big.shape}", file=sys.stderr)
        return 3
    big = np.ascontiguousarray(big.astype(np.float32))
    n_ivf = max(1, min(int(16 * np.sqrt(big.shape[0])), big.shape[0] // 39))
    index = faiss.index_factory(int(big.shape[1]), "IVF%s,Flat" % n_ivf)
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(big)
    batch = 8192  # upstream's add batch size, kept
    for i in range(0, big.shape[0], batch):
        index.add(big[i : i + batch])
    faiss.write_index(index, out)
    print("EXPORT_INDEX_OK rows=%d dim=%d nlist=%d" % (big.shape[0], big.shape[1], n_ivf))
    return 0


if __name__ == "__main__":
    sys.exit(main())
