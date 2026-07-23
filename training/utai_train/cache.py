"""Dataset fingerprint / extraction-cache invalidation, shared by all training
backends (extracted from rvc/pipeline.py in the SoVITS port — single source of
truth per the no-duplication rule).

The per-file extraction caches (f0 / features / spec / ...) are keyed by SLICE
FILE NAME — after a dataset change the re-sliced wavs reuse the same names with
different content, so stale cache entries would silently mismatch. Fingerprint
change → wipe the caches the backend names.
"""
import hashlib
import logging
import os
import shutil

logger = logging.getLogger(__name__)


def dataset_fingerprint(dataset_dir):
    """Content identity of the imported dataset (name + size + head/tail sample).

    The directory must hold FILES only. Since S76 the dataset lives at the project level and
    is shared by every architecture slot, so a flat-dataset backend (vocoder, sovits_diff)
    can be pointed at a multi-speaker project whose dataset is one subdirectory per speaker.
    That combination must fail LOUDLY here: skipping subdirectories instead would fingerprint
    the empty set — a constant — so `invalidate_extract_caches` would conclude "data never
    changes", keep stale caches forever, and the slicer would then produce zero slices. The
    Rust side refuses the combination first (PROJECT_DATASET_SHAPE); this is the assertion
    behind it.
    """
    h = hashlib.blake2b(digest_size=16)
    for name in sorted(os.listdir(dataset_dir)):
        p = os.path.join(dataset_dir, name)
        if os.path.isdir(p):
            raise RuntimeError(
                "DATASET_SHAPE_UNEXPECTED: %s contains a subdirectory (%s); this backend "
                "expects a flat dataset" % (dataset_dir, name)
            )
        st = os.stat(p)
        h.update(name.encode("utf-8"))
        h.update(str(st.st_size).encode())
        with open(p, "rb") as f:
            h.update(f.read(65536))
            if st.st_size > 131072:
                f.seek(-65536, 2)
                h.update(f.read(65536))
    return h.hexdigest()


def invalidate_extract_caches(exp_dir, fingerprint, cache_subdirs):
    """Compare against the stored fingerprint; on change delete cache_subdirs
    (paths relative to exp_dir), then store the new fingerprint."""
    fp_file = os.path.join(exp_dir, "dataset.fingerprint")
    old = None
    if os.path.exists(fp_file):
        with open(fp_file, encoding="utf-8") as f:
            old = f.read().strip()
    if old != fingerprint:
        if old is not None:
            logger.info("dataset changed — clearing stale extraction caches")
        for sub in cache_subdirs:
            d = os.path.join(exp_dir, sub)
            if os.path.isdir(d):
                shutil.rmtree(d)
    with open(fp_file, "w", encoding="utf-8") as f:
        f.write(fingerprint)
