"""Dataset identity, shared by all five training backends (extracted from rvc/pipeline.py in the
SoVITS port — single source of truth per the no-duplication rule).

The per-file extraction caches (f0 / features / spec / ...) are keyed by SLICE FILE NAME — after a
dataset change the re-sliced wavs reuse the same names with different content, so stale cache
entries would silently mismatch. That is why the products of one preprocessing identity must
never be mixed with another's.

⚠ §F2⒝ moved the *consequence* of an identity change out of this module. `invalidate_extract_caches`
used to compare the stored fingerprint and `shutil.rmtree` the named cache directories on a
mismatch — flip `loudnorm`, lose the slices; flip it back, pay for them again, with one
`logger.info` line as the only trace. The identity now NAMES a directory instead
(`utai_train.pool`), so a different identity is a sibling pool and nothing is deleted. What stays
here is the identity itself, because it is the one thing all five chains genuinely share.
"""
import hashlib
import logging
import os

logger = logging.getLogger(__name__)


def dataset_fingerprint(dataset_dir):
    """Content identity of the imported dataset (name + size + head/tail sample).

    The directory must hold FILES only. Since S76 the dataset lives at the project level and
    is shared by every architecture slot, so a flat-dataset backend (vocoder, sovits_diff)
    can be pointed at a multi-speaker project whose dataset is one subdirectory per speaker.
    That combination must fail LOUDLY here: skipping subdirectories instead would fingerprint
    the empty set — a constant — so every run would resolve to the SAME pool no matter what the
    speakers' data is, keep stale products forever, and the slicer would then produce zero
    slices. The
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
