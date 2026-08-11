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

# S134 (§F7 笔 5):`<n>.<ext>.part` 是 Rust 导入阶段 stage-then-rename 的**崩溃残留**。
# `dsmanifest.rs` 的 rule 2 原话:「Copies land on a `.part` name in the same directory and are
# renamed into place, so a crash mid-copy cannot leave a truncated wav that `has_dataset` would
# accept and a run would then slice.」—— 那句承诺靠**每一个读者都跳过它**才成立,而 S78 记的
# 「读侧三处跳过」只兑现在 Rust 侧(`dsmanifest.rs:236/247` 与 `mod.rs:2779-2795`);
# python 侧**四处一个都不跳**,`has_dataset` 自己也不跳。可达序列:导入时被硬杀(copy↔rename
# 之间)→ 用户直接开训 → ⑴ 指纹把 `.part` 算进去 ⇒ 解析到一个**新池** ⇒ 白重跑几小时预处理;
# ⑵ 切片器把那个**截断的 wav** 真的送进 ffmpeg;⑶ 声码器链更是**硬崩**而不是换池 ——
# `_probe_sr` 读不出 header 时返回 None,`slice_dataset` 的 sr 闸对 None **放行**,
# 随后 `_decode` 没有 try 保护。
# ⇒ 单一真源放这里(五条链都已经 import 本模块的 `dataset_fingerprint`)。
PART_SUFFIX = ".part"


def dataset_entries(dataset_dir):
    """数据集目录里【真正算数】的条目名,按名字排序,跳过 `.part` 崩溃残留。

    ⛔ 只跳 `.part`,不跳子目录 —— 子目录是「多歌手项目喂给平铺数据集后端」这条要
    **响亮失败**的形状,由 `dataset_fingerprint` 自己抛(见它的 docstring)。
    这里少跳一样东西都会把那条断言变成静默通过。
    """
    return sorted(n for n in os.listdir(dataset_dir) if not n.endswith(PART_SUFFIX))


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
    for name in dataset_entries(dataset_dir):
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
