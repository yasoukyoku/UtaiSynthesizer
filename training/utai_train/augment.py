"""PSOLA data augmentation shared by all training backends (S41).

Engine decision (evidence archive: TESTING/utai-v2-testing/research/
s41_two_features_design.md B0 + aug_engine_ab/ listening pack): TD-PSOLA via
parselmouth — formant-preserving (measured 0.00 st envelope shift at +/-5 st),
best pitch fidelity of all engines tested (median 1.4-3.0 cents on clean
vocals), zero resynthesis coloration (time-domain grains of the original
waveform). Every "augmentation hurts quality" report in the ecosystem traces
to resampling / spectrum-scaling engines (SingingVocoders key_aug =
torchaudio.Resample; so-vits diff aug = scaled-window mel keyshift) — none
used PSOLA. Range is capped at +/-3 st: PSOLA itself degrades beyond ~3-4 st
and the pitch-timbre covariation critique (Sinsy) is engine-independent.

Invariants (red-team ruling, design doc appendix A):
  - file naming: <stem>_aug<copy_idx>.wav, DIGIT index only; the keyshift
    lives in the meta json (a float in the name would break the RVC
    filelist's split('.')[0] key intersection)
  - meta jsons live in a SEPARATE dir (exp_dir/aug_meta), never in product
    dirs (same split('.')[0] ghost-key hazard)
  - draws use a per-(slice, copy) seeded RNG — independent of processing
    order and cache state (unlike the diff aug's sequential stream)
  - wav writes are atomic (tmp + os.replace); meta is written AFTER the wav,
    so wav+matching-meta together mean "complete" (a truncated wav from a
    mid-write kill must never become a permanent skip-if-exists cache hit)
  - source enumeration REJECTS aug names (no aug-of-aug amplification)
  - copies=0 performs zero generation and zero writes, but stale-aug cleanup
    STILL runs (turning the knob 2 -> 0 must leave the tree byte-identical
    to a fresh copies=0 run)
  - augmented slices never enter val splits or retrieval/cluster/index
    assets — is_aug_name() below is THE single predicate for all of it
"""
import json
import logging
import os
import re
import random
import traceback

import numpy as np

logger = logging.getLogger(__name__)

AUG_STEM_RE = re.compile(r"_aug\d+$")
AUG_PARSE_RE = re.compile(r"^(?P<src>.+)_aug(?P<idx>\d+)$")
AUG_RANGE_MIN = 0.75  # near-zero draws are wasted near-duplicates
AUG_RANGE_MAX = 3.0   # literature safety line; PSOLA's own quality knee
# PSOLA's Manipulation pitch analysis ceiling — the binding constraint for the
# gate's high-pitch headroom accounting (sources whose shifted f0 would exceed
# it cannot be tracked faithfully by the engine OR the detectors)
PITCH_CEILING_HZ = 1100.0
GATE_MEDIAN_CENTS = 30.0
GATE_P90_CENTS = 100.0
GATE_MIN_VOICED_RATIO = 0.30  # breath/noise slices: do not augment at all
GATE_MIN_FRAMES = 10          # too little voiced overlap to verify -> reject
# sources below this are never augmented (审查修复 PY-1): praat's Manipulation
# needs >= 3/60s of signal and throws a raw PraatError below ~50ms — an
# uncaught throw would kill the whole run over OUR OWN product (the A4
# degrade ruling covers generation too). 0.3s aligns with the filelist floor:
# a shorter aug copy could never enter train.txt anyway, so augmenting one is
# pure waste on top of the crash risk.
MIN_SOURCE_SECS = 0.3


def is_aug_name(filename):
    """True when the file belongs to an augmented slice. THE shared predicate
    (source enumeration / slice-deletion exclusion / val-split exclusion /
    retrieval-cluster-index exclusion / stale cleanup) — do not re-implement.
    Works on any product name by checking the first dot-segment:
    '003_001_aug1.wav', '003_001_aug1.wav.soft.pt', '003_001_aug1.spec.pt',
    '003_001_aug1.npz' all match."""
    first = os.path.basename(str(filename)).split(".")[0]
    return AUG_STEM_RE.search(first) is not None


def parse_aug_stem(stem):
    """'003_001_aug2' -> ('003_001', 2); None when not an aug stem."""
    m = AUG_PARSE_RE.match(stem)
    if not m:
        return None
    return m.group("src"), int(m.group("idx"))


def draw_keyshift(seed, source_name, copy_idx):
    """Per-(slice, copy) seeded draw — reproducible regardless of processing
    order or which files happen to be cached. uniform(0.75, 3.0) x random sign."""
    rng = random.Random("%s|aug|%s|%d" % (seed, source_name, copy_idx))
    k = rng.uniform(AUG_RANGE_MIN, AUG_RANGE_MAX)
    if rng.random() < 0.5:
        k = -k
    return k


def psola_shift(x, sr, semitones):
    """TD-PSOLA pitch shift, formant-preserving. x: mono f32 [-1,1].
    Returns f32 with len(y) <= len(x) (trimmed; tolerance-checked)."""
    import parselmouth
    from parselmouth.praat import call as praat_call

    snd = parselmouth.Sound(
        np.asarray(x, dtype=np.float64), sampling_frequency=float(sr)
    )
    manip = praat_call(snd, "To Manipulation", 0.01, 60, PITCH_CEILING_HZ)
    tier = praat_call(manip, "Extract pitch tier")
    praat_call(
        tier, "Multiply frequencies", snd.xmin, snd.xmax, 2.0 ** (semitones / 12.0)
    )
    praat_call([tier, manip], "Replace pitch tier")
    out = praat_call(manip, "Get resynthesis (overlap-add)")
    y = np.asarray(out.values[0], dtype=np.float32)

    if np.any(~np.isfinite(y)):
        raise RuntimeError("PSOLA 输出包含非有限样本")
    tol = max(int(0.0125 * sr), 512)  # ~one 512@44.1k hop
    if abs(len(y) - len(x)) > tol:
        raise RuntimeError(
            "PSOLA 输出长度漂移过大: %d -> %d" % (len(x), len(y))
        )
    if len(y) > len(x):
        y = y[: len(x)]
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.995:
        y = y * (0.995 / peak)
    return y


def _atomic_write(path, write_fn):
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def _meta_path(meta_dir, aug_stem):
    return os.path.join(meta_dir, aug_stem + ".json")


def _read_meta(meta_dir, aug_stem):
    try:
        with open(_meta_path(meta_dir, aug_stem), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_meta(meta_dir, aug_stem, source_name, keyshift):
    os.makedirs(meta_dir, exist_ok=True)

    def w(tmp):
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"source": source_name, "keyshift": round(float(keyshift), 6)},
                f,
                ensure_ascii=False,
            )

    _atomic_write(_meta_path(meta_dir, aug_stem), w)


def _meta_matches(meta, source_name, expected_keyshift):
    return (
        isinstance(meta, dict)
        and meta.get("source") == source_name
        and isinstance(meta.get("keyshift"), (int, float))
        and abs(float(meta["keyshift"]) - expected_keyshift) < 1e-6
    )


def read_wav(path):
    """Shared slice reader (int16 PCM or float32 wav -> f32 [-1,1] mono)."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sr)


def _seed_state_guard(slice_dir, meta_dir, seed, copies, remove_products_fn):
    """Aug file NAMES are seed-independent but their CONTENT is not — a seed
    change with surviving name-keyed downstream caches (rvc feature dirs,
    sovits companions) would silently pair old features with new audio.
    Production seed is constant (Rust hardcodes it), so this is a belt for
    manual/gate fiddling: on seed change, de-aug EVERYTHING first."""
    state_path = os.path.join(meta_dir, "_state.json")
    prev = None
    try:
        with open(state_path, encoding="utf-8") as f:
            prev = json.load(f)
    except Exception:
        pass
    if prev is not None and prev.get("seed") != seed:
        logger.warning("aug seed changed (%s -> %s): removing all aug products",
                       prev.get("seed"), seed)
        stems = set()
        for n in os.listdir(slice_dir):
            if n.endswith(".wav") and is_aug_name(n):
                stems.add(os.path.splitext(n)[0])
        if os.path.isdir(meta_dir):
            for n in os.listdir(meta_dir):
                if n.endswith(".json") and not n.startswith("_"):
                    stems.add(n[: -len(".json")])
        for stem in sorted(stems):
            remove_products_fn(stem)
    if copies > 0:
        os.makedirs(meta_dir, exist_ok=True)

        def _write_state(tmp):
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps({"seed": seed}))

        _atomic_write(state_path, _write_state)
    else:
        # copies=0 leaves ZERO aug artifacts behind (knob-down must equal a
        # fresh copies=0 tree)
        try:
            os.remove(state_path)
        except OSError:
            pass
        try:
            os.rmdir(meta_dir)  # only succeeds when empty — exactly right
        except OSError:
            pass


def augment_slices(
    slice_dir,
    copies,
    seed,
    meta_dir,
    read_fn,          # (path) -> (f32 mono, sr)
    write_fn,         # (tmp_path, samples_f32, sr) -> None  (backend disk format)
    remove_products_fn,  # (aug_stem) -> None  (delete wav + ALL companions + meta)
    reporter,
    stop,
    companion_write_fn=None,  # (aug_stem, samples_f32, sr) after the wav lands
                              # (rvc's 16k twin; owns its own atomic write)
):
    """Generate the _aug copies for every non-aug .wav in slice_dir; prune stale
    ones (idx > copies, missing/mismatching meta, orphaned source). Runs its
    stale cleanup even when copies == 0 (knob-down must fully de-augment).
    Returns (generated, reused, pruned)."""
    _seed_state_guard(slice_dir, meta_dir, seed, copies, remove_products_fn)
    sources = [
        n
        for n in sorted(os.listdir(slice_dir))
        if n.endswith(".wav") and not n.startswith(".") and not is_aug_name(n)
    ]
    source_stems = {os.path.splitext(n)[0] for n in sources}

    # --- stale cleanup (always, including copies == 0) --------------------
    pruned = 0
    for name in sorted(os.listdir(slice_dir)):
        if not name.endswith(".wav") or not is_aug_name(name):
            continue
        stem = os.path.splitext(name)[0]
        parsed = parse_aug_stem(stem)
        stale = parsed is None
        if not stale:
            src_stem, idx = parsed
            src_name = src_stem + ".wav"
            meta = _read_meta(meta_dir, stem)
            stale = (
                idx < 1
                or idx > copies
                or src_stem not in source_stems
                or not _meta_matches(meta, src_name, draw_keyshift(seed, src_name, idx))
            )
        if stale:
            remove_products_fn(stem)
            pruned += 1
    # metas whose wav is gone (rvc wipes its slice dirs every run — these metas
    # are rewritten right after generation; also covers crash windows)
    if os.path.isdir(meta_dir):
        for name in sorted(os.listdir(meta_dir)):
            if not name.endswith(".json") or name.startswith("_"):
                continue
            stem = name[: -len(".json")]
            if not os.path.exists(os.path.join(slice_dir, stem + ".wav")):
                try:
                    os.remove(os.path.join(meta_dir, name))
                except OSError:
                    pass

    if copies <= 0:
        reporter.stage(
            "augment", done=1, total=1, message="数据增强已关闭（份数 0）", force=True
        )
        return 0, 0, pruned

    # too-short sources are excluded from the plan (PY-1); their stems are
    # gone from the plan so any pre-existing aug of theirs was already swept
    # as stale above only if idx>copies — sweep them here explicitly
    def _long_enough(name):
        try:
            import soundfile as sf

            info = sf.info(os.path.join(slice_dir, name))
            return info.frames >= int(MIN_SOURCE_SECS * info.samplerate)
        except Exception:
            return False  # unreadable source: never augment it

    skipped_short = 0
    plan_sources = []
    for n in sources:
        if _long_enough(n):
            plan_sources.append(n)
        else:
            skipped_short += 1
            stem = os.path.splitext(n)[0]
            for idx in range(1, copies + 1):
                if os.path.exists(os.path.join(slice_dir, "%s_aug%d.wav" % (stem, idx))):
                    remove_products_fn("%s_aug%d" % (stem, idx))
    if skipped_short:
        logger.info("augment: skipped %d sub-%.1fs sources", skipped_short, MIN_SOURCE_SECS)

    # --- generation (skip-if-exists = wav + matching meta) ----------------
    total = len(plan_sources) * copies
    generated = 0
    reused = 0
    failed = 0
    n_done = 0
    for src_name in plan_sources:
        src_stem = os.path.splitext(src_name)[0]
        src_audio = None
        src_sr = None
        for idx in range(1, copies + 1):
            stop.check()
            aug_stem = "%s_aug%d" % (src_stem, idx)
            aug_path = os.path.join(slice_dir, aug_stem + ".wav")
            keyshift = draw_keyshift(seed, src_name, idx)
            n_done += 1
            reporter.stage("augment", done=n_done, total=total, message=aug_stem)
            meta = _read_meta(meta_dir, aug_stem)
            if os.path.exists(aug_path) and _meta_matches(meta, src_name, keyshift):
                reused += 1
                continue
            try:
                if src_audio is None:
                    src_audio, src_sr = read_fn(os.path.join(slice_dir, src_name))
                y = psola_shift(src_audio, src_sr, keyshift)
            except Exception:
                # OUR OWN product must never take the run down (A4 degrade
                # ruling, extended to generation by 审查修复 PY-1) — drop the
                # copy, keep the original slice
                logger.warning(
                    "augment failed for %s\n%s", aug_stem, traceback.format_exc()
                )
                remove_products_fn(aug_stem)
                failed += 1
                continue
            _atomic_write(aug_path, lambda tmp: write_fn(tmp, y, src_sr))
            if companion_write_fn is not None:
                companion_write_fn(aug_stem, y, src_sr)
            _write_meta(meta_dir, aug_stem, src_name, keyshift)
            generated += 1
    msg = "生成 %d 片 · 复用 %d 片" % (generated, reused)
    if skipped_short:
        msg += " · 跳过 %d 个过短源" % skipped_short
    if failed:
        msg += " · %d 份生成失败已跳过" % failed
    reporter.stage(
        "augment",
        done=total,
        total=max(total, 1),
        message=msg,
        force=True,
    )
    return generated, reused, pruned


def list_aug_entries(slice_dir, meta_dir):
    """(source_stem, aug_stem, keyshift) for every complete aug slice on disk."""
    entries = []
    if not os.path.isdir(slice_dir):
        return entries
    for name in sorted(os.listdir(slice_dir)):
        if not name.endswith(".wav") or not is_aug_name(name):
            continue
        stem = os.path.splitext(name)[0]
        parsed = parse_aug_stem(stem)
        if parsed is None:
            continue
        meta = _read_meta(meta_dir, stem)
        if not isinstance(meta, dict) or "keyshift" not in meta:
            continue
        entries.append((parsed[0], stem, float(meta["keyshift"])))
    return entries


# frames this close to a voiced-run edge are excluded from the gate stats:
# rmvpe legitimately disagrees by octaves on onset/offset frames between the
# source and its shifted copy, and on real material those 1-2 frames per note
# pushed p90 past the line on copies whose median was near-perfect (S41
# live-test forensics: with this erosion 10/12 borderline rejects pass while
# 0/8 severe rejects do — it separates measurement artifact from real damage)
GATE_EDGE_ERODE = 2


def _edge_distance(mask):
    """Per-frame distance to the nearest voiced/unvoiced transition."""
    edges = np.where(np.diff(mask.astype(np.int8)) != 0)[0]
    if len(edges) == 0:
        return np.full(len(mask), np.inf)
    idx = np.arange(len(mask))
    return np.min(np.abs(idx[:, None] - edges[None, :]), axis=1)


def run_f0_gate(entries, load_f0_fn, remove_products_fn, reporter, stop,
                report_path=None):
    """Post-hoc quality gate over extracted f0 products.
    entries: [(source_stem, aug_stem, keyshift)]
    load_f0_fn(stem) -> (f0_hz f64[N], voiced bool[N]) or None when the
        product is missing/broken (missing aug product -> reject)
    Rejection deletes the aug slice AND all companions via remove_products_fn.
    Criteria (design B2 + live-test recalibration): both-voiced frames after
    min-length truncation and EDGE EROSION (see GATE_EDGE_ERODE), median
    |cents| <= 30 and p90 <= 100 vs source_f0 * 2^(k/12); sources with voiced
    ratio < 30% are not augmentable; frames whose shifted target exceeds the
    PSOLA pitch ceiling are excluded from the p90 arm (headroom).
    Per-copy details go to `report_path` (json) — NOT the log stream: a 50%
    rejection run would flood the frontend ring with a thousand lines."""
    if not entries:
        reporter.stage("aug_check", done=1, total=1, message="无增强样本", force=True)
        return 0, 0

    kept = 0
    rejected = 0
    reasons = {}
    headroom_frames = 0
    copies = []
    src_cache = {}
    for n, (src_stem, aug_stem, keyshift) in enumerate(entries):
        stop.check()
        reporter.stage("aug_check", done=n, total=len(entries), message=aug_stem)
        if src_stem not in src_cache:
            src_cache[src_stem] = load_f0_fn(src_stem)
        src = src_cache[src_stem]
        aug = load_f0_fn(aug_stem)
        reject_reason = None
        med = p90 = None
        if src is None or aug is None:
            reject_reason = "f0 产物缺失"
        else:
            f0_s, v_s = src
            f0_a, v_a = aug
            if float(np.mean(v_s)) < GATE_MIN_VOICED_RATIO:
                reject_reason = "源片浊帧过少"
            else:
                n_frames = min(len(f0_s), len(f0_a))
                mask = v_s[:n_frames] & v_a[:n_frames]
                # edge erosion: keep only frames comfortably INSIDE voiced
                # runs of both tracks
                mask = (
                    mask
                    & (_edge_distance(v_s[:n_frames]) > GATE_EDGE_ERODE)
                    & (_edge_distance(v_a[:n_frames]) > GATE_EDGE_ERODE)
                )
                factor = 2.0 ** (keyshift / 12.0)
                target = f0_s[:n_frames] * factor
                headroom = mask & (target > PITCH_CEILING_HZ)
                headroom_frames += int(headroom.sum())
                eff = mask & ~headroom
                if int(eff.sum()) < GATE_MIN_FRAMES:
                    reject_reason = "有效浊帧不足"
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        cents = 1200.0 * np.log2(f0_a[:n_frames][eff] / target[eff])
                    cents = cents[np.isfinite(cents)]
                    if len(cents) < GATE_MIN_FRAMES:
                        reject_reason = "有效浊帧不足"
                    else:
                        med = float(np.median(np.abs(cents)))
                        p90 = float(np.percentile(np.abs(cents), 90))
                        if med > GATE_MEDIAN_CENTS or p90 > GATE_P90_CENTS:
                            reject_reason = "f0 偏差超限"
        entry = {"aug": aug_stem, "source": src_stem, "keyshift": round(keyshift, 4)}
        if med is not None:
            entry["median_cents"] = round(med, 2)
            entry["p90_cents"] = round(p90, 2)
        if reject_reason is not None:
            entry["verdict"] = "rejected"
            entry["reason"] = reject_reason
            reasons[reject_reason] = reasons.get(reject_reason, 0) + 1
            remove_products_fn(aug_stem)
            rejected += 1
        else:
            entry["verdict"] = "kept"
            kept += 1
        copies.append(entry)

    if report_path:
        report = {
            "kept": kept,
            "rejected": rejected,
            "reasons": reasons,
            "thresholds": {
                "median_cents": GATE_MEDIAN_CENTS,
                "p90_cents": GATE_P90_CENTS,
                "min_voiced_ratio": GATE_MIN_VOICED_RATIO,
                "edge_erode_frames": GATE_EDGE_ERODE,
                "pitch_ceiling_hz": PITCH_CEILING_HZ,
            },
            "headroom_frames": headroom_frames,
            "copies": copies,
        }

        def w(tmp):
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=1)

        try:
            _atomic_write(report_path, w)
        except OSError:
            logger.warning("aug gate report write failed: %s", report_path)

    msg = "增强质检：保留 %d 片 · 剔除 %d 片" % (kept, rejected)
    if reasons:
        msg += "（%s）" % " · ".join("%s %d" % (k, v) for k, v in sorted(reasons.items()))
    if headroom_frames:
        msg += "（高音截顶帧 %d 已豁免）" % headroom_frames
    if rejected > kept:
        msg += " ⚠ 剔除过半，素材可能不适合增强"
    logger.info(
        "aug gate: kept %d rejected %d%s", kept, rejected,
        " (report: %s)" % report_path if report_path else "",
    )
    reporter.stage("aug_check", done=len(entries), total=len(entries), message=msg, force=True)
    return kept, rejected
