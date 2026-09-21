"""S172: stable CODEs for the RVC preprocessing stages, and the policy behind them.

WHY THIS FILE EXISTS. All three RVC prep stages (`preprocess` → `extract_f0` →
`extract_feature`) used to swallow every per-item failure into a counter and raise only when
EVERY item failed. Downstream, `rvc/filelist.py` builds the training set as a 4-way set
INTERSECTION over the four product directories, so any slice missing a product is dropped
without a word: 369 of 370 f0 slices failing produced a run that trained on the remainder and
reported "completed". A model quietly trained on 1/370 of the data is worse than a run that
died, because nothing tells the user which one they got.

THE POLICY, which is not invented here — the SoVITS lanes have had it since S41
(`sovits/extract.py`, see its comment) and RVC was the outlier:

  * a BASE slice failing is fatal. The filelists reference it, and tolerating it silently
    shrinks the dataset. Fail on the first one: the user learns in seconds instead of after
    every remaining slice has also failed.
  * an `_aug` slice failing is NOT fatal. Those are our own generated copies; dropping one
    degrades augmentation rather than the dataset. RVC needs no product cleanup for them —
    filelist.py's intersection already excludes anything missing a product — but the count
    must be reported, never swallowed.
  * a SOURCE FILE failing to preprocess is not fatal either (one corrupt import should not
    kill the run) but it means the user is training on less audio than they handed us, so it
    is reported with a count.

⚠ These are plain top-level literals on purpose: the cross-language CODE gate parses them as
the single source (same contract as device.py::AMD_GPU_NOT_FOUND_CODE).
"""

#: A base (non-aug) slice failed in a preprocessing stage. Fatal, raised on the first one.
#: Detail carries the stage and the slice name.
SLICE_PREP_FAILED_CODE = "TRAINING_SLICE_PREP_FAILED"

#: N augmented copies failed and were dropped. Warning — the run continues on the base set.
AUG_SLICES_DROPPED_CODE = "TRAINING_AUG_SLICES_DROPPED"

#: N source audio files could not be preprocessed and produced no slices at all. Warning —
#: the run continues, but on less material than the user imported.
SOURCE_FILES_SKIPPED_CODE = "TRAINING_SOURCE_FILES_SKIPPED"

#: The augmentation quality gate could not extract f0 for a SINGLE pair. Fatal: a gate that
#: never ran must not be read as "every augmented copy is bad" — that verdict deletes 100%
#: of the augmentation and lets the run continue looking normal (closed-gate iron rule).
AUG_GATE_UNUSABLE_CODE = "TRAINING_AUG_GATE_UNUSABLE"

#: How many failures get a full traceback before the log switches to one line each.
#: S172: a field report had 370 failures x 39 lines = 2.20 MB, 99.65% of the whole log, which
#: evicted the startup section and the device pick from the 2000-entry log panel and pushed
#: the three root-cause MIOpen lines out of the 30-line stderr tail the error card shows. The
#: failures are identical by construction, so the first few carry all the evidence.
FULL_TRACEBACKS = 3


# ── S172 round 4: the material/preprocessing refusals that used to raise hardcoded Chinese ──
# They belong here rather than in config_codes.py because they are this module's own subject:
# every one of them is a verdict about the USER'S AUDIO after a preprocessing stage looked at it.

#: The pitch-shift engine returned unusable output for ONE augmented copy (non-finite samples,
#: or a length drift past tolerance). Deliberately de-escalated in the catalog: psola_shift's
#: only caller swallows this and drops the copy, so the run is not in danger and the user should
#: only act if the log is full of them.
PSOLA_OUTPUT_INVALID_CODE = "TRAINING_PSOLA_OUTPUT_INVALID"

#: The four RVC product directories contradict each other -- no single slice has all of audio,
#: feature, f0 and f0nsf -- so no filelist can be built. ⚠ NOT the same as FEATURE_DIR_EMPTY:
#: build_index runs BEFORE build_filelist, so an empty feature directory reds there first.
#: Reaching this line means the products are inconsistent, which should not happen.
PREP_PRODUCTS_MISMATCH_CODE = "TRAINING_PREP_PRODUCTS_MISMATCH"

#: A feature-product directory holds nothing for any non-aug slice, so the retrieval/cluster
#: index cannot be built. Shared by the RVC and SoVITS lanes: both list a feature directory,
#: both exclude aug copies, both mean "no original slice has a feature".
FEATURE_DIR_EMPTY_CODE = "TRAINING_FEATURE_DIR_EMPTY"

#: Slicing ran and wrote nothing usable. Almost always near-silent or far too quiet material
#: (the slicer treats below -40 dB as silence), or files that are all too short.
NO_USABLE_SLICES_CODE = "TRAINING_NO_USABLE_SLICES"

#: EVERY source file threw while being decoded/sliced/resampled -- as opposed to
#: SOURCE_FILES_SKIPPED_CODE above, which is the same axis with some survivors. Corrupt files,
#: an undecodable format, or files held open by another program.
SOURCE_FILES_ALL_FAILED_CODE = "TRAINING_SOURCE_FILES_ALL_FAILED"

#: Vocoder fine-tuning was handed material below 44.1 kHz. Fatal on purpose and not a warning:
#: upsampling cannot restore high frequencies that were never recorded, and the fine-tuned
#: vocoder would inherit the defect -- a silently worse model is the outcome we refuse.
SOURCE_SR_TOO_LOW_CODE = "TRAINING_SOURCE_SR_TOO_LOW"
