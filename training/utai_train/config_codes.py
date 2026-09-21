"""S172: stable CODEs for the run-config invariants the trainer refuses to start on.

WHY THIS FILE EXISTS. These conditions used to raise hardcoded Chinese, so an English or
Japanese user met a Chinese sentence at the exact moment their training died. Every one of them
now leads with a CODE that `src/lib/backendError.ts` maps into the user's own language, and
carries an English machine detail after the colon for the log.

⚠ FOUR OF THESE ARE NOT OURS TO NAME. `TRAINING_BACKEND_UNSUPPORTED`, `TRAINING_BAD_SAMPLE_RATE`,
`TRAINING_BAD_SOVITS_VERSION` and `TRAINING_ASSET_MISSING` are already emitted by Rust
(`src-tauri/src/training/mod.rs`) for the byte-identical predicate, BEFORE run.json is written.
The python check is defence-in-depth on the same invariant, so it reuses the same CODE. Minting
a near-synonym here would give one condition two names and two catalog entries that could drift
apart -- the no-duplication rule, applied across languages.

WHY THE TEXTS SAY "SEND US THE LOG" SO OFTEN. Most of these are unreachable through the UI: the
speech encoder is derived from the version, the f0 method has no control, the asset paths are
written by Rust and verified before launch. Telling a user to "check your settings" would send
them hunting for a control that does not exist. An honest dead end beats a helpful-sounding one.

These are plain top-level literals on purpose: the cross-language CODE gate parses them as the
single source (same contract as `prep_codes.py`, `stage_codes.py` and `device.py`).
"""

#: run.json named a training backend the runner has no lane for. Mirrors the Rust emitter at
#: src-tauri/src/training/mod.rs:1855 -- same five-value set, same condition.
BACKEND_UNSUPPORTED_CODE = "TRAINING_BACKEND_UNSUPPORTED"

#: RVC sample rate outside 32k/40k/48k. Mirrors src-tauri/src/training/mod.rs:1785.
BAD_SAMPLE_RATE_CODE = "TRAINING_BAD_SAMPLE_RATE"

#: SoVITS version outside the lane's allowed set. Mirrors src-tauri/src/training/mod.rs:1793.
#: The allowed set differs per lane, so it lives in the DETAIL and never in the sentence.
BAD_SOVITS_VERSION_CODE = "TRAINING_BAD_SOVITS_VERSION"

#: An asset path is known but the file is not there. Mirrors src-tauri/src/training/mod.rs:2127.
#: ⚠ Distinct from ASSET_PATH_UNSET below: there the user has nothing to put back.
ASSET_MISSING_CODE = "TRAINING_ASSET_MISSING"

#: An asset path key in run.json came through EMPTY. Rust always writes it and verifies the file
#: before launch, so this can only be our bug or a hand-edited run config.
ASSET_PATH_UNSET_CODE = "TRAINING_ASSET_PATH_UNSET"

#: config named a speech encoder no lane implements. Derived from the version automatically and
#: unreachable from the UI -- an internal invariant, not a user choice.
UNKNOWN_SPEECH_ENCODER_CODE = "TRAINING_UNKNOWN_SPEECH_ENCODER"

#: config named an f0 extractor no lane implements. No UI control exists for it either.
UNKNOWN_F0_METHOD_CODE = "TRAINING_UNKNOWN_F0_METHOD"

#: config asks for a vocoder we do not bundle. Utai's own writer always emits nsf-hifigan, so
#: this means a hand-edited config or a training folder from another project.
VOCODER_NOT_BUNDLED_CODE = "TRAINING_VOCODER_NOT_BUNDLED"

#: shallow diffusion was asked for on a multi-speaker target. A real product limit, and the one
#: code in this module the user can actually act on -- they can train it in a single-singer
#: project instead.
DIFF_MULTI_SPEAKER_CODE = "TRAINING_DIFF_MULTI_SPEAKER"

#: the diffusion base checkpoint loads but carries no 'model' key, i.e. it is not a diffusion
#: base model. Usually the wrong file at that path, or a truncated download.
DIFF_BASE_FORMAT_CODE = "TRAINING_DIFF_BASE_FORMAT"
