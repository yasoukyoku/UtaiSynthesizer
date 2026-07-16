import i18n from "../i18n";

/**
 * Stable Rust backend CODEs → i18n keys. THE single app-wide backend-error mapping — every display
 * funnel (toasts, node error banners, settings panels, training page) consults this before showing a
 * raw message; never fork a per-component copy (NO-duplication hard rule). Domain-specific mappers
 * (vocalRenderErrorMessage's VOCAL_* family, Settings' pyenv L() panel) keep their own payload-aware
 * handling and DELEGATE here for everything else.
 *
 * Codes are matched as substrings because UtaiError Display prefixes ("Audio processing error: …",
 * "Inference error: …") and command wrappers ride along; an optional ": detail" suffix after the code
 * is appended to the localized text in parentheses (detail stays raw/technical by convention).
 *
 * `busy: true` marks transient interlock rejections (another task holds a guard — retry later): the
 * funnels show these as INFO toasts, everything else as errors. Keep the flag accurate — an error
 * mis-flagged busy silently downgrades a real failure.
 *
 * NOT in this table on purpose: the cancel sentinel (Cancelled / 已取消 / CANCELLED) — engine
 * isCancelMessage swallows it silently and mapping it here would localize it BEFORE the swallow check,
 * turning user cancels into visible errors; and CLEANUP_BUSY, which only surfaces in Settings and is
 * mapped by its local L() table (that file's sanctioned convention).
 */
interface CodeEntry {
  key: string;
  busy?: boolean;
}

const CODE_KEYS: Record<string, CodeEntry> = {
  // FlightGuard / interlock busy rejections (audition.rs BUSY_RETRY_MSG — 10 commands inherit it).
  APP_BUSY: { key: "common.busyRetry", busy: true },
  // Model import/delete refused while an audition holds the model open (models.rs).
  MODEL_BUSY_AUDITION: { key: "common.busyRetry", busy: true },
  // Separation single-slot guard (separation/mod.rs) — the TOCTOU backstop behind the pre-flight.
  SEPARATION_BUSY: { key: "workflow.separationBusy", busy: true },
  MSST_MODEL_NOT_CONVERTED: { key: "workflow.errSeparationNotConverted" },
  // Transpose node (utai-stretch wrapper) codes.
  TRANSPOSE_INPUT_MISSING: { key: "workflow.errTransposeInput" },
  TRANSPOSE_RANGE: { key: "workflow.errTransposeRange" },
  // ── Generated from the S62 full-sweep manifests (4 conversion clusters) — one entry per stable
  // Rust CODE; texts live under backend.* in src/i18n/{zh,en,ja}.json (TRAINING_NO_DATA reuses the
  // pre-existing training.needData key). Keep alphabetical; busy flags mark interlock rejections. ──
  ASSET_DL_BUSY: { key: "backend.ASSET_DL_BUSY", busy: true },
  ASSET_DL_FAILED: { key: "backend.ASSET_DL_FAILED" },
  CUDA_DOWNLOAD_BUSY: { key: "backend.CUDA_DOWNLOAD_BUSY", busy: true },
  CUDA_GPU_REQUIRED: { key: "backend.CUDA_GPU_REQUIRED" },
  // S66 poisoned-proxy guard (download.rs): a GH proxy answered a download with an HTML page.
  DOWNLOAD_HTML_RESPONSE: { key: "backend.DOWNLOAD_HTML_RESPONSE" },
  // S66 CUDA local-file install (settings.rs install_cuda_runtime_local).
  CUDA_LOCAL_NO_FILES: { key: "backend.CUDA_LOCAL_NO_FILES" },
  CUDA_LOCAL_UNRECOGNIZED: { key: "backend.CUDA_LOCAL_UNRECOGNIZED" },
  CUDA_LOCAL_BAD_FILE: { key: "backend.CUDA_LOCAL_BAD_FILE" },
  // S66 conversion single-flight + heavy-job interlock (lib.rs acquire_convert_slot).
  CONVERT_BUSY: { key: "backend.CONVERT_BUSY", busy: true },
  CONVERT_RENDER_BUSY: { key: "backend.CONVERT_RENDER_BUSY", busy: true },
  // S66 MSST conversion errors (msst_models.rs, formerly prose strings).
  MSST_ARCH_UNKNOWN: { key: "backend.MSST_ARCH_UNKNOWN" },
  MSST_CONVERT_FAILED: { key: "backend.MSST_CONVERT_FAILED" },
  MSST_FILE_NOT_FOUND: { key: "backend.MSST_FILE_NOT_FOUND" },
  // S66/O5: the render commands write the wav Rust-side; disk write failure is its own code.
  RENDER_WRITE_FAILED: { key: "backend.RENDER_WRITE_FAILED" },
  AUDIO_EMPTY_INPUT: { key: "backend.AUDIO_EMPTY_INPUT" },
  AUDIO_TOO_SHORT_HIGHPASS: { key: "backend.AUDIO_TOO_SHORT_HIGHPASS" },
  AUDITION_BACKEND_UNSUPPORTED: { key: "backend.AUDITION_BACKEND_UNSUPPORTED" },
  AUDITION_BAD_CANDIDATE_PATH: { key: "backend.AUDITION_BAD_CANDIDATE_PATH" },
  AUDITION_BAD_TYPE: { key: "backend.AUDITION_BAD_TYPE" },
  AUDITION_MODEL_MISSING: { key: "backend.MODEL_NOT_FOUND" },
  AUDITION_CLIP_MISSING: { key: "backend.AUDITION_CLIP_MISSING" },
  AUDITION_CONFIG_HOP_ZERO: { key: "backend.AUDITION_CONFIG_HOP_ZERO" },
  AUDITION_CONFIG_NO_FEATURES_DIM: { key: "backend.AUDITION_CONFIG_NO_FEATURES_DIM" },
  AUDITION_CONFIG_NO_SAMPLE_RATE: { key: "backend.AUDITION_CONFIG_NO_SAMPLE_RATE" },
  AUDITION_CONFIG_PARSE_FAILED: { key: "backend.AUDITION_CONFIG_PARSE_FAILED" },
  AUDITION_CONFIG_READ_FAILED: { key: "backend.AUDITION_CONFIG_READ_FAILED" },
  AUDITION_DIR_CREATE_FAILED: { key: "backend.AUDITION_DIR_CREATE_FAILED" },
  AUDITION_HOST_MODEL_MISSING: { key: "backend.AUDITION_HOST_MODEL_MISSING" },
  AUDITION_RENDER_BUSY: { key: "backend.AUDITION_RENDER_BUSY", busy: true },
  AUDITION_TASK_PANICKED: { key: "backend.AUDITION_TASK_PANICKED" },
  AUDITION_TRAINING_ACTIVE: { key: "backend.AUDITION_TRAINING_ACTIVE", busy: true },
  AUDITION_VOCODER_NONSTANDARD: { key: "backend.AUDITION_VOCODER_NONSTANDARD" },
  AUDITION_WAV_WRITE_FAILED: { key: "backend.AUDITION_WAV_WRITE_FAILED" },
  AUTO_F0_FILE_MISSING: { key: "backend.AUTO_F0_FILE_MISSING" },
  AUTO_F0_FRAMES_MISMATCH: { key: "backend.AUTO_F0_FRAMES_MISMATCH" },
  AUTO_F0_NOT_EXPORTED: { key: "backend.AUTO_F0_NOT_EXPORTED" },
  AUTO_F0_NO_OUTPUT: { key: "backend.AUTO_F0_NO_OUTPUT" },
  AUX_FILE_MISSING: { key: "backend.AUX_FILE_MISSING" },
  AUX_VOCODER_MEL_MISSING: { key: "backend.AUX_VOCODER_MEL_MISSING" },
  CONTENTVEC_INPUT_TOO_SHORT: { key: "backend.CONTENTVEC_INPUT_TOO_SHORT" },
  CONTENTVEC_NO_OUTPUT: { key: "backend.CONTENTVEC_NO_OUTPUT" },
  CONTENTVEC_RESHAPE_FAILED: { key: "backend.CONTENTVEC_RESHAPE_FAILED" },
  CONTENTVEC_SHAPE: { key: "backend.CONTENTVEC_SHAPE" },
  DELETE_TASK_FAILED: { key: "backend.DELETE_TASK_FAILED" },
  DELETE_WHILE_ENVTEST: { key: "backend.DELETE_WHILE_ENVTEST", busy: true },
  DELETE_WHILE_INSTALLING: { key: "backend.DELETE_WHILE_INSTALLING", busy: true },
  DIFFUSION_COND_SHAPE: { key: "backend.DIFFUSION_COND_SHAPE" },
  DIFFUSION_DENOISER_NO_OUTPUT: { key: "backend.DIFFUSION_DENOISER_NO_OUTPUT" },
  DIFFUSION_DENOISER_SHAPE: { key: "backend.DIFFUSION_DENOISER_SHAPE" },
  DIFFUSION_DIM_MISMATCH: { key: "backend.DIFFUSION_DIM_MISMATCH" },
  DIFFUSION_ENCODER_NO_OUTPUT: { key: "backend.DIFFUSION_ENCODER_NO_OUTPUT" },
  DIFFUSION_FILE_MISSING: { key: "backend.DIFFUSION_FILE_MISSING" },
  DIFFUSION_GEOMETRY_MISMATCH: { key: "backend.DIFFUSION_GEOMETRY_MISMATCH" },
  DIFFUSION_KSTEP_EXCEEDS_MAX: { key: "backend.DIFFUSION_KSTEP_EXCEEDS_MAX" },
  DIFFUSION_KSTEP_MIN: { key: "backend.DIFFUSION_KSTEP_MIN" },
  DIFFUSION_KSTEP_ZERO: { key: "backend.DIFFUSION_KSTEP_ZERO" },
  DIFFUSION_MEL_SHAPE: { key: "backend.DIFFUSION_MEL_SHAPE" },
  DIFFUSION_NOT_ATTACHED: { key: "backend.DIFFUSION_NOT_ATTACHED" },
  DIFFUSION_REPLACE_FAILED: { key: "backend.DIFFUSION_REPLACE_FAILED" },
  DIFFUSION_SAMPLER_UNKNOWN: { key: "backend.DIFFUSION_SAMPLER_UNKNOWN" },
  DIFFUSION_SCHEDULE_UNSUPPORTED: { key: "backend.DIFFUSION_SCHEDULE_UNSUPPORTED" },
  DIFFUSION_SHALLOW_ONLY: { key: "backend.DIFFUSION_SHALLOW_ONLY" },
  DIFFUSION_SIDECAR_FIELD_MISSING: { key: "backend.DIFFUSION_SIDECAR_FIELD_MISSING" },
  DIFFUSION_SPEAKER_OUT_OF_RANGE: { key: "backend.DIFFUSION_SPEAKER_OUT_OF_RANGE" },
  DIFFUSION_SPEEDUP_TOO_FEW_STEPS: { key: "backend.DIFFUSION_SPEEDUP_TOO_FEW_STEPS" },
  DIFFUSION_SWAP_FAILED: { key: "backend.DIFFUSION_SWAP_FAILED" },
  DIFFUSION_T_OUT_OF_RANGE: { key: "backend.DIFFUSION_T_OUT_OF_RANGE" },
  DIFF_VERSION_MISMATCH: { key: "backend.DIFF_VERSION_MISMATCH" },
  DIFF_WIPE_FAILED: { key: "backend.DIFF_WIPE_FAILED" },
  DOWNLOAD_CANCELLED: { key: "backend.DOWNLOAD_CANCELLED" },
  DOWNLOAD_INCOMPLETE: { key: "backend.DOWNLOAD_INCOMPLETE" },
  DOWNLOAD_NO_SOURCE: { key: "backend.DOWNLOAD_NO_SOURCE" },
  DOWNLOAD_OVERSIZE: { key: "backend.DOWNLOAD_OVERSIZE" },
  DOWNLOAD_RANGE_INVALID: { key: "backend.DOWNLOAD_RANGE_INVALID" },
  DOWNLOAD_REQUEST_FAILED: { key: "backend.DOWNLOAD_REQUEST_FAILED" },
  DOWNLOAD_SHA256_MISMATCH: { key: "backend.DOWNLOAD_SHA256_MISMATCH" },
  DOWNLOAD_STALLED: { key: "backend.DOWNLOAD_STALLED" },
  DOWNLOAD_STREAM_INTERRUPTED: { key: "backend.DOWNLOAD_STREAM_INTERRUPTED" },
  ENHANCER_F0_EMPTY: { key: "backend.ENHANCER_F0_EMPTY" },
  ENVTEST_BUSY: { key: "backend.ENVTEST_BUSY", busy: true },
  ENVTEST_CANCELLED: { key: "backend.ENVTEST_CANCELLED" },
  ENVTEST_CRASHED: { key: "backend.ENVTEST_CRASHED" },
  ENVTEST_FAILED: { key: "backend.ENVTEST_FAILED" },
  ENVTEST_REPORT_CLEAR_FAILED: { key: "backend.ENVTEST_REPORT_CLEAR_FAILED" },
  ENVTEST_REPORT_CONTRADICTION: { key: "backend.ENVTEST_REPORT_CONTRADICTION" },
  ENVTEST_SCRIPT_MISSING: { key: "backend.ENVTEST_SCRIPT_MISSING" },
  ENVTEST_SPAWN_FAILED: { key: "backend.ENVTEST_SPAWN_FAILED" },
  ENVTEST_TIMEOUT: { key: "backend.ENVTEST_TIMEOUT" },
  ENVTEST_WAIT_FAILED: { key: "backend.ENVTEST_WAIT_FAILED" },
  // S63 audio/score export (commands/export_audio.rs + export_score.rs). Longest-code-first matching
  // keeps EXPORT_FFMPEG_MISSING ahead of the training-side FFMPEG_MISSING, and EXPORT_SCORE_WRITE_FAIL
  // ahead of EXPORT_WRITE_FAIL. The score codes reuse the export.* dialog keys (one text, two funnels).
  EXPORT_BAD_PCM: { key: "backend.EXPORT_BAD_PCM" },
  EXPORT_ENCODE_FAIL: { key: "backend.EXPORT_ENCODE_FAIL" },
  EXPORT_FFMPEG_MISSING: { key: "backend.EXPORT_FFMPEG_MISSING" },
  EXPORT_FORMAT_UNSUPPORTED: { key: "backend.EXPORT_FORMAT_UNSUPPORTED" },
  EXPORT_NO_PCM: { key: "backend.EXPORT_NO_PCM" },
  EXPORT_SCORE_EMPTY: { key: "export.errScoreEmpty" },
  EXPORT_SCORE_UNSUPPORTED: { key: "backend.EXPORT_FORMAT_UNSUPPORTED" },
  EXPORT_SCORE_WRITE_FAIL: { key: "export.errScoreWrite" },
  EXPORT_WRITE_FAIL: { key: "backend.EXPORT_WRITE_FAIL" },
  EXTRACT_FAILED: { key: "backend.EXTRACT_FAILED" },
  EXTRACT_TASK_FAILED: { key: "backend.EXTRACT_TASK_FAILED" },
  F0_EMPTY_INPUT: { key: "backend.F0_EMPTY_INPUT" },
  F0_TASK_PANICKED: { key: "backend.F0_TASK_PANICKED" },
  FEATURES_DIM_UNSUPPORTED: { key: "backend.FEATURES_DIM_UNSUPPORTED" },
  FFMPEG_MISSING: { key: "backend.FFMPEG_MISSING" },
  FILE_READ_FAILED: { key: "backend.FILE_READ_FAILED" },
  INDEX_LOAD_FAILED: { key: "backend.INDEX_LOAD_FAILED" },
  // S67c loud guard: DML new-shape compile refused below the system-commit floor —
  // replaces the OS silently killing the process mid-allocation on low-memory machines.
  INFERENCE_LOW_MEMORY: { key: "backend.INFERENCE_LOW_MEMORY" },
  INFER_TASK_PANICKED: { key: "backend.INFER_TASK_PANICKED" },
  INSTALL_BUSY: { key: "backend.INSTALL_BUSY", busy: true },
  INSTALL_CANCELLED: { key: "backend.INSTALL_CANCELLED" },
  INTERNAL_EMPTY_FEATURES: { key: "backend.INTERNAL_EMPTY_FEATURES" },
  INTERNAL_ENHANCER_NO_VOCODER: { key: "backend.INTERNAL_ENHANCER_NO_VOCODER" },
  INTERNAL_NO_OUTPUT_PATH: { key: "backend.INTERNAL_NO_OUTPUT_PATH" },
  INTERNAL_UNIPC_ORDER: { key: "backend.INTERNAL_UNIPC_ORDER" },
  INTERNAL_UNIPC_SINGULAR: { key: "backend.INTERNAL_UNIPC_SINGULAR" },
  INTERP_MODE_UNSUPPORTED: { key: "backend.INTERP_MODE_UNSUPPORTED" },
  JSON_PARSE_FAILED: { key: "backend.JSON_PARSE_FAILED" },
  LOCAL_FILE_BAD_DIR: { key: "backend.LOCAL_FILE_BAD_DIR" },
  LOCAL_FILE_BAD_NAME: { key: "backend.LOCAL_FILE_BAD_NAME" },
  LOCAL_FILE_BAD_TYPE: { key: "backend.LOCAL_FILE_BAD_TYPE" },
  LOCAL_PARTS_GAP: { key: "backend.LOCAL_PARTS_GAP" },
  LOCAL_PARTS_NOT_FOUND: { key: "backend.LOCAL_PARTS_NOT_FOUND" },
  MANIFEST_BAD_ID: { key: "backend.MANIFEST_BAD_ID" },
  MANIFEST_BAD_PART_NAME: { key: "backend.MANIFEST_BAD_PART_NAME" },
  MANIFEST_BAD_SHA256: { key: "backend.MANIFEST_BAD_SHA256" },
  MANIFEST_FETCH_FAILED: { key: "backend.MANIFEST_FETCH_FAILED" },
  MANIFEST_ID_MISMATCH: { key: "backend.MANIFEST_ID_MISMATCH" },
  MANIFEST_NO_PARTS: { key: "backend.MANIFEST_NO_PARTS" },
  MANIFEST_PARSE_FAILED: { key: "backend.MANIFEST_PARSE_FAILED" },
  MANIFEST_READ_FAILED: { key: "backend.MANIFEST_READ_FAILED" },
  MANIFEST_REQUEST_FAILED: { key: "backend.MANIFEST_REQUEST_FAILED" },
  MODEL_HOP_SIZE_ZERO: { key: "backend.MODEL_HOP_SIZE_ZERO" },
  MODEL_LEGACY_EXPORT: { key: "backend.MODEL_LEGACY_EXPORT" },
  MODEL_NOT_FOUND: { key: "backend.MODEL_NOT_FOUND" },
  NPY_LOAD_FAILED: { key: "backend.NPY_LOAD_FAILED" },
  MODEL_NOT_LOADED: { key: "backend.MODEL_NOT_LOADED" },
  PACK_BAD_ID: { key: "backend.PACK_BAD_ID" },
  PACK_DELETE_FAILED: { key: "backend.PACK_DELETE_FAILED" },
  PACK_EMPTY: { key: "backend.PACK_EMPTY" },
  PACK_FORMAT_INVALID: { key: "backend.PACK_FORMAT_INVALID" },
  PACK_JSON_BAD_ID: { key: "backend.PACK_JSON_BAD_ID" },
  PACK_JSON_PARSE_FAILED: { key: "backend.PACK_JSON_PARSE_FAILED" },
  PACK_JSON_READ_FAILED: { key: "backend.PACK_JSON_READ_FAILED" },
  PACK_NOT_FOUND: { key: "backend.PACK_NOT_FOUND" },
  PACK_NO_DOWNLOAD_SOURCE: { key: "backend.PACK_NO_DOWNLOAD_SOURCE" },
  PACK_NO_PYTHON: { key: "backend.PACK_NO_PYTHON" },
  PACK_UNKNOWN: { key: "backend.PACK_UNKNOWN" },
  PART_MISSING: { key: "backend.PART_MISSING" },
  PART_SHA256_MISMATCH: { key: "backend.PART_SHA256_MISMATCH" },
  PART_SIZE_MISMATCH: { key: "backend.PART_SIZE_MISMATCH" },
  PROBE_CONNECT_FAILED: { key: "backend.PROBE_CONNECT_FAILED" },
  PROBE_CONNECT_TIMEOUT: { key: "backend.PROBE_CONNECT_TIMEOUT" },
  PROBE_HTTP_ERROR: { key: "backend.PROBE_HTTP_ERROR" },
  PROBE_TIMEOUT: { key: "backend.PROBE_TIMEOUT" },
  RENAME_FAILED: { key: "backend.RENAME_FAILED" },
  RENAME_RETRY_EXHAUSTED: { key: "backend.RENAME_RETRY_EXHAUSTED" },
  RESUME_KSTEP_MISMATCH: { key: "backend.RESUME_KSTEP_MISMATCH" },
  RESUME_PARAMS_MISMATCH: { key: "backend.RESUME_PARAMS_MISMATCH" },
  RESUME_SPEAKER_COUNT_MISMATCH: { key: "backend.RESUME_SPEAKER_COUNT_MISMATCH" },
  RESUME_SPEAKER_SET_MISMATCH: { key: "backend.RESUME_SPEAKER_SET_MISMATCH" },
  RESUME_TARGET_REACHED_DIFF: { key: "backend.RESUME_TARGET_REACHED_DIFF" },
  RESUME_TARGET_REACHED_VOCODER: { key: "backend.RESUME_TARGET_REACHED_VOCODER" },
  RESUME_VOL_EMBEDDING_MISMATCH: { key: "backend.RESUME_VOL_EMBEDDING_MISMATCH" },
  RMVPE_FRAMES_MISMATCH: { key: "backend.RMVPE_FRAMES_MISMATCH" },
  RMVPE_MEL_SHAPE: { key: "backend.RMVPE_MEL_SHAPE" },
  RMVPE_NO_OUTPUT: { key: "backend.RMVPE_NO_OUTPUT" },
  RUNTIME_PACK_REQUIRED: { key: "backend.RUNTIME_PACK_REQUIRED" },
  RUNTIME_PATH_NON_ASCII: { key: "backend.RUNTIME_PATH_NON_ASCII" },
  RUNTIME_ROOT_UNINIT: { key: "backend.RUNTIME_ROOT_UNINIT" },
  RVC_CHUNK_TOO_SHORT: { key: "backend.RVC_CHUNK_TOO_SHORT" },
  RVC_F0_FRAMES_SHORT: { key: "backend.RVC_F0_FRAMES_SHORT" },
  RVC_MIN_FRAMES: { key: "backend.RVC_MIN_FRAMES" },
  RVC_NO_OUTPUT: { key: "backend.RVC_NO_OUTPUT" },
  RVC_SR_NOT_100FPS: { key: "backend.RVC_SR_NOT_100FPS" },
  SCORE2CV_DIM_UNSUPPORTED: { key: "backend.SCORE2CV_DIM_UNSUPPORTED" },
  SCORE2CV_NO_OUTPUT: { key: "backend.SCORE2CV_NO_OUTPUT" },
  SCORE2CV_SHAPE: { key: "backend.SCORE2CV_SHAPE" },
  SCORE2SVC_ZERO_FRAMES: { key: "backend.SCORE2SVC_ZERO_FRAMES" },
  SHARED_POOL_REUSED: { key: "backend.SHARED_POOL_REUSED" },
  SOVITS_NO_OUTPUT: { key: "backend.SOVITS_NO_OUTPUT" },
  SOVITS_VOL_FRAMES_MISMATCH: { key: "backend.SOVITS_VOL_FRAMES_MISMATCH" },
  SPEECH_ENCODER_UNSUPPORTED: { key: "backend.SPEECH_ENCODER_UNSUPPORTED" },
  SPK_MIX_DIFFUSION: { key: "vocalEditor.render.spkMixDiffusion" },
  STORAGE_JOIN: { key: "backend.STORAGE_JOIN" },
  TAR_ENTRY_BAD_PATH: { key: "backend.TAR_ENTRY_BAD_PATH" },
  TAR_ENTRY_CORRUPT: { key: "backend.TAR_ENTRY_CORRUPT" },
  TAR_READ_FAILED: { key: "backend.TAR_READ_FAILED" },
  TRAINING_ACTIVE: { key: "backend.TRAINING_ACTIVE", busy: true },
  TRAINING_ALREADY_RUNNING: { key: "backend.TRAINING_ALREADY_RUNNING", busy: true },
  TRAINING_ASSET_MISSING: { key: "backend.TRAINING_ASSET_MISSING" },
  TRAINING_AUG_COPIES_MAX: { key: "backend.TRAINING_AUG_COPIES_MAX" },
  TRAINING_BACKEND_UNSUPPORTED: { key: "backend.TRAINING_BACKEND_UNSUPPORTED" },
  TRAINING_BAD_RVC_VERSION: { key: "backend.TRAINING_BAD_RVC_VERSION" },
  TRAINING_BAD_SAMPLE_RATE: { key: "backend.TRAINING_BAD_SAMPLE_RATE" },
  TRAINING_BAD_SOVITS_VERSION: { key: "backend.TRAINING_BAD_SOVITS_VERSION" },
  TRAINING_BAD_VOCODER_FORMAT: { key: "backend.TRAINING_BAD_VOCODER_FORMAT" },
  TRAINING_CROP_FRAMES_ZERO: { key: "backend.TRAINING_CROP_FRAMES_ZERO" },
  TRAINING_DATA_FILE_MISSING: { key: "backend.TRAINING_DATA_FILE_MISSING" },
  // S67 loud-degradation guard (device.py require_wanted_accelerator → protocol error).
  TRAINING_GPU_UNAVAILABLE: { key: "backend.TRAINING_GPU_UNAVAILABLE" },
  TRAINING_IMPORT_COPY_FAILED: { key: "backend.TRAINING_IMPORT_COPY_FAILED" },
  TRAINING_INTERNAL_ASSET_BRANCH: { key: "backend.TRAINING_INTERNAL_ASSET_BRANCH" },
  TRAINING_KILL_FAILED: { key: "backend.TRAINING_KILL_FAILED" },
  TRAINING_MULTI_BACKEND: { key: "backend.TRAINING_MULTI_BACKEND" },
  TRAINING_NAME_EMPTY: { key: "backend.TRAINING_NAME_EMPTY" },
  TRAINING_NO_DATA: { key: "training.needData" },
  TRAINING_NO_SHARED_POOL: { key: "backend.TRAINING_NO_SHARED_POOL" },
  TRAINING_PROCESS_CRASHED: { key: "backend.TRAINING_PROCESS_CRASHED" },
  TRAINING_PYTHON_SPAWN_FAILED: { key: "backend.TRAINING_PYTHON_SPAWN_FAILED" },
  TRAINING_SAVE_INTERVAL_ZERO: { key: "backend.TRAINING_SAVE_INTERVAL_ZERO" },
  TRAINING_SPEAKER_LIMIT: { key: "backend.TRAINING_SPEAKER_LIMIT" },
  TRAINING_SPEAKER_NAME_DUP: { key: "backend.TRAINING_SPEAKER_NAME_DUP" },
  TRAINING_SPEAKER_NAME_EMPTY: { key: "backend.TRAINING_SPEAKER_NAME_EMPTY" },
  TRAINING_SPEAKER_NO_DATA: { key: "backend.TRAINING_SPEAKER_NO_DATA" },
  TRAINING_SR_FIXED_44K: { key: "backend.TRAINING_SR_FIXED_44K" },
  TRAINING_THREAD_SPAWN_FAILED: { key: "backend.TRAINING_THREAD_SPAWN_FAILED" },
  TRAINING_TOTAL_STEPS_ZERO: { key: "backend.TRAINING_TOTAL_STEPS_ZERO" },
  TRAINING_UNKNOWN_ERROR: { key: "backend.TRAINING_UNKNOWN_ERROR" },
  UPDATE_CHECK_FAILED: { key: "backend.UPDATE_CHECK_FAILED" },
  UPDATE_DOWNLOAD_FAILED: { key: "backend.UPDATE_DOWNLOAD_FAILED" },
  UPDATE_INSTALL_FAILED: { key: "backend.UPDATE_INSTALL_FAILED" },
  UPDATE_NO_PENDING: { key: "backend.UPDATE_NO_PENDING" },
  VERIFY_TASK_FAILED: { key: "backend.VERIFY_TASK_FAILED" },
  VOCAL_BACKEND_UNKNOWN: { key: "backend.VOCAL_BACKEND_UNKNOWN" },
  VOCAL_F0_FRAMES_MISMATCH: { key: "backend.VOCAL_F0_FRAMES_MISMATCH" },
  VOCAL_F0_LEN_MISMATCH: { key: "backend.VOCAL_F0_LEN_MISMATCH" },
  VOCAL_SEGMENT_TOO_LONG: { key: "backend.VOCAL_SEGMENT_TOO_LONG" },
  VOCAL_TASK_PANICKED: { key: "backend.VOCAL_TASK_PANICKED" },
  VOCAL_TOO_MANY_NOTES: { key: "backend.VOCAL_TOO_MANY_NOTES" },
  VOCODER_CONFIG_FIELD_MISSING: { key: "backend.VOCODER_CONFIG_FIELD_MISSING" },
  VOCODER_CONFIG_MISSING: { key: "backend.VOCODER_CONFIG_MISSING" },
  VOCODER_F0_FRAMES_SHORT: { key: "backend.VOCODER_F0_FRAMES_SHORT" },
  VOCODER_FILTER_SHAPE_MISMATCH: { key: "backend.VOCODER_FILTER_SHAPE_MISMATCH" },
  VOCODER_GEOMETRY_MISMATCH: { key: "backend.VOCODER_GEOMETRY_MISMATCH" },
  VOCODER_JSON_PARSE_FAILED: { key: "backend.VOCODER_JSON_PARSE_FAILED" },
  VOCODER_JSON_REQUIRED: { key: "backend.VOCODER_JSON_REQUIRED" },
  VOCODER_MEL_FORMAT_MISMATCH: { key: "backend.VOCODER_MEL_FORMAT_MISMATCH" },
  VOCODER_MEL_MISSING: { key: "backend.VOCODER_MEL_MISSING" },
  VOCODER_NOT_FOUND: { key: "backend.VOCODER_NOT_FOUND" },
  VOCODER_NO_OUTPUT: { key: "backend.VOCODER_NO_OUTPUT" },
  VOCODER_PCNSF_UNSUPPORTED: { key: "backend.VOCODER_PCNSF_UNSUPPORTED" },
  WARN_AUTO_F0_COPY_FAILED: { key: "backend.WARN_AUTO_F0_COPY_FAILED" },
  WARN_AUTO_F0_MISSING: { key: "backend.WARN_AUTO_F0_MISSING" },
  WARN_AVATAR_IMPORT_FAILED: { key: "backend.WARN_AVATAR_IMPORT_FAILED" },
  WARN_AVATAR_MISSING: { key: "backend.WARN_AVATAR_MISSING" },
  WARN_CLUSTER_CONVERT_FAILED: { key: "backend.WARN_CLUSTER_CONVERT_FAILED" },
  WARN_CLUSTER_COPY_FAILED: { key: "backend.WARN_CLUSTER_COPY_FAILED" },
  WARN_CLUSTER_DIR_FAILED: { key: "backend.WARN_CLUSTER_DIR_FAILED" },
  WARN_CLUSTER_EMPTY_OUTPUT: { key: "backend.WARN_CLUSTER_EMPTY_OUTPUT" },
  WARN_CLUSTER_FILE_MISSING: { key: "backend.WARN_CLUSTER_FILE_MISSING" },
  WARN_CLUSTER_MULTI_COPY_FAILED: { key: "backend.WARN_CLUSTER_MULTI_COPY_FAILED" },
  WARN_CLUSTER_TYPE_UNSUPPORTED: { key: "backend.WARN_CLUSTER_TYPE_UNSUPPORTED" },
  WARN_DIFFUSION_CONVERT_FAILED: { key: "backend.WARN_DIFFUSION_CONVERT_FAILED" },
  WARN_DIFFUSION_COPY_FAILED: { key: "backend.WARN_DIFFUSION_COPY_FAILED" },
  WARN_DIFFUSION_DIM_MISMATCH: { key: "backend.WARN_DIFFUSION_DIM_MISMATCH" },
  WARN_DIFFUSION_FILE_MISSING: { key: "backend.WARN_DIFFUSION_FILE_MISSING" },
  WARN_DIFFUSION_SOVITS_ONLY: { key: "backend.WARN_DIFFUSION_SOVITS_ONLY" },
  WARN_INDEX_CONVERT_FAILED: { key: "backend.WARN_INDEX_CONVERT_FAILED" },
  WARN_INDEX_COPY_FAILED: { key: "backend.WARN_INDEX_COPY_FAILED" },
  WARN_INDEX_MISSING: { key: "backend.WARN_INDEX_MISSING" },
  WARN_INDEX_TYPE_UNSUPPORTED: { key: "backend.WARN_INDEX_TYPE_UNSUPPORTED" },
  WARN_SIDECAR_REGENERATED: { key: "backend.WARN_SIDECAR_REGENERATED" },
  WARN_SIDECAR_SYNTHESIZED: { key: "backend.WARN_SIDECAR_SYNTHESIZED" },
  WORKSPACE_BACKEND_MISMATCH: { key: "backend.WORKSPACE_BACKEND_MISMATCH" },
  WORKSPACE_MANIFEST_MISSING: { key: "backend.WORKSPACE_MANIFEST_MISSING" },
  WORKSPACE_DELETE_FAILED: { key: "backend.WORKSPACE_DELETE_FAILED" },
  WORKSPACE_WIPE_FAILED: { key: "backend.WORKSPACE_WIPE_FAILED" },
  ZSTD_INIT_FAILED: { key: "backend.ZSTD_INIT_FAILED" },
};

/** Longest-first so a code that happens to be a prefix of another can never shadow it. */
const CODES = Object.keys(CODE_KEYS).sort((a, b) => b.length - a.length);

function findCode(msg: string): { entry: CodeEntry; detail: string } | null {
  for (const code of CODES) {
    const at = msg.indexOf(code);
    if (at < 0) continue;
    const detail = msg.slice(at + code.length).replace(/^[:：]\s*/, "").trim();
    return { entry: CODE_KEYS[code]!, detail };
  }
  return null;
}

/** Localized text for a backend error carrying a known CODE, or null when it carries none (the caller
 *  falls back to its own default display). */
export function backendErrorMessage(e: unknown): string | null {
  const hit = findCode(String(e));
  if (!hit) return null;
  const text = i18n.t(hit.entry.key);
  return hit.detail ? `${text} (${hit.detail})` : text;
}

/** True iff the error is a transient busy/interlock rejection (show as INFO, not error). */
export function isBusyError(e: unknown): boolean {
  return findCode(String(e))?.entry.busy === true;
}

/** THE cancel-sentinel check (single source — the workflow engine and every toast funnel share it).
 *  Backend cancel rejections arrive as "Inference error: 已取消" (legacy) or the stable "CANCELLED"
 *  code; the frontend sentinel is the bare "Cancelled". A user cancel is never an error — funnels
 *  swallow it silently, and it must be checked BEFORE code localization. */
export function isCancelError(e: unknown): boolean {
  const msg = String(e);
  return msg === "Cancelled" || msg.includes("已取消") || msg.includes("CANCELLED");
}
