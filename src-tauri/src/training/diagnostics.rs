//! S115 §F5-2 — diagnostic mode: the ONE enumeration of what it turns on, plus the banner
//! that writes that enumeration into the log.
//!
//! WHY THIS IS A MODULE AND NOT TWO `if`s AT THE SPAWN POINT
//! --------------------------------------------------------
//! The feature exists so a user can reproduce a failure we cannot see and mail us the log.
//! That only works if the log says WHAT DIAGNOSTIC MODE CONTAINED IN THEIR BUILD — six months
//! and three releases from now, a file that merely says "diagnostic mode: on" is unreadable.
//! So the set is enumerated once here, and `banner()` prints that same enumeration with the
//! app version; `diagnostic_banner_lists_every_variable` keeps the two from drifting.
//!
//! WHAT IS DELIBERATELY *NOT* IN THE SET
//! -------------------------------------
//! * `TORCH_USE_CUDA_DSA` — the obvious second knob, and it is inert TWICE OVER on what we
//!   ship (measured S115 from the pack's own bytes, not from memory):
//!     1. it is not an env var at all. `c10_cuda.dll` reads `PYTORCH_USE_CUDA_DSA` /
//!        `PYTORCH_CUDA_DSA_STACKTRACING`; the token `TORCH_USE_CUDA_DSA` appears only inside
//!        the human sentence "Compile with `TORCH_USE_CUDA_DSA` to enable device-side
//!        assertions" — it is a COMPILE flag;
//!     2. our cu130 build has that branch compiled OUT (the whole device-side-assertion
//!        report block — `dsa_get_device_id`, "Thread ID that failed assertion", … — is
//!        present in the amd pack's `c10_hip.dll` and has ZERO occurrences in the nv pack's
//!        `c10_cuda.dll`).
//!   Shipping it would put a variable nothing reads into a list that claims to describe the
//!   run. An inert member of a self-describing set is worse than no set.
//! * A log-level change. The `[train-py]` stderr forwarding is `tracing::debug!` with an
//!   explicit `target: "utai"`, and the FILE layer's filter is `warn,utai=debug` — so python
//!   tracebacks ALREADY reach `utai.log.<date>` and the in-app panel today. Raising it would
//!   fix nothing and would dilute the two real warning CODEs.

use std::sync::atomic::{AtomicBool, Ordering};

/// Set from `config.json` at startup (`load_and_apply_config`) and by `set_diagnostic_mode`.
/// Same shape as `inference::engine::CUDA_MEM_LIMIT_MB`: config is the source of truth on
/// disk, this static is what a run reads.
static DIAGNOSTIC_MODE: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(on: bool) {
    DIAGNOSTIC_MODE.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    DIAGNOSTIC_MODE.load(Ordering::Relaxed)
}

/// One environment variable diagnostic mode sets on the training sidecar.
pub struct DiagnosticVar {
    pub name: &'static str,
    pub value: &'static str,
    /// Why THIS runtime needs THIS name — the two GPU builds do not share one.
    pub why: &'static str,
}

/// What diagnostic mode sets for a given runtime. Empty = nothing useful exists for it, and
/// the caller must then say so rather than pretend.
///
/// ⛔ Keyed on the PACK VARIANT, never on `device_backend`: that one collapses amd → "cuda"
/// (torch-hip owns the `torch.cuda.*` namespace), and the two builds read DIFFERENT names.
/// Measured S115 by extracting each pack's c10 DLL and counting strings:
///   nv  `c10_cuda.dll`: CUDA_LAUNCH_BLOCKING ×2, AMD_SERIALIZE_KERNEL ×0,
///        advice sentence "For debugging consider passing CUDA_LAUNCH_BLOCKING=1"
///   amd `c10_hip.dll` : AMD_SERIALIZE_KERNEL ×2, CUDA_LAUNCH_BLOCKING ×0,
///        advice sentence "For debugging consider passing AMD_SERIALIZE_KERNEL=3"
///   xpu `c10_xpu.dll` : neither, and no equivalent found — 【unverified whether one exists】
/// So `if device_backend == "cuda"` would have set a no-op on every ROCm run while the banner
/// claimed otherwise.
///
/// `variant == None` means a non-pack interpreter (dev venv / manual slot / bare python).
/// Those tiers are claimed for NVIDIA+CPU ONLY — by the same rule `available_training_variants`
/// states — so a `None` with backend "cuda" is a CUDA torch and gets the NVIDIA knob.
pub fn vars_for(variant: Option<&str>, device_backend: &str) -> Vec<DiagnosticVar> {
    const CUDA: DiagnosticVar = DiagnosticVar {
        name: "CUDA_LAUNCH_BLOCKING",
        value: "1",
        why: "makes CUDA kernel launches synchronous, so the reported stack is the kernel that \
               actually failed — CUDA errors are otherwise reported asynchronously and the \
               traceback points at whatever ran next",
    };
    const ROCM: DiagnosticVar = DiagnosticVar {
        name: "AMD_SERIALIZE_KERNEL",
        value: "3",
        why: "the torch-hip build's own equivalent of CUDA_LAUNCH_BLOCKING (it does not read \
               that name); 3 = serialize before and after every kernel",
    };
    match variant {
        Some("amd") => vec![ROCM],
        Some(v) if v.starts_with("nv") => vec![CUDA],
        Some(_) => Vec::new(), // xpu / cpu: nothing equivalent exists
        None if device_backend == "cuda" => vec![CUDA],
        None => Vec::new(),
    }
}

/// Stage diagnostic mode onto a training-sidecar command, if it is on. Returns what was set —
/// EMPTY both when the mode is off and when this runtime has no knob, so the caller must not
/// treat "empty" as "off" (that is why `enabled()` is checked here and not by the caller).
///
/// ⛔ Only the TRAINING spawn calls this. `util::python_command` is shared with the converter,
/// MSST export and envtest spawns, and serializing kernel launches there would slow unrelated
/// work for no diagnostic value.
pub fn apply(
    cmd: &mut std::process::Command,
    variant: Option<&str>,
    device_backend: &str,
) -> Vec<DiagnosticVar> {
    if !enabled() {
        return Vec::new();
    }
    let vars = vars_for(variant, device_backend);
    for v in &vars {
        cmd.env(v.name, v.value);
    }
    vars
}

/// The line that goes into the log when a diagnostic run starts. It MUST name every variable
/// in `vars` and carry the app version — a diagnostic log that cannot say which build's
/// diagnostic set produced it is not evidence.
pub fn banner(vars: &[DiagnosticVar], variant: Option<&str>, device_backend: &str) -> String {
    let set = if vars.is_empty() {
        "nothing to set for this runtime (no equivalent knob exists)".to_string()
    } else {
        vars.iter().map(|v| format!("{}={}", v.name, v.value)).collect::<Vec<_>>().join(", ")
    };
    format!(
        "DIAGNOSTIC MODE is ON for this run — UtaiSynthesizer {} — {} (runtime {}, backend {}). \
         The run will be noticeably slower; turn it off in Settings when you are done.",
        env!("CARGO_PKG_VERSION"),
        set,
        variant.unwrap_or("dev/manual interpreter"),
        device_backend,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every (variant, backend) a run can actually reach. `training_interpreter*` produces
    /// exactly these pairings — pack variants carry their own backend via `variant_backend`
    /// (amd → cuda), non-pack tiers carry cuda or cpu.
    const COMBOS: &[(Option<&str>, &str)] = &[
        (Some("nv-cu130"), "cuda"),
        (Some("amd"), "cuda"),
        (Some("xpu"), "xpu"),
        (Some("cpu"), "cpu"),
        (None, "cuda"),
        (None, "cpu"),
    ];

    /// ★THE self-describing contract. Add a variable to `vars_for` and forget the banner —
    /// or hand-write the banner text — and this goes red. Without it the feature ships logs
    /// that cannot be read by the person who receives them.
    #[test]
    fn s115_diagnostic_banner_lists_every_variable() {
        for (variant, backend) in COMBOS {
            let vars = vars_for(*variant, backend);
            let line = banner(&vars, *variant, backend);
            for v in &vars {
                assert!(
                    line.contains(v.name) && line.contains(v.value),
                    "banner for ({variant:?}, {backend}) does not name {}={}: {line}",
                    v.name,
                    v.value
                );
            }
            assert!(
                line.contains(env!("CARGO_PKG_VERSION")),
                "banner must carry the app version: {line}"
            );
            // …and it must never be silently empty-looking when there IS nothing to set.
            if vars.is_empty() {
                assert!(line.contains("nothing to set"), "{line}");
            }
        }
    }

    /// The bug this whole keying exists to prevent: amd and nv share `device_backend == "cuda"`
    /// but NOT the variable. Pinned as an inequality so "simplifying" the match arms reddens.
    #[test]
    fn s115_rocm_and_nvidia_do_not_get_the_same_variable() {
        let nv = vars_for(Some("nv-cu130"), "cuda");
        let amd = vars_for(Some("amd"), "cuda");
        assert_eq!(nv.len(), 1);
        assert_eq!(amd.len(), 1);
        assert_eq!(nv[0].name, "CUDA_LAUNCH_BLOCKING");
        assert_eq!(amd[0].name, "AMD_SERIALIZE_KERNEL");
        assert_ne!(
            nv[0].name, amd[0].name,
            "both runtimes report device_backend=cuda; keying on that would set a no-op on ROCm"
        );
    }

    /// Runtimes with no equivalent knob must produce an EMPTY set, not a plausible-looking one.
    #[test]
    fn s115_xpu_and_cpu_get_nothing_rather_than_a_no_op() {
        assert!(vars_for(Some("xpu"), "xpu").is_empty());
        assert!(vars_for(Some("cpu"), "cpu").is_empty());
        assert!(vars_for(None, "cpu").is_empty());
    }

    /// ★The mechanism itself, on a REAL `Command`. Everything else here tests the decision;
    /// this tests that the decision reaches the process that has to obey it. Asserted through
    /// `Command::get_envs()` rather than by reading the three lines that do it — S115's whole
    /// §G15 round was about production text that described something the code no longer did.
    ///
    /// ⚠ These tests share one process-wide static, so each one restores it. `enabled()`
    /// defaults false, and `apply` is deliberately the thing that checks it: a caller that
    /// forgot the check would otherwise inject on every run.
    #[test]
    fn s115_apply_stages_the_env_only_when_the_mode_is_on() {
        let staged = |variant: Option<&str>, backend: &str| -> Vec<(String, Option<String>)> {
            let mut cmd = std::process::Command::new("python");
            super::apply(&mut cmd, variant, backend);
            cmd.get_envs()
                .map(|(k, v)| {
                    (k.to_string_lossy().into_owned(), v.map(|v| v.to_string_lossy().into_owned()))
                })
                .collect()
        };

        set_enabled(false);
        assert!(staged(Some("nv-cu130"), "cuda").is_empty(), "OFF must stage nothing at all");

        set_enabled(true);
        let nv = staged(Some("nv-cu130"), "cuda");
        assert_eq!(nv, vec![("CUDA_LAUNCH_BLOCKING".to_string(), Some("1".to_string()))]);
        let amd = staged(Some("amd"), "cuda");
        assert_eq!(amd, vec![("AMD_SERIALIZE_KERNEL".to_string(), Some("3".to_string()))]);
        // …and the runtime with no knob must get a CLEAN command, not a placebo variable.
        assert!(staged(Some("xpu"), "xpu").is_empty());
        assert!(staged(Some("cpu"), "cpu").is_empty());
        set_enabled(false);
    }

    /// The user-visible half. A persisted flag whose only symptom is slowness MUST be able to
    /// say so in every language, in both places it appears — otherwise the feature manufactures
    /// the "your update made training slower" report it exists to prevent. Same `include_str!`
    /// cross-language shape as the S114 warning-code contract.
    #[test]
    fn s115_diagnostic_mode_is_explained_in_all_three_locales() {
        for (lang, raw) in [
            ("zh", include_str!("../../../src/i18n/zh.json")),
            ("en", include_str!("../../../src/i18n/en.json")),
            ("ja", include_str!("../../../src/i18n/ja.json")),
        ] {
            let v: serde_json::Value = serde_json::from_str(raw).unwrap();
            let msg = v
                .pointer("/training/diagnosticOn")
                .and_then(|m| m.as_str())
                .unwrap_or_else(|| panic!("{lang}.json is missing training.diagnosticOn"));
            assert!(
                msg.chars().count() >= 30,
                "training.diagnosticOn in {lang}.json is too short to be the real warning: {msg:?}"
            );
        }
        // The Settings panel keeps its own inline trilingual table, so the JSON check above
        // cannot see it. Pin that the switch and its cost sentence exist there in all three.
        const SETTINGS_TSX: &str = include_str!("../../../src/components/common/Settings.tsx");
        for key in ["diagTitle", "diagToggle", "diagNote", "diagOnNote"] {
            let line = SETTINGS_TSX
                .lines()
                .find(|l| l.trim_start().starts_with(&format!("{key}:")))
                .unwrap_or_else(|| panic!("Settings.tsx lost the {key} label"));
            // Multi-line entries keep their locales on the following lines, so look at a window
            // after the key. ⚠ Take CHARACTERS, not bytes: these labels are full of CJK and a
            // byte slice lands inside a multi-byte char (it did, first run).
            let idx = SETTINGS_TSX.find(line).unwrap();
            let block: String = SETTINGS_TSX[idx..].chars().take(600).collect();
            for tag in ["zh:", "en:", "ja:"] {
                assert!(block.contains(tag), "Settings.tsx {key} has no {tag} locale");
            }
        }
        // And the training page must actually render the banner from that key.
        const TRAINING_PAGE: &str = include_str!("../../../src/components/training/TrainingPage.tsx");
        assert!(
            TRAINING_PAGE.contains("training.diagnosticOn"),
            "TrainingPage.tsx no longer renders the diagnostic banner — a persisted slow mode \
             would then be invisible, which is the whole failure this guards"
        );
    }
}
