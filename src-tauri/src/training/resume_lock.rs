//! What a 续训 may and may not change — ONE table, and the guard that enforces it.
//!
//! ## Why a table
//!
//! The rule lived as eight `if` blocks inside `try_start` and, in three other places, as
//! somebody's memory of them: the run-step's pre-start diff dialog, the project page's
//! form-restore, and (from S78) the parameters page, which has to render the locked fields
//! read-only. Four copies of one rule is four chances to disagree — and the failure mode is
//! the worst kind: a dialog that promises「继续训练」and a start that refuses it, or a field the
//! UI lets you edit that silently makes the slot unresumable.
//!
//! So: [`resume_locked_fields`] is the table, [`check_resume_locks`] is the only enforcement,
//! and a unit test drives ONE through the OTHER — for every `Locked` row, a request differing
//! in exactly that field must be refused with exactly that CODE; for every `Costly` row it must
//! NOT be refused. A field added to the guard without a table row (or vice versa) fails there.
//! `src/lib/resumeLockParity.test.ts` extends the same rule across the language boundary.
//!
//! ## Two tiers, because there are two different truths
//!
//! * **Locked** — the value is baked into artifacts that already exist (graph shape, wire
//!   inputs, emb_g rows, the cached ContentVec space). Changing it cannot be reconciled, so the
//!   start is refused and 重训 is the only way through.
//! * **Costly** — changing it is legitimate but re-fingerprints the dataset, so the next run
//!   redoes slicing and feature extraction. Nothing is lost and no progress is destroyed. These
//!   are NOT refused; the UI says what it will cost.
//!
//! Making the Costly ones Locked would be the easy-looking mistake: it converts today's
//! "slow but fine" into a refusal for a case users legitimately hit (adding augmentation, or
//! adding a few takes to an existing singer).

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum LockTier {
    /// Refused outright on resume — only 重训 (which wipes the slot) unlocks it.
    Locked,
    /// Allowed; invalidates the extraction caches, so the next run re-preprocesses.
    Costly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LockedField {
    /// Stable id shared with the frontend's rendering table (parity-tested).
    pub id: &'static str,
    pub tier: LockTier,
    /// The CODE `check_resume_locks` returns for a `Locked` field; "" for `Costly`.
    pub code: &'static str,
}

const fn locked(id: &'static str, code: &'static str) -> LockedField {
    LockedField { id, tier: LockTier::Locked, code }
}
const fn costly(id: &'static str) -> LockedField {
    LockedField { id, tier: LockTier::Costly, code: "" }
}

/// The fields a resume of `backend` may not (or may not cheaply) change.
///
/// `version`/`sample_rate` are Locked everywhere: they choose the graph, the sample rate of
/// every cached slice and — for a diffusion run — the ContentVec space of the `.soft.pt` files
/// AND of the main model the result will be attached to.
pub fn resume_locked_fields(backend: &str) -> Vec<LockedField> {
    // A diffusion run inside a live sovits slot reports its own CODE: 重训(仅扩散) cannot
    // unlock the version there — the main model pins it — so the text must not suggest it.
    // (With no main model in the workspace it is an ordinary resume mismatch; the dedicated
    // test below covers both branches.)
    let ver_code = if backend == "sovits_diff" {
        "DIFF_VERSION_MISMATCH"
    } else {
        "RESUME_PARAMS_MISMATCH"
    };
    let mut v = vec![locked("version", ver_code), locked("sampleRate", ver_code)];
    match backend {
        // 响度嵌入 changes the generator's inputs, so it is part of the graph.
        "sovits" => {
            v.push(locked("volEmbedding", "RESUME_VOL_EMBEDDING_MISMATCH"));
        }
        _ => {}
    }
    if matches!(backend, "sovits" | "rvc" | "sovits_v2") {
        // count and ORDER both: the position IS the emb_g row.
        v.push(locked("speakerCount", "RESUME_SPEAKER_COUNT_MISMATCH"));
        v.push(locked("speakerSet", "RESUME_SPEAKER_SET_MISMATCH"));
    }
    if backend == "sovits_diff" {
        // pins the training distribution t ~ [0,k) and the exported sidecar contract — but only
        // once there IS diffusion progress; before that the slot is free.
        v.push(locked("kStepMax", "RESUME_KSTEP_MISMATCH"));
    }
    // ── Costly ────────────────────────────────────────────────────────────────────────────
    if backend == "sovits" {
        // folded into the dataset fingerprint; a diff run inherits it rather than choosing.
        v.push(costly("loudnorm"));
    }
    // Every backend. For a diffusion run it is honoured only when the sovits slot holds no main
    // model (diff-first); when it IS inherited the request field is simply ignored, and "allowed"
    // remains the truthful answer either way.
    v.push(costly("augCopies"));
    // Every backend: adding or removing audio re-fingerprints the shared dataset.
    v.push(costly("dataset"));
    v
}

/// Everything the guard needs to know that is not in the request.
pub struct ResumeState<'a> {
    /// The slot's `run_manifest.json`, or None when it has never run.
    pub manifest: Option<&'a serde_json::Value>,
    /// A main-model checkpoint shares this workspace (decides the diff wording).
    pub has_main: bool,
    /// Max numbered diffusion checkpoint; 0/None = no diffusion progress yet.
    pub max_diffusion_step: Option<u64>,
    /// The slot's frozen `(slug, name)` pairs — see `training::frozen_speakers`.
    pub frozen_speakers: &'a [super::dsmanifest::DsSpeaker],
}

/// THE resume guard. Returns the CODE (with its payload) to refuse with, or None to allow.
///
/// `enforce` is `!req.fresh || diff_partial_wipe`: a full 重训 wipes the slot, so nothing is
/// baked in any more — but the diffusion partial wipe KEEPS the manifest, so a mismatched
/// version could never train afterwards; deleting first would destroy hours of diffusion
/// progress and only THEN refuse.
pub fn check_resume_locks(
    req: &super::StartTrainingRequest,
    st: &ResumeState<'_>,
    enforce: bool,
) -> Option<String> {
    if !enforce {
        return None;
    }
    let old = st.manifest?;
    let old_ver = old["version"].as_str().unwrap_or("");
    let old_sr = old["sample_rate"].as_str().unwrap_or("");
    // An ABSENT key fails open on purpose: a pre-S37 workspace records neither, and demanding a
    // match would refuse every one of them forever.
    if (!old_ver.is_empty() && old_ver != req.version)
        || (!old_sr.is_empty() && old_sr != req.sample_rate)
    {
        return Some(if req.backend == "sovits_diff" && st.has_main {
            // 重训(仅扩散) cannot unlock the version — it is pinned by the main model, so
            // don't suggest it.
            format!(
                "DIFF_VERSION_MISMATCH: {}/{} -> {}/{}",
                old_ver, old_sr, req.version, req.sample_rate
            )
        } else {
            format!(
                "RESUME_PARAMS_MISMATCH: {}/{} -> {}/{}",
                old_ver, old_sr, req.version, req.sample_rate
            )
        });
    }
    if req.backend == "sovits" {
        if let Some(old_vol) = old["vol_embedding"].as_bool() {
            if old_vol != req.vol_embedding {
                return Some(format!(
                    "RESUME_VOL_EMBEDDING_MISMATCH: {} -> {}",
                    if old_vol { "on" } else { "off" },
                    if req.vol_embedding { "on" } else { "off" }
                ));
            }
        }
    }
    // ①c: n_speakers + the ordered speaker set are baked into the emb_g rows — resuming with a
    // different count / order / set would silently mis-assign every speaker's timbre. (Old
    // single-speaker manifests have no n_speakers key -> 1, which matches a single-speaker
    // resume = no false rejection.)
    if matches!(req.backend.as_str(), "sovits" | "rvc" | "sovits_v2") {
        let old_n = old["n_speakers"].as_u64().unwrap_or(1);
        let cur_n = if req.speakers.len() > 1 { req.speakers.len() as u64 } else { 1 };
        if old_n != cur_n {
            return Some(format!("RESUME_SPEAKER_COUNT_MISMATCH: {} -> {}", old_n, cur_n));
        }
        if cur_n > 1 {
            // Compare DISPLAY NAMES by position, not recomputed slugs: the slug is a
            // `DefaultHasher` derivative, so judging identity by it means a toolchain bump
            // reports every existing co-trained project as "different speakers".
            let named = st.frozen_speakers.len() == cur_n as usize
                && st.frozen_speakers.iter().all(|s| !s.name.is_empty());
            let same = if named {
                st.frozen_speakers
                    .iter()
                    .zip(req.speakers.iter())
                    .all(|(f, s)| f.name == s.name)
            } else {
                // No name survives anywhere (pre-`speaker_names` AND a diff run rewrote run.json
                // without the key). Fall back to the slug comparison this guard always did — the
                // same toolchain exposure, but it is the only identity left.
                let old_slugs: Vec<String> = old["speakers"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                old_slugs
                    == super::assign_speaker_slugs(&req.speakers)
                        .into_iter()
                        .map(|(_, s)| s)
                        .collect::<Vec<_>>()
            };
            if !same {
                return Some("RESUME_SPEAKER_SET_MISMATCH".into());
            }
        }
    }
    // k_step_max pins the diffusion TRAINING distribution and the exported sidecar contract.
    // The fresh partial-wipe path resets the progress, so it may change there.
    if req.backend == "sovits_diff" && !req.fresh {
        if let (Some(old_k), Some(max_step)) = (old["diff_k_step_max"].as_u64(), st.max_diffusion_step)
        {
            if max_step > 0 && old_k != req.k_step_max as u64 {
                let show = |k: u64| if k == 0 { "full-diffusion".to_string() } else { k.to_string() };
                return Some(format!(
                    "RESUME_KSTEP_MISMATCH: {} -> {}",
                    show(old_k),
                    show(req.k_step_max as u64)
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{dsmanifest::DsSpeaker, StartTrainingRequest};

    fn req(json: serde_json::Value) -> StartTrainingRequest {
        serde_json::from_value(json).expect("request fixture")
    }

    fn base(backend: &str) -> serde_json::Value {
        serde_json::json!({
            "model_name": "m", "backend": backend,
            "version": if backend == "rvc" { "v2" } else { "4.1" },
            "sample_rate": if backend == "rvc" { "40k" } else { "44k" },
            // must MATCH `manifest()` — an "unchanged" fixture that is not actually
            // unchanged makes every assertion below vacuous (caught by the guard on first run)
            "k_step_max": 100,
            "dataset_files": [], "total_epoch": 1, "batch_size": 1,
        })
    }

    fn manifest(backend: &str) -> serde_json::Value {
        serde_json::json!({
            "backend": backend,
            "version": if backend == "rvc" { "v2" } else { "4.1" },
            "sample_rate": if backend == "rvc" { "40k" } else { "44k" },
            "vol_embedding": false,
            "loudnorm": false,
            "aug_copies": 0,
            "diff_k_step_max": 100,
        })
    }

    fn state<'a>(m: &'a serde_json::Value, frozen: &'a [DsSpeaker]) -> ResumeState<'a> {
        ResumeState {
            manifest: Some(m),
            has_main: true,
            max_diffusion_step: Some(500),
            frozen_speakers: frozen,
        }
    }

    /// ★ THE anti-drift test: drive the TABLE through the GUARD.
    ///
    /// Every `Locked` row must actually refuse — with its own CODE — when only that field
    /// differs; every `Costly` row must actually be allowed. A guard added without a row, or a
    /// row without a guard, fails here rather than in a user's half-trained slot.
    #[test]
    fn every_locked_field_refuses_and_every_costly_field_does_not() {
        let two = |a: &str, b: &str| {
            serde_json::json!([{"name": a, "files": []}, {"name": b, "files": []}])
        };
        for backend in ["rvc", "sovits", "sovits_v2", "sovits_diff", "vocoder"] {
            let m = manifest(backend);
            let frozen = vec![
                DsSpeaker { slug: "a_1".into(), name: "A".into() },
                DsSpeaker { slug: "b_2".into(), name: "B".into() },
            ];
            // the unchanged request must ALWAYS pass, or every assertion below is vacuous
            let ok = req(base(backend));
            assert_eq!(
                check_resume_locks(&ok, &state(&m, &[]), true),
                None,
                "{backend}: an unchanged resume must be allowed"
            );

            for f in resume_locked_fields(backend) {
                // build a request that differs in exactly this field
                let mut j = base(backend);
                let mut mf = m.clone();
                let mut frz: &[DsSpeaker] = &[];
                match f.id {
                    "version" => j["version"] = serde_json::json!("v1"),
                    "sampleRate" => j["sample_rate"] = serde_json::json!("32k"),
                    "volEmbedding" => j["vol_embedding"] = serde_json::json!(true),
                    "speakerCount" => j["speakers"] = two("A", "B"),
                    "speakerSet" => {
                        // same COUNT as the manifest, different order
                        mf["n_speakers"] = serde_json::json!(2);
                        mf["speakers"] = serde_json::json!(["a_1", "b_2"]);
                        j["speakers"] = two("B", "A");
                        frz = &frozen;
                    }
                    "kStepMax" => j["k_step_max"] = serde_json::json!(200),
                    "loudnorm" => j["loudnorm"] = serde_json::json!(true),
                    "augCopies" => j["aug_copies"] = serde_json::json!(3),
                    "dataset" => continue, // not a request field — the fingerprint decides
                    other => panic!("table row `{other}` has no test case"),
                }
                let got = check_resume_locks(&req(j), &state(&mf, frz), true);
                match f.tier {
                    LockTier::Locked => {
                        let msg = got.unwrap_or_else(|| {
                            panic!("{backend}/{}: table says Locked but the guard allowed it", f.id)
                        });
                        assert!(
                            msg.starts_with(f.code),
                            "{backend}/{}: expected {}, got {msg}",
                            f.id,
                            f.code
                        );
                    }
                    LockTier::Costly => assert_eq!(
                        got, None,
                        "{backend}/{}: table says Costly but the guard refused it",
                        f.id
                    ),
                }
            }
        }
    }

    /// 重训 unlocks everything — that is the ONLY way out of a Locked field, so it must work.
    #[test]
    fn a_fresh_run_is_never_guarded() {
        let m = manifest("sovits");
        let mut j = base("sovits");
        j["version"] = serde_json::json!("4.0");
        j["fresh"] = serde_json::json!(true);
        assert_eq!(check_resume_locks(&req(j), &state(&m, &[]), false), None);
    }

    /// A manifest that predates a field must not be read as「不匹配」— every one of those
    /// workspaces would become unresumable forever.
    #[test]
    fn absent_manifest_keys_fail_open() {
        let bare = serde_json::json!({ "backend": "sovits" });
        let j = base("sovits");
        assert_eq!(check_resume_locks(&req(j), &state(&bare, &[]), true), None);
        // …and so does a slot that never ran at all
        let none = ResumeState {
            manifest: None,
            has_main: false,
            max_diffusion_step: None,
            frozen_speakers: &[],
        };
        assert_eq!(check_resume_locks(&req(base("rvc")), &none, true), None);
    }

    /// Diffusion depth is only pinned once there IS progress to contradict.
    #[test]
    fn k_step_is_free_until_the_diffusion_has_run() {
        let m = manifest("sovits_diff");
        let mut j = base("sovits_diff");
        j["k_step_max"] = serde_json::json!(200);
        let fresh_slot = ResumeState {
            manifest: Some(&m),
            has_main: true,
            max_diffusion_step: Some(0),
            frozen_speakers: &[],
        };
        assert_eq!(check_resume_locks(&req(j.clone()), &fresh_slot, true), None);
        assert!(check_resume_locks(&req(j), &state(&m, &[]), true).is_some());
    }

    /// The diff run's version refusal names a different CODE, because 重训(仅扩散) cannot
    /// unlock it — the main model pins it.
    #[test]
    fn a_diffusion_version_mismatch_says_so_specifically() {
        let m = manifest("sovits_diff");
        let mut j = base("sovits_diff");
        j["version"] = serde_json::json!("4.0");
        let msg = check_resume_locks(&req(j.clone()), &state(&m, &[]), true).unwrap();
        assert!(msg.starts_with("DIFF_VERSION_MISMATCH"), "{msg}");
        // without a main model in the workspace it is an ordinary resume mismatch
        let no_main = ResumeState {
            manifest: Some(&m),
            has_main: false,
            max_diffusion_step: Some(500),
            frozen_speakers: &[],
        };
        let msg = check_resume_locks(&req(j), &no_main, true).unwrap();
        assert!(msg.starts_with("RESUME_PARAMS_MISMATCH"), "{msg}");
    }
}
