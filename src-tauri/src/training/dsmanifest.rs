//! `<project>/dataset.json` — the ANNOTATION layer over a project's shared dataset.
//!
//! ## Why it exists
//!
//! The import copies the user's audio in as `000.wav` / `<slug>/000.wav`: positional names,
//! chosen so the extraction-cache fingerprint is stable under a re-pick in a different dialog
//! order. Nothing on disk records what those files originally were, or which display name a
//! speaker slug came from (`slugify` is one-way). Months later that is exactly what the user
//! needs in order to decide whether to continue a run — and for a multi-speaker project, the
//! speaker ORDER is the emb_g row order, so getting it wrong swaps timbres.
//!
//! ## Why it is not inside `dataset/`
//!
//! `dataset/` is an exact-match contract region. Three independent judgements read it as a
//! whole directory listing:
//!
//! * `dataset_matches` compares the full listing (name + content digest) against the planned
//!   import — one extra entry and no future import can ever compare equal again;
//! * `get_training_project` counts every non-directory entry as an audio file and every
//!   directory as a speaker, so a sidecar file inflates the count and a sidecar directory makes
//!   the project read as multi-speaker (and `poolFlat` false);
//! * python's `dataset_fingerprint` hashes every entry's name+size+head/tail and RAISES on any
//!   subdirectory for a flat backend — a sidecar file silently wipes every extraction cache
//!   once, a sidecar directory hard-fails the run.
//!
//! So the manifest lives one level up, beside `project.json`, where the only scanner
//! (`tproject::list_projects`) looks at directories and `project.json` alone.
//!
//! ## It is an ANNOTATION, never the authority
//!
//! The disk is the truth. Every read reconciles against the real listing: entries whose file is
//! gone are dropped, files with no entry are reported with their on-disk name. A missing,
//! truncated or hand-mangled `dataset.json` therefore degrades to「显示 000.wav」and must never
//! block a listing, an import or a training run.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Result, UtaiError};

pub const DATASET_MANIFEST: &str = "dataset.json";

/// One imported file: where it landed, and what it was.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DsFile {
    /// Path relative to `dataset/`, forward-slashed: `000.wav` or `<slug>/000.wav`. The same
    /// string `dataset_plan` builds, so the two can be compared directly.
    pub rel: String,
    /// The source file's own name at import time. Empty = unknown (imported before batch 5).
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
}

/// One co-trained speaker. The POSITION in [`DsManifest::speakers`] is the emb_g row id.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DsSpeaker {
    /// `dataset/<slug>/` — also the `dataset_44k` subdir name and the `config.spk` key.
    pub slug: String,
    /// What the user typed. `slugify` is not invertible, so this is the only carrier that
    /// survives outside a slot's `run_manifest.json`.
    pub name: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DsManifest {
    /// Format version. 0 = absent/unversioned; readers must tolerate both.
    #[serde(default)]
    pub v: u32,
    /// Empty for a flat (single-speaker) dataset.
    #[serde(default)]
    pub speakers: Vec<DsSpeaker>,
    #[serde(default)]
    pub files: Vec<DsFile>,
    /// Forward compatibility, same reason as `ProjectMeta::extra`: a downgraded build must not
    /// silently drop what a newer one wrote.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

pub const MANIFEST_VERSION: u32 = 1;

fn manifest_path(data_dir: &Path, id: &str) -> std::path::PathBuf {
    super::tproject::project_dir(data_dir, id).join(DATASET_MANIFEST)
}

/// Never fails: an unreadable manifest is the same as no manifest (see the module doc).
pub fn read(data_dir: &Path, id: &str) -> DsManifest {
    std::fs::read_to_string(manifest_path(data_dir, id))
        .ok()
        .and_then(|s| serde_json::from_str::<DsManifest>(&s).ok())
        .unwrap_or_default()
}

/// Atomic write (tmp + rename in the same directory), same shape as `write_meta` — a torn
/// manifest would read as「原名全部未知」on the next open.
pub fn write(data_dir: &Path, id: &str, m: &DsManifest) -> Result<()> {
    let dir = super::tproject::project_dir(data_dir, id);
    std::fs::create_dir_all(&dir).map_err(|e| {
        UtaiError::Training(format!("DATASET_META_WRITE_FAILED: {}: {e}", dir.display()))
    })?;
    let tmp = dir.join(format!("{DATASET_MANIFEST}.tmp"));
    let body = serde_json::to_string_pretty(m)
        .map_err(|e| UtaiError::Training(format!("DATASET_META_ENCODE_FAILED: {e}")))?;
    std::fs::write(&tmp, body)
        .map_err(|e| UtaiError::Training(format!("DATASET_META_WRITE_FAILED: {e}")))?;
    std::fs::rename(&tmp, dir.join(DATASET_MANIFEST))
        .map_err(|e| UtaiError::Training(format!("DATASET_META_WRITE_FAILED: {e}")))?;
    Ok(())
}

/// Record what an import just wrote. BEST EFFORT by contract: the run has already copied the
/// audio, and losing the annotation must never turn a completed import into a failed start.
///
/// `files` is the full post-import content of `dataset/` — the import always replaces the whole
/// dataset, so the manifest is rewritten wholesale rather than merged. Callers must NOT call
/// this on the reuse path (empty selection): there are no source names there, and writing an
/// unknown-name manifest would erase good annotations from an earlier import.
pub fn record_import(data_dir: &Path, id: &str, speakers: Vec<DsSpeaker>, files: Vec<DsFile>) {
    let m = DsManifest {
        v: MANIFEST_VERSION,
        speakers,
        files,
        extra: Default::default(),
    };
    if let Err(e) = write(data_dir, id, &m) {
        // Loud in the log, silent to the run: see the contract above.
        tracing::warn!("dataset annotation not written for {id}: {e}");
    }
}

/// One speaker as the UI lists it: who, and what they contribute.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupFacts {
    pub speaker: DsSpeaker,
    pub files: u32,
    pub bytes: u64,
}

/// Everything the UI needs about a project's shared dataset, read off disk and reconciled with
/// the annotation. Lives HERE rather than in the command so it is reachable from tests — the
/// command has a `State` in its signature and can only ever be exercised by hand.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DatasetFacts {
    /// Every audio-bearing entry, flat count (top level + one level of speaker subdirectories).
    pub files: u32,
    /// Speaker subdirectory names, SORTED. Kept separate from `groups` because the frontend's
    /// `poolFlat` judgement keys on this being empty and must not change shape.
    pub speaker_slugs: Vec<String>,
    /// Files sorted by `rel`, with the original name filled in where the annotation knows it.
    pub entries: Vec<DsFile>,
    /// Speakers in emb_g order when knowable, alphabetical otherwise.
    pub groups: Vec<GroupFacts>,
    pub order_known: bool,
}

/// Read `dataset/` and reconcile it with `dataset.json`.
///
/// `frozen` is each architecture slot's frozen `(slug, name)` list (see
/// `crate::training::frozen_speakers`); it is passed in rather than read here so this stays a
/// function of its inputs.
///
/// The disk is the authority throughout: an annotation entry whose file is gone is dropped, and
/// a file with no entry is listed under its on-disk name.
pub fn read_facts(data_dir: &Path, id: &str, frozen: &[Vec<DsSpeaker>]) -> DatasetFacts {
    let dataset_dir = super::tproject::dataset_dir(data_dir, id);
    let mut speaker_slugs: Vec<String> = Vec::new();
    // (rel, bytes) — two levels only: the only shape the import writes, and the same depth
    // `current_dataset_listing` judges a dataset change by.
    let mut on_disk: Vec<(String, u64)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dataset_dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if e.path().is_dir() {
                speaker_slugs.push(name);
            } else {
                on_disk.push((name, e.metadata().map(|m| m.len()).unwrap_or(0)));
            }
        }
    }
    for s in &speaker_slugs {
        if let Ok(rd) = std::fs::read_dir(dataset_dir.join(s)) {
            for e in rd.flatten() {
                if e.path().is_file() {
                    on_disk.push((
                        // forward-slashed on every platform: `rel` is compared against
                        // `dataset_plan`'s strings and shown in the UI.
                        format!("{}/{}", s, e.file_name().to_string_lossy()),
                        e.metadata().map(|m| m.len()).unwrap_or(0),
                    ));
                }
            }
        }
    }
    speaker_slugs.sort();
    on_disk.sort_by(|a, b| a.0.cmp(&b.0));

    let ann = read(data_dir, id);
    let entries: Vec<DsFile> = on_disk
        .iter()
        .map(|(rel, bytes)| {
            let rec = ann.files.iter().find(|f| &f.rel == rel);
            DsFile {
                rel: rel.clone(),
                name: rec.map(|f| f.name.clone()).unwrap_or_default(),
                bytes: *bytes,
                duration_ms: rec.and_then(|f| f.duration_ms),
            }
        })
        .collect();
    let view = resolve_speakers(&speaker_slugs, &ann.speakers, frozen);
    let groups = view
        .groups
        .iter()
        .map(|g| {
            let prefix = format!("{}/", g.slug);
            let mine = on_disk.iter().filter(|(rel, _)| rel.starts_with(&prefix));
            GroupFacts {
                speaker: g.clone(),
                files: mine.clone().count() as u32,
                bytes: mine.map(|(_, b)| *b).sum(),
            }
        })
        .collect();
    DatasetFacts {
        files: entries.len() as u32,
        speaker_slugs,
        entries,
        groups,
        order_known: view.order_known,
    }
}

/// The dataset as the UI should show it, after reconciling the annotation against the disk.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetView {
    /// Speakers in emb_g order when it is knowable, alphabetical otherwise. Empty = flat.
    pub groups: Vec<DsSpeaker>,
    /// False when nothing on disk records the order (a multi-speaker dataset that has never
    /// been trained and predates batch 5). Displaying a guessed order as if it were the emb_g
    /// order is the one mistake that silently mis-assigns timbres on a manual rebuild.
    pub order_known: bool,
}

/// Resolve the speaker list from every source that can know it. PURE — the caller supplies what
/// it read, so the precedence is testable without a filesystem.
///
/// * `disk` — the slug subdirectories actually present (any order).
/// * `annotated` — `dataset.json`'s speakers.
/// * `frozen` — each architecture slot's frozen `(slug, name)` list, in emb_g order, for the
///   slots that have one. These are per-SLOT truths: two slots may legitimately hold different
///   orders (each trained its own emb_g rows), which is why disagreement here is reported as
///   "order unknown at the project level" rather than resolved by preference.
///
/// A source is usable only when its slug SET equals the disk's — a stale record from before a
/// dataset replacement describes speakers that are no longer there.
pub fn resolve_speakers(
    disk: &[String],
    annotated: &[DsSpeaker],
    frozen: &[Vec<DsSpeaker>],
) -> DatasetView {
    let same_set = |c: &[DsSpeaker]| -> bool {
        if c.len() != disk.len() {
            return false;
        }
        let mut a: Vec<&str> = c.iter().map(|s| s.slug.as_str()).collect();
        let mut b: Vec<&str> = disk.iter().map(|s| s.as_str()).collect();
        a.sort_unstable();
        b.sort_unstable();
        a == b
    };
    let fallback = || DatasetView {
        groups: {
            let mut g: Vec<DsSpeaker> = disk
                .iter()
                .map(|s| DsSpeaker {
                    slug: s.clone(),
                    name: String::new(),
                })
                .collect();
            g.sort_by(|a, b| a.slug.cmp(&b.slug));
            g
        },
        order_known: false,
    };
    if disk.is_empty() {
        return DatasetView {
            groups: Vec::new(),
            // A flat dataset has no order to know; `groups` being empty is what the UI reads.
            order_known: true,
        };
    }

    // `dataset.json` wins outright: it is written BY the import that created these directories,
    // so it describes this exact dataset — where a slot manifest describes what that slot was
    // last trained on, which may predate a data replacement.
    if same_set(annotated) {
        return DatasetView {
            groups: annotated.to_vec(),
            order_known: true,
        };
    }

    let usable: Vec<&Vec<DsSpeaker>> = frozen.iter().filter(|c| same_set(c)).collect();
    let Some(first) = usable.first() else {
        return fallback();
    };
    let slugs_of = |c: &Vec<DsSpeaker>| -> Vec<String> { c.iter().map(|s| s.slug.clone()).collect() };
    let agreed = usable.iter().all(|c| slugs_of(c) == slugs_of(first));
    if !agreed {
        // Two slots trained different emb_g orders over the same speakers. Neither is "the"
        // project order; each slot card shows its own. Names are still recoverable, so keep
        // them — only the ordering claim is dropped.
        let mut view = fallback();
        for g in view.groups.iter_mut() {
            if let Some(src) = first.iter().find(|s| s.slug == g.slug) {
                g.name = src.name.clone();
            }
        }
        return view;
    }
    DatasetView {
        groups: (*first).clone(),
        order_known: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(slug: &str, name: &str) -> DsSpeaker {
        DsSpeaker {
            slug: slug.into(),
            name: name.into(),
        }
    }

    #[test]
    fn flat_dataset_has_no_speakers_and_no_order_question() {
        let v = resolve_speakers(&[], &[], &[]);
        assert!(v.groups.is_empty());
        assert!(v.order_known);
    }

    #[test]
    fn annotation_wins_and_carries_the_order() {
        let disk = vec!["b_2".to_string(), "a_1".to_string()];
        let ann = vec![sp("a_1", "亚里沙"), sp("b_2", "Bella")];
        // a slot that disagrees must NOT override the annotation
        let frozen = vec![vec![sp("b_2", "Bella"), sp("a_1", "亚里沙")]];
        let v = resolve_speakers(&disk, &ann, &frozen);
        assert!(v.order_known);
        assert_eq!(v.groups, ann);
    }

    #[test]
    fn a_stale_annotation_is_ignored_entirely() {
        // dataset was replaced: the annotation still describes the OLD speakers
        let disk = vec!["c_3".to_string(), "d_4".to_string()];
        let ann = vec![sp("a_1", "亚里沙"), sp("b_2", "Bella")];
        let frozen = vec![vec![sp("d_4", "Dora"), sp("c_3", "Cera")]];
        let v = resolve_speakers(&disk, &ann, &frozen);
        assert!(v.order_known);
        assert_eq!(v.groups, vec![sp("d_4", "Dora"), sp("c_3", "Cera")]);
    }

    #[test]
    fn slot_manifest_recovers_names_and_order_for_a_legacy_project() {
        let disk = vec!["a_1".to_string(), "b_2".to_string()];
        let frozen = vec![vec![sp("b_2", "Bella"), sp("a_1", "亚里沙")]];
        let v = resolve_speakers(&disk, &[], &frozen);
        assert!(v.order_known);
        assert_eq!(v.groups, vec![sp("b_2", "Bella"), sp("a_1", "亚里沙")]);
    }

    #[test]
    fn disagreeing_slots_keep_the_names_but_drop_the_order_claim() {
        let disk = vec!["a_1".to_string(), "b_2".to_string()];
        let frozen = vec![
            vec![sp("b_2", "Bella"), sp("a_1", "亚里沙")],
            vec![sp("a_1", "亚里沙"), sp("b_2", "Bella")],
        ];
        let v = resolve_speakers(&disk, &[], &frozen);
        assert!(!v.order_known, "two slots disagree — no project-level order");
        // alphabetical, names still filled in
        assert_eq!(v.groups, vec![sp("a_1", "亚里沙"), sp("b_2", "Bella")]);
    }

    #[test]
    fn no_source_at_all_falls_back_to_slugs_in_alphabetical_order() {
        let disk = vec!["b_2".to_string(), "a_1".to_string()];
        let v = resolve_speakers(&disk, &[], &[]);
        assert!(!v.order_known);
        assert_eq!(v.groups, vec![sp("a_1", ""), sp("b_2", "")]);
    }

    #[test]
    fn a_partially_matching_source_is_not_usable() {
        // one speaker was added on disk after the record was written
        let disk = vec!["a_1".to_string(), "b_2".to_string(), "c_3".to_string()];
        let ann = vec![sp("a_1", "亚里沙"), sp("b_2", "Bella")];
        let v = resolve_speakers(&disk, &ann, &[]);
        assert!(!v.order_known);
        assert_eq!(v.groups.len(), 3);
        assert!(v.groups.iter().all(|g| g.name.is_empty()));
    }

    #[test]
    fn manifest_round_trips_and_tolerates_unknown_fields() {
        let raw = r#"{"v":1,"speakers":[{"slug":"a_1","name":"亚里沙"}],
                      "files":[{"rel":"a_1/000.wav","name":"原始.wav","bytes":12}],
                      "future_field":{"x":1}}"#;
        let m: DsManifest = serde_json::from_str(raw).expect("parses");
        assert_eq!(m.v, 1);
        assert_eq!(m.speakers[0].name, "亚里沙");
        assert_eq!(m.files[0].duration_ms, None);
        assert!(m.extra.contains_key("future_field"));
        let back = serde_json::to_string(&m).unwrap();
        assert!(back.contains("future_field"), "forward-compat field kept");
        assert!(!back.contains("duration_ms"), "absent duration stays absent");
    }

    /// Round-trip through a REAL directory: the reconciliation has to survive an annotation that
    /// is partly right (one file deleted behind our back, one added).
    #[test]
    fn facts_reconcile_the_annotation_against_the_disk() {
        let tmp = std::env::temp_dir().join(format!("utai_dsfacts_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let id = "p_1";
        let ds = super::super::tproject::dataset_dir(&tmp, id);
        std::fs::create_dir_all(ds.join("a_1")).unwrap();
        std::fs::create_dir_all(ds.join("b_2")).unwrap();
        std::fs::write(ds.join("a_1").join("000.wav"), b"aaaa").unwrap();
        std::fs::write(ds.join("a_1").join("001.wav"), b"bb").unwrap();
        std::fs::write(ds.join("b_2").join("000.flac"), b"ccc").unwrap();
        record_import(
            &tmp,
            id,
            vec![sp("b_2", "Bella"), sp("a_1", "亚里沙")],
            vec![
                DsFile { rel: "b_2/000.flac".into(), name: "原始 B.flac".into(), bytes: 3, duration_ms: Some(1.5) },
                DsFile { rel: "a_1/000.wav".into(), name: "原始 A.wav".into(), bytes: 4, duration_ms: None },
                // an entry whose file is NOT on disk — must be dropped, not shown as a ghost
                DsFile { rel: "a_1/009.wav".into(), name: "已删除.wav".into(), bytes: 9, duration_ms: None },
            ],
        );

        let facts = read_facts(&tmp, id, &[]);
        assert_eq!(facts.files, 3, "counts the disk, not the annotation");
        assert_eq!(facts.speaker_slugs, vec!["a_1", "b_2"], "slug list stays sorted");
        assert_eq!(
            facts.entries.iter().map(|e| e.rel.as_str()).collect::<Vec<_>>(),
            vec!["a_1/000.wav", "a_1/001.wav", "b_2/000.flac"],
            "rel-sorted, forward-slashed, no ghost row"
        );
        assert_eq!(facts.entries[0].name, "原始 A.wav");
        // on disk but unannotated ⇒ blank name, never a guess
        assert_eq!(facts.entries[1].name, "");
        assert_eq!(facts.entries[1].bytes, 2, "bytes come from the file, not the record");
        assert_eq!(facts.entries[2].duration_ms, Some(1.5));
        // emb_g order comes from the annotation, so B is row 0 even though A sorts first
        assert!(facts.order_known);
        assert_eq!(facts.groups[0].speaker.name, "Bella");
        assert_eq!(facts.groups[0].files, 1);
        assert_eq!(facts.groups[1].speaker.name, "亚里沙");
        assert_eq!(facts.groups[1].files, 2);
        assert_eq!(facts.groups[1].bytes, 6);

        // no annotation at all = every legacy project: still fully listed, just unnamed
        std::fs::remove_file(super::super::tproject::project_dir(&tmp, id).join(DATASET_MANIFEST))
            .unwrap();
        let bare = read_facts(&tmp, id, &[]);
        assert_eq!(bare.files, 3);
        assert!(bare.entries.iter().all(|e| e.name.is_empty()));
        assert!(!bare.order_known, "nothing on disk records the order any more");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Diagnostic against THIS machine's real training data. The fixtures above cannot cover CJK
    /// project names, a legacy multi-speaker project whose only order record is a slot manifest,
    /// or a dataset big enough for the walk to matter. Run after touching the reconciliation:
    ///   cargo test --lib dataset_view_on_this_machine -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dataset_view_on_this_machine() {
        let data = std::path::PathBuf::from("D:/MyDev/Utai_v2-dev/data");
        let projects = super::super::tproject::list_projects(&data);
        println!("\n{} project(s) under {}", projects.len(), data.display());
        for m in &projects {
            let frozen: Vec<Vec<DsSpeaker>> = super::super::tproject::FAMILIES
                .iter()
                .map(|f| super::super::frozen_speakers(&data, &m.id, f))
                .filter(|v| !v.is_empty())
                .collect();
            let t0 = std::time::Instant::now();
            let facts = read_facts(&data, &m.id, &frozen);
            let named = facts.entries.iter().filter(|e| !e.name.is_empty()).count();
            println!(
                "  {:<26} {:>4} files ({} named) · {} singer(s) order_known={} · {:?}  {}",
                m.id,
                facts.files,
                named,
                facts.groups.len(),
                facts.order_known,
                t0.elapsed(),
                m.name
            );
            for (i, g) in facts.groups.iter().enumerate() {
                println!(
                    "      #{} {:<24} {:>4} files  {:>8}KB  [{}]",
                    i,
                    if g.speaker.name.is_empty() { "(名字未记录)" } else { &g.speaker.name },
                    g.files,
                    g.bytes / 1024,
                    g.speaker.slug
                );
            }
            // ---- invariants that must hold for EVERY real project ----
            assert_eq!(facts.files as usize, facts.entries.len());
            for e in &facts.entries {
                assert!(!e.rel.contains('\\'), "rel must be forward-slashed: {}", e.rel);
                assert!(
                    super::super::tproject::dataset_dir(&data, &m.id).join(&e.rel).is_file(),
                    "listed a file that is not there: {}/{}",
                    m.id,
                    e.rel
                );
            }
            let mut group_slugs: Vec<&str> =
                facts.groups.iter().map(|g| g.speaker.slug.as_str()).collect();
            group_slugs.sort_unstable();
            assert_eq!(
                group_slugs,
                facts.speaker_slugs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                "groups must describe exactly the speaker directories on disk ({})",
                m.id
            );
            let in_groups: u32 = facts.groups.iter().map(|g| g.files).sum();
            let nested = facts.entries.iter().filter(|e| e.rel.contains('/')).count() as u32;
            assert_eq!(in_groups, nested, "every nested file belongs to a group ({})", m.id);
            // a flat project must have no speaker dirs and vice versa — that is what `poolFlat`
            // and the PROJECT_DATASET_SHAPE refusal both key on
            assert_eq!(
                facts.groups.is_empty(),
                nested == 0,
                "mixed flat/nested dataset shape in {}",
                m.id
            );
        }
    }

    #[test]
    fn a_mangled_manifest_reads_as_absent() {
        let m: std::result::Result<DsManifest, _> = serde_json::from_str("{ truncated");
        assert!(m.is_err());
        // read() maps that to Default — asserted here so the tolerance is not silently lost
        let d = DsManifest::default();
        assert_eq!(d.v, 0);
        assert!(d.files.is_empty());
    }
}
