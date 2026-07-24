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

/// What was actually on disk — the difference matters to WRITERS, never to readers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestState {
    Missing,
    /// Present but unparseable. It has been renamed to `dataset.json.bad` so a human can still
    /// salvage the names from it; the caller proceeds with a fresh manifest.
    Salvaged,
    Ok,
}

/// Never fails: an unreadable manifest is the same as no manifest (see the module doc).
pub fn read(data_dir: &Path, id: &str) -> DsManifest {
    read_state(data_dir, id).0
}

/// Read, and say what was there.
///
/// ★ A CORRUPT manifest is moved aside rather than silently overwritten. Both writers do
/// read-modify-write, so without this a single unparseable byte would make the next add-or-delete
/// destroy every original file name the project had recorded — the exact silent-loss shape this
/// module's own contract forbids. Readers keep degrading quietly (they call `read`); only the
/// write path pays the cost of being careful.
pub fn read_state(data_dir: &Path, id: &str) -> (DsManifest, ManifestState) {
    let p = manifest_path(data_dir, id);
    let Ok(raw) = std::fs::read_to_string(&p) else {
        return (DsManifest::default(), ManifestState::Missing);
    };
    match serde_json::from_str::<DsManifest>(&raw) {
        Ok(m) => (m, ManifestState::Ok),
        Err(e) => {
            let bad = p.with_extension("json.bad");
            let _ = std::fs::rename(&p, &bad);
            tracing::warn!(
                "unreadable {} for project {id} ({e}) — kept as {} and starting a fresh one",
                DATASET_MANIFEST,
                bad.display()
            );
            (DsManifest::default(), ManifestState::Salvaged)
        }
    }
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

/// The group a UI label refers to — matched by DISPLAY NAME first, then by slug.
///
/// ★ The slug fallback is not defensive padding: when no carrier records the names, the view
/// shows the slug and the UI therefore sends back the slug. Matching on the name alone made
/// adding files to an EXISTING singer look like creating a NEW one, which the frozen-structure
/// guard then refused ("我并没有更改歌手结构,只是改了数据量" — user, S78).
pub fn find_group<'a>(facts: &'a DatasetFacts, label: &str) -> Option<&'a GroupFacts> {
    let l = label.trim();
    facts
        .groups
        .iter()
        .find(|g| !g.speaker.name.is_empty() && g.speaker.name == l)
        .or_else(|| facts.groups.iter().find(|g| g.speaker.slug == l))
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

// ───────────────────────── mutation (S76 批 5b) ─────────────────────────
//
// The dataset used to be written by exactly one thing (`run_worker`'s import stage), which is why
// `has_dataset` can assume a non-empty directory is a complete one. Managing data outside a run
// needs two more writers, and both obey the same three rules:
//
//  1. **Never renumber.** Deleting `001.wav` leaves a hole. Renaming the survivors would rewrite
//     files the user did not touch, and nothing downstream needs dense numbering — both python
//     preprocessors enumerate `sorted(os.listdir(...))` and derive their own indices. Holes also
//     make deletion crash-safe for free: there is no multi-file rename to interrupt.
//  2. **Nothing partial is ever visible inside `dataset/`.** Copies land on a `.part` name in the
//     same directory and are renamed into place, so a crash mid-copy cannot leave a truncated wav
//     that `has_dataset` would accept and a run would then slice.
//  3. **The annotation follows the disk, never leads it.** It is rewritten AFTER the files move;
//     a crash in between leaves a stale entry, which `read_facts` drops on the next read.

/// Highest positional index already used in a directory, +1. Names that are not `<digits>.<ext>`
/// are ignored (they cannot collide with what we mint).
fn next_index(existing_names: &[String]) -> usize {
    existing_names
        .iter()
        .filter_map(|n| {
            let stem = n.split('.').next()?;
            (!stem.is_empty() && stem.chars().all(|c| c.is_ascii_digit()))
                .then(|| stem.parse::<usize>().ok())
                .flatten()
        })
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// Where each source will land, in the order given. Pure so the naming can be tested without
/// touching a disk.
pub fn plan_append(
    existing_names: &[String],
    slug: Option<&str>,
    srcs: &[String],
) -> Vec<(String, String)> {
    let start = next_index(existing_names);
    srcs.iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), super::dataset_rel(slug, start + i, s)))
        .collect()
}

/// Copy files into the project's dataset, appending. `slug` selects a speaker subdirectory
/// (created when new); `None` is the flat dataset.
///
/// `probe_ms` is injected so the command can pay for duration probing (an ffmpeg spawn per
/// non-wav file) while tests stay instant.
pub fn append_files(
    data_dir: &Path,
    id: &str,
    slug: Option<&str>,
    speaker_name: Option<&str>,
    srcs: &[String],
    probe_ms: &dyn Fn(&Path) -> Option<f64>,
) -> Result<()> {
    let root = super::tproject::dataset_dir(data_dir, id);
    let dir = match slug {
        Some(s) => root.join(s),
        None => root.clone(),
    };
    std::fs::create_dir_all(&dir)
        .map_err(|e| UtaiError::Training(format!("DATASET_WRITE_FAILED: {e}")))?;
    let existing: Vec<String> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_file())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    let plan = plan_append(&existing, slug, srcs);

    let mut added: Vec<DsFile> = Vec::new();
    for (src, rel) in &plan {
        let src_path = Path::new(src);
        let dst = root.join(rel);
        // rule 2: land on a sibling `.part`, then rename into place (same directory ⇒ atomic)
        let part = dst.with_extension(format!(
            "{}.part",
            dst.extension().and_then(|e| e.to_str()).unwrap_or("bin")
        ));
        std::fs::copy(src_path, &part).map_err(|e| {
            let _ = std::fs::remove_file(&part);
            UtaiError::Training(format!("DATASET_COPY_FAILED: {}: {e}", src_path.display()))
        })?;
        std::fs::rename(&part, &dst).map_err(|e| {
            let _ = std::fs::remove_file(&part);
            UtaiError::Training(format!("DATASET_COPY_FAILED: {}: {e}", dst.display()))
        })?;
        added.push(DsFile {
            rel: rel.clone(),
            name: src_path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            bytes: std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0),
            duration_ms: probe_ms(&dst),
        });
    }

    // rule 3: annotation last. `read_state` moves a corrupt manifest aside first, so the
    // read-modify-write below can never be the thing that destroys recorded names.
    let mut m = read_state(data_dir, id).0;
    m.v = MANIFEST_VERSION;
    m.files.retain(|f| !added.iter().any(|a| a.rel == f.rel));
    m.files.extend(added);
    if let (Some(s), Some(name)) = (slug, speaker_name) {
        if !m.speakers.iter().any(|sp| sp.slug == s) {
            m.speakers.push(DsSpeaker {
                slug: s.to_string(),
                name: name.to_string(),
            });
        }
    }
    write(data_dir, id, &m)
}

/// What a delete would do — computed BEFORE anything is removed so a refusal costs nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct DeletePlan {
    /// Speaker directories that would be left with no files at all.
    pub emptied_speakers: Vec<String>,
    pub files: usize,
    pub bytes: u64,
}

pub fn plan_delete(facts: &DatasetFacts, rels: &[String]) -> DeletePlan {
    let going: std::collections::HashSet<&str> = rels.iter().map(|s| s.as_str()).collect();
    let mut emptied = Vec::new();
    for g in &facts.groups {
        let prefix = format!("{}/", g.speaker.slug);
        let survives = facts
            .entries
            .iter()
            .any(|e| e.rel.starts_with(&prefix) && !going.contains(e.rel.as_str()));
        if !survives && g.files > 0 {
            emptied.push(g.speaker.slug.clone());
        }
    }
    let hit: Vec<&DsFile> = facts
        .entries
        .iter()
        .filter(|e| going.contains(e.rel.as_str()))
        .collect();
    DeletePlan {
        emptied_speakers: emptied,
        files: hit.len(),
        bytes: hit.iter().map(|e| e.bytes).sum(),
    }
}

/// Remove files from the dataset. `rels` must be entries the caller just listed.
///
/// `drop_speaker_dirs` empties are removed only when the caller has established that no slot has
/// frozen this speaker set — an emptied-but-present directory would still read as a speaker, and
/// a removed one changes the speaker SET, which is resume-locked.
pub fn delete_files(
    data_dir: &Path,
    id: &str,
    rels: &[String],
    drop_empty_speaker_dirs: bool,
) -> Result<()> {
    let root = super::tproject::dataset_dir(data_dir, id);
    for rel in rels {
        // Path-escape guard: these strings come from the frontend. Only the two shapes the import
        // writes are acceptable — one segment, or `<slug>/<name>`.
        let parts: Vec<&str> = rel.split('/').collect();
        if parts.len() > 2
            || parts.iter().any(|p| {
                p.is_empty() || *p == "." || *p == ".." || p.contains('\\') || p.contains(':')
            })
        {
            return Err(UtaiError::Training(format!("DATASET_REL_INVALID: {rel}")));
        }
        let p = root.join(rel);
        if !p.is_file() {
            // already gone (a stale list, a second click) — not an error, the goal is reached
            continue;
        }
        std::fs::remove_file(&p)
            .map_err(|e| UtaiError::Training(format!("DATASET_DELETE_FAILED: {rel}: {e}")))?;
    }
    if drop_empty_speaker_dirs {
        if let Ok(rd) = std::fs::read_dir(&root) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir()
                    && std::fs::read_dir(&p)
                        .map(|mut d| d.next().is_none())
                        .unwrap_or(false)
                {
                    let _ = std::fs::remove_dir(&p);
                }
            }
        }
    }
    let (mut m, state) = read_state(data_dir, id);
    let gone: std::collections::HashSet<&str> = rels.iter().map(|s| s.as_str()).collect();
    m.files.retain(|f| !gone.contains(f.rel.as_str()));
    // a speaker whose directory is gone is no longer part of this dataset
    m.speakers
        .retain(|sp| root.join(&sp.slug).is_dir() || m.files.iter().any(|f| f.rel.starts_with(&format!("{}/", sp.slug))));
    // Nothing was ever recorded and nothing survives to record ⇒ do not mint an empty file. A
    // project whose data predates the annotation would otherwise grow a `{"v":0,...}` shell on
    // its first delete, which claims「有清单」while knowing nothing.
    if m.files.is_empty() && m.speakers.is_empty() && state != ManifestState::Ok {
        return Ok(());
    }
    m.v = MANIFEST_VERSION;
    write(data_dir, id, &m)
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

    // ★ A slot's FROZEN record wins over the annotation. The frozen order is not an opinion: it
    // is the emb_g row order the model was actually trained with, and a resume refuses anything
    // else. The annotation's order is merely the order files happened to be imported in — which
    // for a project that already trained can differ (append to teto first and its list reads
    // [teto, sayo]), and publishing THAT as the emb_g order is precisely the mistake that makes
    // a manual rebuild swap every singer's voice.
    //
    // Staleness is handled by `same_set`, not by precedence: a frozen record from before a data
    // replacement describes speakers that are no longer on disk, so it is not usable at all.
    let usable: Vec<&Vec<DsSpeaker>> = frozen.iter().filter(|c| same_set(c)).collect();
    let Some(first) = usable.first() else {
        // Nothing has trained on this shape yet ⇒ the annotation IS the order: it is what the
        // data page shows, what a first run will declare, and therefore what gets frozen.
        if same_set(annotated) {
            return DatasetView {
                groups: annotated.to_vec(),
                order_known: true,
            };
        }
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

    /// ★ The frozen order OUTRANKS the annotation, and the reason is not a preference.
    ///
    /// The annotation's order is「文件是按什么顺序导进来的」— append to B first and its list
    /// reads [B, A]. The frozen order is the emb_g row order the model was TRAINED with, and a
    /// resume refuses anything else. Publishing the import order as the emb_g order is exactly
    /// how a manual rebuild ends up swapping every singer's voice.
    ///
    /// (Caught on real data: a project trained as [sayo, teto] whose annotation only had `sayo`
    /// in it — one more append to `teto` and the annotation would have become "complete" and
    /// started overriding the truth.)
    #[test]
    fn a_trained_order_outranks_the_import_order() {
        let disk = vec!["b_2".to_string(), "a_1".to_string()];
        let ann = vec![sp("a_1", "亚里沙"), sp("b_2", "Bella")];
        let frozen = vec![vec![sp("b_2", "Bella"), sp("a_1", "亚里沙")]];
        let v = resolve_speakers(&disk, &ann, &frozen);
        assert!(v.order_known);
        assert_eq!(
            v.groups,
            vec![sp("b_2", "Bella"), sp("a_1", "亚里沙")],
            "the trained slot decides; the annotation only records how files arrived"
        );
    }

    /// …but with nothing trained on this shape, the annotation IS the order: it is what the data
    /// page shows, what the first run declares, and therefore what gets frozen.
    #[test]
    fn the_annotation_is_the_order_until_something_trains_on_it() {
        let disk = vec!["b_2".to_string(), "a_1".to_string()];
        let ann = vec![sp("a_1", "亚里沙"), sp("b_2", "Bella")];
        let v = resolve_speakers(&disk, &ann, &[]);
        assert!(v.order_known);
        assert_eq!(v.groups, ann);
        // a frozen record that describes OTHER speakers is stale, not authoritative
        let stale = vec![vec![sp("x_9", "X"), sp("y_8", "Y")]];
        assert_eq!(resolve_speakers(&disk, &ann, &stale).groups, ann);
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

    /// ★ The rule the whole delete story rests on: numbering is APPEND-ONLY and holes are
    /// permanent. Compacting would rename files the user never touched (and both python
    /// preprocessors index by `sorted(listdir)` position anyway, so nothing gains from density).
    #[test]
    fn numbering_is_append_only_and_holes_are_permanent() {
        assert_eq!(next_index(&[]), 0);
        assert_eq!(next_index(&["000.wav".into(), "001.flac".into()]), 2);
        // the hole left by deleting 001 does NOT get refilled
        assert_eq!(next_index(&["000.wav".into(), "002.flac".into()]), 3);
        // junk names cannot collide with what we mint, so they are ignored
        assert_eq!(next_index(&["notes.txt".into(), "007.wav".into(), ".part".into()]), 8);
        // 4 digits is fine — `{:03}` is a minimum width, not a cap
        assert_eq!(next_index(&["1000.wav".into()]), 1001);

        let plan = plan_append(
            &["000.wav".into(), "002.wav".into()],
            Some("spk_1"),
            &["D:/a/one.FLAC".into(), "D:/a/two.wav".into()],
        );
        assert_eq!(
            plan.iter().map(|(_, r)| r.as_str()).collect::<Vec<_>>(),
            vec!["spk_1/003.flac", "spk_1/004.wav"]
        );
    }

    #[test]
    fn append_copies_lands_atomically_and_records_the_original_names() {
        let tmp = std::env::temp_dir().join(format!("utai_dsappend_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        std::fs::create_dir_all(&src).unwrap();
        let a = src.join("原始 A.wav");
        let b = src.join("B.flac");
        std::fs::write(&a, b"aaaa").unwrap();
        std::fs::write(&b, b"bb").unwrap();
        let id = "p_app";
        let never = |_: &Path| None;

        append_files(
            &tmp,
            id,
            None,
            None,
            &[a.to_string_lossy().into_owned(), b.to_string_lossy().into_owned()],
            &never,
        )
        .unwrap();
        let facts = read_facts(&tmp, id, &[]);
        assert_eq!(
            facts.entries.iter().map(|e| e.rel.as_str()).collect::<Vec<_>>(),
            vec!["000.wav", "001.flac"]
        );
        assert_eq!(facts.entries[0].name, "原始 A.wav");
        assert_eq!(facts.entries[1].bytes, 2);
        // rule 2: no `.part` may survive a successful copy
        let ds = super::super::tproject::dataset_dir(&tmp, id);
        assert!(
            std::fs::read_dir(&ds)
                .unwrap()
                .flatten()
                .all(|e| !e.file_name().to_string_lossy().contains(".part")),
            "a partial copy must never remain inside dataset/"
        );

        // appending again continues the numbering instead of clobbering
        append_files(&tmp, id, None, None, &[a.to_string_lossy().into_owned()], &never).unwrap();
        let facts = read_facts(&tmp, id, &[]);
        assert_eq!(facts.files, 3);
        assert_eq!(facts.entries[2].rel, "002.wav");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn delete_leaves_holes_refuses_path_escapes_and_prunes_emptied_speakers() {
        let tmp = std::env::temp_dir().join(format!("utai_dsdel_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let id = "p_del";
        let ds = super::super::tproject::dataset_dir(&tmp, id);
        std::fs::create_dir_all(ds.join("a_1")).unwrap();
        std::fs::create_dir_all(ds.join("b_2")).unwrap();
        for (p, n) in [("a_1/000.wav", 4), ("a_1/001.wav", 4), ("b_2/000.wav", 3)] {
            std::fs::write(ds.join(p), vec![b'x'; n]).unwrap();
        }
        record_import(
            &tmp,
            id,
            vec![sp("a_1", "A"), sp("b_2", "B")],
            vec![
                DsFile { rel: "a_1/000.wav".into(), name: "one.wav".into(), bytes: 4, duration_ms: None },
                DsFile { rel: "a_1/001.wav".into(), name: "two.wav".into(), bytes: 4, duration_ms: None },
                DsFile { rel: "b_2/000.wav".into(), name: "three.wav".into(), bytes: 3, duration_ms: None },
            ],
        );

        // the plan must see that deleting b_2's only file empties that singer
        let facts = read_facts(&tmp, id, &[]);
        let plan = plan_delete(&facts, &["b_2/000.wav".into()]);
        assert_eq!(plan.emptied_speakers, vec!["b_2"]);
        assert_eq!((plan.files, plan.bytes), (1, 3));
        // and that deleting ONE of a_1's two does not
        let plan = plan_delete(&facts, &["a_1/000.wav".into()]);
        assert!(plan.emptied_speakers.is_empty());

        // path escapes are refused before anything is touched
        for bad in ["../project.json", "a_1/../../x", "a_1/sub/deep.wav", "C:/etc/passwd"] {
            assert!(
                delete_files(&tmp, id, &[bad.into()], false).is_err(),
                "must refuse {bad}"
            );
        }
        assert!(ds.join("a_1/000.wav").is_file(), "a refused delete touches nothing");

        // delete the FIRST of a_1 — the survivor keeps its name (hole at 000)
        delete_files(&tmp, id, &["a_1/000.wav".into()], true).unwrap();
        let facts = read_facts(&tmp, id, &[]);
        assert_eq!(
            facts.entries.iter().map(|e| e.rel.as_str()).collect::<Vec<_>>(),
            vec!["a_1/001.wav", "b_2/000.wav"],
            "no renumbering — 001 stays 001"
        );
        assert_eq!(facts.entries[0].name, "two.wav", "the annotation follows the survivor");

        // emptying b_2 with pruning allowed removes the directory, so it stops being a speaker
        delete_files(&tmp, id, &["b_2/000.wav".into()], true).unwrap();
        let facts = read_facts(&tmp, id, &[]);
        assert!(!ds.join("b_2").exists(), "an emptied singer directory is removed");
        assert_eq!(facts.speaker_slugs, vec!["a_1"]);
        assert_eq!(facts.groups.len(), 1);
        // deleting something already gone is a no-op, not an error
        delete_files(&tmp, id, &["b_2/000.wav".into()], true).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The frozen-structure case: an emptied singer must NOT have its directory removed, because
    /// the caller refuses that delete outright — but if it ever got here, leaving the directory
    /// keeps the speaker SET intact, which is the resume-locked thing.
    #[test]
    fn pruning_can_be_withheld_so_a_frozen_speaker_set_survives() {
        let tmp = std::env::temp_dir().join(format!("utai_dsfrozen_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let id = "p_frozen";
        let ds = super::super::tproject::dataset_dir(&tmp, id);
        std::fs::create_dir_all(ds.join("a_1")).unwrap();
        std::fs::write(ds.join("a_1/000.wav"), b"x").unwrap();
        delete_files(&tmp, id, &["a_1/000.wav".into()], false).unwrap();
        assert!(ds.join("a_1").is_dir(), "directory kept ⇒ the speaker set is unchanged");
        let facts = read_facts(&tmp, id, &[]);
        assert_eq!(facts.speaker_slugs, vec!["a_1"]);
        assert_eq!(facts.files, 0);
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

    /// ★ Regression (user, S78): on a LEGACY multi-speaker project — names recoverable only from
    /// a slot's frozen record, never from an annotation —「给已有歌手加文件」was refused with
    /// `DATASET_SPEAKERS_FROZEN`, i.e. treated as CREATING a singer.
    ///
    /// Two causes, both fixed: the writers read the dataset with an EMPTY frozen list (so their
    /// groups had blank names while the UI's had real ones), and the lookup matched on the name
    /// only (so a view that shows the slug — because nothing records the name — could never be
    /// matched by what it sent back).
    #[test]
    fn an_existing_singer_is_found_by_its_name_or_by_its_slug() {
        let tmp = std::env::temp_dir().join(format!("utai_dsfind_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let id = "p_find";
        let ds = super::super::tproject::dataset_dir(&tmp, id);
        for s in ["sayo_d769c729", "teto_41198cd4"] {
            std::fs::create_dir_all(ds.join(s)).unwrap();
            std::fs::write(ds.join(s).join("000.wav"), b"x").unwrap();
        }

        // (a) no annotation, no frozen record: the view shows SLUGS, so the UI sends a slug back
        let bare = read_facts(&tmp, id, &[]);
        assert!(bare.groups.iter().all(|g| g.speaker.name.is_empty()));
        assert!(find_group(&bare, "sayo").is_none(), "there is no such name to find");
        assert_eq!(
            find_group(&bare, "sayo_d769c729").map(|g| g.speaker.slug.as_str()),
            Some("sayo_d769c729"),
            "…but the slug the view showed must resolve"
        );

        // (b) the real case: a slot froze the names — the WRITERS must read it the same way the
        // detail page does, or the name it displayed resolves to nothing
        let frozen = vec![vec![sp("sayo_d769c729", "sayo"), sp("teto_41198cd4", "teto")]];
        let seen = read_facts(&tmp, id, &frozen);
        assert_eq!(seen.groups[0].speaker.name, "sayo");
        assert_eq!(
            find_group(&seen, "sayo").map(|g| g.speaker.slug.as_str()),
            Some("sayo_d769c729"),
            "adding files to an existing singer must not read as creating one"
        );
        assert_eq!(find_group(&seen, "  teto  ").map(|g| g.speaker.slug.as_str()), Some("teto_41198cd4"));
        assert!(find_group(&seen, "somebody else").is_none());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// ★ Regression: the first delete on a LEGACY project (no annotation) wrote
    /// `{"v":0,"speakers":[],"files":[]}` — a shell that claims「有清单」while knowing nothing.
    /// Caught on the real machine after the user deleted one file from `sayo-RVC_3646751c`.
    #[test]
    fn a_delete_on_an_unannotated_project_mints_no_empty_shell() {
        let tmp = std::env::temp_dir().join(format!("utai_dsshell_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let id = "p_shell";
        let ds = super::super::tproject::dataset_dir(&tmp, id);
        std::fs::create_dir_all(&ds).unwrap();
        std::fs::write(ds.join("000.wav"), b"x").unwrap();
        std::fs::write(ds.join("001.wav"), b"y").unwrap();
        delete_files(&tmp, id, &["000.wav".into()], true).unwrap();
        assert!(
            !super::super::tproject::project_dir(&tmp, id).join(DATASET_MANIFEST).exists(),
            "no annotation existed and none could be written — do not leave a shell"
        );
        // the delete itself still happened, and the survivor keeps its number
        let facts = read_facts(&tmp, id, &[]);
        assert_eq!(facts.entries.iter().map(|e| e.rel.as_str()).collect::<Vec<_>>(), vec!["001.wav"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// ★ A corrupt annotation must never be destroyed by the next write — both writers
    /// read-modify-write, so a single bad byte would otherwise erase every recorded name.
    #[test]
    fn a_corrupt_manifest_is_moved_aside_not_overwritten() {
        let tmp = std::env::temp_dir().join(format!("utai_dsbad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let id = "p_bad";
        let ds = super::super::tproject::dataset_dir(&tmp, id);
        std::fs::create_dir_all(&ds).unwrap();
        std::fs::write(ds.join("000.wav"), b"x").unwrap();
        let mpath = super::super::tproject::project_dir(&tmp, id).join(DATASET_MANIFEST);
        std::fs::write(&mpath, "{ \"files\": [ truncated").unwrap();

        let (m, state) = read_state(&tmp, id);
        assert_eq!(state, ManifestState::Salvaged);
        assert!(m.files.is_empty(), "the caller starts fresh");
        let bad = mpath.with_extension("json.bad");
        assert!(bad.is_file(), "the unreadable content is kept for salvage");
        assert!(
            std::fs::read_to_string(&bad).unwrap().contains("truncated"),
            "kept VERBATIM — the point is that a human can still read the names out of it"
        );
        assert!(!mpath.exists(), "and it is out of the way");
        // readers still degrade quietly
        assert!(read(&tmp, id).files.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
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
