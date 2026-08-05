//! S101 — the dictionary DISTRIBUTION chain, driven the way the installed app drives it.
//!
//! WHY THIS EXISTS AS AN INTEGRATION TEST (§user, 2026-08-03): the first version of this coverage
//! was a unit test that called `sync_bundled_dictionaries` on two temp directories and checked the
//! bytes. That proves the FUNCTION works; it does not prove the PROGRAM goes through. And the whole
//! feature is dead code on the dev machine — a debug build resolves `app_dir` to the repo root, so
//! source and destination are the same directory and the sync returns immediately. Left like that,
//! the first real execution of this path would have been on a user's install.
//!
//! So this drives the real chain, in a real process, in the real order the app uses:
//!
//!   1. build an INSTALL tree  `<tmp>/install/data/dictionaries/*.tsv`  (= what NSIS lays down)
//!   2. build a MIGRATED data root `<tmp>/data/dictionaries/` holding a STALE fr.tsv
//!      (= the S83 fault: `migrate_data_dir` copied once, every later update refreshed only
//!       the install copy)
//!   3. `commands::settings::sync_bundled_dictionaries(app_dir, data_dir)`  — the same call
//!      `lib.rs` setup() makes, with the same two arguments
//!   4. `inference::g2p::set_dict_dir(<data root>/dictionaries)` — the same call the command
//!      layer makes (`commands/inference.rs`, `data_dir.join("dictionaries")`)
//!
//! ⚠ S110 — "the SAME call", not "the EXACT call", and the difference is the whole reason two more
//! gates had to be written. This file builds both argument values itself, as literals. So:
//!   · it passes with `lib.rs`'s call site DELETED — nothing here goes through setup(). That hole is
//!     now covered by `boot_steps_with_a_single_call_site_stay_wired_into_setup`
//!     (`commands/settings.rs`), which reads lib.rs as text.
//!   · step 4 uses a path this test computed, whereas production computes
//!     `state.models.models_dir().parent().join("dictionaries")` — a re-derivation that has to land
//!     on the same directory the sync just wrote. That link is covered by
//!     `data_root_derivations_agree` (same module).
//! Neither gap makes the chain below less real; they are the two joints this shape structurally
//! cannot reach, and they are named here so the next reader does not conclude they are covered.
//!   5. `inference::g2p::resolve_score(...)` through `GlobalDicts` — the render's own entry point,
//!      including the lazy `Box::leak` load. Assert the FRENCH note comes out with the post-D6
//!      phones.
//!
//! Step 5 is the part that matters: it is the only thing that can distinguish "the file on disk is
//! right" from "the running program sings the right thing". `set_dict_dir` is a first-call-wins
//! `OnceLock` and a loaded dictionary is leaked for the process lifetime, which is precisely why
//! this must be its own test BINARY (own process) and why the sync must be synchronous in setup().
//!
//! Skip-if-absent, same contract as `s94_en_onset_vote_gate`: `data/dictionaries` is a gitignored
//! generated asset (MBS2H `build_dictionaries.py`), so a bare checkout must not go red.
//!
//! ★ BOTH LAYERS WERE MUTATION-PROBED, separately — "green" is not evidence (S89):
//!   · sync disabled outright  ⇒ dies at the FILE layer ("fr.tsv was not refreshed…"), and the
//!     legacy-root test dies too. But that mutation never reaches step 5, so it does NOT show the
//!     render assertion works.
//!   · `UTAI_MUTANT_STALE_INSTALL=1` (below) makes the distribution SUCCEED while carrying the old
//!     content — the exact "files copied fine, program still sings the old thing" class step 5
//!     exists for ⇒ dies at "the loaded dictionary still has the pre-D6 reading". That hook is kept
//!     so the claim stays re-checkable instead of being a sentence in a commit message. It only
//!     rewrites this test's own fixture; there is deliberately NO env hook in production code.
//!
//! Run:  cargo test --test dictionary_distribution -- --nocapture

use std::path::{Path, PathBuf};

use utai_lib::commands::settings::{dictionary_fingerprint_for, sync_bundled_dictionaries};
// `DictSource` must be in scope for `GlobalDicts.words(..)` — it is the trait the render resolves
// through, so importing it here is the same shape the production call sites have.
use utai_lib::inference::g2p::{self, DictSource, GlobalDicts, Lang, ResolvedKind, ScoreEvt};
use utai_lib::inference::g2p_alias::PhonemeSet;

const SHIPPED: [&str; 8] = [
    "en.tsv", "de.tsv", "fr.tsv", "es.tsv", "it.tsv",
    "zh_syllables.tsv", "zh_chars.tsv", "zh_phrases.tsv",
];

/// The word the fr D6 mirror is defined by: upstream says `a p s t ə ɲ i ʁ`, the training labels
/// (and now the shipped dictionary) say `a p s t ə n i ʁ`.
const WORD: &str = "abstenir";

fn repo_dictionaries() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries")
}

/// ⚠ Per-test SIBLING roots, never nested (S109 review). Both tests live in one binary and libtest
/// runs them on concurrent threads; when the second test's base was a CHILD of the first's, the
/// first's two `remove_dir_all(&base)` calls could delete the second's fixture mid-run. The
/// interleaving was not observed in practice, but "not observed" is not a guard.
fn tmp_root(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("utai_dictdist_{tag}_{}", std::process::id()))
}

#[test]
fn migrated_data_root_gets_the_new_dictionary_and_the_render_sings_it() {
    let shipped = repo_dictionaries();
    if !shipped.join("fr.tsv").is_file() {
        eprintln!(
            "[dict-dist] SKIPPED — {} not present (gitignored generated asset; run MBS2H build_dictionaries.py)",
            shipped.display()
        );
        return;
    }
    let base = tmp_root("migrated");
    let _ = std::fs::remove_dir_all(&base);
    let app_dir = base.join("install");
    let data_dir = base.join("data");
    let install_dicts = app_dir.join("data").join("dictionaries");
    let root_dicts = data_dir.join("dictionaries");
    std::fs::create_dir_all(&install_dicts).unwrap();
    std::fs::create_dir_all(&root_dicts).unwrap();

    // 1. the install tree = exactly what the bundle ships.
    for name in SHIPPED {
        std::fs::copy(shipped.join(name), install_dicts.join(name)).unwrap();
    }
    let mutant_stale_install = std::env::var("UTAI_MUTANT_STALE_INSTALL").is_ok();
    // 2. the migrated data root = a copy taken BEFORE the D6 mirror. Reconstructed from the shipped
    //    file by undoing the knife, so the stale arm is the real dictionary minus this one change
    //    (not a toy file that would never exercise the loader's vote/syllabify passes).
    let fresh_fr = std::fs::read_to_string(install_dicts.join("fr.tsv")).unwrap();
    let stale_fr = fresh_fr.replace(" n i", " \u{0272} i");
    assert_ne!(stale_fr, fresh_fr, "failed to build a stale arm — the fixture would prove nothing");
    for name in SHIPPED {
        if name == "fr.tsv" {
            std::fs::write(root_dicts.join(name), &stale_fr).unwrap();
        } else {
            std::fs::copy(shipped.join(name), root_dicts.join(name)).unwrap();
        }
    }
    // A user file living beside them must survive (the sync is not a mirror-and-delete).
    std::fs::write(root_dicts.join("notes.txt"), "user file").unwrap();
    if mutant_stale_install {
        std::fs::write(install_dicts.join("fr.tsv"), &stale_fr).unwrap();
    }

    // The carrier must SEE the difference — otherwise nothing downstream can ever invalidate a bake.
    let fp_stale = dictionary_fingerprint_for(&root_dicts);
    let fp_install = dictionary_fingerprint_for(&install_dicts);
    if !mutant_stale_install {
        assert_ne!(fp_stale, fp_install, "the fingerprint cannot tell a stale dictionary from a fresh one");
    }

    // 3. THE CALL lib.rs setup() makes.
    sync_bundled_dictionaries(&app_dir, &data_dir);

    // The data root now holds the shipped bytes, and nothing else was disturbed.
    for name in SHIPPED {
        assert_eq!(
            std::fs::read(root_dicts.join(name)).unwrap(),
            std::fs::read(install_dicts.join(name)).unwrap(),
            "{name} was not refreshed into the migrated data root"
        );
    }
    assert_eq!(std::fs::read_to_string(root_dicts.join("notes.txt")).unwrap(), "user file");
    assert_eq!(
        dictionary_fingerprint_for(&root_dicts),
        fp_install,
        "after the sync the data root must carry the install's fingerprint — this is what makes an \
         existing bake dirty exactly once and then stop"
    );

    // 4. + 5. the command layer's own call, then the RENDER's own entry point.
    g2p::set_dict_dir(root_dicts.clone());
    let trad = GlobalDicts
        .words(Lang::Fr)
        .expect("fr dictionary loads from the synced data root")
        .lookup(WORD)
        .expect("abstenir is in fr.tsv");
    assert_eq!(
        trad,
        "a p s t ə n i ʁ".split(' ').map(str::to_string).collect::<Vec<_>>(),
        "the loaded dictionary still has the pre-D6 reading — the sync did not reach the loader"
    );

    let evts = [ScoreEvt {
        lyric: WORD,
        note_num: 62,
        frames: 64,
        lang: Lang::Fr,
        phoneme_input: None,
        phoneme_set: PhonemeSet::Words,
    }];
    let resolved = g2p::resolve_score(&evts, &GlobalDicts).expect("strict resolve of a French word");
    let ResolvedKind::Phones(phones) = &resolved[0].kind else {
        panic!("expected sung phones, got {:?}", resolved[0].kind)
    };
    assert!(
        phones.windows(2).any(|w| w[0] == "n" && w[1] == "i"),
        "the render wire did not produce `n i`: {phones:?}"
    );
    assert!(
        !phones.windows(2).any(|w| w[0] == "\u{0272}" && w[1] == "i"),
        "the render wire still produces the untrained bigram: {phones:?}"
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// The OTHER stuck population: a legacy AppData root has never had a `dictionaries` directory at
/// all (nothing but `migrate_data_dir` ever created one), so those installs do not sing a stale
/// French — they fail the render outright with VOCAL_DICT_MISSING. The sync has to CREATE the
/// directory, not merely refresh it. File-level only: `set_dict_dir` is first-call-wins and the
/// test above owns it in this process.
#[test]
fn absent_data_root_directory_is_created_not_skipped() {
    let shipped = repo_dictionaries();
    if !shipped.join("fr.tsv").is_file() {
        eprintln!("[dict-dist] SKIPPED — no shipped dictionaries");
        return;
    }
    let base = tmp_root("legacy");
    let _ = std::fs::remove_dir_all(&base);
    let app_dir = base.join("install");
    let data_dir = base.join("appdata");
    let install_dicts = app_dir.join("data").join("dictionaries");
    std::fs::create_dir_all(&install_dicts).unwrap();
    std::fs::create_dir_all(&data_dir).unwrap();
    for name in SHIPPED {
        std::fs::copy(shipped.join(name), install_dicts.join(name)).unwrap();
    }
    assert!(!data_dir.join("dictionaries").exists(), "precondition: no dictionaries dir yet");

    sync_bundled_dictionaries(&app_dir, &data_dir);

    for name in SHIPPED {
        assert!(
            data_dir.join("dictionaries").join(name).is_file(),
            "{name} was not created in a legacy AppData root — that install still cannot render"
        );
    }
    let _ = std::fs::remove_dir_all(&base);
}
