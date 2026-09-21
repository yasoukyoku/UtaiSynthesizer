use std::path::{Path, PathBuf};

/// The manual portable-python slot `<app_dir>/python/python.exe` — ONE definition
/// shared by `pyenv::training_interpreter` (training) and `pyenv::converter_python`
/// (converter), so the two roles can never drift onto different "manual slot"
/// locations. (Phase B retired the role-agnostic `find_python` once training moved to
/// pyenv's variant-aware resolver — the same move the converter made in S42.)
pub fn manual_python_slot(app_dir: &Path) -> PathBuf {
    app_dir.join("python").join("python.exe")
}

/// Windows `CREATE_NO_WINDOW` process-creation flag — pass to `Command::creation_flags(...)` so spawned
/// console tools (ffmpeg, powershell) don't flash a black console window. Was the bare magic
/// `0x08000000` repeated at every spawn site.
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Build a `std::process::Command` for a bundled Python tool with the shared spawn hygiene applied.
/// Single source of truth for EVERY python spawn (converter, index extractor, training):
///   - `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`: a PIPED stdout/stderr on Windows defaults to the
///     ANSI codepage, so one CJK character in a `print()` raises UnicodeEncodeError AFTER the tool
///     already wrote its artifacts — the spawn "fails" with the files on disk (phantom import).
///   - `CREATE_NO_WINDOW`: no console flash.
/// Async call sites convert with `tokio::process::Command::from(python_command(...))` — the flags
/// and envs carry over.
pub fn python_command(python: &Path) -> std::process::Command {
    let mut cmd = std::process::Command::new(python);
    cmd.env("PYTHONIOENCODING", "utf-8");
    cmd.env("PYTHONUTF8", "1");
    // Isolate from the HOST machine's Python environment (S42): a user-set
    // PYTHONHOME makes the embedded runtime-pack interpreter fail at startup
    // ("init_fs_encoding"), and an inherited PYTHONPATH / user-site can shadow the
    // pack's site-packages with foreign versions (e.g. a numpy 2.x that breaks the
    // pack's numpy-1.26 C-API wheels) — silently, which is worse. `-E` would also
    // kill our OWN env vars above, so strip the two inherited ones explicitly and
    // disable user-site. Dev venvs are unaffected (venvs need neither variable).
    cmd.env_remove("PYTHONHOME");
    cmd.env_remove("PYTHONPATH");
    cmd.env("PYTHONNOUSERSITE", "1");
    // AMD/ROCm (MIOpen) tuning — read ONLY by the rocm torch build; NVIDIA (cuDNN),
    // CPU and Intel/XPU ignore these, so setting them once in the single spawn helper
    // is harmless everywhere else and keeps them out of every call site (S44).
    //   FIND_MODE=5: ⚠ S172 correction — this line is a NO-OP and its old comment was wrong
    //     twice. 5 is DYNAMIC_HYBRID (FastHybrid was the deprecated 4) and it is already
    //     MIOpen's own default (measured on runtime-amd-v2: with the var unset the pack
    //     logs `MIOPEN_FIND_MODE = DYNAMIC_HYBRID(5)`), and the exhaustive search the old
    //     comment meant to avoid is NORMAL=1, not the default. Kept as an explicit pin so a
    //     future upstream default change cannot silently move us off it.
    //   LOG_LEVEL=4 (Warning): ⛔ S172 — this was pinned to 3 (Error), and that deleted the
    //     ONLY line that ever says WHY a kernel build failed. MIOpen prints the hipRTC
    //     compiler text at Warning (comgr.cpp `MIOPEN_LOG_W(ex.text)`), so a community
    //     report arrived with 740 `HIPRTC_ERROR_COMPILATION` lines and no reason at all —
    //     the root cause (`miopen_type_traits.hpp:151: fatal error: 'type_traits' file not
    //     found`) was invisible, and attribution nearly broke on it.
    //     The old rationale — muting a "~70 line" first-step Warning burst — does not
    //     reproduce on runtime-amd-v2: measured, level 4 costs exactly +1 line on a healthy
    //     training-shaped run, and +7 lines per FAILED kernel build, which are precisely the
    //     lines worth having. ⚠ Do not raise it further: level 5 (Info) measured 1,888
    //     MIOpen lines for a single GAN-shaped forward/backward and would flood the
    //     200-line stderr ring this pin exists to protect.
    cmd.env("MIOPEN_FIND_MODE", "5");
    cmd.env("MIOPEN_LOG_LEVEL", "4");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Directory holding the C++ standard headers the AMD runtime's kernel compiler needs.
/// Shipped as a Tauri resource next to `utai_train` (see tauri.conf.json), and therefore also
/// listed in `training::tproject::RESERVED_TRAINING_DIRS`.
pub const ROCM_CXX_HEADER_DIR: &str = "rocm_cxx_headers";

/// S172 — the value for `HIPRTC_COMPILE_OPTIONS_APPEND` on the ROCm lane.
///
/// WHY THIS EXISTS. MIOpen compiles its kernels at run time through hipRTC, and those sources
/// reach `#include <type_traits>`. comgr is supposed to carry the libc++ headers for exactly
/// that — every compile line it emits already contains `-idirafter "include\c++\v1"` — but our
/// pinned nightly ships with the payload EMPTY (measured with AMD_COMGR_EMIT_VERBOSE_LOGS=1:
/// `Embedded 2 libc++ headers`, `Embedded 0 clang builtins`; fixed upstream by
/// ROCm/llvm-project#4241 on 2026-09-08, after our pin). So the compile only succeeds on a
/// machine that happens to have Microsoft's MSVC headers installed — two community machines
/// had none, and one of them had them and UNINSTALLED Visual Studio. We supply the headers the
/// toolchain was already looking for.
///
/// ⚠ `-idirafter`, NEVER `-I`. `-I` puts our headers AHEAD of the host's, so a machine that
/// HAS a C++ toolchain gets a MIXED standard library — half ours, half the host's — whenever
/// our payload is missing anything the host would have supplied. That is not hypothetical: it
/// was measured on this box while the payload still lacked `<utility>`
/// (`<type_traits>` from ours + `<utility>` from MSVC ⇒
/// `MSVC\...\utility:111:60: error: no template named '_Is_swappable'`), i.e. the fix BROKE a
/// machine that worked before it. Completing the payload made that particular probe pass
/// again — which is exactly why `-I` is still the wrong instrument: it makes correctness on
/// every already-working machine depend on our payload being complete forever, and the next
/// MIOpen kernel that reaches a header we did not ship brings the failure straight back.
/// `-idirafter` appends instead, so where a host C++ library exists ours is never consulted
/// at all (measured inert: same 27 headers opened, identical digest, with and without it) and
/// the mixing cannot happen by construction. ⚠ Also NEVER `-idirafter=<dir>`: that form is
/// silently ignored, so the payload would never be consulted and a green would be a lie.
///
/// ⚠ ORDER IS LOAD-BEARING: `v1` must precede `clangres`. libc++ reaches the C headers with
/// `#include_next`, so reversing them breaks `<cstddef>`
/// ("tried including <stddef.h> but didn't find libc's <stddef.h>").
///
/// ⚠ RELATIVE ON PURPOSE, and also load-bearing: `HIPRTC_COMPILE_OPTIONS_APPEND` is split on
/// WHITESPACE and does NOT honour quotes (measured: a quoted `"-IC:\Program Files\..."` arrives
/// as the fragment `"-IC:\Program` and clang rejects it). Both spawn sites that can reach
/// MIOpen set cwd to `<app_dir>/training`, so a path relative to that is space-free however the
/// user named their install directory — which is what dissolves the whole 8.3-short-name
/// problem instead of papering over it. The flag is written CONCATENATED with its directory
/// for the same reason: one token, no space to be split on.
///
/// Composed with, never clobbering, whatever the user already had in the variable.
pub fn hiprtc_cxx_include_options() -> String {
    let ours = format!(
        "-idirafter{d}/v1 -idirafter{d}/clangres",
        d = ROCM_CXX_HEADER_DIR
    );
    match std::env::var("HIPRTC_COMPILE_OPTIONS_APPEND") {
        Ok(prev) if !prev.trim().is_empty() => format!("{prev} {ours}"),
        _ => ours,
    }
}

/// How many headers each shipped directory must have for the payload to be usable.
/// Not a guess: `v1` and `clangres` were grown empirically until a `-nostdinc` compile of a
/// real MIOpen kernel succeeded, and they landed at 189 and 11. Half a C++ standard library
/// is as dead as none of it, so a SHORT payload has to read as broken, not as present.
/// `scripts/verify-install.ps1` asserts the same two floors against an installed tree.
const ROCM_CXX_MIN_V1: usize = 189;
const ROCM_CXX_MIN_CLANGRES: usize = 11;

/// Write one line saying whether the hipRTC C++ header payload is actually THERE, and what
/// we told the compiler about it. Called once per sidecar spawn, from both sites.
///
/// WHY. `-idirafter <dir>` on a directory that does not exist is **silently ignored** — no
/// warning, no error, nothing on any stream. So an install where the resource never unpacked,
/// or was quarantined by antivirus, or where a refactor moved the spawn cwd out from under the
/// relative path, produces a log that is indistinguishable from the pre-fix bug: a wall of
/// `HIPRTC_ERROR_COMPILATION` and no way to tell "this person never got the fix" apart from
/// "this person got the fix and something else is wrong". That ambiguity is what turns one
/// bug report into three rounds of mailing the reporter diagnostic scripts, so the shipped
/// app answers it itself, in advance, in the file every reporter already sends us.
///
/// Deliberately unconditional rather than AMD-only: the spawn sites cannot cheaply know the
/// lane, the line is one per run, and a uniform line means a log can be read the same way
/// whoever sent it. On a non-AMD machine the payload is simply never consulted.
/// Files under `dir`, at any depth. `None` means the directory could not be read at all —
/// distinct from `Some(0)`, which means it was read and was empty.
///
/// ⚠ RECURSIVE ON PURPOSE. libc++ is not a flat directory: `v1` holds 15 headers at the top
/// level and 189 across 11 subdirectories (`__algorithm/`, `__type_traits/`, …). The first
/// version of this counted only the top level, which put a healthy install at 15 against a
/// floor of 189 and would have logged "headers MISSING OR INCOMPLETE" on EVERY run. A
/// diagnostic that cries wolf every time teaches its reader to skip it, and then it is also
/// skipped the one time it is true — so this is worse than not logging at all.
/// `verify-install.ps1` counts the same tree with `-Recurse`; the two must agree.
fn count_files_recursive(dir: &std::path::Path) -> Option<usize> {
    count_files_to_depth(dir, 8)
}

/// ⚠ `DirEntry::file_type()`, never `Path::is_file()`: the latter FOLLOWS reparse points, and a
/// junction placed inside the install directory would make this walk leave the tree or loop
/// forever. That is not a hypothetical on this project — S96 destroyed the repository through a
/// self-made junction. Symlinked directories are counted as entries and not descended into, and
/// the depth cap is a second belt: the real payload is 2 levels deep, so 8 can only ever be hit
/// by something pathological.
fn count_files_to_depth(dir: &std::path::Path, depth: u32) -> Option<usize> {
    let rd = std::fs::read_dir(dir).ok()?;
    let mut n = 0usize;
    for e in rd.flatten() {
        match e.file_type() {
            Ok(t) if t.is_symlink() => n += 1, // counted, never followed
            Ok(t) if t.is_dir() => {
                if depth > 0 {
                    n += count_files_to_depth(&e.path(), depth - 1).unwrap_or(0);
                }
            }
            Ok(_) => n += 1,
            Err(_) => {}
        }
    }
    Some(n)
}

pub fn log_hiprtc_cxx_provenance(training_dir: &std::path::Path, options: &str) {
    let root = training_dir.join(ROCM_CXX_HEADER_DIR);
    let count = |sub: &str| -> Option<usize> { count_files_recursive(&root.join(sub)) };
    // Spot-check the same four paths `verify-install.ps1` names. A short count is not
    // actionable on its own; naming the absent file is, and the reporter can confirm it by hand.
    let named: Vec<&str> = ["v1/type_traits", "v1/utility", "clangres/stddef.h", "LICENSE.TXT"]
        .into_iter()
        .filter(|rel| !root.join(rel).exists())
        .collect();
    match (count("v1"), count("clangres")) {
        (Some(v1), Some(cr))
            if v1 >= ROCM_CXX_MIN_V1 && cr >= ROCM_CXX_MIN_CLANGRES && named.is_empty() =>
        {
            tracing::info!(
                "hipRTC C++ headers OK: {} (v1={}, clangres={}, counted recursively) cwd={} -> HIPRTC_COMPILE_OPTIONS_APPEND={}",
                root.display(),
                v1,
                cr,
                // Not decoration: the include paths are relative, so the directory alone does
                // not establish that the compiler will find them. See this module's doc.
                training_dir.display(),
                options
            );
        }
        (v1, cr) => {
            // tracing::error! on purpose. This is the one fact that decides how the next
            // report gets triaged, and the file log keeps `utai=debug`, so it is guaranteed
            // to be there — and greppable — even when the run afterwards looks like every
            // other MIOpen failure.
            tracing::error!(
                "hipRTC C++ headers MISSING OR INCOMPLETE at {} (v1={}, clangres={}; need >={} and >={}, counted recursively){} cwd={}. \
                 On an AMD GPU every MIOpen kernel build will fail from here on and f0 extraction \
                 dies first — the shipped resource did not install, or was removed by antivirus. \
                 NVIDIA and CPU runs are unaffected. HIPRTC_COMPILE_OPTIONS_APPEND={}",
                root.display(),
                v1.map_or("unreadable".into(), |n| n.to_string()),
                cr.map_or("unreadable".into(), |n| n.to_string()),
                ROCM_CXX_MIN_V1,
                ROCM_CXX_MIN_CLANGRES,
                if named.is_empty() {
                    String::new()
                } else {
                    format!(" MISSING: {}", named.join(", "))
                },
                training_dir.display(),
                options
            );
        }
    }
}

/// Free bytes available to THIS process on the volume holding `p` (S68d). Uses
/// `GetDiskFreeSpaceExW`'s `lpFreeBytesAvailableToCaller` — quota-aware, and the path is
/// resolved by the filesystem itself, so junctions/mount points report the true host
/// volume (a drive-letter lookup would not). `None` = probe failed (path gone, exotic
/// volume) — callers MUST fail open: this feeds preflight checks, never correctness.
#[cfg(windows)]
pub fn free_bytes_at(p: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    let mut wide: Vec<u16> = p.as_os_str().encode_wide().collect();
    wide.push(0);
    let mut avail: u64 = 0;
    unsafe {
        windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
            windows::core::PCWSTR(wide.as_ptr()),
            Some(&mut avail),
            None,
            None,
        )
    }
    .ok()
    .map(|()| avail)
}

#[cfg(not(windows))]
pub fn free_bytes_at(_p: &Path) -> Option<u64> {
    None
}

/// Extract every `.dll` entry whose archive path satisfies `matches` from `zip_path` into `dest_dir`
/// (flattened to its basename). Single source for the CUDA-runtime downloader's nupkg + wheel DLL
/// extraction — previously `extract_nupkg_dlls` / `extract_wheel_dlls`, byte-identical except for
/// `starts_with` vs `contains`, now expressed by the caller's `matches` closure.
pub fn extract_zip_dlls(zip_path: &Path, dest_dir: &Path, matches: impl Fn(&str) -> bool) -> crate::Result<()> {
    let file = std::fs::File::open(zip_path)?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| crate::UtaiError::Audio(format!("Zip open: {}", e)))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| crate::UtaiError::Audio(format!("Zip entry: {}", e)))?;
        let name = entry.name().to_string();
        if name.ends_with(".dll") && matches(&name) {
            let filename = name.rsplit('/').next().unwrap_or(&name);
            let dest = dest_dir.join(filename);
            let mut out = std::fs::File::create(&dest)?;
            std::io::copy(&mut entry, &mut out)?;
            tracing::info!("Extracted: {}", dest.display());
        }
    }
    Ok(())
}

/// Rename with exponential-backoff retry on ACCESS_DENIED(5) / SHARING_VIOLATION(32).
///
/// Returns the failure as a `CODE: detail` string so each domain can wrap it in its own
/// `UtaiError` variant; the CODEs (`RENAME_FAILED` / `RENAME_RETRY_EXHAUSTED`) are already in
/// the frontend's trilingual map.
///
/// Originally written for SINGLE-FILE renames (the pack.json commit marker), where a hold, if
/// any, is on that one fresh file and clears fast. It is explicitly NOT a fix for renaming a
/// big FRESHLY-EXTRACTED directory tree — Defender's async inspection of thousands of new PE
/// files holds handles below such trees continuously for minutes, far beyond any sane retry
/// window (live-failed S42 even with 10 s of backoff; that is why the install commit protocol
/// is a marker FILE, not a directory rename — see `extract_and_commit`).
///
/// The S76 training-layout migration DOES rename directory trees through it, and that is a
/// different situation, not a relapse: those trees are cold (written minutes-to-months ago,
/// long past any AV inspection queue) and the migration runs at startup before this process
/// opens anything under them. The retry is there for the remaining real holders — an Explorer
/// window, a backup agent — and a failure is answered by a rollback, not by pressing on.
pub fn rename_with_retry(from: &Path, to: &Path, what: &str) -> std::result::Result<(), String> {
    let mut delay_ms = 250u64;
    let mut last: Option<std::io::Error> = None;
    for attempt in 0..8 {
        match std::fs::rename(from, to) {
            Ok(()) => {
                if attempt > 0 {
                    // `what` is a short English phase token (PACK_MOVE_OUT / INSTALL_COMMIT /
                    // PACK_ROLLBACK / INSTALL_RECOVERY / TRAINING_MIGRATE_*) carried as the
                    // CODE's detail payload; the paths carry the context here.
                    tracing::info!("rename succeeded on retry {attempt}: {} -> {}", from.display(), to.display());
                }
                return Ok(());
            }
            Err(e) => {
                let retriable = matches!(e.raw_os_error(), Some(5) | Some(32));
                if !retriable {
                    return Err(format!("RENAME_FAILED: {what} ({} -> {}): {e}", from.display(), to.display()));
                }
                tracing::warn!("rename denied (attempt {attempt}) {} -> {}: {e} — retrying in {delay_ms}ms", from.display(), to.display());
                last = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                delay_ms = (delay_ms * 2).min(2000);
            }
        }
    }
    Err(format!(
        "RENAME_RETRY_EXHAUSTED: {what} ({} -> {}): {}",
        from.display(),
        to.display(),
        last.map(|e| e.to_string()).unwrap_or_default()
    ))
}

/// `remove_dir_all` with one assisted retry: strip READONLY attributes throughout (files
/// planted by user tooling/backup restores may carry them; a read-only file fails both the
/// in-tree delete and tar's overwrite path — `remove_file` before re-create — with os error
/// 5), pause briefly for transient handle races (AV inspection), then try once more. Call
/// sites stay best-effort — see the torn-dir comment in `pyenv::extract_and_commit`.
pub fn remove_dir_all_robust(dir: &Path) -> std::io::Result<()> {
    fn clear_readonly(dir: &Path) {
        // The ROOT dir's own attribute matters too (review S68d): std deletes children
        // first and the root last, so a readonly root alone defeats both attempts.
        if let Ok(md) = std::fs::metadata(dir) {
            let mut perm = md.permissions();
            if perm.readonly() {
                perm.set_readonly(false);
                let _ = std::fs::set_permissions(dir, perm);
            }
        }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            if let Ok(md) = e.metadata() {
                let mut perm = md.permissions();
                if perm.readonly() {
                    perm.set_readonly(false);
                    let _ = std::fs::set_permissions(&e.path(), perm);
                }
                if md.is_dir() {
                    clear_readonly(&e.path());
                }
            }
        }
    }
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(_) => {
            clear_readonly(dir);
            std::thread::sleep(std::time::Duration::from_millis(300));
            std::fs::remove_dir_all(dir)
        }
    }
}

#[cfg(test)]
mod s172_tests {
    /// The shipped payload must clear its own floors, counted the way the shipped code counts.
    ///
    /// This is the test that would have caught the non-recursive counter before it reached a
    /// user's log: against the real tree it returns 15 for `v1`, and 15 < 189 is a red test
    /// instead of a red line in every customer's log file.
    #[test]
    fn s172_shipped_cxx_header_payload_clears_its_floors() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("training")
            .join(super::ROCM_CXX_HEADER_DIR);
        let v1 = super::count_files_recursive(&root.join("v1")).expect("v1 unreadable");
        let cr = super::count_files_recursive(&root.join("clangres")).expect("clangres unreadable");
        assert!(
            v1 >= super::ROCM_CXX_MIN_V1,
            "v1 has {v1} files, floor is {}",
            super::ROCM_CXX_MIN_V1
        );
        assert!(
            cr >= super::ROCM_CXX_MIN_CLANGRES,
            "clangres has {cr} files, floor is {}",
            super::ROCM_CXX_MIN_CLANGRES
        );
        // The anti-vacuity half: `v1` must actually be NESTED, or a future non-recursive
        // counter would pass this test by accident and fail in the field exactly as before.
        let flat = std::fs::read_dir(root.join("v1"))
            .expect("v1 unreadable")
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .count();
        assert!(
            flat < super::ROCM_CXX_MIN_V1,
            "v1 is flat ({flat} files at the top level) — this test no longer proves the counter \
             recurses; check why the payload layout changed"
        );
    }

    /// A directory that is not there reads as `None`, never as `Some(0)` — the log line's two
    /// branches depend on telling "could not look" apart from "looked and it was empty".
    #[test]
    fn s172_missing_header_dir_is_none_not_zero() {
        let missing = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("training")
            .join("rocm_cxx_headers_does_not_exist");
        assert_eq!(super::count_files_recursive(&missing), None);
    }

    use super::*;

    /// S172. Two properties, and both are load-bearing rather than cosmetic:
    ///   * the tokens carry NO whitespace, because `HIPRTC_COMPILE_OPTIONS_APPEND` is split on
    ///     whitespace and ignores quoting — a path with a space arrives as a broken fragment
    ///     and clang rejects the whole compile (measured);
    ///   * the paths stay RELATIVE, which is what keeps that true no matter where the user
    ///     installed the app. Both spawn sites set cwd to `<app_dir>/training`.
    #[test]
    fn hiprtc_include_options_are_relative_and_have_no_whitespace_in_a_token() {
        let v = hiprtc_cxx_include_options();
        let tokens: Vec<&str> = v.split_whitespace().collect();
        assert_eq!(tokens.len(), 2, "unexpected token count in {v:?}");
        // -idirafter, not -I: -I would shadow the host C++ library and break machines that
        // have Visual Studio (measured). And CONCATENATED, so each flag stays one token.
        assert!(
            tokens[0].ends_with("/v1") && tokens[1].ends_with("/clangres"),
            "v1 must precede clangres (libc++ reaches the C headers with #include_next): {v:?}"
        );
        for t in &tokens {
            assert!(!t.starts_with("-idirafter="), "the '=' form is silently ignored: {t}");
            let path = t
                .strip_prefix("-idirafter")
                .unwrap_or_else(|| panic!("not an -idirafter flag: {t}"));
            assert!(!path.is_empty(), "empty include path in {v:?}");
            assert!(
                !path.starts_with('/') && !path.starts_with('\\') && !path.contains(':'),
                "include path must stay relative to the spawn cwd, got {path:?}"
            );
            assert!(path.starts_with(ROCM_CXX_HEADER_DIR), "unexpected dir in {path:?}");
        }
    }

    /// A user who already set the variable keeps it — we append, never clobber. (Their value
    /// is allowed to contain spaces: it is theirs, and it is already whitespace-split.)
    #[test]
    fn hiprtc_include_options_compose_with_an_existing_user_value() {
        // Safety: single-threaded within this test, and the value is restored before return.
        let prev = std::env::var("HIPRTC_COMPILE_OPTIONS_APPEND").ok();
        std::env::set_var("HIPRTC_COMPILE_OPTIONS_APPEND", "-DUSER_SET=1");
        let v = hiprtc_cxx_include_options();
        match prev {
            Some(p) => std::env::set_var("HIPRTC_COMPILE_OPTIONS_APPEND", p),
            None => std::env::remove_var("HIPRTC_COMPILE_OPTIONS_APPEND"),
        }
        assert!(v.starts_with("-DUSER_SET=1 "), "user value was dropped: {v:?}");
        assert!(v.contains(ROCM_CXX_HEADER_DIR), "our includes went missing: {v:?}");
    }
}
