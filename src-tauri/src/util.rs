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
    //   FIND_MODE=5 (FastHybrid): gfx1103 ships no precompiled conv DB (rocm #6335), so
    //     the default exhaustive kernel search is slow / can hang on first encounter —
    //     this bounds it (community-verified on the 780M; validated in the S44 dogfood).
    //   LOG_LEVEL=3 (Error): mute MIOpen's one-time first-step "IsEnoughWorkspace /
    //     CK grouped conv" Warning burst (~70 lines) that would otherwise crowd the
    //     200-line stderr ring; real Error/Fatal still surface. (xnack device-cap notice
    //     is a HIP-runtime line, not MIOpen — a harmless 2-liner this doesn't touch.)
    cmd.env("MIOPEN_FIND_MODE", "5");
    cmd.env("MIOPEN_LOG_LEVEL", "3");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
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
