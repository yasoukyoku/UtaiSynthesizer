use std::path::{Path, PathBuf};

/// Resolve the Python interpreter for the TRAINING sidecar. Priority:
///   1. the tool's own venv:        `<venv_dir>/.venv/Scripts/python.exe`
///   2. the manual portable slot:    `<app_dir>/python/python.exe`
///   3. the system `python` on PATH (dev fallback)
///
/// `venv_dir` is the directory CONTAINING `.venv` (e.g. `app_dir/training`), NOT the `.venv` itself.
/// S42: the CONVERTER role moved to `crate::pyenv::converter_python` (venv → installed runtime
/// packs → manual slot → PATH); training deliberately does NOT consult runtime packs yet — the
/// vendored training code is only validated against packs after the Phase B torch-axis migration
/// (running it on a 2.11 pack before that would break at the first weights_only torch.load).
pub fn find_python(venv_dir: &Path, app_dir: &Path) -> PathBuf {
    let venv = venv_dir.join(".venv").join("Scripts").join("python.exe");
    if venv.exists() {
        return venv;
    }
    let embedded = manual_python_slot(app_dir);
    if embedded.exists() {
        return embedded;
    }
    PathBuf::from("python")
}

/// The manual portable-python slot `<app_dir>/python/python.exe` — ONE definition
/// shared by `find_python` (training) and `pyenv::converter_python` (converter), so
/// the two roles can never drift onto different "manual slot" locations.
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
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
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
