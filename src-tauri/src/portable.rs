//! S68d fullportable: make the Windows registry FOLLOW a moved install.
//!
//! The NSIS installer remembers its install dir in
//! `HKCU\Software\utaisynthesizer\UtaiSynthesizer` (default value, UNQUOTED) and the
//! Add/Remove entry in `HKCU\...\Uninstall\UtaiSynthesizer` (UninstallString /
//! InstallLocation / DisplayIcon, QUOTED — the asymmetry is the template's, mirror it
//! exactly: RestorePreviousInstallLocation copies the default value VERBATIM into
//! $INSTDIR, so quoting it would break every future install). When the user moves the
//! whole folder, those values keep pointing at the dead location — a manually
//! downloaded installer then "restores" onto C:, and Add/Remove points at a missing
//! uninstaller. The in-app updater is fixed independently via /D= (update.rs); this
//! heal covers the MANUAL-installer path and Add/Remove.
//!
//! Guards (adversarially reviewed, S68d — the ORDER matters):
//!   1. Release build + `uninstall.exe` next to the exe — only installer-produced
//!      copies may claim the registry (a `cargo build` run must not).
//!   2. HKCU uninstall entry EXISTS — repair-only, never create: a leftover copy must
//!      not resurrect an entry the user deliberately uninstalled.
//!   3. No HKLM uninstall entry — a perMachine install (not ours today) owns the
//!      machine; don't fight it with per-user values.
//!   4. DEAD-LOCATION PROOF: the recorded dir must NOT contain our main binary. Two
//!      LIVE copies therefore never ping-pong the registry, and running an old backup
//!      copy once cannot hijack the real install — only a move (old dir gone) heals.
//!   5. Compare-first, value-level writes only (the manu key also stores the NSIS
//!      "Installer Language" — never delete/recreate the key), all failures warn-only.

#[cfg(all(windows, not(debug_assertions)))]
pub fn heal_install_registry() {
    match heal_inner() {
        Ok(Some(dir)) => tracing::info!("install registry healed -> {dir}"),
        Ok(None) => {}
        Err(e) => tracing::warn!("install-registry heal skipped: {e}"),
    }
}

#[cfg(not(all(windows, not(debug_assertions))))]
pub fn heal_install_registry() {}

#[cfg(all(windows, not(debug_assertions)))]
fn heal_inner() -> Result<Option<String>, String> {
    use windows::core::PCWSTR;
    use windows::Win32::System::Registry::{
        RegCreateKeyExW, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW, HKEY,
        HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_QUERY_VALUE, KEY_READ, KEY_SET_VALUE,
        KEY_WOW64_64KEY, REG_OPTION_NON_VOLATILE, REG_SZ,
    };

    const MANU_KEY: &str = r"Software\utaisynthesizer\UtaiSynthesizer";
    const UNINST_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\UtaiSynthesizer";

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    struct Key(windows::Win32::System::Registry::HKEY);
    impl Drop for Key {
        fn drop(&mut self) {
            unsafe {
                let _ = windows::Win32::System::Registry::RegCloseKey(self.0);
            }
        }
    }

    fn open(root: HKEY, path: &str, rights: windows::Win32::System::Registry::REG_SAM_FLAGS) -> Option<Key> {
        let w = wide(path);
        let mut h = HKEY::default();
        let rc = unsafe { RegOpenKeyExW(root, PCWSTR(w.as_ptr()), Some(0), rights, &mut h) };
        rc.is_ok().then_some(Key(h))
    }

    fn read_sz(key: &Key, value: &str) -> Option<String> {
        let w = wide(value);
        let mut len: u32 = 0;
        let rc = unsafe {
            RegQueryValueExW(key.0, PCWSTR(w.as_ptr()), None, None, None, Some(&mut len))
        };
        if rc.is_err() || len == 0 {
            return None;
        }
        let mut buf = vec![0u8; len as usize];
        let mut len2 = len;
        let rc = unsafe {
            RegQueryValueExW(key.0, PCWSTR(w.as_ptr()), None, None, Some(buf.as_mut_ptr()), Some(&mut len2))
        };
        if rc.is_err() {
            return None;
        }
        let u16s: Vec<u16> = buf[..len2 as usize]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let end = u16s.iter().position(|&c| c == 0).unwrap_or(u16s.len());
        Some(String::from_utf16_lossy(&u16s[..end]))
    }

    fn write_sz(key: &Key, value: &str, data: &str) -> Result<(), String> {
        let w = wide(value);
        let d = wide(data);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 2)
        };
        let rc = unsafe { RegSetValueExW(key.0, PCWSTR(w.as_ptr()), None, REG_SZ, Some(bytes)) };
        if rc.is_ok() {
            Ok(())
        } else {
            Err(format!("RegSetValueExW {value}: {rc:?}"))
        }
    }

    // ── guard 1: installer-produced copy only ──
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let exe_name = exe
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .ok_or("no exe name")?;
    let dir = exe.parent().ok_or("no exe dir")?;
    if !dir.join("uninstall.exe").exists() {
        return Ok(None);
    }
    let dir_s = dir.to_string_lossy();
    let dir_s = dir_s.strip_prefix(r"\\?\").unwrap_or(&dir_s).to_string();

    // ── guard 3: perMachine install present → hands off ──
    if open(HKEY_LOCAL_MACHINE, UNINST_KEY, KEY_QUERY_VALUE | KEY_WOW64_64KEY).is_some() {
        return Ok(None);
    }
    // ── guard 2: repair-only — the HKCU uninstall entry must already exist ──
    let Some(uninst_ro) = open(HKEY_CURRENT_USER, UNINST_KEY, KEY_READ) else {
        return Ok(None);
    };

    // ── guard 4+5: compare against the recorded location; heal only a DEAD one ──
    let norm = |s: &str| s.trim_end_matches(['\\', '/']).to_lowercase();
    let recorded = open(HKEY_CURRENT_USER, MANU_KEY, KEY_QUERY_VALUE)
        .and_then(|k| read_sz(&k, ""))
        .unwrap_or_default();
    if !recorded.is_empty() && norm(&recorded) == norm(&dir_s) {
        return Ok(None); // already pointing here — nothing to do (idempotency key)
    }
    // Alive probe: prefer the MANU record; when a registry cleaner stripped it, fall
    // back to the uninstall entry's own InstallLocation (quoted by the installer) —
    // otherwise the dead-location proof would be vacuously skipped and a stale backup
    // copy could claim the registry over a live install (review S68d).
    let probe = if recorded.is_empty() {
        read_sz(&uninst_ro, "InstallLocation")
            .map(|s| s.trim_matches('"').to_string())
            .unwrap_or_default()
    } else {
        recorded.clone()
    };
    if !probe.is_empty() && norm(&probe) != norm(&dir_s) {
        if std::path::Path::new(&probe).join(&exe_name).exists() {
            // The recorded install is ALIVE (a second copy) — never steal its registry.
            tracing::info!(
                "portable copy at {dir_s} left registry alone (live install recorded at {probe})"
            );
            return Ok(None);
        }
    }

    // ── the heal: mirror the installer's exact value shapes. WRITE ORDER MATTERS
    // (review S68d): the MANU default is the idempotency key that short-circuits the
    // next boot, so it goes LAST — a partial failure among the uninstall values then
    // leaves the heal retryable instead of permanently half-done. ──
    drop(uninst_ro);
    let uninst = open(HKEY_CURRENT_USER, UNINST_KEY, KEY_SET_VALUE)
        .ok_or("uninstall entry vanished mid-heal")?;
    write_sz(&uninst, "UninstallString", &format!("\"{dir_s}\\uninstall.exe\""))?;
    write_sz(&uninst, "InstallLocation", &format!("\"{dir_s}\""))?;
    write_sz(&uninst, "DisplayIcon", &format!("\"{dir_s}\\{exe_name}\""))?;
    write_sz(&uninst, "DisplayVersion", env!("CARGO_PKG_VERSION"))?;
    let mut manu = HKEY::default();
    let manu_w = wide(MANU_KEY);
    let rc = unsafe {
        RegCreateKeyExW(
            HKEY_CURRENT_USER,
            PCWSTR(manu_w.as_ptr()),
            None,
            None,
            REG_OPTION_NON_VOLATILE,
            KEY_SET_VALUE,
            None,
            &mut manu,
            None,
        )
    };
    if rc.is_err() {
        return Err(format!("open {MANU_KEY}: {rc:?}"));
    }
    let manu = Key(manu);
    write_sz(&manu, "", &dir_s)?; // UNQUOTED — RestorePreviousInstallLocation copies it verbatim
    Ok(Some(dir_s))
}
