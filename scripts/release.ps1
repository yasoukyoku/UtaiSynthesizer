# S64 release build — the ONE command that produces a signed, publishable installer.
# Encodes the v1 installer post-mortem lessons as HARD GATES: version consistency, bundle-input
# existence (incl. ffmpeg ENCODERS, not just presence), full test gates, and updater artifacts.
#
#   powershell -File scripts\release.ps1            # build + verify, no publishing
#   powershell -File scripts\release.ps1 -Publish   # ...then create the GitHub release (+latest.json)
#
# Signing: TAURI_SIGNING_PRIVATE_KEY_PATH is set below (key location recorded in project memory;
# LOSING THAT KEY = shipped installs can never update again).

param(
  [switch]$Publish,
  [string]$Notes = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Fail($msg) { Write-Host "RELEASE GATE FAILED: $msg" -ForegroundColor Red; exit 1 }

# ── 1. version consistency (the v1 "version strings scattered over 7 places" lesson) ──
$confVer  = (Get-Content src-tauri\tauri.conf.json -Raw | ConvertFrom-Json).version
$cargoVer = (Select-String -Path src-tauri\Cargo.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$npmVer   = (Get-Content package.json -Raw | ConvertFrom-Json).version
if ($confVer -ne $cargoVer -or $confVer -ne $npmVer) {
  Fail "version mismatch: tauri.conf=$confVer cargo=$cargoVer npm=$npmVer"
}
if ($confVer -notmatch '^\d+\.\d+\.\d+$') { Fail "version '$confVer' is not strict three-part semver" }

# ── 2. version must be STRICTLY newer than the latest published tag (updater only offers newer) ──
$lastTag = (git tag --list "v*" --sort=-v:refname | Select-Object -First 1)
if ($lastTag) {
  $last = [version]($lastTag -replace '^v', '')
  if ([version]$confVer -le $last) { Fail "version $confVer is not newer than published tag $lastTag" }
}

# ── 3. bundle inputs exist + ffmpeg has the S63 encoder set ──
foreach ($p in @(
  "bin\ffmpeg.exe",
  "runtime\ort\onnxruntime.dll",
  "runtime\ort\onnxruntime_providers_shared.dll",
  "runtime\ort\DirectML.dll",
  "LICENSE", "NOTICE.md",
  "src-tauri\installer\header.bmp", "src-tauri\installer\sidebar.bmp",
  "converter\convert.py", "training\utai_train\runner.py", "training\assets"
)) { if (-not (Test-Path $p)) { Fail "bundle input missing: $p" } }
$dicts = @("zh_syllables","zh_chars","zh_phrases","en","de","fr","es","it")
foreach ($d in $dicts) { if (-not (Test-Path "data\dictionaries\$d.tsv")) { Fail "dictionary missing: $d.tsv" } }
$enc = & bin\ffmpeg.exe -hide_banner -encoders 2>$null | Out-String
foreach ($e in @("libmp3lame", "libvorbis", "libopus", " aac", " flac")) {
  if ($enc -notmatch [regex]::Escape($e)) { Fail "bundled ffmpeg lacks encoder:$e" }
}

# ── 4. purge python bytecode from bundled trees (resources copy directories verbatim) ──
Get-ChildItem -Recurse -Directory -Filter __pycache__ -Path converter, training\utai_train, training\assets -ErrorAction SilentlyContinue |
  ForEach-Object { Remove-Item $_.FullName -Recurse -Force }

# ── 5. full gates ──
Write-Host "gate: tsc" -ForegroundColor Cyan
npx tsc -b; if ($LASTEXITCODE -ne 0) { Fail "tsc" }
Write-Host "gate: vitest" -ForegroundColor Cyan
npx vitest run; if ($LASTEXITCODE -ne 0) { Fail "vitest" }
Write-Host "gate: cargo test --lib" -ForegroundColor Cyan
Push-Location src-tauri; cargo test --lib; $rc = $LASTEXITCODE; Pop-Location
if ($rc -ne 0) { Fail "cargo test" }

# ── 6. build (signed) ── (the CLI wants TAURI_SIGNING_PRIVATE_KEY — content works everywhere,
# the *_PATH variant was not honored by tauri CLI 2.x in practice)
$keyPath = "$env:USERPROFILE\.tauri\utaisynthesizer.key"
if (-not (Test-Path $keyPath)) { Fail "signing key missing (see memory: reference_release_signing_and_publishing)" }
$env:TAURI_SIGNING_PRIVATE_KEY = (Get-Content $keyPath -Raw)
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = ""
Write-Host "building v$confVer ..." -ForegroundColor Cyan
npm run tauri build; if ($LASTEXITCODE -ne 0) { Fail "tauri build" }

# ── 7. artifacts + latest.json ──
$setup = "src-tauri\target\release\bundle\nsis\UtaiSynthesizer_${confVer}_x64-setup.exe"
$sig = "$setup.sig"
if (-not (Test-Path $setup)) { Fail "setup exe not produced: $setup" }
if (-not (Test-Path $sig)) { Fail ".sig not produced (signing key env not seen by the build?)" }
$latest = @{
  version  = $confVer
  notes    = $Notes
  pub_date = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  platforms = @{
    "windows-x86_64" = @{
      signature = (Get-Content $sig -Raw)
      url = "https://github.com/yasoukyoku/UtaiSynthesizer/releases/download/v$confVer/UtaiSynthesizer_${confVer}_x64-setup.exe"
    }
  }
}
$latestPath = "src-tauri\target\release\bundle\nsis\latest.json"
$latest | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 $latestPath
Write-Host "built: $setup" -ForegroundColor Green
Write-Host "       $sig" -ForegroundColor Green
Write-Host "       $latestPath" -ForegroundColor Green

# ── 8. publish (opt-in; the release must be a REAL release — prerelease/draft never becomes `latest`) ──
if ($Publish) {
  Write-Host "publishing v$confVer to GitHub Releases..." -ForegroundColor Cyan
  $relNotes = if ($Notes) { $Notes } else { "UtaiSynthesizer v$confVer" }
  gh release create "v$confVer" $setup $latestPath --title "UtaiSynthesizer v$confVer" --notes $relNotes
  if ($LASTEXITCODE -ne 0) { Fail "gh release create" }
  Write-Host "published: https://github.com/yasoukyoku/UtaiSynthesizer/releases/tag/v$confVer" -ForegroundColor Green
}
