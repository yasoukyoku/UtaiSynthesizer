# S64 — installed-path verification checklist (the v1 lesson: verify the INSTALLED tree, never the
# build tree; scripted + full output, never a verbal "done").
#
#   powershell -File scripts\verify-install.ps1 -InstallDir "$env:LOCALAPPDATA\UtaiSynthesizer"

param([Parameter(Mandatory = $true)][string]$InstallDir)

$ErrorActionPreference = "Continue"
$fails = 0
function Check($label, $ok, $detail = "") {
  if ($ok) { Write-Host ("  OK   " + $label + ($detail ? "  ($detail)" : "")) -ForegroundColor Green }
  else { Write-Host ("  FAIL " + $label + ($detail ? "  ($detail)" : "")) -ForegroundColor Red; $script:fails++ }
}

Write-Host "== UtaiSynthesizer install verification: $InstallDir ==" -ForegroundColor Cyan

# 1. core binary + uninstaller
$exe = Join-Path $InstallDir "UtaiSynthesizer.exe"
Check "UtaiSynthesizer.exe" (Test-Path $exe) $(if (Test-Path $exe) { "{0:N0} bytes" -f (Get-Item $exe).Length })
Check "uninstall.exe" (Test-Path (Join-Path $InstallDir "uninstall.exe"))

# 2. exe version metadata matches the repo version
$repoVer = (Get-Content (Join-Path $PSScriptRoot "..\src-tauri\tauri.conf.json") -Raw | ConvertFrom-Json).version
if (Test-Path $exe) {
  $fileVer = (Get-Item $exe).VersionInfo.ProductVersion
  Check "exe ProductVersion == $repoVer" ($fileVer -like "$repoVer*") "exe reports $fileVer"
}

# 3. ffmpeg present WITH the S63 encoder set (presence alone was a v1 delayed-explosion class)
$ff = Join-Path $InstallDir "ffmpeg.exe"
Check "ffmpeg.exe" (Test-Path $ff)
if (Test-Path $ff) {
  $enc = & $ff -hide_banner -encoders 2>$null | Out-String
  foreach ($e in @("libmp3lame", "libvorbis", "libopus", " aac", " flac")) {
    Check "ffmpeg encoder$e" ($enc -match [regex]::Escape($e))
  }
}

# 4. ORT DirectML runtime — incl. DirectML.dll itself (the ORT DirectML build delay-loads it and the
# Windows inbox copy is too old for ORT 1.24; its absence only explodes on END-USER machines)
Check "runtime\ort\onnxruntime.dll" (Test-Path (Join-Path $InstallDir "runtime\ort\onnxruntime.dll"))
Check "runtime\ort\onnxruntime_providers_shared.dll" (Test-Path (Join-Path $InstallDir "runtime\ort\onnxruntime_providers_shared.dll"))
Check "runtime\ort\DirectML.dll" (Test-Path (Join-Path $InstallDir "runtime\ort\DirectML.dll"))

# 5. dictionaries — exact count AND per-file (a flattened/partial copy was a v1 silent-crash class)
$dictDir = Join-Path $InstallDir "data\dictionaries"
foreach ($d in @("zh_syllables", "zh_chars", "zh_phrases", "en", "de", "fr", "es", "it")) {
  $p = Join-Path $dictDir "$d.tsv"
  Check "dictionary $d.tsv" ((Test-Path $p) -and ((Get-Item $p -ErrorAction SilentlyContinue).Length -gt 100))
}

# 6. converter scripts (runtime-invoked set) + architectures package
foreach ($f in @("convert.py", "extract_index.py", "export_cluster.py", "export_diffusion.py", "export_nsf_hifigan.py", "onnx_fp16.py")) {
  Check "converter\$f" (Test-Path (Join-Path $InstallDir "converter\$f"))
}
$archCount = (Get-ChildItem (Join-Path $InstallDir "converter\architectures") -Filter *.py -ErrorAction SilentlyContinue | Measure-Object).Count
Check "converter\architectures\*.py (>=8)" ($archCount -ge 8) "$archCount files"

# 7. training package tree (spawn: python -m utai_train.runner, cwd=<install>\training)
foreach ($f in @("training\utai_train\runner.py", "training\utai_train\rvc", "training\utai_train\sovits", "training\utai_train\vocoder", "training\assets\mute")) {
  Check $f (Test-Path (Join-Path $InstallDir $f))
}
$pyc = (Get-ChildItem (Join-Path $InstallDir "training") -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue | Measure-Object).Count
Check "no __pycache__ shipped" ($pyc -eq 0) "$pyc dirs"

# 8. Nothing that must NOT ship. data\models is special: the APP creates it (empty) at first run,
# so on a used install only FILES inside it indicate accidentally-bundled payload.
foreach ($bad in @("converter\.venv", "training\.venv", "training\packs", "data\dictionaries\sources")) {
  Check "absent: $bad" (-not (Test-Path (Join-Path $InstallDir $bad)))
}
$modelFiles = (Get-ChildItem (Join-Path $InstallDir "data\models") -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
Check "no bundled model payload in data\models" ($modelFiles -eq 0) "$modelFiles files"

# 9. portability: no absolute paths baked into any bundled text config the app reads at startup
$cfg = Join-Path $InstallDir "config.json"
Check "no pre-baked config.json" (-not (Test-Path $cfg)) "config.json must be created at runtime, not shipped"

Write-Host ""
if ($fails -eq 0) { Write-Host "== ALL CHECKS PASSED ==" -ForegroundColor Green; exit 0 }
else { Write-Host "== $fails CHECKS FAILED ==" -ForegroundColor Red; exit 1 }
