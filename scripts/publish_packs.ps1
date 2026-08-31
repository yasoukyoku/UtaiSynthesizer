# S168: publish the runtime packs to a GitHub release (tag `packs-v1`) as a TRUE mirror of
# the HF dataset — hf-mirror.com 308-redirects our dataset back to huggingface.co, so until
# this release exists the app has no download route that leaves the huggingface.co network
# path (the first community report failed six pack installs exactly there). The catalog in
# src-tauri/src/pyenv/mod.rs and MIRROR_LIST_URLS in commands/settings.rs reference these
# exact asset URLs; the tag is the base directory (part URLs are derived by joining the
# manifest's directory), so the tag name must never change once shipped.
#
# ⛔ --prerelease + --latest=false are LOAD-BEARING, not cosmetic: the updater fetches
#    `releases/latest/download/latest.json`, and a plain release on this tag would become
#    "latest" and hijack the update chain.
#
# Usage:  pwsh scripts/publish_packs.ps1           # verify-only dry run
#         pwsh scripts/publish_packs.ps1 -Publish  # create/refresh the release + upload
param([switch]$Publish)
$ErrorActionPreference = "Stop"

$repo = "yasoukyoku/UtaiSynthesizer"
$tag = "packs-v1"
$dist = Join-Path $PSScriptRoot "..\training\packs\build\dist"
# The four packs the current catalog offers. runtime-amd-v1 stays HF-only on purpose: no
# shipped catalog fetches its manifest any more (v2 replaced it in 0.12.0).
$packs = @("runtime-cpu-v1", "runtime-nv-cu130-v1", "runtime-amd-v2", "runtime-xpu-v1")

# 1. Integrity: every local part must match its manifest's sha256 BEFORE it is published
#    (S103: shipped assets are verified against their own hash table, never assumed).
$assets = @()
foreach ($p in $packs) {
    $manifest = Join-Path $dist "$p.manifest.json"
    if (-not (Test-Path $manifest)) { throw "missing manifest: $manifest" }
    $m = Get-Content $manifest -Raw | ConvertFrom-Json
    foreach ($part in $m.parts) {
        $f = Join-Path $dist $part.name
        if (-not (Test-Path $f)) { throw "missing part: $f" }
        $len = (Get-Item $f).Length
        if ($len -ne $part.size) { throw "size mismatch: $($part.name) ($len vs $($part.size))" }
        $sha = (Get-FileHash $f -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($sha -ne $part.sha256) { throw "sha256 mismatch: $($part.name)" }
        Write-Host "verified $($part.name) ($len bytes, sha256 ok)"
        $assets += $f
    }
    $assets += $manifest
}

# 2. mirrors.json: the GH copy exists to break the bootstrap circle (the HF-hosted original
#    is unreachable for exactly the clients that need the gh proxy list).
$mirrors = Join-Path $dist "mirrors.json"
Invoke-WebRequest -Uri "https://huggingface.co/datasets/yasoukyoku/utai-runtimes/resolve/main/mirrors.json" -OutFile $mirrors
$mj = Get-Content $mirrors -Raw | ConvertFrom-Json
if ($mj.schema -ne 1) { throw "mirrors.json schema != 1 — refusing to republish it" }
Write-Host "verified mirrors.json (schema 1, $((Get-Item $mirrors).Length) bytes)"
$assets += $mirrors

if (-not $Publish) {
    Write-Host "`nDRY RUN ok — would upload to $repo@$tag :"
    $assets | ForEach-Object { Write-Host "  $_" }
    exit 0
}

# 3. Create the release if the tag is new, then ENFORCE the load-bearing flags on every run
#    (reviewed S168: enforcing them only on the create path leaves a hand-flipped release
#    hijacking releases/latest until the next create — which never comes). Upload with
#    --clobber so a re-run refreshes rather than duplicates.
#    ⚠ Content-drift rule: a tar asset must only ever be re-uploaded with byte-identical
#    content (the manifests pin sha256, and the downloader carries a .part across mirror
#    failover assuming identical bytes). A changed pack is a NEW pack id, never a refresh.
gh release view $tag --repo $repo *> $null
if ($LASTEXITCODE -ne 0) {
    gh release create $tag --repo $repo --prerelease --latest=false `
        --title "Runtime packs mirror" `
        --notes "Mirror of the utai-runtimes HF dataset for the in-app runtime-pack installer. Do not rename or delete: shipped builds resolve these asset URLs directly (see src-tauri/src/pyenv/mod.rs CATALOG)."
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed" }
}
gh release edit $tag --repo $repo --prerelease --latest=false *> $null
if ($LASTEXITCODE -ne 0) { throw "gh release edit (prerelease/latest flags) failed" }
foreach ($a in $assets) {
    Write-Host "uploading $a ..."
    gh release upload $tag $a --repo $repo --clobber
    if ($LASTEXITCODE -ne 0) { throw "upload failed: $a" }
}

# 4. Post-publish verification: EVERY url the client derives — manifests, mirrors.json, and
#    the multi-GB parts (the ones this mirror exists for) — must answer 200 with the exact
#    manifest-pinned size (reviewed S168: verifying only the manifests guaranteed nothing).
$checks = @(@{ url = "https://github.com/$repo/releases/download/$tag/mirrors.json"; size = $null })
foreach ($p in $packs) {
    $m = Get-Content (Join-Path $dist "$p.manifest.json") -Raw | ConvertFrom-Json
    $checks += @{ url = "https://github.com/$repo/releases/download/$tag/$p.manifest.json"; size = $null }
    foreach ($part in $m.parts) {
        $checks += @{ url = "https://github.com/$repo/releases/download/$tag/$($part.name)"; size = [int64]$part.size }
    }
}
foreach ($c in $checks) {
    $r = Invoke-WebRequest -Uri $c.url -Method Head -MaximumRedirection 5
    if ($r.StatusCode -ne 200) { throw "verify failed: $($c.url) -> $($r.StatusCode)" }
    $len = [int64]($r.Headers["Content-Length"] | Select-Object -First 1)
    if ($null -ne $c.size -and $len -ne $c.size) {
        throw "verify failed: $($c.url) Content-Length $len != manifest size $($c.size)"
    }
    Write-Host "verified $($c.url) -> 200 ($len bytes)"
}
Write-Host "`npublish complete."
