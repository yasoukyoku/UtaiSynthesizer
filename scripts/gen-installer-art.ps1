# Installer artwork maintenance (S66). Provenance: the original header.bmp / sidebar.bmp were
# generated ad-hoc with ffmpeg during S64 and committed as binaries (no script survived — this
# file is the durable record going forward). NSIS/MUI2 requirements the outputs MUST keep:
#   sidebar.bmp = 164x314, header.bmp = 150x57, both 24bpp BI_RGB with the classic 40-byte
#   BITMAPINFOHEADER (System.Drawing's 24bpp BMP encoder produces exactly that).
#
# Current operation: erase the "Voice Synthesis DAW" subtitle band from the sidebar (user
# decision S66 — it duplicated the UTAI/SYNTHESIZER title). The band is filled with the
# background color sampled just above it; every other pixel stays byte-identical. The
# pre-edit original remains in git history (commit 883f068).
#
# Usage:  pwsh -File scripts\gen-installer-art.ps1
param(
  [string]$Sidebar = "$PSScriptRoot\..\src-tauri\installer\sidebar.bmp"
)

Add-Type -AssemblyName System.Drawing

$src = [System.Drawing.Bitmap]::FromFile((Resolve-Path $Sidebar))
try {
  $out = New-Object System.Drawing.Bitmap($src.Width, $src.Height, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
  $g = [System.Drawing.Graphics]::FromImage($out)
  $g.DrawImageUnscaled($src, 0, 0)

  # Background sample point: inside the plain dark area between "SYNTHESIZER" and the subtitle.
  $bg = $out.GetPixel(80, 226)
  $brush = New-Object System.Drawing.SolidBrush($bg)
  # Subtitle band (measured on the committed art): x 6..158, y 236..258 — clear of the edge
  # accents and of the colored-squares decoration further down-right.
  $g.FillRectangle($brush, 6, 236, 152, 22)
  $g.Dispose()
} finally {
  $src.Dispose()
}

$tmp = "$Sidebar.tmp.bmp"
$out.Save($tmp, [System.Drawing.Imaging.ImageFormat]::Bmp)
$out.Dispose()
Move-Item -Force $tmp $Sidebar

# Sanity: dimensions + bpp + compression from the BMP header.
$b = [IO.File]::ReadAllBytes((Resolve-Path $Sidebar))
$w = [BitConverter]::ToInt32($b, 18); $h = [BitConverter]::ToInt32($b, 22)
$bpp = [BitConverter]::ToInt16($b, 28); $comp = [BitConverter]::ToInt32($b, 30)
"sidebar.bmp: ${w}x${h} ${bpp}bpp compression=$comp size=$($b.Length)B"
if ($w -ne 164 -or $h -ne 314 -or $bpp -ne 24 -or $comp -ne 0) {
  throw "sidebar.bmp no longer matches the NSIS MUI2 contract (164x314 24bpp BI_RGB)"
}
