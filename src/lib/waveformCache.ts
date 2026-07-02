/**
 * Per-source waveform bitmap cache.
 *
 * Rendering a waveform by looping over every visible pixel (moveTo/lineTo per column) once per
 * segment per lane on EVERY redraw was the dominant arrangement scroll/auto-scroll cost: the
 * static layer is rebuilt on every scroll frame, so those loops re-ran continuously. Instead we
 * render each audio source's full waveform ONCE into its own OffscreenCanvas (a fixed-resolution
 * bitmap) and, on every redraw, BLIT the visible slice with a single drawImage — decoupling the
 * waveform cost from scroll entirely (one GPU blit per segment/lane, no per-pixel JS).
 *
 * The cache is zoom-INDEPENDENT: built once at a fixed column resolution, scaled by the blit. A
 * blit at the current zoom is therefore ~1:1 for normal-length clips and only softens for very long
 * clips (whose peaks are capped to MAX_CACHE_W) when zoomed in. Built lazily on first use, then
 * reused across scroll/zoom/playback.
 *
 * Eviction is GENERATION-PROTECTED (see beginWaveformFrame): an entry fetched during the current
 * draw pass is never evicted, so the live working set is always fully cached even if it exceeds the
 * byte budget for a frame — that guarantees no per-frame rebuild "thrash" no matter how many
 * distinct waveforms are visible at once. Only stale (previous-frame) entries are pruned, down to a
 * byte budget, so idle memory stays bounded.
 */

const CACHE_H = 256; // bitmap height; the blit scales it to the segment/lane display height. 256 covers
// the tallest realistic device height (max vZoom ~140 CSS px × dpr 2 ≈ 280) so high-vZoom/hi-dpi
// waveforms aren't upscaled into blur.
// Column cap: bounds the OffscreenCanvas width (browsers cap canvas dimensions at ~32767) and memory.
// Kept just under that hard limit so even multi-minute sources retain plenty of horizontal detail.
const MAX_CACHE_W = 32000;
// Idle memory budget for stale bitmaps. Current-frame entries are exempt (never evicted), so the live
// set can briefly exceed this; it only caps what's retained for scroll-back after it leaves the view.
const MAX_CACHE_BYTES = 384 * 1024 * 1024;

interface Entry {
  canvas: OffscreenCanvas;
  gen: number; // the draw generation in which this entry was last used
  bytes: number;
}

const cache = new Map<string, Entry>();
let totalBytes = 0;
let drawGen = 0;

/** Start a new draw pass. Call ONCE at the top of the draw, before any getWaveformCache calls:
 *  entries fetched after this are stamped with the new generation and protected from eviction for
 *  the rest of the pass, so a working set larger than the budget never thrashes. */
export function beginWaveformFrame() {
  drawGen++;
}

/** O(1) content discriminator: peak count + a few sampled amplitudes. Distinguishes two same-length
 *  peak arrays — e.g. a workflow re-run that overwrites a processed output at the SAME path with new
 *  content — which the length alone would not, preventing a stale cached/rendered waveform. Must be
 *  mirrored into the arrangement's staticKey so the static layer also redraws on a content change. */
export function peaksSignature(peaks: number[]): string {
  const n = peaks.length;
  if (n === 0) return "0";
  const q = n >> 2;
  const f = (i: number) => (peaks[i] ?? 0).toFixed(3);
  return `${n}.${f(0)}.${f(q)}.${f(q * 2)}.${f(q * 3)}.${f(n - 1)}`;
}

function build(peaks: number[], color: string): { canvas: OffscreenCanvas; bytes: number } {
  const cols = Math.max(1, Math.min(peaks.length, MAX_CACHE_W));
  const canvas = new OffscreenCanvas(cols, CACHE_H);
  const ctx = canvas.getContext("2d")!;
  const mid = CACHE_H / 2;
  const amp = CACHE_H / 2 - 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  const per = peaks.length / cols; // peaks per column: >1 when downsampling, ~1 when one col per peak
  for (let c = 0; c < cols; c++) {
    // Max over this column's peak range preserves peak height when downsampling (better than the old
    // floor-sample which dropped peaks); when cols >= peaks.length each column maps to one peak.
    const start = Math.floor(c * per);
    const end = Math.min(peaks.length, Math.max(start + 1, Math.floor((c + 1) * per)));
    let p = 0;
    for (let i = start; i < end; i++) {
      const v = peaks[i]!;
      if (v > p) p = v;
    }
    // Contiguous vertical lines (one per integer column) tile the full width with no gaps, so a
    // downscaling blit averages neighbours instead of losing alpha to transparent gaps.
    const x = c + 0.5;
    ctx.moveTo(x, mid - p * amp);
    ctx.lineTo(x, mid + p * amp);
  }
  ctx.stroke();
  return { canvas, bytes: cols * CACHE_H * 4 };
}

/**
 * Get (building + caching on first use) the waveform bitmap for `peaks` rendered in `color`.
 * `id` is a stable identity for the peak data (a source path or processed-output path); `color`
 * is baked into the bitmap (a source shown in two colours yields two entries — rare in practice).
 * Returns null for empty peaks.
 */
export function getWaveformCache(id: string, peaks: number[], color: string): OffscreenCanvas | null {
  if (peaks.length === 0) return null;
  const key = `${id} ${peaksSignature(peaks)} ${color}`;
  const hit = cache.get(key);
  if (hit) {
    hit.gen = drawGen;
    // Bump recency: re-insert so it moves to the end of the Map's insertion (= LRU) order.
    cache.delete(key);
    cache.set(key, hit);
    return hit.canvas;
  }
  const { canvas, bytes } = build(peaks, color);
  cache.set(key, { canvas, gen: drawGen, bytes });
  totalBytes += bytes;
  if (totalBytes > MAX_CACHE_BYTES) {
    // Evict least-recently-used entries (Map iterates oldest-first) NOT touched this frame, until
    // under budget. Current-frame entries are skipped, so the live working set is never evicted →
    // no per-frame rebuild thrash even when more distinct waveforms are visible than fit the budget.
    for (const [k, ent] of cache) {
      if (totalBytes <= MAX_CACHE_BYTES) break;
      if (ent.gen === drawGen) continue;
      cache.delete(k);
      totalBytes -= ent.bytes;
    }
  }
  return canvas;
}

/** Drop all cached bitmaps (e.g. on project close/switch) so a new project starts clean. */
export function clearWaveformCache() {
  cache.clear();
  totalBytes = 0;
}

/**
 * Blit the visible slice of a cached waveform onto `ctx`. `startRatio`/`endRatio` select the source
 * window (offset → offset+duration as a fraction of the full source, both in [0,1] with
 * startRatio <= endRatio); the destination is clipped to the canvas's visible x-range [0, clipWidth]
 * so a long, mostly off-screen segment costs nothing.
 */
export function blitWaveform(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  wave: OffscreenCanvas,
  x: number,
  y: number,
  w: number,
  h: number,
  startRatio: number,
  endRatio: number,
  clipWidth: number,
) {
  if (w <= 0 || h <= 0 || endRatio <= startRatio) return;
  const destL = Math.max(x, 0);
  const destR = Math.min(x + w, clipWidth);
  if (destR <= destL) return;
  // Fraction of the segment that's visible → the matching source sub-window inside [startRatio,endRatio].
  const fL = (destL - x) / w;
  const fR = (destR - x) / w;
  const span = endRatio - startRatio;
  const srcX = Math.max(0, (startRatio + fL * span) * wave.width);
  const srcW = Math.min((fR - fL) * span * wave.width, wave.width - srcX);
  if (srcW <= 0) return;
  // The blit only ever DOWNSCALES (drawWaveform routes upscaling to the per-pixel path), so bilinear
  // smoothing is correct here: it cleanly averages columns (no aliasing/shimmer while scrolling) and,
  // since we never upscale, it can't blur thin lines into a smear.
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(wave, srcX, 0, srcW, wave.height, destL, y, destR - destL, h);
}

/**
 * Draw a waveform, choosing the rendering that looks best at the current scale:
 *  - DOWNSCALE / ~1:1 (zoomed out — the perf-critical case with many segments/lanes visible): blit
 *    the cached bitmap (one drawImage, no per-pixel JS).
 *  - UPSCALE (zoomed in past the cache's column resolution, where blitting would show blocky
 *    rectangles): draw per-pixel directly from the FULL peaks with linear interpolation → a smooth,
 *    crisp envelope. Few segments are visible when zoomed in, so this is cheap.
 * `id` keys the cache; `peaks` is the raw amplitude data; `color` is the stroke/fill colour.
 */
export function drawWaveform(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  id: string,
  peaks: number[],
  color: string,
  x: number,
  y: number,
  w: number,
  h: number,
  startRatio: number,
  endRatio: number,
  clipWidth: number,
) {
  if (w <= 0 || h <= 0 || endRatio <= startRatio || peaks.length === 0) return;
  const destL = Math.max(x, 0);
  const destR = Math.min(x + w, clipWidth);
  if (destR <= destL) return;

  // How many cache columns back the visible window vs how many device pixels it occupies. If the
  // device span exceeds the available columns, blitting would upscale (blocky) → draw per-pixel.
  const visFrac = (destR - destL) / w;
  const cacheCols = Math.min(peaks.length, MAX_CACHE_W);
  const colsInWindow = visFrac * (endRatio - startRatio) * cacheCols;
  const deviceW = (destR - destL) * devicePixelRatio;

  if (deviceW > colsInWindow) {
    drawWaveformDirect(ctx, peaks, color, x, y, w, h, startRatio, endRatio, destL, destR);
  } else {
    const wave = getWaveformCache(id, peaks, color);
    if (wave) blitWaveform(ctx, wave, x, y, w, h, startRatio, endRatio, clipWidth);
  }
}

/** Per-pixel waveform draw over the visible canvas-x range [destL, destR), with linear interpolation
 *  between adjacent peaks so a zoomed-in envelope is smooth rather than a stair-stepped/blocky one. */
function drawWaveformDirect(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  peaks: number[],
  color: string,
  x: number,
  y: number,
  w: number,
  h: number,
  startRatio: number,
  endRatio: number,
  destL: number,
  destR: number,
) {
  const midY = y + h / 2;
  const amp = h / 2 - 1;
  const span = endRatio - startRatio;
  const n = peaks.length;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let sx = Math.floor(destL); sx < destR; sx++) {
    const ratio = startRatio + ((sx - x) / w) * span;
    const fidx = Math.min(Math.max(ratio * (n - 1), 0), n - 1);
    const i0 = Math.floor(fidx);
    const i1 = Math.min(i0 + 1, n - 1);
    const peak = peaks[i0]! + (peaks[i1]! - peaks[i0]!) * (fidx - i0);
    const px = sx + 0.5;
    ctx.moveTo(px, midY - peak * amp);
    ctx.lineTo(px, midY + peak * amp);
  }
  ctx.stroke();
}
