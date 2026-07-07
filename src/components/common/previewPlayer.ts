/** Single-file preview player (extracted from TrainingPage in S41 so the
 *  audition rows can share it): one AudioContext, decodes into ITS OWN buffer
 *  (not the DAW's shared loadedBuffers cache — a preview must not pin decoded
 *  PCM there for the session). WebAudio sources are one-shot, so pause/seek =
 *  stop + restart at an offset; `seq` guards a stop's onended from a
 *  superseded gesture.
 *
 *  CONSUMER CONTRACT (red-team R19) — the exported `preview` is a singleton
 *  shared by the training data step AND the audition rows (mutual preemption
 *  is the intended behavior):
 *    - on mount / before taking over: call preview.stop(), THEN assign onEnd
 *    - on unmount: preview.stop() and null out onEnd (a stale callback would
 *      drive a dead component's state)
 */
export class PreviewPlayer {
  private ctx: AudioContext | null = null;
  private src: AudioBufferSourceNode | null = null;
  private buffer: AudioBuffer | null = null;
  private startedAt = 0; // ctx time when the current source started
  private offset = 0; // seconds into the buffer at startedAt
  private seq = 0;
  path: string | null = null;
  paused = false;
  onEnd: (() => void) | null = null;

  private ensureCtx(): AudioContext {
    if (!this.ctx) this.ctx = new AudioContext();
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }

  /** Decode on the player's OWN context (one per session) — a fresh AudioContext
   *  per decode would hit the browser's ~6-context cap after a few previews. */
  decode(bytes: Uint8Array): Promise<AudioBuffer> {
    const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    return this.ensureCtx().decodeAudioData(ab as ArrayBuffer);
  }

  get duration(): number {
    return this.buffer?.duration ?? 0;
  }

  get position(): number {
    if (!this.ctx || this.paused || !this.src) return this.offset;
    return Math.min(this.duration, this.offset + (this.ctx.currentTime - this.startedAt));
  }

  private startSource(offsetSec: number) {
    const ctx = this.ensureCtx();
    this.stopSource();
    const src = ctx.createBufferSource();
    src.buffer = this.buffer;
    src.connect(ctx.destination);
    const mySeq = ++this.seq;
    src.onended = () => {
      if (mySeq !== this.seq) return; // stopped for seek/pause/switch, not a real end
      this.path = null;
      this.onEnd?.();
    };
    this.offset = offsetSec;
    this.startedAt = ctx.currentTime;
    src.start(0, offsetSec);
    this.src = src;
    this.paused = false;
  }

  private stopSource() {
    this.seq++; // invalidate the outgoing source's onended
    if (this.src) {
      try {
        this.src.stop();
      } catch {
        /* already stopped */
      }
      this.src = null;
    }
  }

  async play(path: string, buffer: AudioBuffer) {
    this.buffer = buffer;
    this.path = path;
    this.startSource(0);
  }

  pause() {
    if (!this.src || this.paused) return;
    this.offset = this.position;
    this.stopSource();
    this.paused = true;
  }

  resume() {
    if (!this.paused || !this.buffer) return;
    this.startSource(this.offset);
  }

  seek(frac: number) {
    if (!this.buffer) return;
    const target = Math.max(0, Math.min(1, frac)) * this.duration;
    if (this.paused) {
      this.offset = target;
    } else {
      this.startSource(target);
    }
  }

  stop() {
    this.stopSource();
    this.path = null;
    this.paused = false;
    this.offset = 0;
    this.buffer = null; // release the decoded PCM (a long file is hundreds of MB)
  }
}

export const preview = new PreviewPlayer();
