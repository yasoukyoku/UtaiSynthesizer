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
    constructor() {
        Object.defineProperty(this, "ctx", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
        Object.defineProperty(this, "src", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
        Object.defineProperty(this, "buffer", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
        Object.defineProperty(this, "startedAt", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        }); // ctx time when the current source started
        Object.defineProperty(this, "offset", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        }); // seconds into the buffer at startedAt
        Object.defineProperty(this, "seq", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "path", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
        Object.defineProperty(this, "paused", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: false
        });
        Object.defineProperty(this, "onEnd", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
    }
    ensureCtx() {
        if (!this.ctx)
            this.ctx = new AudioContext();
        if (this.ctx.state === "suspended")
            void this.ctx.resume();
        return this.ctx;
    }
    /** Decode on the player's OWN context (one per session) — a fresh AudioContext
     *  per decode would hit the browser's ~6-context cap after a few previews. */
    decode(bytes) {
        const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        return this.ensureCtx().decodeAudioData(ab);
    }
    get duration() {
        return this.buffer?.duration ?? 0;
    }
    get position() {
        if (!this.ctx || this.paused || !this.src)
            return this.offset;
        return Math.min(this.duration, this.offset + (this.ctx.currentTime - this.startedAt));
    }
    startSource(offsetSec) {
        const ctx = this.ensureCtx();
        this.stopSource();
        const src = ctx.createBufferSource();
        src.buffer = this.buffer;
        src.connect(ctx.destination);
        const mySeq = ++this.seq;
        src.onended = () => {
            if (mySeq !== this.seq)
                return; // stopped for seek/pause/switch, not a real end
            this.path = null;
            this.onEnd?.();
        };
        this.offset = offsetSec;
        this.startedAt = ctx.currentTime;
        src.start(0, offsetSec);
        this.src = src;
        this.paused = false;
    }
    stopSource() {
        this.seq++; // invalidate the outgoing source's onended
        if (this.src) {
            try {
                this.src.stop();
            }
            catch {
                /* already stopped */
            }
            this.src = null;
        }
    }
    async play(path, buffer) {
        this.buffer = buffer;
        this.path = path;
        this.startSource(0);
    }
    pause() {
        if (!this.src || this.paused)
            return;
        this.offset = this.position;
        this.stopSource();
        this.paused = true;
    }
    resume() {
        if (!this.paused || !this.buffer)
            return;
        this.startSource(this.offset);
    }
    seek(frac) {
        if (!this.buffer)
            return;
        const target = Math.max(0, Math.min(1, frac)) * this.duration;
        if (this.paused) {
            this.offset = target;
        }
        else {
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
