// wrapper.cpp — minimal extern "C" shim over Signalsmith Stretch (vendored, MIT).
//
// One-shot offline exact-length stretch. The body is a line-for-line mirror of the upstream
// one-call recipe `exact()` (signalsmith-stretch.h:466-491: outputSeek pre-roll → process the
// body → flush the tail, output aligned sample-exactly start-to-start and end-to-end with
// round(inputSamples * timeFactor)) with ONE deliberate difference: the body process() call is
// split at the points where the caller's formant-base schedule changes value, so
// setFormantBase can follow the audio's local fundamental through the piece (S82b — the
// engine's per-block pitch auto-detector chases noise on voiceless consonants, and a single
// whole-call base is wrong for wide-range material). Same engine instance across the slices:
// process() is the upstream streaming API, so the seams do not exist.
//
// exact()'s too-short bail-out (input shorter than the seek length, ~130 ms at 44.1k, would
// return silence) is handled by zero-padding the input FRONT up to the seek length and
// trimming the correspondingly stretched head off the output — the tail stays end-aligned, so
// a tiny clip still comes back as audio.
//
// Interleaved f32 in/out; channels are deinterleaved here because the engine's templated IO
// indexes buffers[channel][sample].

#include "signalsmith-stretch/signalsmith-stretch.h"

#include <algorithm>
#include <cmath>
#include <vector>

extern "C" {

// Returns 0 on success, non-zero on failure. `output` must hold out_samples * channels floats,
// where out_samples = (int)llround(in_samples * time_factor) as computed by the Rust caller.
// `transpose_semitones` = 0 keeps the original pitch (pure time-stretch); non-zero pitch-shifts
// in the spectral domain (the upstream-recommended ~8 kHz tonality limit keeps highs/air
// natural on full mixes).
// `formant_semitones` + `formant_active`: when active != 0 the formant envelope is pinned
// relative to the INPUT spectrum (compensatePitch cancels the transpose's formant drag) and
// then shifted by formant_semitones — 0 keeps formants exactly where the source had them.
// When active == 0 the formant machinery is bypassed entirely and formants follow the
// transpose (the pre-formant-control behavior, zero extra cost).
// `base_track`/`base_len`/`base_step` (only read when active): a sticky per-step schedule of
// the KNOWN fundamental on the (unpadded) input sample grid — entry i covers input samples
// [i*base_step, (i+1)*base_step), positions past the end hold the last entry. Empty (len 0)
// = the engine's auto-detector. Values are Hz; setFormantBase wants freq normalized to the
// SAMPLE RATE (stft.h freqToBin(f) = f * fftSamples), same axis as the tonality limit.
int utai_stretch_exact(const float* input, int in_samples, int channels, float sample_rate,
                       double time_factor, double transpose_semitones,
                       double formant_semitones, int formant_active, const float* base_track,
                       int base_len, int base_step, float* output, int out_samples) {
    if (!input || !output || in_samples <= 0 || out_samples <= 0 || channels <= 0 ||
        sample_rate <= 0.0f || !(time_factor > 0.0) || !std::isfinite(transpose_semitones) ||
        !std::isfinite(formant_semitones)) {
        return 1;
    }
    if (formant_active != 0 && base_len > 0 && (!base_track || base_step <= 0)) {
        return 1;
    }
    try {
        signalsmith::stretch::SignalsmithStretch<float> stretch;
        stretch.presetDefault(channels, sample_rate);
        if (transpose_semitones != 0.0) {
            stretch.setTransposeSemitones((float)transpose_semitones, 8000.0f / sample_rate);
        }
        if (formant_active != 0) {
            stretch.setFormantSemitones((float)formant_semitones, /*compensatePitch=*/true);
        }

        // Front-pad short clips so the exact() recipe never hits its too-short region (see the
        // file header). The seek length depends on the playback rate derived from the TOTAL
        // sizes, and for tiny inputs out_samples = round(in*tf) deviates measurably from tf
        // (64 × 1.4 → 90 is a 1.40625 ratio) — so iterate the pad against the actual padded
        // totals until the requirement is met (converges in a step or two; +2 covers
        // float→int truncation).
        int pad_in = 0;
        for (int guard = 0; guard < 8; ++guard) {
            const int cur_in = in_samples + pad_in;
            const int cur_out =
                std::max((int)std::llround((double)cur_in * time_factor), out_samples);
            const double pr = (double)cur_in / (double)cur_out;
            const int need = (int)stretch.outputSeekLength((float)pr) + 2;
            if (cur_in >= need) break;
            pad_in = need - in_samples;
        }
        const int total_in = in_samples + pad_in;
        const int total_out =
            std::max((int)std::llround((double)total_in * time_factor), out_samples);
        const int pad_out = total_out - out_samples;

        std::vector<std::vector<float>> in_ch((size_t)channels), out_ch((size_t)channels);
        for (int c = 0; c < channels; ++c) {
            in_ch[(size_t)c].assign((size_t)total_in, 0.0f);
            out_ch[(size_t)c].assign((size_t)total_out, 0.0f);
            for (int i = 0; i < in_samples; ++i) {
                in_ch[(size_t)c][(size_t)(pad_in + i)] = input[(size_t)i * channels + c];
            }
        }
        std::vector<float*> ptrs((size_t)channels);
        auto channel_ptrs = [&](std::vector<std::vector<float>>& chans, long long off) {
            for (int c = 0; c < channels; ++c) {
                ptrs[(size_t)c] = chans[(size_t)c].data() + off;
            }
            return ptrs.data();
        };

        // Sticky schedule lookup in PADDED input coordinates (the track is on the unpadded
        // grid — shift by pad_in).
        const bool has_base = formant_active != 0 && base_len > 0;
        auto base_at = [&](long long padded_pos) -> float {
            long long pos = padded_pos - pad_in;
            if (pos < 0) pos = 0;
            long long idx = pos / base_step;
            if (idx >= base_len) idx = base_len - 1;
            return base_track[idx];
        };
        auto apply_base = [&](long long padded_pos) {
            if (has_base) {
                stretch.setFormantBase(base_at(padded_pos) / sample_rate);
            }
        };

        // ── the exact() recipe (header 466-491), float arithmetic mirrored ──
        const float playback_rate = (float)total_in / (float)total_out;
        const int seek_len = stretch.outputSeekLength(playback_rate);
        if (total_in < seek_len) {
            return 3; // unreachable after padding (pad loop used the same formula +2)
        }
        apply_base(0);
        stretch.outputSeek(channel_ptrs(in_ch, 0), seek_len);
        const int output_index = total_out - (int)(seek_len / playback_rate);
        const long long body_in = total_in - seek_len;
        const long long body_out = output_index;

        long long done_in = 0, done_out = 0;
        while (done_in < body_in) {
            long long next_in = body_in;
            if (has_base) {
                // extend the slice to the end of the current base-value run
                const long long pos = seek_len + done_in;
                const float cur = base_at(pos);
                long long boundary = (std::max((long long)0, pos - pad_in) / base_step + 1) *
                                         (long long)base_step +
                                     pad_in;
                while (boundary < (long long)total_in && base_at(boundary) == cur) {
                    boundary += base_step;
                }
                next_in = std::min(body_in, boundary - seek_len);
                if (next_in <= done_in) next_in = body_in; // defensive: never stall
                apply_base(pos);
            }
            const long long next_out =
                (next_in >= body_in)
                    ? body_out
                    : (long long)std::llround((double)next_in * (double)body_out /
                                              (double)body_in);
            float* const* in_p = channel_ptrs(in_ch, seek_len + done_in);
            // channel_ptrs reuses one scratch vector — take output pointers separately
            std::vector<float*> out_p((size_t)channels);
            for (int c = 0; c < channels; ++c) {
                out_p[(size_t)c] = out_ch[(size_t)c].data() + done_out;
            }
            stretch.process(in_p, (int)(next_in - done_in), out_p.data(),
                            (int)(std::max(done_out, next_out) - done_out));
            done_in = next_in;
            done_out = std::max(done_out, next_out);
        }
        stretch.flush(channel_ptrs(out_ch, output_index), total_out - output_index,
                      playback_rate);

        // The padded silence occupies output [0, pad_out); the caller's exact-length result is
        // the end-aligned tail.
        for (int i = 0; i < out_samples; ++i) {
            for (int c = 0; c < channels; ++c) {
                output[(size_t)i * channels + c] = out_ch[(size_t)c][(size_t)(pad_out + i)];
            }
        }
        return 0;
    } catch (...) {
        return 2;
    }
}

} // extern "C"
