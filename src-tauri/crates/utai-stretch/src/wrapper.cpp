// wrapper.cpp — minimal extern "C" shim over Signalsmith Stretch (vendored, MIT).
//
// One-shot offline exact-length stretch via the upstream one-call recipe: stretch.exact()
// (the 1.3.x canonical path — vendor cmd/main.cpp uses it; it runs the outputSeek pre-roll,
// process and flush internally and aligns the output sample-exactly start-to-start and
// end-to-end with round(inputSamples * timeFactor)).
//
// exact() refuses inputs shorter than its seek length (outputSeekLength(playbackRate), on the
// order of 130 ms at 44.1k) by zeroing the output. Those degenerate clips are handled here by
// zero-padding the input FRONT up to the seek length and trimming the correspondingly stretched
// head off the output — the tail stays end-aligned, so a tiny clip still comes back as audio
// instead of silence (the transpose node can be fed arbitrarily short segments).
//
// Interleaved f32 in/out; channels are deinterleaved here because the templated exact()
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
// `formant_base_hz` (only read when active): > 0 pins the formant analysis to a KNOWN
// fundamental instead of the engine's per-block auto-detector — the detector chases noise on
// voiceless consonants and the per-block correction jitter is audible as pops (S82). 0 = auto.
int utai_stretch_exact(const float* input, int in_samples, int channels, float sample_rate,
                       double time_factor, double transpose_semitones,
                       double formant_semitones, int formant_active, double formant_base_hz,
                       float* output, int out_samples) {
    if (!input || !output || in_samples <= 0 || out_samples <= 0 || channels <= 0 ||
        sample_rate <= 0.0f || !(time_factor > 0.0) || !std::isfinite(transpose_semitones) ||
        !std::isfinite(formant_semitones) || !std::isfinite(formant_base_hz)) {
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
            if (formant_base_hz > 0.0) {
                // setFormantBase expects frequency normalized to the SAMPLE RATE
                // (stft.h freqToBin(f) = f * fftSamples), same axis as the tonality limit.
                stretch.setFormantBase((float)(formant_base_hz / sample_rate));
            }
        }

        // Front-pad short clips so exact() never hits its too-short bail-out (see file header).
        // The seek length depends on the playback rate exact() derives from the TOTAL sizes, and
        // for tiny inputs out_samples = round(in*tf) deviates measurably from tf (64 × 1.4 → 90
        // is a 1.40625 ratio) — so iterate the pad against the actual padded totals until the
        // requirement is met (converges in a step or two; +2 covers float→int truncation).
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
        std::vector<float*> in_ptrs((size_t)channels), out_ptrs((size_t)channels);
        for (int c = 0; c < channels; ++c) {
            in_ptrs[(size_t)c] = in_ch[(size_t)c].data();
            out_ptrs[(size_t)c] = out_ch[(size_t)c].data();
        }

        if (!stretch.exact(in_ptrs.data(), total_in, out_ptrs.data(), total_out)) {
            return 3; // input still shorter than the seek length — unreachable after padding
        }

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
