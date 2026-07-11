// wrapper.cpp — minimal extern "C" shim over Signalsmith Stretch (vendored, MIT).
//
// One-shot offline exact-length stretch, mirroring the upstream CLI's canonical recipe
// (vendor cmd/main.cpp): seek() a pre-roll of inputLatency samples, process() the whole body,
// flush() an outputLatency tail, then fold the leading latency block back (reversed & negated)
// and skip it — the output then aligns sample-exactly with round(inputSamples * timeFactor).
//
// Interleaved f32 in/out; channels are deinterleaved here because the templated process()
// indexes buffers[channel][sample].

#include "signalsmith-stretch/signalsmith-stretch.h"

#include <algorithm>
#include <cmath>
#include <vector>

extern "C" {

// Returns 0 on success, non-zero on failure. `output` must hold out_samples * channels floats,
// where out_samples = (int)llround(in_samples * time_factor) as computed by the Rust caller.
int utai_stretch_exact(const float* input, int in_samples, int channels, float sample_rate,
                       double time_factor, float* output, int out_samples) {
    if (!input || !output || in_samples <= 0 || out_samples <= 0 || channels <= 0 ||
        sample_rate <= 0.0f || !(time_factor > 0.0)) {
        return 1;
    }
    try {
        signalsmith::stretch::SignalsmithStretch<float> stretch;
        stretch.presetDefault(channels, sample_rate);
        // tempo-only change: transpose stays 1.0 (the default) — pitch is preserved

        const int in_lat = stretch.inputLatency();
        const int out_lat = stretch.outputLatency();

        std::vector<std::vector<float>> in_ch((size_t)channels), out_ch((size_t)channels);
        for (int c = 0; c < channels; ++c) {
            in_ch[(size_t)c].assign((size_t)(in_samples + in_lat), 0.0f);
            out_ch[(size_t)c].assign((size_t)(out_samples + out_lat), 0.0f);
            for (int i = 0; i < in_samples; ++i) {
                in_ch[(size_t)c][(size_t)i] = input[(size_t)i * channels + c];
            }
        }
        std::vector<float*> in_ptrs((size_t)channels), in_body((size_t)channels),
            out_ptrs((size_t)channels), out_tail((size_t)channels);
        for (int c = 0; c < channels; ++c) {
            in_ptrs[(size_t)c] = in_ch[(size_t)c].data();
            in_body[(size_t)c] = in_ch[(size_t)c].data() + in_lat;
            out_ptrs[(size_t)c] = out_ch[(size_t)c].data();
            out_tail[(size_t)c] = out_ch[(size_t)c].data() + out_samples;
        }

        // stay slightly ahead in the input (upstream cmd/main.cpp lines 77-84)
        stretch.seek(in_ptrs.data(), in_lat, 1.0 / time_factor);
        stretch.process(in_body.data(), in_samples, out_ptrs.data(), out_samples);
        stretch.flush(out_tail.data(), out_lat);

        // fold the leading latency block back into the start (reversed in time and negated),
        // then the exact-length output begins at out_lat. BOUND the fold to the valid region:
        // an input shorter than the engine latency yields out_samples < out_lat, and an
        // unbounded loop would write past out_ch's (out_samples + out_lat) end (heap overflow,
        // audit-caught). For normal-length inputs fold == out_lat → behavior unchanged.
        const int fold = std::min(out_lat, out_samples);
        for (int c = 0; c < channels; ++c) {
            float* ch = out_ch[(size_t)c].data();
            for (int i = 0; i < fold; ++i) {
                ch[out_lat + i] -= ch[out_lat - 1 - i];
            }
        }
        for (int i = 0; i < out_samples; ++i) {
            for (int c = 0; c < channels; ++c) {
                output[(size_t)i * channels + c] = out_ch[(size_t)c][(size_t)(out_lat + i)];
            }
        }
        return 0;
    } catch (...) {
        return 2;
    }
}

} // extern "C"
