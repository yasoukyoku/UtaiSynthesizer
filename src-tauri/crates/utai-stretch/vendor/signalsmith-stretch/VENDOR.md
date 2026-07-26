# Vendored sources — provenance

Both libraries are header-only, MIT licensed (LICENSE.txt files kept in place).

- `signalsmith-stretch.h` + `README.md` + `LICENSE.txt`
  from https://github.com/Signalsmith-Audio/signalsmith-stretch
  main @ `57b93f4e9206a089a45387eaa39bdc9f310d3308` (v1.3.2, 2026-01-24).
  Taken from the repo root (the copy under upstream `include/` is a one-line
  forwarding stub, not the real header).

- `signalsmith-linear/` (`stft.h`, `fft.h`, `README.md`, `LICENSE.txt`)
  from https://github.com/Signalsmith-Audio/linear
  tag `0.3.1` @ `5668673560146a9cfe38c25315071e3fd68c8317` — the exact tag
  signalsmith-stretch v1.3.2 pins in its CMakeLists (do not "upgrade" it
  independently of the stretch header).
  `linear.h` and `platform/` are NOT vendored: the stretch header only pulls
  `stft.h` → `fft.h`, and every `platform/` include sits behind an opt-in
  `SIGNALSMITH_USE_*` macro we never define (plain C++ FFT path).

Upgrading: replace the headers from a matched pair of upstream commits
(stretch release + the linear tag its CMakeLists pins), update the hashes
above, and re-run the utai-stretch crate tests (exact length / pitch /
transpose / formant / short-input contracts).
