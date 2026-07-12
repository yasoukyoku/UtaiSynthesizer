# Third-Party Notices / 第三方声明

UtaiSynthesizer is licensed under the GNU Affero General Public License v3.0 (see `LICENSE`).
It contains code ported from, or written with reference to, the following projects.
(完整致谢与详细清单将随正式文档完善;本文件为分发所需的许可声明。)

## Vendored / ported code in this repository

- **so-vits-svc** (svc-develop-team) — AGPL-3.0
  https://github.com/svc-develop-team/so-vits-svc
  Vendored training port (`training/utai_train/sovits/`), diffusion training, converter export
  architectures (`converter/architectures/sovits_*.py`, `nsf_hifigan_gen.py`), and the reference
  for the Rust inference reimplementation. (This dependency is why the whole repository is AGPL-3.0.)

- **Retrieval-based-Voice-Conversion-WebUI** (RVC-Project) — MIT
  https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
  Vendored training port (`training/utai_train/rvc/`) and converter export architecture
  (`converter/architectures/rvc_v2.py`); reference for the Rust RVC inference.

- **SingingVocoders** (openvpi) — MIT
  https://github.com/openvpi/SingingVocoders
  Vendored vocoder fine-tuning port (`training/utai_train/vocoder/`), including
  `modules/loss/stft_loss.py` (Copyright 2019 Tomoki Hayashi, MIT).

- **Signalsmith Stretch** and **signalsmith-dsp** (Signalsmith Audio) — MIT
  vendored at `src-tauri/crates/utai-stretch/vendor/signalsmith-stretch/` (LICENSE.txt included
  in-tree). Time-stretch / pitch-shift engine.

## Implementation references (no code vendored)

- **OpenUTAU** — MIT — https://github.com/stakira/OpenUtau — ustx/ust score format reference.
- **Music-Source-Separation-Training** (ZFTurbo) and **Ultimate Vocal Remover** — separation
  model architectures reimplemented natively in Rust; model weights are downloaded by the user
  in-app from their original distribution points and are governed by their own licenses.
- **ContentVec** (auspicious3000, MIT) and **RMVPE** — feature-extraction / pitch models exported
  to ONNX for the in-app downloader.

## Model weights (downloaded by the user, NOT bundled)

- **NSF-HiFiGAN** vocoder weights (OpenVPI) — CC BY-NC-SA 4.0. Never bundled; the original
  NOTICE.txt / NOTICE.zh-CN.txt accompany the files wherever the app uses them.
- **GAME** vocal-to-MIDI weights — CC BY-NC-SA. Never bundled; downloaded from the original
  release with the license shown at download time.
- Separation / voice model weights fetched through the in-app downloaders keep their upstream
  licenses; the app stores them locally for the user and does not redistribute them.

## Bundled runtime redistributables

- **ONNX Runtime** (Microsoft, MIT) — `runtime/ort/onnxruntime*.dll`.
- **DirectML** (Microsoft) — `runtime/ort/DirectML.dll`, redistributed under the Microsoft
  DirectML redistributable license (shipped because the Windows inbox copy is older than what
  ONNX Runtime requires).
- **FFmpeg** (GPL build, gyan.dev "essentials") — `ffmpeg.exe`, invoked as a separate process for
  audio decode/encode. Source: https://ffmpeg.org / builds: https://www.gyan.dev/ffmpeg/builds/.

## Bundled dictionary data (`data/dictionaries/`)

Compiled pronunciation dictionaries derived from: CMUdict (BSD-2-Clause), Montreal Forced
Aligner community dictionaries (CC BY 4.0), pinyin-data / phrase-pinyin-data (MIT / CC),
opencpop-extension (Apache-2.0). Detailed per-file attribution ships with the full documentation.
