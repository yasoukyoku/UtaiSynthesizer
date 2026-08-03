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

- **Signalsmith Stretch** (v1.3.2) and **signalsmith-linear** (0.3.1) (Signalsmith Audio) — MIT
  vendored at `src-tauri/crates/utai-stretch/vendor/signalsmith-stretch/` (LICENSE.txt files and
  VENDOR.md provenance included in-tree). Time-stretch / pitch-shift / formant engine.

## Implementation references (no code vendored)

- **OpenUTAU** — MIT — https://github.com/stakira/OpenUtau — ustx/ust score format reference.
- **Music-Source-Separation-Training** (ZFTurbo) and **Ultimate Vocal Remover** — separation
  model architectures reimplemented natively in Rust; model weights are downloaded by the user
  in-app from their original distribution points and are governed by their own licenses.
- **ContentVec** (auspicious3000, MIT) and **RMVPE** — feature-extraction / pitch models exported
  to ONNX for the in-app downloader.

## Models trained by this project (downloaded on demand, NEVER bundled)

These are **our own** weights — not derived from anyone else's checkpoint — but they were trained on
third-party singing corpora, and those corpora's terms carry over to the trained weights.

- **ScoreToCV** — `score2cv_768.onnx` / `score2cv_256.onnx` (core inference pack).
- **Automatic pitch tuning** — `autotune_a1.onnx` (optional `autotune` pack).

Both were trained on the same corpus set (44,947 clips; `train_final` / `val_final`):

| Corpus | Share | License |
|---|---|---|
| GTSinger (English direct; French/German/Italian/Spanish re-aligned) | 49.5% | **CC BY-NC-SA 4.0** — https://github.com/AaronZ345/GTSinger |
| M4Singer (Chinese) | 44.5% | **CC BY-NC-SA 4.0** — https://github.com/M4Singer/M4Singer |
| Namine Ritsu "Kiritan" song DB / Tohoku Itako song DB (SSS LLC) | 1.2% / 1.1% | Non-commercial use permitted; https://zunko.jp/ |
| Natsume Yuuri song DB (ATSUYA) · Ofuton P song DB · Oniku Kurumi "Utagoe" DB | 1.0% / 1.0% / 1.4% | Non-commercial; redistribution of derived voice models requires prior permission from each author |
| PJS: Phoneme-balanced Japanese Singing-voice corpus | 0.4% | **CC BY-SA 4.0** (commercial use permitted) |

**Consequence:** 93.96% of the training set is NonCommercial-licensed, so these two models are
distributed for **non-commercial use only**, with attribution to the corpora above and under
ShareAlike terms inherited from the CC BY-NC-SA sets. This is the same pattern as the NSF-HiFiGAN
weights below. The application source (see `LICENSE`, AGPL-3.0) is a separate matter and does not
grant any rights over these weights, nor do these corpus terms restrict the source code.

Corpora that appear in intermediate data but are **not** in the shipped models' training set
(`ace_opencpop`, `PopCS`, `CSD`, `NUS-48E`) are listed here only to record that they were excluded.

## Third-party model weights (downloaded on demand, NEVER bundled)

No model weights ship inside the installer. The app downloads them on demand, and weights that
carry terms of their own are shown with those terms before the download starts.

- **NSF-HiFiGAN** vocoder weights (OpenVPI) — CC BY-NC-SA 4.0. Two distinct artifacts: the
  inference vocoder (an ONNX export derived from those weights, part of the core inference pack)
  and the fine-tuning base checkpoint (its own `training-vocoder` pack). The original
  NOTICE.txt / NOTICE.zh-CN.txt are downloaded together with the weights and stay beside them on
  disk. A vocoder fine-tuned from the base inherits the same license (non-commercial).
- **GAME** vocal-to-MIDI weights — CC BY-NC-SA. Primary source is the upstream release; the
  license is shown at download time.
- Separation / voice model weights fetched through the in-app downloaders keep their upstream
  licenses and come from their original distribution points.

Where this project mirrors third-party weights for availability (currently the CC BY-NC-SA sets
above, hosted at `huggingface.co/datasets/yasoukyoku/utai-runtimes`), it does so as those licenses
permit — attribution preserved, same terms, non-commercial — and the mirroring conveys no
additional rights. The app's own source license (see LICENSE) covers the application code only,
never the third-party weights it fetches.

## Bundled runtime redistributables

- **ONNX Runtime** (Microsoft, MIT) — `runtime/ort/onnxruntime*.dll`.
- **DirectML** (Microsoft) — `runtime/ort/DirectML.dll`, redistributed under the Microsoft
  DirectML redistributable license (shipped because the Windows inbox copy is older than what
  ONNX Runtime requires).
- **FFmpeg** — `ffmpeg.exe`, invoked as a separate process for audio decode/encode. The shipped
  binary is a **BtbN** win64 GPL build (currently `n8.1.2-34-g9b6c8969e0-20260801`, configured with
  `--enable-gpl --enable-version3` ⇒ **GPL-3.0**). Source: https://ffmpeg.org / builds:
  https://github.com/BtbN/FFmpeg-Builds.

## Bundled dictionary data (`data/dictionaries/`)

Compiled pronunciation dictionaries derived from: CMUdict (BSD-2-Clause), Montreal Forced
Aligner community dictionaries (CC BY 4.0), pinyin-data / phrase-pinyin-data (MIT / CC),
opencpop-extension (Apache-2.0). Detailed per-file attribution ships with the full documentation.
