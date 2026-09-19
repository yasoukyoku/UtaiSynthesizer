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
| Tohoku Kiritan singing DB (東北きりたん歌唱データベース) / Tohoku Itako singing DB (東北イタコ歌唱データベース) — SSS LLC, Meiji University (Masanori Morise) | 1.2% / 1.1% | Non-commercial use under SSS LLC's usage agreement; models built with the data may be incorporated into public software/services only with SSS LLC's prior approval (agreement §3-6); https://zunko.jp/ |
| Natsume Yuuri male singing DB (夏目悠李/男性歌声データベース) — ATSUYA | 1.0% | Non-commercial; distributing models built with the DB requires prior notice to ATSUYA and bundling「夏目悠李の出力音声に関する利用規約」 |
| Ofuton P singing DB (おふとんP歌声データベース) | 1.0% | Non-commercial; distributing voice models built with the DB requires prior inquiry |
| Oniku Kurumi singing DB (御丹宮くるみ歌声データべース) | 1.4% | Credit required; commercial use requires prior inquiry |
| PJS: Phoneme-balanced Japanese Singing-voice corpus (Koguchi & Takamichi); lyrics from the Voice Actor Statistics Corpus (声優統計コーパス, 日本声優統計学会) | 0.4% | **CC BY-SA 4.0** (commercial use permitted) |

Credits required by the Japanese corpora's terms (verbatim):
『©SSS』 · 『歌声DB制作:アマノケイ 音声提供者: 霧野蒼太』 · 『DB制作:おふとんP』 · 『御丹宮くるみ歌声データべース』.

**Audio you generate with these models.** Singing rendered through ScoreToCV / automatic pitch
tuning is output of models trained on the corpora above, so their terms follow it:
- It is the 「出力音声」 (output voice) of the Natsume Yuuri DB and falls under
  「夏目悠李の出力音声に関する利用規約」, published by ATSUYA at
  https://ksdcm1ng.wixsite.com/njksofficial/%E8%A6%8F%E7%B4%84-rules; an unmodified copy is published
  beside the model files on the download mirror (`huggingface.co/datasets/yasoukyoku/utai-runtimes`,
  `models/auxiliary/NATSUME_OUTPUT_VOICE_TERMS.txt`). In particular: commercial use needs
  separate permission, and **generated audio must not be used to build acoustic/pitch models**
  (禁止事項 (8)) — this includes using it as training data in Utai's own training feature.
- Commercial use of the generated singing is not permitted under the corpora's terms.

**Licence position.** We take the position — consistent with Creative Commons' guidance that in many
cases an AI model is not an adaptation of the works it was trained on
(https://creativecommons.org/using-cc-licensed-works-for-ai-training/) — that these trained weights
are not Adapted Material of the CC-licensed corpora above, so the ShareAlike conditions of GTSinger /
M4Singer (CC BY-NC-SA 4.0) and PJS (CC BY-SA 4.0) do not attach to them. The weights are nevertheless
distributed **for non-commercial use only**, with the attribution above: several Japanese corpora's
usage agreements require it, and we respect the NonCommercial terms of the corpora that make up
93.96% of the training set. The application source (see `LICENSE`, AGPL-3.0) is a separate matter
and does not grant any rights over these weights, nor do these corpus terms restrict the source code.

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

Compiled pronunciation dictionaries, one upstream per file:

| File | Upstream | License / attribution |
|---|---|---|
| `en.tsv` | CMUdict (Carnegie Mellon University) | BSD-2-Clause-style; CMU copyright notice retained |
| `de.tsv` · `fr.tsv` · `es.tsv` | Montreal Forced Aligner dictionaries (German/French MFA v3.0.0, Spanish MFA v2.0.0a; McAuliffe & Sonderegger) | declared CC BY 4.0; built largely from WikiPron-scraped Wiktionary transcriptions, so treated as **CC BY-SA 4.0** with attribution to MFA and the Wiktionary contributors |
| `it.tsv` | MFA Italian CV dictionary v2.0.0 (VoxCommunis; Ahn & Chodroff 2022, via Epitran) | CC BY 4.0 (upstream metadata; the model card says CC0) |
| `zh_chars.tsv` · `zh_phrases.tsv` · `zh_syllables.tsv` | pinyin-data v0.15.0 / phrase-pinyin-data (mozillazg), incl. Unicode Unihan `kMandarin` | MIT; Unicode License V3 for the Unihan-derived data |

The build is reproducible from these sources (URLs, sizes and sha256 in the generator's
`dict_sources_manifest.json`); the generator's own deliberate rewrites (merges and primary-reading
overrides that keep the dictionaries consistent with the trained model) are changes to the upstream
data in the sense of CC BY 4.0 / CC BY-SA 4.0.
