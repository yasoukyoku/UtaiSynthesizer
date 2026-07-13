# UtaiSynthesizer

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

[![Release](https://img.shields.io/github/v/release/yasoukyoku/UtaiSynthesizer)](https://github.com/yasoukyoku/UtaiSynthesizer/releases)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
![Platform: Windows](https://img.shields.io/badge/platform-Windows%2010%2F11-informational)
[![QQ Group](https://img.shields.io/badge/QQ-1058227212-1EBAFC)](https://qun.qq.com/universal-share/share?ac=1&authKey=3uD5AoM8e50y00vhOYOZsa2VI341dBNfr07S2IK9wraewz0rcFHpSzONYJ9QrTP7&busi_data=eyJncm91cENvZGUiOiIxMDU4MjI3MjEyIiwidG9rZW4iOiJONGpqQ2MzM3h3N3BDMVBMRzZiSUFOU05YWnRnbHBxdTZDUElZYlZOSGN3VnhCaEc5eWludlJBYlltK3hkdlFwIiwidWluIjoiMjc2Njc2NDM1NSJ9&data=VyWCaG06iaMLBFcfEx_fjE2Tme2X7YvJsUIUjJ51zk6XymaED6Z6TEC_zOvAdm9q2MbzbYbpuO4ukQHZ1GBHLw&svctype=4&tempid=h5_group_info)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2)](https://discord.com/invite/p3fGh942fJ)

A singing-voice synthesis DAW for Windows. Write notes on a piano roll and hear them **sung
directly by an SVC voice model** (score → [Score2ConVec](https://github.com/yasoukyoku/Score2ConVec)
→ SVC decode — no intermediate human vocal needed), render AI covers with a per-clip node
workflow, separate vocals from songs natively, and train your own voice models on-device —
all in one app, with no Python required at inference time.

![UtaiSynthesizer — arrangement view](docs/images/overview-arrangement.png)

## Features

- **Piano-roll vocal synthesis** — UTAU-style free note placement, direct lyric typing with
  phrase distribution, SynthV-style pitch transitions / vibrato / hand-drawn pitch deviation,
  loudness & formant parameter lanes, breath notes, and dictionary-backed lyrics in
  **7 languages** (zh / en / ja / de / fr / es / it) with three-level OOV warnings.
- **AI cover workflow** — every audio clip carries a node graph: source separation
  (BS-Roformer, MelBand Roformer, MDX23C, HTDemucs, VR, legacy MDX-Net — all reimplemented
  natively in Rust) → RVC / So-VITS-SVC 4.0 & 4.1 voice conversion (shallow diffusion,
  NSF-HiFiGAN enhancer/vocoders, multi-speaker blending) → spectral transpose → back onto
  the track as non-destructive sub-lanes.
- **On-device training** — RVC, SoVITS 4.1/4.0, shallow diffusion, and vocoder fine-tuning
  on embedded portable Python runtimes (NVIDIA / AMD / Intel / CPU packs, downloaded in-app;
  self-contained CUDA runtime — no CUDA Toolkit install needed).
- **DAW fundamentals** — multi-track timeline, drag-drop import with smart placement,
  crossfades, clip/track clipboard, BPM & beat-grid detection, pitch-preserving time-stretch
  (Signalsmith), loudness envelopes, minimap, full undo/redo.
- **Vocal-range tooling** — automatic range testing of voice models, comfort-zone tuning, and
  opt-in range extension (out-of-range phrases are synthesized inside the model's comfort zone
  and shifted back with TD-PSOLA).
- **Vocal-to-MIDI** — transcribe a separated vocal stem into editable notes (GAME engine).
- Exports: audio (wav / flac / mp3 / ogg / opus / m4a) with what-you-hear offline mixdown
  parity, and scores (ust / ustx / midi). Imports: ustx / ust / midi and 9 audio formats.
- Trilingual UI (简体中文 / English / 日本語), sha256-verified downloads with CN-friendly
  mirror options, minisign-signed auto-updates.

**Just want to use the app?** Grab the installer from
[Releases](https://github.com/yasoukyoku/UtaiSynthesizer/releases) and read the
**[User Guide](docs/user-guide.en.md)** — no development setup needed. The installed
directory is fully portable (copy it anywhere, data travels with it).

## Development setup

The app is a [Tauri 2](https://tauri.app) project: React frontend + Rust backend in one
process. A fresh clone deliberately excludes every large binary asset, so a few pieces must
be placed manually before `tauri dev` is fully functional.

### Prerequisites

| Tool | Version | Notes |
| --- | --- | --- |
| Windows | 10 / 11 x64 | WebView2 runtime (preinstalled on Win 11) |
| Node.js | 18+ (20 LTS recommended) | npm 7+ (lockfile v3) |
| Rust | 1.77+ stable, **MSVC** toolchain | `rustup default stable-x86_64-pc-windows-msvc` |

The C++ time-stretch crate builds with the `cc` crate via MSVC — no libclang/bindgen needed.

### Assets a fresh clone is missing

| Path | What | Where to get it |
| --- | --- | --- |
| `bin/ffmpeg.exe` | decode fallback + all non-wav export encoding | [gyan.dev "essentials"](https://www.gyan.dev/ffmpeg/builds/) GPL build (mirrored at [GyanD/codexffmpeg](https://github.com/GyanD/codexffmpeg/releases)); must include `libmp3lame/libvorbis/libopus/aac/flac` encoders |
| `runtime/ort/*.dll` | ONNX Runtime **1.24.4 DirectML build** (`onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `DirectML.dll`) | [Microsoft.ML.OnnxRuntime.DirectML 1.24.4](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.24.4) NuGet package (+ the DirectML redistributable DLL). The version must match the `ort` crate's API level — mismatched DLLs deadlock at init |
| `data/dictionaries/*.tsv` | G2P dictionaries (8 files) for zh/en/de/fr/es/it lyrics | currently easiest: install a release build and copy its `data\dictionaries` folder (a build-from-source script is a known gap) |
| `data/models/` | voice/separation/aux models | downloaded in-app (Settings → Model Assets, Resource Manager) |
| `converter/.venv` | Python env for importing `.pth` voice models | optional — the in-app "CPU runtime" pack (Settings → Training Runtime) substitutes |

Everything else (icons, installer art, vendored C++/Python) is tracked in the repo.

### Run / test / build

```powershell
npm install
npm run tauri dev      # full app in a real window (Vite on :1420 + Rust backend)

npm run build          # frontend gate: tsc -b && vite build
npm test               # vitest suite
cd src-tauri; cargo test   # Rust suite (heavier E2E tests are #[ignore]d)

pwsh -File scripts/release.ps1          # gated, signed installer build (see below)
pwsh -File scripts/verify-install.ps1   # 39-point audit of an installed tree
```

Notes:

- In debug builds the app pins its data root to the repo (`<repo>/data`), so dev data never
  mixes with an installed copy.
- `scripts/release.ps1` enforces version sync across `package.json` / `Cargo.toml` /
  `tauri.conf.json`, strict semver, full gates (tsc / vitest / cargo test), and minisign
  update signing. The signing key is **not** in the repo — forks can build unsigned local
  bundles with `npm run tauri build`, but cannot publish updates to existing installs.
- Dev-loop pitfalls, subsystem design notes and verification playbooks live in code comments
  near the relevant modules — read them before touching render lifecycle, undo, or ORT
  session code.

## Architecture

| Layer | Tech |
| --- | --- |
| Shell | Tauri 2 + system WebView2 (no bundled browser) |
| Frontend | React 19 + TypeScript + zustand + Canvas 2D (piano roll / arrangement are OFF-React canvases), @xyflow/react for the node editor |
| Backend | one Rust process; all DSP + inference in-process |
| Inference | ONNX Runtime via `ort` (load-dynamic): **DirectML** shipped by default, optional self-contained **CUDA** runtime downloaded in-app, CPU fallback |
| Training | vendored training ports (RVC / so-vits-svc / SingingVocoders) executed on embedded portable Python runtime packs |

**Vocal synthesis chain** (the piano-roll path):

```
score (notes + lyrics)
  → Rust two-stage G2P     (per-language dictionaries → shared IPA phoneme vocab)
  → Score2ConVec (ONNX)    (phonemes + parameterized f0 → ContentVec-space content vectors)
  → SVC decode (ONNX)      (SoVITS 4.0/4.1 or RVC net_g; optional shallow diffusion + NSF-HiFiGAN)
  → audio on the track
```

[**Score2ConVec**](https://github.com/yasoukyoku/Score2ConVec) is the score-to-content-vector
model that makes this possible: it maps a symbolic score directly into the ContentVec feature
space that SVC models consume, so any ordinary SVC voice model can *sing a score* without a
human guide vocal. It was trained specifically for this project; the repo hosts the model,
training code and details.

Cover mode uses the same SVC decoders with ContentVec as the feature extractor and RMVPE for
pitch. Source separation is a native Rust reimplementation of the MSST/UVR model families
driven by per-model config JSON produced by the Python converter.

Repo map (top level): `src/` frontend · `src-tauri/` Rust backend (`crates/utai-dsp` DSP hot
loops, `crates/utai-stretch` vendored Signalsmith Stretch) · `converter/` Python ONNX export
scripts · `training/` vendored training package + runtime-pack builder · `scripts/` release
tooling · `docs/` user guides.

## Responsible use & disclaimer

UtaiSynthesizer is a creative tool. **You are solely responsible for what you make with it.**

- **Voice rights.** Only train or use voice models when you have the right to do so: your own
  voice, a consenting person's voice, or characters/datasets whose license permits it. Do not
  impersonate real people, and clearly label AI-generated vocals as such where confusion is
  possible.
- **Song rights.** Covers and derived audio of copyrighted songs may require permission from
  the rights holders, especially for monetized or public distribution. Follow the law and
  platform rules that apply to you.
- **No models included.** This repository and the official installer contain **no singer
  voice models**. Auxiliary weights the app can download (e.g. the community NSF-HiFiGAN
  vocoder, the GAME vocal-to-MIDI weights) carry their own licenses — several are
  **CC BY-NC-SA (non-commercial)** — and the app shows/ships those notices with the files.
  Voice models you import or train are governed by the licenses/permissions of their source
  data.
- The developers do not endorse, and accept no liability for, any use of this software that
  infringes voice rights, copyright, or applicable law. See [NOTICE.md](NOTICE.md) for
  third-party attributions.

## Community

- **QQ group**: [1058227212](https://qun.qq.com/universal-share/share?ac=1&authKey=3uD5AoM8e50y00vhOYOZsa2VI341dBNfr07S2IK9wraewz0rcFHpSzONYJ9QrTP7&busi_data=eyJncm91cENvZGUiOiIxMDU4MjI3MjEyIiwidG9rZW4iOiJONGpqQ2MzM3h3N3BDMVBMRzZiSUFOU05YWnRnbHBxdTZDUElZYlZOSGN3VnhCaEc5eWludlJBYlltK3hkdlFwIiwidWluIjoiMjc2Njc2NDM1NSJ9&data=VyWCaG06iaMLBFcfEx_fjE2Tme2X7YvJsUIUjJ51zk6XymaED6Z6TEC_zOvAdm9q2MbzbYbpuO4ukQHZ1GBHLw&svctype=4&tempid=h5_group_info)
- **Discord**: <https://discord.com/invite/p3fGh942fJ>
- **Bugs / feature requests**: [GitHub Issues](https://github.com/yasoukyoku/UtaiSynthesizer/issues)
  (please attach app version + logs — see the in-app Log page)
- Security issues: see [SECURITY.md](SECURITY.md)

## License

[AGPL-3.0](LICENSE). The repository vendors AGPL-3.0 code from
[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) (which is why the whole project
is AGPL) alongside MIT components — full third-party attributions in [NOTICE.md](NOTICE.md).
