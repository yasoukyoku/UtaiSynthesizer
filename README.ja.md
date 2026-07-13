# UtaiSynthesizer

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

[![Release](https://img.shields.io/github/v/release/yasoukyoku/UtaiSynthesizer)](https://github.com/yasoukyoku/UtaiSynthesizer/releases)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
![Platform: Windows](https://img.shields.io/badge/platform-Windows%2010%2F11-informational)
[![QQ グループ](https://img.shields.io/badge/QQ-1058227212-1EBAFC)](https://qun.qq.com/universal-share/share?ac=1&authKey=3uD5AoM8e50y00vhOYOZsa2VI341dBNfr07S2IK9wraewz0rcFHpSzONYJ9QrTP7&busi_data=eyJncm91cENvZGUiOiIxMDU4MjI3MjEyIiwidG9rZW4iOiJONGpqQ2MzM3h3N3BDMVBMRzZiSUFOU05YWnRnbHBxdTZDUElZYlZOSGN3VnhCaEc5eWludlJBYlltK3hkdlFwIiwidWluIjoiMjc2Njc2NDM1NSJ9&data=VyWCaG06iaMLBFcfEx_fjE2Tme2X7YvJsUIUjJ51zk6XymaED6Z6TEC_zOvAdm9q2MbzbYbpuO4ukQHZ1GBHLw&svctype=4&tempid=h5_group_info)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2)](https://discord.com/invite/p3fGh942fJ)

Windows 向けの歌声合成 DAW。ピアノロールに書いた譜面を **SVC ボイスモデルが直接歌います**
(譜面 → [Score2ConVec](https://github.com/yasoukyoku/Score2ConVec) → SVC デコード。
ガイドとなる人間の歌声は不要)。クリップごとのノードワークフローで AI カバーを
レンダリングし、ボーカル抽出もネイティブ実装、自分のボイスモデルのオンデバイス学習まで
——すべてを 1 つのアプリで。推論時に Python は一切不要です。

![UtaiSynthesizer——アレンジビュー](docs/images/overview-arrangement.png)

## 機能

- **ピアノロール歌声合成**——UTAU 流の自由なノート配置、ノートへ直接歌詞入力
  (フレーズ一括割り当て対応)、SynthV 流のピッチ遷移/ビブラート/手描きピッチ、
  ラウドネス&フォルマントのパラメータレーン、ブレスノート、そして辞書ベースの
  **7 言語**歌詞(中/英/日/独/仏/西/伊)+ 3 段階の OOV(発音不能)警告。
- **AI カバーワークフロー**——各オーディオクリップがノードグラフを持ちます:
  ボーカル分離(BS-Roformer、MelBand Roformer、MDX23C、HTDemucs、VR、旧 MDX-Net——
  すべて Rust でネイティブ再実装)→ RVC / So-VITS-SVC 4.0・4.1 の声変換
  (浅い拡散、NSF-HiFiGAN エンハンサー/ボコーダー、複数話者ブレンド)→
  スペクトル移調 → 非破壊のサブレーンとしてトラックへ配置。
- **オンデバイス学習**——RVC、SoVITS 4.1/4.0、浅い拡散、ボコーダー微調整。
  組み込みポータブル Python ランタイム(NVIDIA / AMD / Intel / CPU の 4 種、アプリ内
  ダウンロード)で動作。CUDA ランタイムは完全自己完結——**CUDA Toolkit 不要**。
- **DAW の基本**——マルチトラックタイムライン、スマート配置つきドラッグ&ドロップ
  読み込み、クロスフェード、クリップ/トラックのクリップボード、BPM・拍グリッド検出、
  ピッチ保持タイムストレッチ(Signalsmith)、ラウドネスエンベロープ、ミニマップ、
  完全なアンドゥ/リドゥ。
- **音域ツール**——ボイスモデルの自動音域テスト、快適音域の微調整、オプトインの
  音域拡張(音域外のフレーズをモデルの快適音域内で合成し、TD-PSOLA で元のキーへ
  戻します)。
- **歌声→MIDI**——分離したボーカルのサブレーンを編集可能なノートに書き起こし
  (GAME エンジン)。
- 書き出し:オーディオ(wav / flac / mp3 / ogg / opus / m4a、再生と完全一致の
  オフラインミックスダウン)、譜面(ust / ustx / midi)。読み込み:ustx / ust / midi と
  9 種のオーディオ形式。
- 3 言語 UI(简体中文 / English / 日本語)、sha256 検証つきダウンロードとミラー設定、
  minisign 署名つき自動アップデート。

**アプリを使いたいだけの方へ**:[Releases](https://github.com/yasoukyoku/UtaiSynthesizer/releases)
からインストーラをダウンロードし、**[ユーザーガイド](docs/user-guide.ja.md)** をどうぞ。
開発環境は不要です。インストール先フォルダは丸ごとコピーすればポータブル版として
動きます(データもフォルダに付いて移動します)。

## 開発環境のセットアップ

本プロジェクトは [Tauri 2](https://tauri.app) アプリ(React フロントエンド + Rust
バックエンドの単一プロセス)です。大きなバイナリ資産は意図的に git 管理外のため、
新規 clone では以下をいくつか手動で配置すると `tauri dev` が完全に機能します。

### 前提条件

| ツール | バージョン | 備考 |
| --- | --- | --- |
| Windows | 10 / 11 x64 | WebView2 ランタイム(Win 11 は標準搭載) |
| Node.js | 18+(20 LTS 推奨) | npm 7+(lockfile v3) |
| Rust | 1.77+ stable、**MSVC** ツールチェーン | `rustup default stable-x86_64-pc-windows-msvc` |

C++ タイムストレッチは `cc` crate + MSVC でビルドされ、libclang/bindgen は不要です。

### 新規 clone に無い資産

| パス | 内容 | 入手先 |
| --- | --- | --- |
| `bin/ffmpeg.exe` | デコードのフォールバック + wav 以外の全書き出しエンコード | [gyan.dev "essentials"](https://www.gyan.dev/ffmpeg/builds/) GPL ビルド(ミラー:[GyanD/codexffmpeg](https://github.com/GyanD/codexffmpeg/releases))。`libmp3lame/libvorbis/libopus/aac/flac` エンコーダ必須 |
| `runtime/ort/*.dll` | ONNX Runtime **1.24.4 DirectML ビルド**(`onnxruntime.dll`、`onnxruntime_providers_shared.dll`、`DirectML.dll`) | NuGet [Microsoft.ML.OnnxRuntime.DirectML 1.24.4](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.24.4)(+ DirectML 再頒布 DLL)。`ort` crate の API レベルと一致必須——不一致は初期化デッドロック |
| `data/dictionaries/*.tsv` | 中/英/独/仏/西/伊歌詞用 G2P 辞書(8 ファイル) | 現状はリリース版をインストールし、その `data\dictionaries` をコピーするのが最短(ソースからのビルドスクリプトは既知の未整備) |
| `data/models/` | ボイス/分離/補助モデル | アプリ内ダウンロード(設定 → モデルアセット、リソース管理) |
| `converter/.venv` | `.pth` ボイスモデル取り込み用 Python 環境 | 任意——アプリ内「CPU ランタイム」パック(設定 → トレーニング環境)で代替可 |

それ以外(アイコン、インストーラ画像、vendored C++/Python)はリポジトリに含まれます。

### 実行 / テスト / ビルド

```powershell
npm install
npm run tauri dev      # 実ウィンドウでフル起動(Vite :1420 + Rust バックエンド)

npm run build          # フロントエンドゲート:tsc -b && vite build
npm test               # vitest スイート
cd src-tauri; cargo test   # Rust スイート(重い E2E は #[ignore])

pwsh -File scripts/release.ps1          # ゲートつき署名インストーラビルド(下記参照)
pwsh -File scripts/verify-install.ps1   # インストール済みツリーの 39 項目監査
```

補足:

- debug ビルドはデータルートをリポジトリ直下(`<repo>/data`)に固定するため、開発データが
  インストール版と混ざることはありません。
- `scripts/release.ps1` は 3 箇所のバージョン一致(`package.json` / `Cargo.toml` /
  `tauri.conf.json`)、厳格 semver、全ゲート(tsc / vitest / cargo test)、minisign
  署名を強制します。署名鍵はリポジトリに**ありません**——fork は `npm run tauri build`
  で未署名のローカルバンドルをビルドできますが、既存インストールへの更新配信は
  できません。
- 開発時の落とし穴やサブシステム設計メモは該当モジュール付近のコードコメントに
  あります——レンダリングライフサイクル、undo、ORT セッション周りを触る前に一読を。

## アーキテクチャ

| レイヤ | 技術 |
| --- | --- |
| シェル | Tauri 2 + システム WebView2(ブラウザ同梱なし) |
| フロントエンド | React 19 + TypeScript + zustand + Canvas 2D(ピアノロール/アレンジは React 外キャンバス)、ノードエディタは @xyflow/react |
| バックエンド | 単一 Rust プロセス;DSP・推論はすべてプロセス内 |
| 推論 | `ort`(load-dynamic)経由の ONNX Runtime:標準同梱は **DirectML**、任意でアプリ内ダウンロードの自己完結 **CUDA** ランタイム、CPU フォールバック |
| 学習 | vendored した学習コード移植(RVC / so-vits-svc / SingingVocoders)を組み込みポータブル Python ランタイムパックで実行 |

**歌声合成チェーン**(ピアノロール経路):

```
譜面(ノート + 歌詞)
  → Rust 二段 G2P            (言語別辞書 → 共有 IPA 音素表)
  → Score2ConVec(ONNX)      (音素 + パラメトリック f0 → ContentVec 空間のコンテンツベクトル)
  → SVC デコード(ONNX)      (SoVITS 4.0/4.1 または RVC net_g;任意で浅い拡散 + NSF-HiFiGAN)
  → トラック上のオーディオ
```

[**Score2ConVec**](https://github.com/yasoukyoku/Score2ConVec) は、この仕組みを可能にする
「譜面→コンテンツベクトル」モデルです。シンボリックな譜面を SVC モデルが消費する
ContentVec 特徴空間へ直接写像するため、ごく普通の SVC ボイスモデルが人間のガイドなしで
*譜面どおりに歌えます*。本プロジェクトのために学習されたモデルで、モデル・学習コード・
詳細は同リポジトリにあります。

カバーモードは同じ SVC デコーダを使い、特徴抽出に ContentVec、ピッチに RMVPE を
使用します。ボーカル分離は MSST/UVR モデルファミリーの Rust ネイティブ再実装で、
Python コンバータが生成するモデル別 JSON 設定で駆動されます。

リポジトリ構成(トップレベル):`src/` フロントエンド · `src-tauri/` Rust バックエンド
(`crates/utai-dsp` DSP ホットループ、`crates/utai-stretch` vendored Signalsmith Stretch)·
`converter/` Python ONNX エクスポート · `training/` vendored 学習パッケージ +
ランタイムパックビルダー · `scripts/` リリースツール · `docs/` ユーザーガイド。

## 責任ある利用と免責事項

UtaiSynthesizer は創作ツールです。**制作物への責任はすべて利用者にあります。**

- **声の権利。** ボイスモデルの学習・使用は、権利がある場合に限ってください:
  自分自身の声、明確に同意した人の声、あるいはライセンスが許諾する
  キャラクター/データセット。実在の人物へのなりすましは禁止です。混同のおそれが
  ある場面では、AI 生成の歌声であることを明示してください。
- **楽曲の権利。** 著作権のある楽曲のカバーや派生音源は、特に収益化・公開配布の際、
  権利者の許諾が必要な場合があります。お住まいの地域の法律と各プラットフォームの
  規約に従ってください。
- **モデル同梱なし。** 本リポジトリと公式インストーラには**歌手ボイスモデルを一切
  同梱していません**。アプリがダウンロードできる補助ウェイト(コミュニティ版
  NSF-HiFiGAN ボコーダー、GAME 歌声→MIDI ウェイトなど)にはそれぞれのライセンスが
  あり——一部は **CC BY-NC-SA(非商用)** です——アプリは該当ファイルとともに
  その通知を表示・同梱します。取り込み/学習したボイスモデルは、その元データの
  ライセンスと許諾に従います。
- 声の権利・著作権・適用法に反する本ソフトウェアの利用を、開発者は容認せず、
  一切の責任を負いません。サードパーティの帰属表示は [NOTICE.md](NOTICE.md) を
  参照してください。

## コミュニティ

- **QQ グループ**:[1058227212](https://qun.qq.com/universal-share/share?ac=1&authKey=3uD5AoM8e50y00vhOYOZsa2VI341dBNfr07S2IK9wraewz0rcFHpSzONYJ9QrTP7&busi_data=eyJncm91cENvZGUiOiIxMDU4MjI3MjEyIiwidG9rZW4iOiJONGpqQ2MzM3h3N3BDMVBMRzZiSUFOU05YWnRnbHBxdTZDUElZYlZOSGN3VnhCaEc5eWludlJBYlltK3hkdlFwIiwidWluIjoiMjc2Njc2NDM1NSJ9&data=VyWCaG06iaMLBFcfEx_fjE2Tme2X7YvJsUIUjJ51zk6XymaED6Z6TEC_zOvAdm9q2MbzbYbpuO4ukQHZ1GBHLw&svctype=4&tempid=h5_group_info)
- **Discord**:<https://discord.com/invite/p3fGh942fJ>
- **バグ報告 / 機能要望**:[GitHub Issues](https://github.com/yasoukyoku/UtaiSynthesizer/issues)
  (アプリのバージョンとログを添付してください——アプリ内「ログ」ページ参照)
- セキュリティ関連:[SECURITY.md](SECURITY.md)

## ライセンス

[AGPL-3.0](LICENSE)。本リポジトリは
[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) の AGPL-3.0 コードを
vendored しています(プロジェクト全体が AGPL である理由)。MIT コンポーネントも
含まれます——サードパーティ帰属の全文は [NOTICE.md](NOTICE.md) を参照。
