# MunoAI

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
![Platform: Windows](https://img.shields.io/badge/platform-Windows%2010%2F11-informational)

**MunoAI(造乐之地 — 「音楽創造の地」)** は Windows 向けのオールインワン AI 音楽
ワークステーションです:歌声合成 DAW、ローカル AI ソングスタジオ、マスタリングを
1 つのアプリに統合。ピアノロールに書いた譜面を **SVC ボイスモデルが直接歌います**
(譜面 → [Score2ConVec](https://github.com/yasoukyoku/Score2ConVec) → SVC デコード。
ガイドとなる人間の歌声は不要)。クリップごとのノードワークフローで AI カバーを
レンダリングし、ボーカル抽出もネイティブ実装、**オンデバイスで曲を丸ごと生成**し、
ミックス後のマスタリングもそのまま。自分のボイスモデルの学習まで——すべてローカルで。
推論時に Python は一切不要です。

![MunoAI——アレンジビュー](docs/images/overview-arrangement.png)

## 機能

- **ピアノロール歌声合成**——UTAU 流の自由なノート配置、ノートへ直接歌詞入力
  (フレーズ一括割り当て対応)、SynthV 流のピッチ遷移/ビブラート/手描きピッチ、
  **常時オンの自動ピッチ調教**(自動追従・手動編集で引き継ぎ)、
  ラウドネス&フォルマントのパラメータレーン、ブレスノート、そして辞書ベースの
  **7 言語**歌詞(中/英/日/独/仏/西/伊)+ 3 段階の OOV(発音不能)警告。
- **AI カバーワークフロー**——各オーディオクリップがノードグラフを持ちます:
  ボーカル分離(BS-Roformer、MelBand Roformer、MDX23C、HTDemucs、VR、旧 MDX-Net——
  すべて Rust でネイティブ再実装)→ RVC / So-VITS-SVC 4.0・4.1 の声変換
  (浅い拡散、NSF-HiFiGAN エンハンサー/ボコーダー、複数話者ブレンド)→
  スペクトル移調 → 非破壊のサブレーンとしてトラックへ配置。
- **ソングスタジオ(歌曲制作)**——曲の丸ごと生成をローカルで完結:内蔵 **GGUF 大規模
  言語モデルエンジン**(Yue2 ほか GGUF モデル、アプリ内ダウンロード)が歌詞・創作を
  アシストし、ローカル音楽生成モデル(ACE-Step / HeartMuLa 系)が曲全体をレンダリング。
  クラウド API は一切不要。
- **オンデバイス学習**——RVC、SoVITS 4.1/4.0/4.0-v2、浅い拡散、ボコーダー微調整。**データセット・
  複数話者構成・アーカイブをプロジェクト単位で管理**。組み込みポータブル Python ランタイム(NVIDIA / AMD / Intel / CPU の 4 種、アプリ内
  ダウンロード)で動作。CUDA ランタイムは完全自己完結——**CUDA Toolkit 不要**。
- **マスタリング&ラウドネススイート**——トゥルーピーク / ラウドネス計測(LUFS)、
  クリップ検出、DC オフセット除去、ステレオ幅・位相相関メーター、そして
  バス EQ・サチュレーション・ディザリングを備えたマスタリングチェーン。
- **SoundFont / MIDI 合成**——FluidSynth ベースの **SF2 / SFZ** 音源再生で
  楽器トラックや伴奏を鳴らし、歌声→MIDI 書き起こし(GAME エンジン)にも対応。
- **DAW の基本**——マルチトラックタイムライン、スマート配置つきドラッグ&ドロップ
  読み込み、クロスフェード、クリップ/トラックのクリップボード、BPM・拍グリッド検出、
  ピッチ保持タイムストレッチ(Signalsmith)、ラウドネスエンベロープ、ミニマップ、
  完全なアンドゥ/リドゥ。
- **音域ツール**——ボイスモデルの自動音域テスト、快適音域の微調整、オプトインの
  音域拡張(音域外のフレーズをモデルの快適音域内で合成し、Signalsmith エンジンで
  元のキーへ戻します)。
- **コンプライアンスガード**——書き出し前に、権利侵害の可能性がある・ラベル不備の
  AI 生成コンテンツを検出して警告。公開するものを常にコントロールできます。
- 書き出し:オーディオ(wav / flac / mp3 / ogg / opus / m4a、再生と完全一致の
  オフラインミックスダウン)、譜面(ust / ustx / midi)、**ボイスモデルパッケージ(.zip、別マシンへ
  移して無損で再取り込み)**。読み込み:ustx / ust / midi、9 種のオーディオ形式、モデルパッケージ。
- 3 言語 UI(简体中文 / English / 日本語)、sha256 検証つきダウンロードとミラー設定、
  minisign 署名つき自動アップデート。

**アプリを使いたいだけの方へ**:[Releases](https://github.com/junziai/munoai/releases)
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

C++ タイムストレッチクレートは `cc` クレート経由で MSVC ビルドします。libclang/bindgen は不要です。

### 新規 clone で不足しているアセット

| パス | 内容 | 入手先 |
| --- | --- | --- |
| `bin/ffmpeg.exe` | デコードフォールバック + 非 wav 書き出しの全エンコード | [gyan.dev "essentials"](https://www.gyan.dev/ffmpeg/builds/) GPL ビルド(ミラー:[GyanD/codexffmpeg](https://github.com/GyanD/codexffmpeg/releases));`libmp3lame/libvorbis/libopus/aac/flac` エンコーダ必須 |
| `runtime/ort/*.dll` | ONNX Runtime **1.24.4 DirectML ビルド**(`onnxruntime.dll`、`onnxruntime_providers_shared.dll`、`DirectML.dll`) | NuGet の [Microsoft.ML.OnnxRuntime.DirectML 1.24.4](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.24.4)(DirectML 再配布 DLL 付き)。`ort` クレートの API レベルと一致必須——不一致は初期化時にデッドロック |
| `data/dictionaries/*.tsv` | 中/英/独/仏/西/伊の G2P 辞書(8 ファイル) | 現状最も簡単:release ビルドを入れて `data\dictionaries` をコピー(ソースビルド用スクリプトは既知のギャップ) |
| `data/models/` | 声/分離/補助/曲生成モデル | アプリ内ダウンロード(設定 → モデルアセット、リソースマネージャー) |
| `converter/.venv` | `.pth` ボイスモデル取込用の Python 環境 | 任意——アプリ内「CPU ランタイム」パック(設定 → 学習ランタイム)で代替可 |

それ以外(アイコン、インストーラ用アート、vendored C++/Python)はリポジトリに含まれます。

### 実行 / テスト / ビルド

```powershell
npm install
npm run tauri dev      # 実ウィンドウでアプリ全体(Vite :1420 + Rust バックエンド)

npm run build          # フロントエンド gate: tsc -b && vite build
npm test               # vitest スイート
cd src-tauri; cargo test --workspace   # Rust スイート(重い E2E は #[ignore])

pwsh -File scripts/release.ps1          # ゲート付き署名インストーラビルド(下記)
pwsh -File scripts/verify-install.ps1   # インストール済みツリーの 39 項目監査
```

注意:

- debug ビルドはデータルートをリポジトリ(`<repo>/data`)に固定するため、開発データが
  インストール版と混ざりません。
- `scripts/release.ps1` は `package.json` / `Cargo.toml` / `tauri.conf.json` の
  バージョン同期、厳密 semver、全 gate(tsc / vitest / cargo test)、minisign 更新署名を
  強制します。署名キーは**リポジトリにありません**——fork は `npm run tauri build` で
  未署名のローカルバンドルを作れますが、既存インストールへの更新配信はできません。
- 開発ループの落とし穴・サブシステム設計メモ・検証プレイブックは各モジュール近くの
  コードコメントにあります。レンダーライフサイクル、undo、ORT session を触る前に読んでください。

## アーキテクチャ

| 層 | 技術 |
| --- | --- |
| シェル | Tauri 2 + システム WebView2(ブラウザ同梱なし) |
| フロントエンド | React 19 + TypeScript + zustand + Canvas 2D(ピアノロール/アレンジは React 外キャンバス)、ノードエディタは @xyflow/react |
| バックエンド | 単一 Rust プロセス;すべての DSP と推論をプロセス内で実行 |
| 推論 | `ort`(load-dynamic)駆動の ONNX Runtime:既定で **DirectML**、任意でアプリ内ダウンロードの自己完結 **CUDA** ランタイム、CPU フォールバック |
| 曲生成 | ローカル GGUF LLM エンジン(Yue2)による歌詞・アシスト + ACE-Step / HeartMuLa 音楽モデルによる全体レンダリング |
| 音源 | FluidSynth(FFI)による SF2 / SFZ 楽器再生 |
| 学習 | vendored 学習移植(RVC / so-vits-svc / SingingVocoders)を組み込みポータブル Python ランタイムパック上で実行 |

**歌声合成チェーン**(ピアノロール経路):

```
譜面(ノート + 歌詞)
  → Rust 2 段階 G2P          (言語別辞書 → 共通 IPA 音素ボキャブラリ)
  → Score2ConVec(ONNX)    (音素 + パラメータ化 f0 → ContentVec 空間のコンテンツベクトル)
  → SVC デコード(ONNX)     (SoVITS 4.0/4.1 または RVC net_g;任意で浅い拡散 + NSF-HiFiGAN)
  → トラック上のオーディオ
```

[**Score2ConVec**](https://github.com/yasoukyoku/Score2ConVec) はこれを可能にする
「譜面→コンテンツベクトル」モデルです:記号譜を SVC モデルが消費する ContentVec
特徴空間へ直接マッピングするため、通常の SVC ボイスモデルがガイドボーカルなしで
*譜面を歌えます*。元の UtaiSynthesizer プロジェクトのために特訓されたモデルで、
モデル・学習コード・詳細は同リポジトリにあります。

カバーモードは同じ SVC デコーダーを使い、特徴抽出は ContentVec、ピッチは RMVPE です。
ボーカル分離は MSST/UVR モデル群の Rust ネイティブ再実装で、Python コンバーターが
生成するモデル別 config JSON によって駆動されます。

リポジトリマップ(トップレベル):`src/` フロントエンド · `src-tauri/` Rust バックエンド
(`crates/utai-dsp` DSP ホットループ、`crates/utai-stretch` vendored Signalsmith Stretch)·
`converter/` Python ONNX エクスポートスクリプト · `training/` vendored 学習パッケージ +
ランタイムパックビルダー · `scripts/` リリースツール · `docs/` ユーザーガイド。

## 責任ある利用と免責事項

MunoAI は創作ツールです。**あなたが作った内容の責任はすべてあなたにあります。**

- **声の権利。** ボイスモデルの学習・利用は権利がある場合のみ:自分の声、同意を得た
  人の声、ライセンスが許すキャラクター/データセット。実在人物へのなりすましは禁止。
  混乱が生じうる場面では AI 生成であることを明示してください。
- **楽曲の権利。** 著作権のある楽曲のカバーや派生音声は、特に商用・公開配信では
  権利者の許可が必要な場合があります。適用される法律とプラットフォームのルールを
  守ってください。
- **モデルは同梱されません。** 本リポジトリと公式インストーラは**歌手ボイスモデルを
  一切含みません**。アプリがダウンロードする補助ウェイト(コミュニティ NSF-HiFiGAN
  ボコーダー、GAME 歌声→MIDI ウェイトなど)には独自ライセンスがあり、複数は
  **CC BY-NC-SA(非商用)** です——アプリはそれらの通知をファイルとともに表示します。
  あなたが取り込む・学習するボイスモデルは、その元データのライセンス/許諾に従います。
- 声の権利・著作権・適用法律を侵害する利用について、開発者は推奨せず、責任も負いません。
  サードパーティの帰属は [NOTICE.md](NOTICE.md) を参照してください。

## 既知の制限とオープンな課題

私たちが**把握しており、意図的に残している**問題です。ぜひ一緒に取り組みたい部分でもあります。

- **大きなマシンでは学習スループットを引き出しきれていません。** データローダーは
  固定数ではなくマシンの**利用可能コミットメモリ**で worker/プリフェッチ数を決めますが、
  これは**減らす方向のみ**:余裕のあるマシンでも既定値のまま上がりません。大きいマシンで
  **上げる**のはパフォーマンス主張であり、責任を持って結論できません——worker 数と
  速度は単調ではなく、実データセットを異なる RAM/CPU の複数マシンで回して初めて
  決まります。
  **より良いハードウェアをお持ちの方**:実測(実データセットでの step 時間 vs worker 数)、
  issue、PR を歓迎します。ノブと論拠は `training/utai_train/loader_budget.py` にあります。
- **AMD ランタイムパックは現在 gfx1103 系 iGPU のみ対応**(Radeon 780M / 760M / 740M)。
  パックの汎用コンピュートカーネルはその 1 ターゲット専用のため、ダウンロードゲートも
  それに合わせて誠実に隠れます——**ゲートは正直、パックが狭い**のです。RDNA3/RDNA4
  ディスクリートや他世代 iGPU は未対応。拡張にはアーキテクチャ別カーネルホイール
  (~100MB/個)追加**と**能力ゲートの厳格さ維持が必須で、手元にないハードウェアでは
  検証できません。
- **Intel XPU サポートは実験的で、実機未検証です** —— Intel GPU を持っていません。
  正しさは構成とアプリ内ランタイム自己テスト(`envtest`)に依拠。コミュニティの
  自己テスト報告が届くまで「実験的」ラベルは外しません。

## コミュニティ

- **QQ グループ**:4446804
- **バグ / 機能要望**:[GitHub Issues](https://github.com/junziai/munoai/issues)
  (アプリバージョンとログを添付してください——アプリ内「ログ」ページ参照)
- セキュリティ問題:[SECURITY.md](SECURITY.md) を参照

## ライセンス

[AGPL-3.0](LICENSE)。リポジトリは [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
の AGPL-3.0 コードを vendored しており(そのためプロジェクト全体が AGPL)、
MIT コンポーネントも併用しています——完全なサードパーティ帰属は
[NOTICE.md](NOTICE.md) にあります。
