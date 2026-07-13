# UtaiSynthesizer

[English](README.md) | **简体中文** | [日本語](README.ja.md)

[![Release](https://img.shields.io/github/v/release/yasoukyoku/UtaiSynthesizer)](https://github.com/yasoukyoku/UtaiSynthesizer/releases)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
![Platform: Windows](https://img.shields.io/badge/platform-Windows%2010%2F11-informational)
[![QQ 群](https://img.shields.io/badge/QQ-1058227212-1EBAFC)](https://qun.qq.com/universal-share/share?ac=1&authKey=3uD5AoM8e50y00vhOYOZsa2VI341dBNfr07S2IK9wraewz0rcFHpSzONYJ9QrTP7&busi_data=eyJncm91cENvZGUiOiIxMDU4MjI3MjEyIiwidG9rZW4iOiJONGpqQ2MzM3h3N3BDMVBMRzZiSUFOU05YWnRnbHBxdTZDUElZYlZOSGN3VnhCaEc5eWludlJBYlltK3hkdlFwIiwidWluIjoiMjc2Njc2NDM1NSJ9&data=VyWCaG06iaMLBFcfEx_fjE2Tme2X7YvJsUIUjJ51zk6XymaED6Z6TEC_zOvAdm9q2MbzbYbpuO4ukQHZ1GBHLw&svctype=4&tempid=h5_group_info)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2)](https://discord.com/invite/p3fGh942fJ)

Windows 平台的歌声合成 DAW。在钢琴卷帘上写谱,**直接让 SVC 声音模型开口唱**
(乐谱 → [Score2ConVec](https://github.com/yasoukyoku/Score2ConVec) → SVC 解码,
不需要任何人声干声做中介);用逐片段节点工作流渲染 AI 翻唱;原生人声分离;
本地训练自己的声音模型——全部在一个程序里完成,推理阶段完全不依赖 Python。

![UtaiSynthesizer——编排视图](docs/images/overview-arrangement.png)

## 功能

- **钢琴卷帘歌声合成**——UTAU 式自由音符摆放、直接在音符上打歌词(整句自动分配)、
  SynthV 式音高过渡/颤音/手绘音高偏差、响度与共振腔参数带、呼吸音符,以及
  **7 种语言**(中/英/日/德/法/西/意)的词典化歌词与三级 OOV(无法发音)警示。
- **AI 翻唱工作流**——每个音频片段自带节点图:人声分离
  (BS-Roformer、MelBand Roformer、MDX23C、HTDemucs、VR、老式 MDX-Net——全部用 Rust
  原生重实现)→ RVC / So-VITS-SVC 4.0 与 4.1 声音转换(浅扩散、NSF-HiFiGAN
  增强器/声码器、多歌手声线混合)→ 频谱移调 → 以非破坏性子轨道回落到轨道上。
- **本地训练**——RVC、SoVITS 4.1/4.0、浅扩散、声码器微调,跑在内嵌便携 Python
  运行时上(NVIDIA / AMD / Intel / CPU 四种包,应用内下载;CUDA 运行时全自包含,
  **无需安装 CUDA Toolkit**)。
- **DAW 基本功**——多轨时间线、拖放导入与智能落位、交叉淡化、片段/轨道剪贴板、
  BPM 与节拍网格检测、保音高变速(Signalsmith)、响度包络、小地图、完整撤销/重做。
- **音域工具链**——声音模型自动音域测试、舒适区微调、可选的音域扩展
  (超出音域的乐句先在模型舒适区内合成,再用 TD-PSOLA 移回原调)。
- **人声转 MIDI**——把分离出的人声子轨道转写成可编辑音符(GAME 引擎)。
- 导出:音频(wav / flac / mp3 / ogg / opus / m4a,离线混音与播放所听严格一致)、
  乐谱(ust / ustx / midi)。导入:ustx / ust / midi 及 9 种音频格式。
- 三语界面(简体中文 / English / 日本語),下载全程 sha256 校验并内置对中国大陆
  友好的镜像选项,自动更新带 minisign 签名校验。

**只是想用这个软件?** 直接从 [Releases](https://github.com/yasoukyoku/UtaiSynthesizer/releases)
下载安装包,并阅读 **[使用指南](docs/user-guide.zh-CN.md)**——不需要任何开发环境。
安装目录整体可拷走当便携版用(数据跟着目录走)。

## 开发环境配置

本项目是 [Tauri 2](https://tauri.app) 应用:React 前端 + Rust 后端在同一进程。
出于体积考虑,大体积二进制资产一律不进 git——新 clone 需要手动补几样东西,
`tauri dev` 才能完整可用。

### 前置条件

| 工具 | 版本 | 说明 |
| --- | --- | --- |
| Windows | 10 / 11 x64 | 需要 WebView2 运行时(Win 11 自带) |
| Node.js | 18+(推荐 20 LTS) | npm 7+(lockfile v3) |
| Rust | 1.77+ stable,**MSVC** 工具链 | `rustup default stable-x86_64-pc-windows-msvc` |

C++ 变速库用 `cc` crate 走 MSVC 编译,不需要 libclang/bindgen。

### 新 clone 缺少的资产

| 路径 | 是什么 | 从哪来 |
| --- | --- | --- |
| `bin/ffmpeg.exe` | 解码兜底 + 全部非 wav 导出编码 | [gyan.dev "essentials"](https://www.gyan.dev/ffmpeg/builds/) GPL 构建(镜像:[GyanD/codexffmpeg](https://github.com/GyanD/codexffmpeg/releases));必须含 `libmp3lame/libvorbis/libopus/aac/flac` 编码器 |
| `runtime/ort/*.dll` | ONNX Runtime **1.24.4 DirectML 构建**(`onnxruntime.dll`、`onnxruntime_providers_shared.dll`、`DirectML.dll`) | NuGet 包 [Microsoft.ML.OnnxRuntime.DirectML 1.24.4](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.24.4)(附 DirectML 再分发 DLL)。版本必须与 `ort` crate 的 API 级别匹配——版本不配会在初始化时死锁 |
| `data/dictionaries/*.tsv` | 中/英/德/法/西/意歌词的 G2P 词典(8 个文件) | 目前最省事:装一份 release 版,把它的 `data\dictionaries` 拷过来(源码侧构建脚本是已知缺口) |
| `data/models/` | 声音/分离/辅助模型 | 应用内下载(设置 → 模型资产、资源管理) |
| `converter/.venv` | 导入 `.pth` 声音模型用的 Python 环境 | 可选——应用内「CPU 运行时」包(设置 → 训练环境)可替代 |

其余(图标、安装器美术、vendored C++/Python)都在仓库里。

### 运行 / 测试 / 构建

```powershell
npm install
npm run tauri dev      # 完整应用真窗口(Vite :1420 + Rust 后端)

npm run build          # 前端 gate:tsc -b && vite build
npm test               # vitest 测试
cd src-tauri; cargo test   # Rust 测试(重型 E2E 用 #[ignore] 标注)

pwsh -File scripts/release.ps1          # 带门禁的签名安装包构建(见下)
pwsh -File scripts/verify-install.ps1   # 装机目录 39 项核对
```

说明:

- debug 构建会把数据根钉在仓库目录(`<repo>/data`),开发数据不会和已安装副本混用。
- `scripts/release.ps1` 强制三处版本一致(`package.json` / `Cargo.toml` /
  `tauri.conf.json`)、严格 semver、全量 gate(tsc / vitest / cargo test)和 minisign
  更新签名。签名私钥**不在**仓库里——fork 可以用 `npm run tauri build` 构建本地
  未签名包,但无法向已有安装推送更新。
- 开发环节的坑、子系统设计说明和验证手册都写在相关模块附近的代码注释里——
  动渲染生命周期、undo、ORT session 相关代码前先读。

## 架构

| 层 | 技术 |
| --- | --- |
| 外壳 | Tauri 2 + 系统 WebView2(不捆绑浏览器) |
| 前端 | React 19 + TypeScript + zustand + Canvas 2D(钢琴卷帘/编排是脱-React 画布),节点编辑器用 @xyflow/react |
| 后端 | 单个 Rust 进程;全部 DSP 与推理在进程内 |
| 推理 | `ort`(load-dynamic)驱动 ONNX Runtime:默认随包 **DirectML**,可选应用内下载的全自包含 **CUDA** 运行时,CPU 兜底 |
| 训练 | vendored 训练移植(RVC / so-vits-svc / SingingVocoders),跑在内嵌便携 Python 运行时包上 |

**歌声合成链**(钢琴卷帘路径):

```
乐谱(音符 + 歌词)
  → Rust 两级 G2P          (分语言词典 → 共享 IPA 音素表)
  → Score2ConVec(ONNX)    (音素 + 参数化 f0 → ContentVec 空间的内容向量)
  → SVC 解码(ONNX)        (SoVITS 4.0/4.1 或 RVC net_g;可选浅扩散 + NSF-HiFiGAN)
  → 轨道上的音频
```

[**Score2ConVec**](https://github.com/yasoukyoku/Score2ConVec) 是让这一切成立的
「乐谱→内容向量」模型:它把符号乐谱直接映射进 SVC 模型消费的 ContentVec 特征空间,
因此任何普通 SVC 声音模型都能*照谱唱歌*,不需要人声引导。该模型为本项目专门训练,
模型、训练代码与细节都在其仓库里。

翻唱模式复用同一套 SVC 解码器,特征提取用 ContentVec,音高用 RMVPE。人声分离是
MSST/UVR 模型家族的 Rust 原生重实现,由 Python 转换器产出的逐模型 JSON 配置驱动。

仓库地图(顶层):`src/` 前端 · `src-tauri/` Rust 后端(`crates/utai-dsp` DSP 热循环、
`crates/utai-stretch` vendored Signalsmith Stretch)· `converter/` Python ONNX 导出脚本 ·
`training/` vendored 训练包 + 运行时包构建器 · `scripts/` 发版工具 · `docs/` 使用指南。

## 负责任使用与免责声明

UtaiSynthesizer 是一个创作工具。**你用它做出的内容,责任完全在你自己。**

- **声音权利。** 只在有权利的前提下训练或使用声音模型:你自己的声音、明确同意者的
  声音、或许可条款允许的角色/数据集。不得假冒真实人物;在可能造成混淆的场合,
  请明确标注内容为 AI 生成。
- **歌曲权利。** 对有版权歌曲的翻唱及衍生音频,尤其是商业化或公开传播时,可能需要
  取得权利方许可。请遵守你所在地区的法律与平台规则。
- **不含任何模型。** 本仓库与官方安装包**不包含任何歌手声音模型**。应用可下载的
  辅助权重(如社区 NSF-HiFiGAN 声码器、GAME 人声转 MIDI 权重)各有自己的许可——
  其中若干为 **CC BY-NC-SA(禁止商用)**——应用会随文件一并展示/分发这些声明。
  你导入或训练的声音模型,受其源数据的许可与授权约束。
- 对任何侵犯声音权利、著作权或违反适用法律的使用方式,开发者不予认可,亦不承担
  任何责任。第三方致谢见 [NOTICE.md](NOTICE.md)。

## 社区

- **QQ 群**:[1058227212](https://qun.qq.com/universal-share/share?ac=1&authKey=3uD5AoM8e50y00vhOYOZsa2VI341dBNfr07S2IK9wraewz0rcFHpSzONYJ9QrTP7&busi_data=eyJncm91cENvZGUiOiIxMDU4MjI3MjEyIiwidG9rZW4iOiJONGpqQ2MzM3h3N3BDMVBMRzZiSUFOU05YWnRnbHBxdTZDUElZYlZOSGN3VnhCaEc5eWludlJBYlltK3hkdlFwIiwidWluIjoiMjc2Njc2NDM1NSJ9&data=VyWCaG06iaMLBFcfEx_fjE2Tme2X7YvJsUIUjJ51zk6XymaED6Z6TEC_zOvAdm9q2MbzbYbpuO4ukQHZ1GBHLw&svctype=4&tempid=h5_group_info)
- **Discord**:<https://discord.com/invite/p3fGh942fJ>
- **Bug / 功能建议**:[GitHub Issues](https://github.com/yasoukyoku/UtaiSynthesizer/issues)
  (请附应用版本和日志——见应用内「日志」页)
- 安全问题:见 [SECURITY.md](SECURITY.md)

## 许可

[AGPL-3.0](LICENSE)。仓库 vendored 了
[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) 的 AGPL-3.0 代码
(整个项目因此采用 AGPL),同时包含若干 MIT 组件——完整第三方致谢见
[NOTICE.md](NOTICE.md)。
