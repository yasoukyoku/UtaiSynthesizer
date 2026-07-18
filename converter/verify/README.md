# MSST 验证工具箱（S31 方法论，2026-07-03）

S31 用这套方法从"BSRoformer 人声泄露"一路查修到四架构全验证。**任何转换器/管线改动、
新架构接入（如 UVR）、新 fp16 开放，都必须过这里的对应关卡** —— S7 的教训：
"自我一致性验证"（重实现 vs 自己的导出）证明不了正确性，必须对照 ORIGINAL。

原版参照代码：`D:\MyDev\ARCHIVE\MSSTRVCv2\MSST\`（modules/、configs/ 按 ckpt 名的 yaml）。
python 环境：`converter/.venv/Scripts/python.exe`（torch CPU + onnxruntime + einops +
rotary_embedding_torch + librosa + pyyaml + onnxconverter-common）。
注意：个别 ARCHIVE yaml 本身是错的（如 bs_roformer_ep_937）—— catalog configUrl 的在线 yaml
与 ckpt kwargs 才是更高优先级的真源。

## 关卡 1 — 架构等价（converter 重实现 vs 原版）
**随机权重移植法**（无需下载真权重）：用 yaml/kwargs 实例化【原版】类 → `state_dict()` →
`strict=True` 载入【我们的】重实现 → 同一随机输入比输出。**max_abs_diff < 1e-5 才算过**；
不过就逐模块二分（band_split → transformer → mask_estim）。有真 ckpt 时优先真权重
（还能覆盖 detect_config）。三方对照定位漂移来源：原版 torch vs 我们 torch vs 我们 ONNX。

## 关卡 2 — E2E 管线（Rust vs 原版 demix 语义）
`e2e_py_ref.py`：按原版 MSST demix 语义（chunk/step/border reflect/梯形窗/首尾特例）驱动
我们的 ONNX 的 python 参照实现。Rust 侧用测试 harness 跑同素材：
```
UTAI_SEP_INPUT=<wav> [UTAI_SEP_MODEL=<onnx>] [UTAI_SEP_DEVICE=auto] \
  cargo test --test separation_pipeline separate_env_wav -- --ignored --nocapture
```
（默认 CPU EP 保证数值纯净；auto = CUDA。**Conv 类架构（mdx23c/htdemucs）在裸 harness 上
需要 `PATH="/d/MyDev/Utai_v2-dev/runtime/cuda:$PATH"` 前置**，否则 cudnn 子 DLL 找不到。）
`e2e_compare.py` 比 stems：**SNR > ~40 dB = 管线忠实**（S31 实测 BS 73.9/78.3 dB）。
注意：overlap-add 的 split-jitter 主导管线变体间差异（python demix 对自身直推都只有 26 dB），
45-65 dB 区间的 SNR 不能当独立噪声底解读，必须同 chunk 几何对比。

## 关卡 3 — fp16 质量门（每个架构单独过，不许连坐）
`onnx_fp16.py` 转换 → fp32/fp16 各跑一次 GPU E2E → `e2e_compare.py` 比 stems。
**通过线：非安静 stem 全部 > 45 dB**（安静 stem 看绝对误差底是否与其他 stem 一致）。
**硬规则：必须在 CUDA EP 上跑 —— CPU EP 用 fp32 模拟 fp16，会假阳性**（htdemucs 的两个
NaN 级 bug 就是 CUDA-only 的；S33 再证：CPU EP 对 fp16 MultiHeadAttention 没有原生
kernel，CastFloat16Transformer 会静默升回 fp32 算）。过门后同步更新 convert.py
`FP16_VERIFIED_TYPES` 与 msst-catalog.ts `MSST_FP16_ARCHS`（镜像对，注释里有各架构的门值）。
S33 融合图门值：BS 65.8/70.2 dB、MelBand 67.8/63.0 dB —— 比旧解构图高 ~9-12 dB，
因为 RotaryEmbedding cache 存的是 cos/sin 的**值**（值域 [-1,1]，fp16 舍入无害），
而旧图的 fp16 是在 fp16 里算大角度（t·f 可到数百 rad，fp16 分辨率 0.5 rad）再取 cos。

## 关卡 4 — 波形审计（怀疑"输出没换/覆盖/串轨"时）
`cross_model_check.py`：md5 + 两两 corr/SNR 矩阵，一分钟定案。读数参考（S31 实测，S32 修正）：
- 同模型同精度重跑：**roformer/mdx23c 在 CUDA 上逐位一致；htdemucs 不是**（cudnn 卷积
  算法按计时选型，S32 实测同一二进制两次 GPU 运行 6/6 stem md5 全不同）——**htdemucs 的
  bitwise A/B 门必须在 CPU EP 上跑**（确定性；20s 素材即可）。GPU md5 对 htdemucs 出 DIFF
  先怀疑这个，不要先怀疑代码。（roformer 的逐位一致 S33 在融合图上重校过：
  com.microsoft MHA/RotaryEmbedding 的 CUDA kernel 跨运行 fp32/fp16 全部 md5 一致。）
- 同模型 fp32 vs fp16 ≈ corr 1.0000 / ~50 dB；
- 不同架构分同一首歌 ≈ corr 0.98 / 14-15 dB（波形缩略图看不出差别，这是正常态）。
配套检查 autosave.json 的 lane 路径是否指向各节点**最新** run 目录。

S32 增补的管线改动 A/B 手法：cargo 会把旧的测试 exe 留在 `target/release/deps/
separation_pipeline-<hash>.exe` —— 新旧 exe 直接各跑一遍比 md5，免 git stash；harness 另有
`UTAI_SEP_OVERLAP=<n>` 覆盖模型 JSON 的 num_overlap（chunk 几何实验用，同 UI 滑条语义）。

跨几何（如 ov4 vs ov2）质量门：`overlap_ab_check.py <mix> <dirA> <dirB> <stepB>` ——
每 stem corr/电平 + sum(stems)-vs-mix 可加性残差 + 步长边界接缝探针。读数校准（S32 htdemucs
实测）：响 stem corr 0.998+/SNR 25-33 dB = 正常 split-jitter；安静 stem（RMS<0.02）的低
corr/SNR 是效应假象，看绝对误差底是否与响 stem 一致；接缝探针 ratio ~1.0 过 / >2 查边界。

## 改导出图的完整关卡组合（S33 attention 融合的实跑清单，照抄即可）
S33 把 roformer attention 重写成 com.microsoft::MultiHeadAttention + RotaryEmbedding
（外置 RoPE/gating，torch 前向保持纯数学）。任何"改图不改语义"的导出改动照这个清单：
1. **spike 先行**：onnx.helper 手搭最小图，contrib op vs torch 参照数学 < 1e-5 再动转换器。
2. **关卡 1'（传递性）**：旧代码 git show 出来 vs 新代码，真 ckpt 同输入 —— 0.0 bit-exact
   则 S31 的"旧≡原版"结论直接传递，不用再搭原版对照环境。
3. **CPU EP E2E A/B**（Rust harness，同几何）：新旧 onnx 同素材，>130 dB = 图级等价。
4. **CUDA A/B + 关卡 3 重过 + VRAM 峰值**（vram_poll 轮询 nvidia-smi）+ 确定性重校
   （同配置跑两遍比 md5 —— 融合 kernel 的确定性不继承旧图的读数）。
5. **DML 冒烟**：一次性 venv 装 onnxruntime-directml，跑 DmlExecutionProvider vs CPU 的
   SNR + 耗时比 —— **speedup 接近 1× = 静默回退到了 CPU**（DML 对缺 kernel 的节点是
   逐节点回退不报错），几十倍 = 真在 GPU 上。release 构建走 DML，必须冒烟。
6. 部署时**同步替换 TESTING 副本**（S31 stale-artifact 教训），convert.py/catalog 的
   门值注释同步刷新。

## 关卡 5 — stem 顺序 / 端口映射
模型真实输出顺序 = json `stem_names`（converter 从 kwargs/yaml training.instruments 读出）。
catalog 手写 stems 只是未安装时的展示兜底。**E2E 测试按 json 命名存文件，永远抓不到前端
端口映射错位** —— 新架构接入后必须人工核一遍"端口标签 vs 实际内容"（htdemucs_6s 教训：
模型介绍页顺序 ≠ 权重顺序，vocals 曾被标到 Piano 口）。

## UVR VR-arch / 遗留 MDX-Net 接入实录（S34，全关卡数值）
原版参照就在本地：`D:\MyDev\ARCHIVE\MSSTRVCv2\MSST\modules\vocal_remover\`（vr_separator +
uvr_lib_v5，与 TESTING\Utai\MSST 副本逐字节一致）+ `configs\vr_modelparams\*.json`。
模型识别 = UVR 式尾部 hash（文件末 10000*1024 字节的 md5）→ converter 内嵌注册表
（architectures/uvr_vr.py / mdx_net.py；未知 hash 拒绝转换，绝不猜参数）。
- **关卡 1 双档法**（AdaptiveAvgPool→ReduceMean 这类"参数无关的 op 改写"通用）：
  tier1 把改写点临时换回原版 op → 与原版真权重 **0.0 bit-exact**（9/9，证明结构零偏差）；
  tier2 出货语义 < 1e-5（README 阈值；实测 2.8e-7~9.5e-6，池化求和顺序是唯一残差源）。
- **关卡 2 分层对照**（重采样器选型差异会污染读数，必须隔离）：python 参照跑两版——
  原版 sinc_fastest 合成 + 强制 polyphase 合成（= Rust 选型 = UVR macOS-ARM 行为）。
  实测（20s 素材，CPU EP）：Rust vs REFPOLY **109-128 dB**（DSP 链等价）；
  Rust vs 原版 REF **53-61 dB corr 1.000000**（残差全部来自 sinc_fastest↔polyphase，
  远超 40dB 忠实线）。MDX-Net：Rust vs 原版 **122-135 dB**（图级）。
- **CUDA**：VR/MDX 四模型跨运行 md5 全部一致（conv 架构里的例外——htdemucs 才是非确定的）；
  CUDA vs CPU 61-88 dB = TF32 正常区间。裸 harness 跑 CUDA 记得 `runtime/cuda` 前置 PATH。
- **DML 冒烟**：4 模型 speedup 46-58×（真 GPU，v5.1 的 LSTM 未回退）、SNR 111-113 dB。
- **fp16 未开放**（FP16_VERIFIED_TYPES / MSST_FP16_ARCHS 均未加 uvr_vr/mdx_net）——
  要开必须单独过关卡 3（CUDA EP）。
- 语义坑备忘：DeNoise 的 primary=**Noise**（干净音频在第二口）；KARA 主输出=主音、
  KARA_2 主输出=伴奏+和声（互为镜像）；5_HP 配置里的 stereo_n 在 v5.0 推理下**被忽略**
  （per-band 变换仅 v5.1 生效）；6_HP 的 mid_side_b2 在**波形域**做。
- python 参照复现：脚本存档在 **verify/uvr/**（gate1_vr / e2e_vr_ref / e2e_mdx_ref /
  e2e_compare_uvr / dml_smoke_uvr / gen_dsp_refs；venv 需 pip install samplerate；
  模型/素材路径按会话 scratchpad 硬编码，复用时改路径）。

—— 脚本里的路径多为 S31 会话的硬编码（scratchpad / 新宝島 素材），复用时改路径即可；
方法本身照抄。历史细节见 memory：project_v2_session31 / project_v2_separation_vocal_leak。

## 关卡 3 增补 — fp16 归一化统计保护(S68c,2026-07-18)

**背景**:旧 fp16 配方在真 fp16 kernel 上,roformer 的 F.normalize(ReduceL2→Clip→Expand→Div)
会在静音/近静音 chunk 上全输出 NaN(0.5.0 RVC 20% 闪退的毒源)。三重机理:①Clip 下限 1e-12
被 onnxconverter 钳到 1e-7=fp16 亚正规数,GPU FTZ 后=0;②Σx² 在 fp16 累加上溢/下溢;
③库把岛内 fp32 常数经 fp16 张量中转喂 Clip(1e-12→0,图数学级 0/0)。修复=onnx_fp16.py
_norm_stats_block_list(统计脊整体 fp32,walk 止步于 Reduce)+cast 往返坍缩+死 cast 修剪
(悬挂死 cast 对会把 AMD 780M DML 第二跑直接挂死 887A0006——N 卡容忍 A 卡不容忍)。

**新工具(verify/fp16/)**:
- `dml_smoke.py <fp32> <fp16old> <fp16new>`:双 DML 适配器 × 5 输入档(zeros/tiny/真实STFT/
  randn×2)非有限计数+SNR;CPU(fp32-emu)档=图数学 vs kernel 判别器。**零/微输入档必须跑**
  ——真实歌曲素材永远测不到静音 chunk,这正是旧配方三个 gate 全绿却在用户机上炸的原因。
- `check_fp16_graph.py`:静态验 Clip 下限(dtype+值+**是否经 fp16 张量中转**——只看常数
  原值会被中转盲区骗过)。
- `nan_bisect.py <model> <dev> <regime>`:中间张量分批重导出为输出,二分定位第一个非有限张量。
- `minimal_norm_repro.py`:8 元素最小图,fp16-norm vs fp32-island 三 EP 对照。

**S68c 门值(fused 图+保护配方)**:CUDA E2E BS 72.8/67.9 dB、MelBand 68.6/63.4 dB
(vs 未保护配方 ≤0.1dB=保护零代价);MDX23C 空名单字节等同(md5);htdemucs 走原
S31 路径逐字节不动(坍缩豁免)。**改 fp16 配方的完整关卡=本节冒烟(双卡+零输入档)
+CUDA E2E 45dB+静态中转检查+MDX23C 哈希等同+htdemucs 路径字节不动。**
