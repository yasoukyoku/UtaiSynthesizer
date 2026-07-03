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
NaN 级 bug 就是 CUDA-only 的）。过门后同步更新 convert.py `FP16_VERIFIED_TYPES` 与
msst-catalog.ts `MSST_FP16_ARCHS`（镜像对，注释里有各架构的门值）。

## 关卡 4 — 波形审计（怀疑"输出没换/覆盖/串轨"时）
`cross_model_check.py`：md5 + 两两 corr/SNR 矩阵，一分钟定案。读数参考（S31 实测，S32 修正）：
- 同模型同精度重跑：**roformer/mdx23c 在 CUDA 上逐位一致；htdemucs 不是**（cudnn 卷积
  算法按计时选型，S32 实测同一二进制两次 GPU 运行 6/6 stem md5 全不同）——**htdemucs 的
  bitwise A/B 门必须在 CPU EP 上跑**（确定性；20s 素材即可）。GPU md5 对 htdemucs 出 DIFF
  先怀疑这个，不要先怀疑代码。
- 同模型 fp32 vs fp16 ≈ corr 1.0000 / ~50 dB；
- 不同架构分同一首歌 ≈ corr 0.98 / 14-15 dB（波形缩略图看不出差别，这是正常态）。
配套检查 autosave.json 的 lane 路径是否指向各节点**最新** run 目录。

S32 增补的管线改动 A/B 手法：cargo 会把旧的测试 exe 留在 `target/release/deps/
separation_pipeline-<hash>.exe` —— 新旧 exe 直接各跑一遍比 md5，免 git stash；harness 另有
`UTAI_SEP_OVERLAP=<n>` 覆盖模型 JSON 的 num_overlap（chunk 几何实验用，同 UI 滑条语义）。

## 关卡 5 — stem 顺序 / 端口映射
模型真实输出顺序 = json `stem_names`（converter 从 kwargs/yaml training.instruments 读出）。
catalog 手写 stems 只是未安装时的展示兜底。**E2E 测试按 json 命名存文件，永远抓不到前端
端口映射错位** —— 新架构接入后必须人工核一遍"端口标签 vs 实际内容"（htdemucs_6s 教训：
模型介绍页顺序 ≠ 权重顺序，vocals 曾被标到 Piano 口）。

—— 脚本里的路径多为 S31 会话的硬编码（scratchpad / 新宝島 素材），复用时改路径即可；
方法本身照抄。历史细节见 memory：project_v2_session31 / project_v2_separation_vocal_leak。
