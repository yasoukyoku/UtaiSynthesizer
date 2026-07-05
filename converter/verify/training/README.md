# 训练链验证关卡（①b，S37 起）

方法论同 `converter/verify/README.md`：**参照物永远是原版仓库的真实执行，绝不自证**。
本目录覆盖 RVC 训练链的两道关卡（SoVITS/扩散/声码器各阶段落地时在此追加）。

原版参照：`D:\MyDev\RVC\RVC20240604Nvidia`（20240604 NVIDIA 整合包，自带 runtime =
python3.9 + torch 2.0.0+cu118 + fairseq 0.12.2 + librosa 0.9.1）。
测试数据：`D:\MyDev\TESTING\Kazano_Sayo\dataset1.wav` 切出的 3×60s 段
（`D:\MyDev\TESTING\utai-v2-testing\gate_dataset`，48k 立体声干声，未切片）。
所有中间产物在 `D:\MyDev\TESTING\utai-v2-testing\`（不进 git）。

## 关卡0 —— 预处理对拍（切片/f0/特征/filelist/index）

```
# ① 原版侧（ground truth，用原版自带 runtime，cwd=RVC 根）：
cd D:\MyDev\RVC\RVC20240604Nvidia
runtime\python.exe infer\modules\train\preprocess.py <gate_dataset> 48000 4 <rvc_orig> True 3.7
runtime\python.exe infer\modules\train\extract\extract_f0_rmvpe.py 1 0 0 <rvc_orig> True
runtime\python.exe infer\modules\train\extract_feature_print.py cuda 1 0 <rvc_orig> v2 False
# ② f0 fp32 参照（原版 CPU 分支，喂"我们的"16k wav → rvc_B2_orig）：
runtime\python.exe infer\modules\train\extract\extract_f0_print.py <rvc_B2_orig> 1 rmvpe
# ③ 特征 fp32 参照（真 fairseq CPU，生成 rvc_fairseq_fp32 —— 见本 README 末尾内联脚本）
# ④ 我方侧 + 对拍：
training\.venv\Scripts\python.exe converter\verify\training\gate0_run_ours.py
training\.venv\Scripts\python.exe converter\verify\training\gate0_compare.py
```

**S37 实测读数（2026-07-05，全 PASS）：**

| 层 | 项 | 读数 |
|---|---|---|
| A | 0_gt_wavs 切片链（解码→48Hz高通→slicer→3.7s窗→归一） | **逐位 0.0**（51/51 文件） |
| A | 1_16k_wavs | min SNR 39.1 dB（librosa 版本轴，见下） |
| A | f0 / coarse / 特征 | 0.75% / 1.9% / cos 0.985（GPU+重采样轴叠加，松线） |
| C | f0 定审（双方 fp32 CPU） | **0/10534 帧超 0.5Hz、0 清浊翻转、max 0.24 mHz** |
| C | 特征定审（真 fairseq fp32 CPU vs 我们 ContentVec onnx） | **max 7.7e-4，min cos = 1−1e-9**（51 文件） |

### 调查中钉死的数值轴（复跑对拍时勿再踩）
1. **librosa 0.9.1 (kaiser_best) vs ≥0.10 (soxr_hq)**：16k 重采样单行同名调用，
   版本默认 res_type 不同 → ~39dB。代码同构，属环境轴。
2. **原版特征脚本永远 `.to("cuda")`**（argv 的 device 会被 `torch.cuda.is_available()`
   覆盖）→ cudnn TF32 卷积噪声 ~1e-2 弥散差；且 **`CUDA_VISIBLE_DEVICES=`（空值）在
   Windows 上等于删除变量**（Windows 无空环境变量），根本藏不住 GPU —— 要禁用必须
   `CUDA_VISIBLE_DEVICES=-1`。我们 runner 的 CPU 模式因此用 `"-1"` 哨兵。
3. **原版 f0 的 is_half 是字符串**（`"False"` 恒真）→ NVIDIA 上事实恒 half。我们
   extract_f0 默认 CUDA=half 对齐该行为；fp32 CPU 定审用显式参数。
4. 半精度/跨 torch 版本的 f0 残差是清浊边界/八度歧义帧（10/10534 帧，含一个
   257↔130Hz 八度翻转），训练数据容差内。

## 关卡1 —— 训练等价（逐 step loss 轨迹 vs 原版 train.py）

同一份预处理产物（关卡0 的 rvc_ours）+ 同 filelist 行序 + 同 seed(1234) + 同底模
(f0G48k/f0D48k v2) + **双方 fp32 CPU（确定性）** + 同一个 torch(2.5.1，我们的 venv 里
跑原版脚本 —— 隔离代码轴)。原版侧读 tensorboard events（stdout 只有 3 位小数）。

```
training\.venv\Scripts\python.exe converter\verify\training\gate1_prepare.py
# 原版侧（cwd=RVC 根；USE_LIBUV=0 是 torch≥2.4 Windows 的 TCPStore 必需）：
USE_LIBUV=0 <utai>\training\.venv\Scripts\python.exe infer\modules\train\train.py ^
  -e gate1 -sr 48k -f0 1 -bs 4 -g -1 -te 2 -se 1 -pg <f0G48k> -pd <f0D48k> -l 1 -c 0 -sw 0 -v v2
# 我方侧 + 对拍：
training\.venv\Scripts\python.exe converter\verify\training\gate1_run_ours.py > gate1_ours_steps.jsonl
training\.venv\Scripts\python.exe converter\verify\training\gate1_compare.py
```

**S37 实测读数（全 PASS，30/30 step 对齐）：**

| 分量 | max 相对差 | mean 相对差 |
|---|---|---|
| loss/g/total | 1.2e-8 | 5.9e-9 |
| loss/d/total | 1.3e-7 | 6.7e-8 |
| loss/g/fm | 8.6e-8 | 3.8e-8 |
| loss/g/mel | 1.9e-8 | 8.2e-9 |
| loss/g/kl | 1.8e-7 | 6.5e-8 |

= float32/TB 序列化噪声级：vendored 训练循环对原版逐 step 复刻。

### 复跑注意
- 训练 venv 的 **matplotlib 必须 ==3.7.5**（原版 utils.plot 用 `np.fromstring/
  tostring_rgb`，mpl≥3.8 删除；我们 vendored 版已换 `buffer_rgba` 双兼容）。
- 原版 train.py 正常完训是 `os._exit(2333333)`；我们的 runner 在发完协议 `done`
  之后 `os._exit(0)`（DataLoader spawn worker 在 Windows 上会吊死解释器退出）。
- 协议 stdout 一律 UTF-8（Reporter 构造时 reconfigure —— 本机是 cp932 控制台，
  没这条中文 message 直接 UnicodeEncodeError）。
- 两侧数据顺序由 filelist 行序+seed 决定，与路径字符串无关（gate1_prepare 校验
  行序一致）。

## 已知有意偏离（vendored vs 原版，全部有注释标注在对应文件头）
- 预处理串行化（逐文件数学不变）；每次运行重建切片目录（防换数据集后的陈旧切片
  污染 filelist —— 上游同款 bug 的修复）。
- ContentVec 提取改用项目自有 onnx（本关卡 C 层定审 7.7e-4/cos 1−1e-9），顺带
  消灭 fairseq 依赖，并保证训练特征空间 == 推理特征空间（同一张图）。
- 检索库跳过 faiss：运行时本来就是暴力精确检索原始矩阵（total_fea 语义不变，
  >2e5 行 MiniBatchKMeans 压缩保留）；faiss 的 ANSI 窄字符路径雷一并消失。
- 检索库在特征提取后立即生成（原版在训练完成后）——早停也必有 index。
- best 快照 = loss_mel 的 EMA(~100 step) 启发式（GAN 无验证集，见训练页 UI 标注）；
  ckpt 原子写；停止旗标每 step 检查。
