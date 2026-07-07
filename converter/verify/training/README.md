# 训练链验证关卡（①b，S37 起）

方法论同 `converter/verify/README.md`：**参照物永远是原版仓库的真实执行，绝不自证**。
本目录覆盖 RVC 训练链（关卡0/1）、SoVITS 训练链（gate0_sovits_* / gate1_sovits_*）
与浅扩散 train_diff 链（gate0_diff_* / gate1_diff_* + regress_extract_sovits，S39，
见文末章节）；声码器阶段落地时在此追加。

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

---

# SoVITS 训练关卡（S38 起，gate0_sovits_* / gate1_sovits_*）

原版参照：`D:\MyDev\so-vits-svc\so-vits-svc`（4.1-Stable @ 730930d，代码零改动）。
**「原版时代环境」= RVC 整合包 runtime**（python3.9 + torch 2.0.0 + torchaudio
2.0.1+cpu + fairseq 0.12.2 + librosa 0.9.1）—— so-vits requirements 自己钉的就是
librosa==0.9.1 / fairseq==0.12.2，与 RVC 整合包同代；so-vits 伴生 venv 是半空的
（无 fairseq/librosa），fairseq 无 Windows wheel，此 runtime 是本机唯一能真跑
原版预处理的环境。中间产物在 `D:\MyDev\TESTING\utai-v2-testing\sovits_*`。

⚠️ **rmvpe 是两个血统**：`aux/rmvpe.pt` = RVC 版（裸 state_dict 的 E2E）；so-vits
vendored 的是 yxlllc/RMVPE 分支（E2E0，多 60 个 `unet.tf.*` 键，`{'model': sd}`
包装）→ 训练资产 `data/models/training/sovits/rmvpe.pt`（yxlllc release 230917 的
model.pt）。两个文件**不可互换**（本关卡首跑就是被这个炸出来的）。

## 关卡0 —— 预处理对拍

```
# ① 切片（双方共同输入；上游无切片器，README 指定同款 openvpi 工具）：
training\.venv\Scripts\python.exe converter\verify\training\gate0_sovits_prepare.py
# ② 原版侧（RVC runtime 原样跑 resample/flist/hubert_f0 + vencoder oracle，CPU fp32）：
D:\MyDev\RVC\RVC20240604Nvidia\runtime\python.exe converter\verify\training\gate0_sovits_orig.py
# ③ 我方侧 + C1 + 对拍：
training\.venv\Scripts\python.exe converter\verify\training\gate0_sovits_run_ours.py
D:\MyDev\RVC\RVC20240604Nvidia\runtime\python.exe converter\verify\training\gate0_sovits_c_resample.py
training\.venv\Scripts\python.exe converter\verify\training\gate0_sovits_compare.py
```

原版侧 harness 补丁（零数值影响，脚本头注释登记）：loguru 桩、ProcessPoolExecutor
→串行内联（runpy __main__ 无法被 spawn unpickle）、repo configs 快照/恢复、
pretrain/rmvpe.pt 补入（双方同一权重文件）。

**S38 实测读数（2026-07-05，全 PASS；gate 数据 = 3×60s 切出 33 切片）：**

| 层 | 项 | 读数 |
|---|---|---|
| C1 | resample 链代码轴（双方 librosa 0.9.1） | **逐位 0**（33/33 文件） |
| C2 | ContentVec 768（同16k输入，onnx vs 真 fairseq fp32 CPU） | max 1.99e-4，min cos 0.99999984 |
| C2 | ContentVec 256（同上） | max 7.4e-5，min cos 0.99999991 |
| C3 | f0 定审（同44k输入，双方 fp32 CPU，torch 2.0↔2.5 轴） | **0/12389 帧超 0.5Hz、0 uv 翻转、max 0.6mHz** |
| C4/C5 | spec / vol 定审（同44k输入，torch 2.0↔2.5 轴） | **逐位 0.0 / 0.0** |
| A | 44k wav（librosa 0.9.1 kaiser_best vs 0.11 soxr_hq 轴） | min SNR 52.3 dB |
| A | soft / spec / vol（输入轴叠加） | cos 0.987 / 55.6 dB / 94.6 dB |
| A | f0 浊帧判据 | 浊帧 0.67% 超 0.5Hz，uv 翻转 0.15% |

### 关卡0 钉死的数值轴/度量备忘
1. **A 层 f0 必须只判双浊帧**：so-vits 后处理在清音区做线性插值填充，uv 边界帧随
   输入(-52dB)漂移一帧即把锚点差扩散到整段清音区（全帧口径虚高到 9.3%，浊帧口径
   0.67% —— 与 RVC A 层 0.75% 同级）。C3 已证同输入下代码 0 帧差。
2. spec/vol 在 torch 2.0 vs 2.5 之间**逐位一致**（CPU fp32 的 stft/unfold 未变）。

## 关卡1 —— 训练等价（逐 step loss 轨迹 vs 原版 train.py）

```
training\.venv\Scripts\python.exe converter\verify\training\gate1_sovits_prepare.py
training\.venv\Scripts\python.exe converter\verify\training\gate1_sovits_run_orig.py
training\.venv\Scripts\python.exe converter\verify\training\gate1_sovits_run_ours.py ^
    > D:\MyDev\TESTING\utai-v2-testing\gate1_sovits_ours_steps.jsonl
training\.venv\Scripts\python.exe converter\verify\training\gate1_sovits_compare.py
```

同一份关卡0 我方预处理产物（filelist 绝对路径双方直读）+ 同 seed(1234) + 同底模
(vec768 G_0/D_0) + **双方 fp32 CPU + 同一个 torch(2.5.1)**（我们 venv 跑原版
train.py 未改文件 —— 隔离代码轴，RVC 关卡1 同款）。原版侧 shim（只动执行环境：
faiss 桩 / Tensor·Module.cuda→恒等 / DDP 剥 device_ids / 绕过 mp.spawn 直调
run(0,1,hps) / gloo env + USE_LIBUV=0）。config: all_in_mem=true（双方
num_workers=0）、vol_embedding+vol_aug=true（覆盖响度增强随机路径）、log_interval=1。

**S38 实测读数（全 PASS，16/16 step 对齐）：**

| 分量 | max 相对差 | mean 相对差 |
|---|---|---|
| loss/g/total | 1.6e-8 | 5.8e-9 |
| loss/d/total | 1.7e-7 | 9.7e-8 |
| loss/g/fm | 1.6e-7 | 6.2e-8 |
| loss/g/mel | 2.4e-8 | 1.0e-8 |
| loss/g/kl | 1.7e-7 | 7.5e-8 |
| loss/g/lf0 | 2.4e-4 | 6.9e-5 |

lf0 的 2.4e-4 是相对量纲效应：lf0 值域小（~1e-2），绝对差 ~1e-6 与其他分量同级；
g_total（包含 lf0，值域 ~40）1.6e-8 证明合成完整。so-vits **没有** RVC 的
mel>75/kl>9 显示夹取（TB 记原始值），对拍脚本无 clamp。

### SoVITS 关卡复跑注意
- 关卡1 前置依赖关卡0 我方产物（sovits_ours 的 filelists/config/特征）。
- 原版侧 gate1 在 so-vits repo 里留下 `logs/gate1_sovits/`（gitignore 外目录，
  复跑由 prepare 清理重建；repo 代码零改动）。
- SoVITS 训练侧 vendored 有意偏离（全部登记在对应文件头）：预处理串行化 / 切片
  统一用 slicer2（上游要求用户手切，同款工具同默认参数）/ 响度归一默认关（上游
  默认开但其 README 自认损音质；关卡里双方都开）/ ContentVec 用项目 onnx /
  filelist 种子化 shuffle + UTF-8（上游 locale 写 UTF-8 读 = CJK mojibake 雷修复）/
  rmvpe 每 run 构造一次（上游每文件重载 180MB）/ kmeans 用上游 use_minibatch 代码
  路径（默认全量 KMeans 万级中心不可行）且 n_clusters 截到行数 / DataLoader
  persistent_workers（Windows spawn 每 epoch 重启 worker 之灾）/ 停止旗标逐 step /
  ckpt 原子写 / best=EMA(mel) / 完训补存 latest G/D + release 导出
  （compress_model 语义：去 enc_q + fp16）。

---

# 浅扩散 train_diff 关卡（S39 起，gate0_diff_* / gate1_diff_*）

原版参照：`D:\MyDev\so-vits-svc\so-vits-svc`（4.1-Stable @ 730930d，代码零改动）。
gate0 原版侧仍 = RVC 整合包 runtime（torch 2.0.0 / torchaudio 2.0.1+cpu /
librosa 0.9.1）；gate1 双方 = 我们的 venv（torch 2.5.1，隔离代码轴，S37/S38 同款）。
输入复用 S38 sovits gate 的 33 切片（同 44k wav 喂双方 = C 层；aug 抽样上界依赖
max|wav|，输入轴会使 aug 产物不可比）。

## gate0_diff —— --use_diff 预处理对拍（vol/mel/aug_mel/aug_vol）

```
training\.venv\Scripts\python.exe converter\verify\training\gate0_diff_prepare.py
D:\MyDev\RVC\RVC20240604Nvidia\runtime\python.exe converter\verify\training\gate0_diff_orig.py
training\.venv\Scripts\python.exe converter\verify\training\gate0_diff_run_ours.py
training\.venv\Scripts\python.exe converter\verify\training\gate0_diff_compare.py
```

原版侧 harness（零数值影响，脚本头登记）：预置 soft/f0/spec（skip-if-exists 跳过
→ 无需 fairseq/GPU）+ 逐文件直调 process_one（绕开 spawn 执行器与 shuffle——
spawn 子进程不继承种子、shuffle 先消耗随机流）+ random.seed(1234) + sorted 文件序
+ configs 快照/恢复 + CUDA_VISIBLE_DEVICES=-1。我方侧 = extract_all(diff_mode=True,
aug_seed=1234)（random.Random(1234) 与全局 random.seed(1234) 同 MT19937 流）。

**S39 实测读数（2026-07-06，全 PASS；33 切片）：**

| 项 | 读数 |
|---|---|
| .vol.npy | **逐位 0.0** |
| .mel.npy（torch 2.0↔2.5 轴） | max_abs 9.5e-7 |
| .aug_mel keyshift（随机流对齐证明） | **33/33 逐 draw 一致** |
| .aug_mel 响亮位（ln-mel > -10） | max_abs 7.6e-6 |
| .aug_mel 近 clamp 位 | max_abs 1.1e-4（12/21632 项，全部 ln<-10.1 = S36 记档的近 clamp ln 放大；变调路径非 2 幂 FFT 的 torch 版本轴） |
| .aug_vol.npy（同响度 shift） | **逐位 0.0** |
| librosa mel 滤波器组 0.9.1↔0.11 | **逐位一致**（melbasis_091.npy 留档） |

## gate1_diff —— 训练等价（逐 step loss 轨迹 vs 原版 train_diff.py）

```
training\.venv\Scripts\python.exe converter\verify\training\gate1_diff_prepare.py
training\.venv\Scripts\python.exe converter\verify\training\gate1_diff_run_orig.py
training\.venv\Scripts\python.exe converter\verify\training\gate1_diff_run_ours.py
training\.venv\Scripts\python.exe converter\verify\training\gate1_diff_compare.py
```

同 gate0 我方产物 + 同 vec768 底模 + 同 yaml（fp32 CPU / num_workers 0 /
cache_all_data / batch 4 / interval_val 8 / 3 epochs = 24 步 = 我方 total_steps，
完成判定与自然结束重合）。原版侧 harness：runpy 原样跑 train_diff.py + 种子 +
loguru/faiss 桩 + librosa.get_duration(filename=)→path= shim（librosa 0.11 环境轴）。
**≥2 个 interval_val 边界是硬要求**：第一个边界含 NsfHifiGAN 懒加载 Generator
构造的 RNG 大块消耗，之后的不含——两段都逐 step 对齐才证明 RNG 消耗模型完整
（vendored 代码严禁"预热优化"该懒加载）。

**S39 实测读数（全 PASS）：**

| 项 | 读数 |
|---|---|
| train/loss（24/24 步） | **max_rel 0.0（逐位一致）** |
| validation/loss（交集 8/16） | **max_rel 0.0** |
| validation/loss step 24 | 原版 TB 缓冲丢失（原版不 close SummaryWriter，flush_secs=120）→ 用原版 stdout 3 位小数补核：0.081 vs 0.081112 ✓ |

## regress_extract_sovits —— 共享文件回归（S38 基线）

```
training\.venv\Scripts\python.exe converter\verify\training\regress_extract_sovits.py
```

extract.py 是 S38 主链共享文件（本次加 diff_mode 分支 + vol 门扩展）：对同输入
全新跑非 diff 模式，与 S38 存档产物比对。**S39 读数：132 产物精确相等（.pt 按
张量精确相等——S38 基线早于原子写修复，zip 档案根名不同属序列化元数据轴；
.npy 逐字节），非 diff 模式零杂散 diff 产物。**

## 冒烟（runner 直驱，TESTING/smoke_diff4{0,1}*.json）

- run A：共享工作区（S38 smoke_sovits41）30 步完训——soft/f0/spec 缓存 mtime
  与 S38 时代逐秒一致（增量承诺兑现），val 10/20/30，best/final，encoder_dim=768。
- run B：续训至 60——首步 31 零跳号；存档清扫后恰保留 0/20/40/60/best（幸存者
  网格 = 里程碑谓词）。
- run C：优雅停 @130——stop 存档**含 optimizer**（periodic 不含 = save_opt 拆分
  偏离），done(stopped)。
- run D：停后续训至 150——best@110 跨续训保留（diffusion/best_state.json）。
- 4.0：vec256 无底模从零训 + k_step_max=100（浅扩散 config）完训。
- 转换链：export_diffusion.py 直接吃 model_150.pt + expdir/config.yaml（自动解析）
  ——schedule 重算逐位 0、ORT sanity ≤3.1e-6、sidecar speakers = 中文显示名。
- 三段式：主模型（S38 smoke）在 diff 之后续训 2 epochs 正常（G_91 续、GAN loss
  健康、diffusion/ 产物无扰）。

### train_diff 复跑注意
- gate1 prepare 会清双侧 expdir；gate0 prepare 会清 diff_orig/diff_ours。
- 原版侧 gate0 需要 diff_orig 里已预置 soft/f0/spec（prepare 负责）。
- Reporter.stage 有 0.4s 节流：一次性通知（如"无扩散底模，将从零训练"）必须
  force=True，否则会被同窗口的前一条吞掉（S39 冒烟实锤后已修）。

---

# 声码器微调（backend "vocoder"，S40）

原版参照：`D:\MyDev\SingingVocoders`（openvpi/SingingVocoders @4d0889c 2026-03-08，MIT，
PyTorch Lightning）。vendored → `training/utai_train/vocoder/`（登记偏离全在
pipeline.py 模块头 + base_task_gan.py / training_utils.py 头注；设计+红队裁决全文 =
`D:\MyDev\TESTING\utai-v2-testing\research\s40_vocoder_train_design.md` 附录 A）。

## 环境轴（与 RVC/SoVITS 的"原版时代环境"不同——本链双侧同 venv）

SingingVocoders 无 requirements/无钉版（librosa 仅 load+filters.mel、torch 2.x 兼容、
parselmouth 无版本断言）→ gate0/gate1 双侧同 training/.venv：对拍面 = 纯代码轴。
单列证据链：librosa 0.9.1↔0.11 滤波器组逐位（S39 留档）、torch CPU fp32 stft
2.0↔2.5 逐位（S38 C4）、**parselmouth 0.4.2↔0.4.7 = gate0b 本次实测**（双版本同嵌
Praat 6.1.38；9 切片 8094 浊帧 f0 逐位 0 差、0 清浊翻转）：
```
<second venv>  gate0b_parselmouth_xenv.py --dump pm_old.npz
training venv  gate0b_parselmouth_xenv.py --dump pm_new.npz && --compare pm_old pm_new
```
lightning 钉版 ==2.6.5（requirements.txt 注释：vendored get_strategy 走 lightning
私有 API——升级 lightning = 重跑 gate1 的事件）。

## gate0 — 预处理对拍（wav2spec：mel/f0/audio/uv/pe）

```
training\.venv\Scripts\python.exe converter\verify\training\gate0_vocoder.py
```
原版侧 = 原仓库 process.wav2spec 直调（绕 ProcessPoolExecutor 编排层，S39 教训）；
C 层同输入 = smoke_vocoder 的 9 个 44.1k 切片。**S40 读数：audio/mel/f0/uv/pe 五字段
9 切片全部逐位相等（bitwise 0）**。
(b) 48k 源用例（红队 A1 回归）：440Hz 正弦 48k 源经我们 slice 阶段（统一重采样
44100 = 偏离 #9）后 **f0 中位数 = 440.00Hz**；对照演示：上游原始路径（48k 直喂
wav2spec）f0 = 404.25Hz = 440×44100/48000——上游 mel 用重采样后音频、f0 用原采样率
数组按 44100 解读（process.py:34-49 + wav2F0.py:61-73），>44.1k 源整库 f0 系统性
错标且 gate0 双侧同代码结构性抓不到 → 这就是切片阶段统一 44.1k 的存在理由。

## gate1 — 训练 loss 轨迹对拍（fp32 CPU）

```
training\.venv\Scripts\python.exe converter\verify\training\gate1_vocoder_prepare.py
training\.venv\Scripts\python.exe converter\verify\training\gate1_vocoder_run_orig.py
training\.venv\Scripts\python.exe converter\verify\training\gate1_vocoder_run_ours.py
training\.venv\Scripts\python.exe converter\verify\training\gate1_vocoder_compare.py
```
- 小型化（双侧同值）：batch 2 / crop 16 / ds_workers 0 / log_interval 1（红队 A11：
  默认 100 下 24 global 步只有 step0 一个点 = 空交集假 PASS——compare 先断言点数）/
  val_check_interval 5（3 个 val 边界@global 0/10/20，跨 3 个 epoch）/ max_updates 24
  （global = 2×实际步：lightning manual-opt GAN 的 D、G 各计一步）/ seed 1234 /
  finetune 底模 = 正式 2024.02 ckpt（CPU 重存副本——原版裸 torch.load 在 CPU 下炸，
  见下"坑"）。
- 原版侧 = 原仓库 train.py 真实执行（repo 代码零改动）+ 执行环境 shim 三件
  （run_orig 头注：get_strategy→"auto"[lightning 2.6 删私有 API]、dataloader
  workers==0 合法化[= vendored A2 镜像]、CUDA 屏蔽）；work_dir 必须在 repo 树内
  （/experiments/ gitignored）——DsModelCheckpoint 的 relative_to(cwd) 出树即
  ValueError（红队 A4 实弹，原版侧首跑撞出；我方 vendored _display_path 已修）。
- **S40 读数：11 tags；training/* 9 分量 × 15 步全部 max_rel = 0.000e+00（逐位）；
  validation/{stft_loss,total_loss} × 4 边界(global 0/10/20/30) max_rel = 0.000e+00；
  步轴逐点一致。**（gate 配置必须落在存档网格上——离网 total 会触发我方侧的
  尾验偏离 #11 而原版侧没有,点数就不对齐;compare 的点数断言会响亮抓住。）

## gate2 — 导出/导入链

- converter 全量回归：`converter\.venv\Scripts\python.exe converter\verify\voice\gate1_nsf_hifigan.py`
  （S40 复跑全 PASS：gate(a) 真权重 8.3e-07/corr 1.000000；gate(b) det ORT 3.5e-06 +
  动态 T 扫描；gate(c) mel DSP；gate(d) 旧 CLI 原样重建 aux——sidecar 精确相等/
  mel npy 逐位/双跑噪声活性。**export_nsf_hifigan.py 参数化(--stem)+自检(--no-selfcheck
  可关)对默认路径零扰动的机器证明**）。
- 导出脚本自检（每次导入用户机上跑）：deterministic 孪生 torch-vs-ORT ≤1e-4 +
  corr>0.9999 双 T + 动态 T + 活图噪声活性（S40 实测 3.9e-06~4.5e-06）。
- 三形态输入实测：{'generator':sd}（so-vits pretrain 2022.12，CJK stem）、
  lightning ckpt（v0.0.2 底模，剥 generator. 前缀）、训练 weights/ 快照
  （冒烟 vocoder_best.ckpt + 伴随 config.json）→ 全过；mini_nsf（pc 2025.02）
  中文拒绝（导入面与 exporter 双层）。

## 冒烟（runner 直驱，TESTING/smoke_vocoder/run*.json）

素材 = ikanaiteyo vocal.wav（126s 44.1k 干声 → 9×≤15s 切片）。GPU：
- run A：30 步完训——协议 JSONL 零污染；periodic@10/20/30 = weights/ 快照
  （vocoder_<实际步>.ckpt + config.json）；best = 真 val loss（0.3593→0.3562 递减）；
  lightning 工作区档 = model_ckpt_steps_{40,60}（global=2×实锤；keep=2 清扫生效）。
- run B：总步 60 续训——31→60 零跳号，best 跨续训延续（0.3511→0.3494），final@60。
- run C：优雅停 @81（离网）——stop 档 model_ckpt_steps_162 补存（离网尾段不丢，
  红队 A8），weights/vocoder_81 + kind=stop；清扫仍 keep=2。
- run D：死胡同守卫——总步 60 < 进度 81 → 响亮拒绝（Rust 侧另有 //2 口径 guard）。
- run F：freeze_mpd=true 10 步——**MPD 参数 vs 底模逐位不变、MSD/G 均变化**（断言
  过；freezing_enabled 联动 = 红队 A3）。
- SovitsOptions serde 兼容单测（tests/voice_pipeline.rs）：缺键/null→None（默认
  声码器路径）、CJK 名透传、未知键不炸。cargo test 全绿（41+5+3+1）。

## 坑（S40 新增）

1. **上游 wav2spec 的 f0/mel 采样率错配**（gate0(b) 有数值实证）——切片统一 44.1k
   是数学正确性问题，不是便利问题。
2. **lightning global_step = 2×实际步**（manual-opt GAN，D/G 各计一步）：ckpt 文件名、
   TB 步轴、max_steps、log_interval 全是 global 口径；协议/UI/run.json 全是实际步。
   任何比较 total_steps 的地方漏 //2 = 步数翻倍级 bug（Rust guard 有注释）。
3. **val_check_interval（int）+ check_val_every_n_epoch=None = 跨 epoch 累计 batch 数**
   = 实际步口径——与 save_every_steps 直接对齐（gate1 3 epoch 交叉实证）。
4. 原版 DsModelCheckpoint 的 relative_to(cwd)：出树工作区首存档即崩（vendored
   _display_path 修；原版侧 gate 用 repo 内 work_dir）。
5. 原版 get_strategy 深走 lightning ≤2.5 私有 accelerator_connector API——2.6 直接
   AttributeError（我方 pipeline 传 "auto" 登记偏离；单设备语义等价）。
6. CUDA 存档底模 + CPU 训练：上游裸 torch.load 崩（vendored map_location="cpu"）。
7. 上游 print_arch 用裸 print 打 stdout —— 协议卫生：UtaiNsfTask 覆写为 logging；
   root logger 必须在任何 vendored/lightning import 前配置（basicConfig(stdout) 才是 no-op）。
8. 函数体内的缩进 import（`from utils import ...`）会躲过行首锚定的 vendored 重写
   正则——全包 grep 收尾（vendor_vocoder.py 教训）。
9. 连续歌声在 openvpi 默认切片参数下可能只出个位数巨型切片（126s → 2×60s 实测）
   → 切片 ≤15s 上限（偏离 #10，兼收 val 全长前向显存）。
10. **尾验偏离 #11（用户 S40 走查提出）**：自然完训停在存档网格之间时 lightning
   不跑收尾 val（优雅停会跑——实测两场），final 档从未与 best 比较 → pipeline
   post-fit 补 `trainer.validate(task, verbose=False)`（verbose=False 硬要求：
   结果表裸 print 打 stdout=协议）。实测（total 13/save 5）：修前 final metric=None、
   best 卡在 10；修后 val 0.363→0.3614→0.3585→**0.3577@13** 且 best 正确更新到 13。
   ⚠️ 连环坑：上游 `build_model()` **无 return**（只赋 self.generator/discriminator），
   `self.model` 永远是 None——setup 幂等守卫拿 self.model 当哨兵永不生效，
   trainer.validate 二次 setup 重建随机权重 + 因工作区有 ckpt 跳过底模播种，
   尾验得 0.88（随机 G 的成绩）：哨兵必须用 `self.generator`。
11. 上游 spec_to_figure 用 plt.pcolor 在 1025×万列网格逐格画 quad——每次 val 10 张图
   ≈ 用户实测 ~2min 边界停顿的大头；vendored 登记偏离改 pcolormesh
   （openvpi 自家 DiffSinger HEAD commit #302 同款修法），单图 0.20s。

---

# S41：PSOLA 数据增强关卡（2026-07-07）

方法论提醒：增强是我们自己的设计（音频域 PSOLA 变调副本，歌声领域零先例——证据链与
红队裁决全在 `D:\MyDev\TESTING\utai-v2-testing\research\s41_two_features_design.md`），
没有上游端到端参照，所以验证拆层：引擎语义关卡 + 份数0 逐字节 noop（vs git HEAD 真旧码
冷跑）+ 管线不变量阶梯 + 跨代 extract 回归 + runner 冒烟。

## gate_aug_semantic.py（引擎/门语义，27 检查全 PASS）
- 干净真人素材（kaz mp3）生产切片链 ×2 份：f0 目标 worst median 3.5 / p90 30.6 cents；
  时长守恒 ≤1 hop；跨 run 逐位确定（parselmouth 0.4.7 版本内性质）。
- 共振峰：PSOLA ±3st 位移 est=0.00st；度量锚（重采样 +3st）est=+3.00st（度量自身测得准，
  V11 反自证）。
- 真实脏片双臂：human(OpenUtau 渲染) p90 臂 1/1 剔除；kazane@30s median 臂 1/3 剔除；
  逐片分布档 gate_aug_semantic_dist.json。
- ⚠ V9 裁决实据：**parselmouth 对 PSOLA 毛刺失明**（同批片 rmvpe p90=323/245 cents，
  praat 读 12/17 且浊覆盖率不变=连续性先验平滑）→ 生产门四链统一 rmvpe 血统
  （vocoder 门对音频现算 rmvpe，禁用其自产 parselmouth npz f0）；part4 = 盲区在案断言
  （praat 若某天看见了会翻红提醒重评估）。
- 单元语义：全清拒/忠实留/不变调拒/高音截顶豁免（sweep 700→1000Hz +3st）。

## gate_aug0_noop.py（份数0 = 逐字节 no-op，四链全 PASS）
冷跑协议（V1 反自证）：git worktree 检出基线（默认 HEAD）跑 pipeline.run 编排层（V3，
gate_aug0_driver.py，CPU 钉死 CUDA_VISIBLE_DEVICES=-1）→ 同路径快照 → 现行代码冷跑 →
按后缀比对（V6：wav/npy/txt/json=字节；.pt=字节→张量级降级[torch zip 档案名轴]；
.wav 字节不等时=采样级降级[**libsndfile float32 wav 的 PEAK chunk 带写入时间戳**，
vocoder 切片跨 run 恒差 1 字节，本次实测定责]）。读数：sovits 21/21、rvc 43/43、
vocoder 12/12、sovits_diff 37/37 文件全等。复跑：
`.venv\Scripts\python.exe ..\converter\verify\training\gate_aug0_noop.py --backend <b>`

## gate_aug_pipeline.py（管线不变量，52 检查全 PASS）
每链 0→2→2(rerun)→3→1→0 档位阶梯：val 与份数0 逐字节同且永无 aug；检索/index 资产
与份数0 逐字节同（原片-only 拍板）；meta==幸存 aug 数；rerun 逐位稳定+缓存 mtime 不变
（rvc 每 run 重算但逐位相同=名键特征缓存有效性的实证）；降档产物连坐清除（vocoder 含
npz 侧）；**2→…→0 树 == fresh-0**。dirty 混合集：≥1 剔除 + 幸存片全材料化 + 零残渣。
diff 继承：增量路径 aug 不动 + 缓存失效重建再生 aug + diff 产物齐全。

## 既有关卡复跑协议（V13）
- 基线归档：`TESTING/utai-v2-testing/sovits_ours_s38_archive`（永久只读参照）。
- **regress_extract_sovits.py 实跑 PASS**：新 extract.py（含 aug 失败降级改动）vs S38
  时代存档 132 产物逐字节 0 失配、零杂散。
- **传递规则生效**：四链 noop 树等价 + 训练循环文件（train.py/solver/data_loaders/
  losses/harness）git diff 为空 ⇒ S37-40 的 gate1 结论直接传递，不重跑（S33 先例）。
- 旧 gate 脚本调用面：既有阶段函数签名零改动（build_flist_and_config 保留为兼容包装；
  extract_all 仅新增返回值）。

## smoke_aug.py（runner 直驱真训练，21 检查全 PASS）
sovits(dirty 混合, 剔除消息实证)→sovits_diff(同工作区继承)→rvc→vocoder 各 aug=1 短训完训：
协议全 JSON、augment/aug_check 阶段齐、done=completed、step 消息正常。JSONL 存档
`TESTING/utai-v2-testing/gate_aug/smoke_*.jsonl`。

## 坑（本次新增）
- libsndfile float32 WAV 的 PEAK chunk 时间戳（见上）——凡逐字节比较 soundfile 写的
  float wav 都要有采样级降级；scipy wavfile / int16 无此轴。
- serde_json::json! 宏递归深度：run.json 字面量加键顶爆默认 128 上限 → lib.rs
  `#![recursion_limit = "256"]`（纯编译期）。
- augment 的 seed 状态防线（aug_meta/_state.json）：aug 文件名不含 keyshift，seed 变更
  会让名键下游缓存静默错配——状态守卫先全清再生成（生产 seed 恒 1234，属 belt）。
