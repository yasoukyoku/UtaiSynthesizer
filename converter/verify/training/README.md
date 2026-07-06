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
