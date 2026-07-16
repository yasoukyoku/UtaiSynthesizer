# 声音后端验证关卡（voice/ — RVC / SoVITS / f0 链）

方法论同 `converter/verify/README.md`（S31）：**对照 ORIGINAL 代码，自我一致性不算数**；
真权重优先，没有真 ckpt 就随机权重移植（原版类 `state_dict()` → strict=True 载入我们的
重实现）；torch-vs-torch 过线 **< 1e-5**，torch-vs-ORT fp32 过线 **< 1e-4**；不过就逐模块二分。
python：`converter/.venv/Scripts/python.exe`，原版参照 repo 路径写死在各脚本头部。

## gate1_rvc.py — RVC 合成器（v1 256 / v2 768，仅 f0 变体）
对照 `D:\MyDev\RVC\RVC20240604Nvidia\infer\lib\infer_pack\models.py`。2026-07-04 实测全过：

- **(a) v2 真权重**（lengv2.3.pth，48k）：原版 `SynthesizerTrnMs768NSFsid.infer()` vs
  `architectures/rvc_v2.py`，零噪声 + 固定噪声两档，m_p/logs_p/z_p/z 全 **0.0 bit-exact**，
  全前向 audio **8.8e-6 / 8.6e-6**（唯一非零源 = SineGen 相位重排，见下）。
- **(b) v1 随机移植**（"40k" 字符串 sr 配置，flow post 卷积重新随机化避免零初始化假阳性）：
  audio **1.9e-7 / 2.2e-7**。
- **(b2) 拒绝档**：nof0 → 中文 ValueError「暂不支持无音高(nof0)的 RVC 模型」（库层 + CLI
  exit 1 stderr 两层都验）；version 标签与 emb_phone 维度矛盾 → "mislabeled" 拒绝。
- **(c) torch vs ONNX**（deterministic 导出，T=200 描迹）：T=200 rnd=0 **6.8e-6**、固定 rnd
  **6.4e-6**（< 1e-4 主门）；动态 T 扫描 311/137/57/32/22/16/13/12 全部 < 5e-4（扫描档容差
  放宽的依据：ORT 与 torch 的 fp32 conv 舍入差经 30 层 dec 偶发放大 ~6×，已二分确认非结构性
  — T=12 seed=112 的 1.8e-4 outlier：har 1e-8 / z 3e-5 / audio 1.8e-4）。json `min_frames: 12`
  = 保守验证下界（T=10/11 实测也过，但走的是负 Pad 裁剪路径，不作承诺）。
- **(d) 出货导出**（noise 全在图内，非 deterministic）：`convert.py --type rvc` CLI →
  `test_output/lengv2.3.onnx` + sidecar json 全字段校验 + ORT 有限性/长度/幅度双 T 检查。

**SineGen 相位的唯一蓄意偏差**（rvc_v2.py forward 内有完整注释）：原版 `%1` 绕回检测 +
逐采样 cumsum 的相位记账**只在生产/消费两端逐位同舍入时**保持有界 — ORT 的帧级 cumsum
差几个 ULP，绕回检测被放大成漏/多 ±1 周期，2 秒音频相位漂到 **1531 周**、fp32 sin 完全
去相关（max 0.98，出货前实测）。改用数学恒等式（差整数周期，sin 不变）：
`phase(t,j) = frac(Σ_{t'<t} rad·upp)_fp64 + rad[t]·(j+1)`，帧级累加 fp64（无 fp64 kernel 的
EP 该少数节点自动落回 CPU）。与逐字 fp32 torch 差 **4.5e-7**(T=200)/**1.3e-6**(T=2000)，
且任意长度下都比原方案更贴近 fp64 精确参照 — 这既是过门手段也是长音频音质修复。

导出契约（破坏性变更，Rust 侧新契约）：
`phone[1,T,dim] f32, phone_lengths[1] i64, pitch[1,T] i64, pitchf[1,T] f32, sid[1] i64,
rnd[1,192,T] f32 → audio[1,1,T*upp]`；**rnd = N(0,1) × noise_scale 由调用方预乘**
（原版默认 0.66666，sidecar `noise.default_scale`）；帧率 10ms（`hop_ms`）。

## gate_rmvpe.py — RMVPE f0 链（先行会话已建）
对照原版 `infer\lib\rmvpe.py`：mel 结构 f64 隔离档 (<1e-4 log 域)、mel fp32 噪声档
(<1e-5 线性域)、真人声全链 f0 相对误差 <0.1%@99% + uv 一致率 >99%、动态 T、onnx 隔离档。
细节见脚本头注释。

## gate1_sovits.py — SoVITS 合成器（4.0 vec256l9 / 4.1 vec768l12，nsf-hifigan）
对照 `D:\MyDev\so-vits-svc\so-vits-svc\models.py`（+ modules\、vdecoder\hifigan\、
onnxexport\model_onnx_speaker_mix.py）。2026-07-04 实测全过：

- **(a) 4.0 真权重**（akiko_320000.pth，ssl 256/gin 256，含 enc_q+f0_decoder，双侧
  strict=True）：原版 `SynthesizerTrn.infer()` vs `architectures/sovits_v4.py`，零噪声 +
  固定噪声两档，m_p/logs_p 全 **0.0 bit-exact**（z_p 固定噪声档 4.8e-7 = ×noice_scale 的
  乘法舍入序），全前向 audio **6.3e-6 / 6.0e-6**（唯一非零源 = SineGen 相位重排，同 RVC）。
- **(a2) 随机移植补档**（真 ckpt 覆盖不到的结构：`flow_share_parameter=True` 共享 WN +
  vol_embedding，flow post 卷积重新随机化）：audio **8.9e-8**。
- **(b) 4.1 真权重**（东雪莲，ssl 768/gin 768/vol_embedding，**压缩版无 enc_q**（原版侧
  strict=False 验 missing⊆enc_q，我们侧按权重条件构建 enc_q 保 strict=True）；vol 取自
  原版 `utils.Volume_Extractor(512)` 真提取）：audio **2.2e-6 / 2.7e-6**。中文路径全程存活。
- **(b2) 拒绝档**（排期项全部中文 ValueError）：use_depthwise_conv / use_transformer_flow /
  speech_encoder 超出 {vec768l12, vec256l9} / vocoder_name 非 nsf-hifigan；config 与权重
  矛盾（ssl_dim、hop_length vs 上采样积）→「配置文件与模型不匹配」。**无 config 回退**：
  从权重推断超参（warn），与 config 构建**逐位一致**（0.0），版本按 ssl_dim 判定。
- **(c) torch vs ONNX**（deterministic 导出，T=200 描迹，两个版本各一遍）：akiko T=200
  **1.2e-6**（noise=0 / 固定 noise 两档），东雪莲 **4.0e-7 / 4.4e-7**（< 1e-4 主门）；动态 T
  扫描 311/137/57/20/10/6 全部 < 2e-6（门 5e-4，容差依据同 RVC 扫描档）。json
  `min_frames: 6`（= window_size 4 + 2，rel-pos 描迹分支下界）。
- **(d) 出货导出**（noise 全在图内，非 deterministic）：CLI → `test_output/akiko_320000.onnx`
  + `test_output/Sovits4.1东雪莲主模型.onnx`（**中文文件名端到端**）+ sidecar 全字段校验 +
  ORT 双 T 有限性/长度/幅度 + 活噪声证明（同输入两跑必须不同，实测 6.5e-2 / 1.1e-1）。
- **(e) 聚类/检索资产**（export_cluster.py CLI）：合成 kmeans .pt（**中文 speaker 键**）→
  `<speaker>.centers.npy` 逐位相等；合成 faiss "IVF12,Flat"（utils.train_index 配方）.pkl →
  `<speaker_id>.index_vectors.npy` 与原始向量**逐位相等**（reconstruct_n 按 add 序返回
  Flat 原向量；IVF 需 make_direct_map，脚本内已处理）。faiss-cpu 1.14.3 已装入 venv。

**复用与偏差**：LayerNorm/WN/MultiHeadAttention/FFN/attentions.Encoder/ResBlock1/2/
SineGen/SourceModuleHnNSF 直接复用 rvc_v2.py 的移植（2026-07-04 与 so-vits 原版逐段
diff 过 = 数学同源；window_size 是配置差：so-vits 4 vs RVC 10，显式传参）。SineGen 继承
rvc_v2 的**唯一蓄意偏差**（ONNX 稳相位恒等式，见 gate1_rvc 一节）。so-vits 特有结构
（Flip 反向裸返回、耦合层 wn 共享、TextEncoder f0_emb、F0Decoder+FFT（仅 strict 载入，
predict_f0 用户决定暂缓不入图）、vdecoder Generator 的 (k-u+1)//2 与 (stride_f0+1)//2
padding、conv_pre/conv_post 带 weight_norm、8 谐波）逐字移植在 sovits_v4.py。

导出契约（Rust 侧新契约）：
`c[1,T,ssl_dim] f32（已按 f0 帧数展开——mel2ph/repeat_expand 在 Rust 侧做，插值模式见
sidecar unit_interpolate_mode）, f0[1,T] f32（Hz，f0_to_coarse 在图内）, uv[1,T] f32,
noise[1,192,T] f32, sid[1] i64 [, vol[1,T] f32 —— 仅 vol_embedding 模型有此输入]
→ audio[1,1,T*hop_size]`；**noise = N(0,1) × noice_scale 由调用方预乘**（原版默认 0.4，
sidecar `noise.default_scale`）；vol = 原版 Volume_Extractor 语义（hop=hop_size 的帧 RMS）。
sidecar：type/version(4.0|4.1)/features_dim/speech_encoder/sample_rate/hop_size/
vol_embedding/unit_interpolate_mode/n_speakers/speakers{名:id}/noise/inputs/min_frames，
utf-8 + ensure_ascii=False（中文 speaker 名原样保留）。

**坑备忘**：python onnxruntime 在 Windows 上**打不开含中文的 session 路径**（locale ACP
问题，实测 RuntimeException "Unicode …"）——convert.py 的 ORT 自检与 gate 一律
`InferenceSession(path.read_bytes())` 从内存加载；Rust ort crate 走宽字符路径不受影响。
so-vits 原版 repo 的 vdecoder 在模块层 import matplotlib（推理用不到）——venv 已装
matplotlib 3.10.9 以便原版对照可导入。

## 关卡 2 — E2E 全链（Rust 全管线 vs 原版管线语义），2026-07-04 实测全过
方法论同 `converter/verify/README.md` 关卡 2（MSST）：**python 参照按原版编排语义驱动，
Rust 侧用 harness 跑同素材，e2e_compare SNR 比波形。SNR > 40 dB = 管线忠实。**
关卡 1 已单独证过各组件（合成器 <1e-4、ContentVec cos>0.9999、RMVPE f0<0.1%），关卡 2
专测 **Rust 编排移植**（分块/protect/KNN/2x 上采样/filtfilt/rms-mix/pad-trim）+ 合成器
onnx-vs-torch 的端到端积分误差。

**素材**：`ikanaiteyo\vocal.wav`（真人歌唱）切 20s，预重采样成 16k mono（RVC）/ 44.1k mono
（SoVITS）—— 两侧吃同一个文件，**杜绝输入重采样方差**（RVC 的 16k→16k、SoVITS 的
native→target 都是恒等）。

**Rust harness**：`src-tauri\tests\voice_pipeline.rs` 的 `voice_env_wav`（#[ignore] env 门，
镜像 separation_pipeline.rs）。直接构造 RvcModel/SovitsModel，读 sidecar json 取模型事实，
aux 模型（contentvec/rmvpe/mel）从 `data\models\aux` 解析。默认 CPU EP（数值纯净）；
`UTAI_VOICE_DEVICE=auto` = GPU 链。
```bash
# 确定性 det 导出（不覆盖出货 data\models）——SineGen 图内噪声清零，Rust 侧 noise_scale=0：
converter\.venv\Scripts\python.exe convert.py --input <pth> --output converter\test_output\det\<x>.det.onnx --type rvc|sovits --deterministic
# Rust（bash，CUDA 要 runtime\cuda 前置 PATH）：
UTAI_VOICE_KIND=rvc UTAI_VOICE_INPUT=<16k.wav> UTAI_VOICE_MODEL=<...\lengv2.3.det.onnx> \
UTAI_VOICE_INDEX=<data\models\rvc\lengv2.3.npy> \
UTAI_VOICE_OPTS='{"noise_scale":0.0,"index_ratio":0.0,"protect":0.33,"rms_mix_rate":0.25}' \
UTAI_VOICE_OUT=<out.wav> cargo test --test voice_pipeline voice_env_wav -- --ignored --nocapture
```
**python 参照**（`e2e_rvc_ref.py` / `e2e_sovits_ref.py`）：驱动**原版编排**（RVC 真 `Pipeline`
对象的真 `vc()`/`get_f0()`；SoVITS 逐字转写的 slice_inference 单段路径）+ **原版 torch 合成器**
（真权重，torch.randn/randn_like 清零 = Rust det + noise_scale=0）。以下替换均已被关卡 1 对
真原版证过，故属可归因基准：
- ContentVec：fairseq hubert 在 Windows 无 wheel → 注入我们的 contentvec_*.onnx（Rust 同款
  onnx，此级残差≈0；gate_contentvec cos>0.9999）。
- f0：注入 rmvpe_e2e.onnx，python log-mel 前端逐字镜像 f0.rs；原版 coarse 量化（RVC np.rint）
  / post_process（SoVITS resize+uv+gap 插值）逐字转写（gate_rmvpe 证 onnx vs rmvpe.pt）。
- 输出保持 float（跳过原版尾部 int16 量化，Rust 为 DAW 留 f32，既定偏差）。
- RVC index-on：faiss **IndexFlatL2** 精确检索，与 Rust 暴力 top-8 用**同一批** `lengv2.3.npy`
  向量（两侧都精确 KNN）。

**e2e_compare_voice.py**：`<label> <ref.wav> <rust.wav> …` 三元组，出 SNR/corr/峰值/rms +
中段 SNR<65 的分频带误差。

### 实测（2026-07-04，CPU EP，20s 真人声）
| 用例 | SNR | corr | 归因 |
|---|---|---|---|
| RVC index-off | **61.34 dB** | 1.000000 | 合成器 onnx-vs-torch(~1e-4，SineGen 稳相位恒等式 + 30 层 dec conv fp 舍入) + fp；管线纯净 |
| RVC index-on (0.75) | **49.43 dB** | 0.999994 | 同上合成器残差，在 index 混合特征工作点上多解相关 ~12 dB；**KNN 已排除**（见下二分）|
| SoVITS 4.0 变体 B（resample_poly=Rust） | **66.39 dB** | 1.000000 | 合成器 onnx-vs-torch + numpy-FFT-vs-rustfft；lag 0 |
| SoVITS 4.1 变体 B（resample_poly=Rust） | **66.69 dB** | 1.000000| 同上（vol_embedding 链） |
| SoVITS 4.0/4.1 变体 A（原版 torchaudio 内部 16k 重采样） | 0.70 / 1.33 dB | 0.58 / 0.63 | **全部是重采样选型**（见下） |

**index-on 49dB 的二分定位（KNN 排除）**：`e2e_rvc_ref.py --knn expanded` 让参照的检索改用
Rust 的展开范数 L2（`|q|²−2q·v+|v|²`，clamp 1e-9）替代 faiss 直算 `‖q−v‖²`。结果 Rust vs
参照(faiss) 与 vs 参照(expanded) **逐位同为 49.43 dB / max_diff 4.291e-2** —— KNN 距离式选型
对结果零影响（合成好的混合特征 ref≡rust 到 ~128 dB）。故 49dB 的源在检索之后 = **合成器
torch-vs-onnx** 数值残差（两 ref 变体都用 torch 合成器，故同值）。（另：400 帧真实尺度合成查询上
faiss 与展开范数 top-8 集合 0% 分歧、混合特征 128 dB —— 良分离向量下两式等价，close-neighbor 的
灾难消去在本素材未触发。）

**变体 A 低 SNR = 已完整归因、非 Rust 缺陷**：`e2e_sovits_ref.py` 出两版内部 44.1k→16k 重采样
—— A=原版 torchaudio（f0 路 width=128、hubert 路 default），B=scipy.signal.resample_poly（=Rust
选型，单个 wav16k 喂 f0+hubert）。关键旁证：**refA-vs-refB（同一原版 torch 合成器，只差重采样器）
= 0.70/1.33 dB，与 refA-vs-Rust 逐位同值** —— 即变体 A 的 ~65 dB 落差 100% 在参照内部由重采样器
制造，与 Rust 无关。机理：NSF 声码器的 SineGen 谐波源相位是 f0 的累积积分，重采样器换选型使 f0
轨微变（+~8 采样组延迟），20s 后谐波相位去相关 → **波形 SNR≈1 dB 但 mel-log-SNR（相位无关）
仍 22–26 dB**（听感等价）。整链既定标准化到 resample_poly（见 features.rs 注释），故**判据取变体 B
（同重采样器）= 66 dB 过线**。RVC 输入即 16k、无内部重采样，故 index-off 直接 61 dB。

### CUDA EP 冒烟（RVC det，UTAI_VOICE_DEVICE=auto，runtime\cuda 前置 PATH）
四个 session 全 `device=Auto → using CUDA`，输出无 NaN。CUDA vs CPU-EP **42.17 dB / corr
0.999970**（vs 参照 41.95 dB）。比分离链（关卡 4 的 61-88 dB）低是**声码器特性**：TF32 微扰
rmvpe f0 → 同上 NSF 相位去相关（误差集中在 10-24k 高频带 -25.8 dB）；corr 0.99997 + 无 NaN =
TF32 正常、链路稳定。

**坑/教训**：① 我们的 rmvpe onnx 输出 `[1,T]`，原版 `RMVPE.infer_from_audio` 返回 1-D —— 参照
shim 必须 flatten，否则 `f0[:p_len]` 切到 size-1 batch 轴变 no-op，vc() 里报 2198≠2201 形状崩。
② pipeline.py 顶层 import parselmouth/pyworld/torchcrepe（我们不用的 f0 法，未装）→ 参照里
`sys.modules.setdefault` 塞空模块即可。③ torchaudio 无匹配 torch 2.12 的 wheel，`pip install
torchaudio --no-deps` 装 2.11.0（Resample 是纯 torch functional，无需 libtorchaudio C 扩展，可用）
—— 仅变体 A 需要。④ 中文路径导出走 `PYTHONUTF8=1`（cp932 控制台会崩在打印路径处）。

---

## S36 质量路径关卡（浅扩散 / NSF-HiFiGAN / 自动f0 / E2E 全变体，2026-07-05 实测全过）

### gate1_diffusion.py — 扩散模型（Unit2Mel + GaussianDiffusion + WaveNet）
对照 `D:\MyDev\so-vits-svc\so-vits-svc\diffusion\{unit2mel,diffusion,wavenet}.py`（dpm/unipc
采样档 lazy-import 原版 solver 模块驱动）。真权重 = 东雪莲扩散模型.pt（191 tensors，yaml
vocoder.ckpt 缺失 → 临时补丁 yaml 指向 pretrain/nsf_hifigan 绝对路径的先例在此建立）。
- **(a) 全采样 torch-vs-torch**（真 gt mel + 真 Volume_Extractor + 固定噪声）：naive/ddim/
  pndm/dpm-solver/dpm-solver++/unipc + 纯扩散 **全部 0.0 bit-exact**（8 档；含 dpm
  `lower_order_final steps<10` 两分支）。
- **(c) torch vs ORT**：encoder 2.0e-6；denoiser 整数/分数 t（13.7/456.789）3.2e-6–9.6e-6
  < 1e-4；动态 T 扫描 < 5e-4。schedule 重算（f64 np.linspace→f32）vs ckpt buffer **逐位 0.0**
  —— Rust 侧 f64 重算调度的依据。
- **(f) 随机移植补档**：n_spk=8 one-hot MatMul（**蓄意偏差**：`spk_mix@W` 替代 Embedding
  gather，0-based forward:161 语义）vs 原版 sid 路径 0.0；分数 mix 2.4e-7；k_step_max=100
  shallow-only 守卫两侧同 raise。
- 图边界：**encoder.onnx（条件嵌入）+ denoiser.onnx（ε 网络，time 输入 f32 —— dpm/unipc
  喂非整数 t）两图；采样循环全在 Rust 宿主**（`betas[:t]` 截断使 schedule 依赖运行时 k_step，
  不可烘图）。sidecar 契约见 `export_diffusion.py`/已装 diffusion.json。

### gate1_nsf_hifigan.py — NSF-HiFiGAN 声码器（aux，扩散+增强器共用）
对照 `vdecoder\nsf_hifigan\models.py`，真权重 pretrain/nsf_hifigan/model（14.2M 参数）。
SineGen 复用 rvc_v2 稳定相位重排（同一蓄意偏差）：零噪声 sine 2.98e-6 / audio 8.3e-7
corr 1.000000；torch vs ORT det 3.5e-6，动态 T 全 < 5e-4；活噪声两跑差 1.2e-1（证噪声在图）。
nvSTFT get_mel 原版 vs 独立 numpy f64 参照 4.6e-6/7.3e-7（**注意**：真实音频上近 clamp(1e-5)
的 mel bin 经 ln 放大 fp 舍入至 ~1.8e-4 —— mel 对拍必须用带噪声底的合成信号，静音素材
不可卡紧线）。Rust `inference/mel.rs` 参考向量由此关卡产出（实测精度 ~2e-6，测试线 1e-4）。

### gate_autof0.py — 自动 f0（<stem>.f0.onnx 独立小图）
对照 models.py:520,523-527（predict_f0 分支）。akiko(4.0)+东雪莲(4.1) 双真权重：
- torch wrapper vs `orig.infer(predict_f0=True)` 返回的 f0：**0.0 bitwise**（含全无声守卫）。
  normalize_f0 的 `uv_sum[uv_sum==0]=9999` → torch.where 改写（蓄意偏差）恒等 0.0。
- ORT：lf0 域 <1.2e-6（**容差按网络输出域（lf0）与相对 Hz 判**，f0 是 O(400Hz) 量纲，
  裸 Hz 读数天然 ×440 —— coarse-bin 翻转 0/200 是下游无害证明）。
- **链路档（接线序证明）**：f0.onnx→主图（f0=f0_pred，uv 不变）vs 原版整体 infer：
  c-torch audio ≤2.1e-5；c-ort ≤1.2e-3（f0 亚 mHz 扰动经 SineGen 相位积分放大，固有敏感、
  非接线错，硬帽 1e-2）。主 onnx 重建 **MD5 与已装文件一致**（主图零扰动证明）。

### e2e_quality_ref.py — 关卡2 E2E 全变体（vs 原版 torch 转写，REFPOLY 判定路径）
两侧同一 20s 44.1k 素材（vocal.wav offset 5s，原版 slicer 实证单块）；ZeroNoise ↔ Rust
`debug_zero_noise`（SovitsOptions 隐藏测试口）+ det 主图/det 声码器；Rust harness 用
`UTAI_VOICE_DIFF/VOCODER/F0PRED` env 直驱（tests/voice_pipeline.rs）。2026-07-05 实测：

| 变体 | SNR (dB) | corr |
|---|---|---|
| 浅扩散 dpm-solver++ k100 sp10 | 57.37 | 0.999999 |
| 浅扩散 naive k100 | 59.17 | 0.999999 |
| 浅扩散 unipc k100 sp10 | 57.63 | 0.999999 |
| 纯扩散 dpm++ sp10 (t=1000) | 62.69 | 1.000000 |
| 浅扩散+二次编码 | 57.86 | 0.999999 |
| 自动f0 东雪莲 / akiko | 48.89 / 42.09 | 0.999994 / 0.999969 |
| 增强器 key=0 / key=+4 | 54.10 / 54.63 | 0.999998 |

全部 > 40 过线；残差解剖：浅扩散 57-59 ≈ VITS onnx-vs-torch 基线残差主导（纯扩散无 VITS
→ 62.7 最高）；autof0 42-49 = f0 微扰相位积分（gate_autof0 已归因）。
**second_encoding 实证裁决**（原版 infer_tool.py:313 缺 unsqueeze 疑似 bug）：逐字执行不崩
—— 2-D c 被 torch 广播救回，与规范 batched 版差 7.6e-6（纯求和序）→ 非 bug，Rust 用规范
布局，无偏差。
**CUDA 冒烟**（V1）：36.8 dB / corr 0.9999 / 无 NaN；vs 同二进制 CPU 输出同值同频带剖面
（TF32 → NSF 相位去相关，误差集中 4-24kHz）→ 100% EP 数值。**校准更新：SoVITS 浅扩散链
CUDA 基线 ≈ 37 dB/corr 0.9999（比 S35 RVC 的 42 dB 更低是链更长），CUDA 回归看
corr/无NaN/CPU自比，不拿波形 SNR 卡线。**


## gate1_sovits_v2.py — SoVITS 4.0-v2 / VISinger2（主图 + .f0 companion + ConviSTFT）

对照 D:\MyDev\TESTING\SoVITS-4.0_v2\src\so-vits-svc（官方 svc-develop-team @ cf5a8fb,分支
4.0-v2,后改名 Moe-SVC-V2;快照与分支终版零代码差异,溯源见 TESTING research/s68_sovits_v2_design.md §0）。
2026-07-17 全 tier PASS,真权重 = chika G_73000（epoch402,社区分发件）+ 官方底模 G_0。

- (i) ConviSTFT vs torch.istft(hann, center=True, length=T·hop)：随机 amp/相位全程 **8.0e-8**。
- (a) chika 真权重 torch-vs-torch（固定 z/相位,原版打 randn_like+Generator_Noise 补丁）：
  text_encoder / predict_mel / prior m_p+logs_p **逐位 0.0**;dec_harm 4.1e-5(稳定相位恒等式);
  audio ours-vs-fp64参照 **2.5e-6**,而原版 fp32 自身漂移 2.4e-4 → 支配判据 ~100× 通过。
- (a2) G_0 底模同层：逐位 0.0 ×3;audio ours-vs-fp64 **1.2e-6**（原版 fp32 漂 1.6e-5）。
- (b2) 无 config 构建与 config 构建 **逐字节同 audio**;4.x ckpt 被 v2 builder 响亮拒绝;
  is_v2_state_dict 判定正确（路由 = convert_sovits 开头按 state_dict 命名空间分派）。
- (c) torch(ours) vs ONNX shipping 导出（v2 无图内随机,shipping 即确定图）：
  T=200 **2.6e-5** / 扫 T∈{137,64,23,7,6} ≤ 2.8e-5,min_frames=6 直接实测。
- (f) autof0：wrapper vs 原版 predict_f0 分支（normalize_f0 factor 钉 1）**逐位 0.0**;
  .f0.onnx ORT Hz 相对 **1.2e-6**。

**复用与偏差**：复用 rvc_v2（LayerNorm/WN/attn Encoder/ResBlock1/2 等）+ sovits_v4
（FFT/Flip/耦合层/normalize_f0/config 发现）;v2 特有类 vendored（ConvReluNormV2/双 FFT 解码器/
DDSP 谐波+iSTFT 噪声/下采样条件 HiFiGAN）。六项登记偏差（架构文件头注,全部 gated）：
①z_p 噪声显式输入 ②相位显式输入（原版 onnxexport 留图内 rand=非确定）③ConviSTFT 窗=hann
（原版 onnxexport ctor 默认 hamming 是上游 bug,训练用 torch.istft+hann）④istft 对齐=前裁
n_fft/2+按 length 取尾（原版 onnxexport 对称裁 768 = 与训练错位 256 采样,实测 0.14,不复刻）
⑤companion normalize_f0 用 random_scale=False（v2 infer 漏传该旗=训练期抖动漏进推理,4.x 分支
已修的语义）⑥稳定相位（rvc_v2.SineGen 惯例:fp64 帧级 cumsum frac+帧内 ramp,逐谐波 frac;
原版 fp32 逐采样 cumsum 在 30s×64 次谐波达 ~4e5 rad,sin 去相关且 torch↔ORT 累加发散,
实测 0.24-0.54 → 修后 2.6e-5）。

```
导出契约（sidecar type=sovits, version="4.0-v2"）:
  c[1,T,256] f32(Rust 预扩帧) + f0[1,T] Hz + noise[1,192,T](N(0,1)·0.4 预缩放)
  + phase[1,1025,T](uniform·2·3.14-3.14;零=确定) + sid[1] i64|spk_mix[1,n_spk]
  -> audio[1,1,T·512]
  主图无 uv/vol;.f0.onnx companion = c/f0/uv/sid -> f0_pred[1,T] Hz
  sidecar 新增 "phase" 块 {"phase_input":[1,1025,"T"]};min_frames=6
```

**坑备忘**：①上游包不带 config.json（底模/chika 均无）→ 按官方模板重建/无 config 推断路径
gate 证逐字节等价;chika spk 名从 kmeans_10000.pt 键取证 ②torch≥2 跑原版参照需 istft
view_as_complex 垫片+抢占 root logger（原版 utils import 时设 DEBUG=numba 洪水）③torch.istft
`length` 语义=前裁 n_fft/2 后直接取 length（尾部是真 OLA 内容,不是补零——踩过）。