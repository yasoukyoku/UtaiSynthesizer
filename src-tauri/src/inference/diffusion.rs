//! Shallow-diffusion host-side math — faithful port of so-vits-svc
//! (D:\MyDev\so-vits-svc\so-vits-svc\diffusion\):
//!   - diffusion.py      GaussianDiffusion: linear schedule buffers, norm/denorm_spec,
//!                       q_sample, naive DDPM p_sample loop, ddim, pndm(PLMS)
//!   - dpm_solver_pytorch.py  NoiseScheduleVP('discrete') + DPM_Solver
//!                       (algorithm_type "dpmsolver" / "dpmsolver++", multistep order 2)
//!   - uni_pc.py         UniPC (variant='bh2', data_prediction → predict_x0)
//!
//! The denoiser (WaveNet) and the condition encoder live in ONNX
//! (`<stem>.diffusion/{encoder,denoiser}.onnx`); everything else — schedule, norm/denorm,
//! q_sample, the sampling loops — is host math here, because the solvers rebuild their
//! noise schedule over betas[..t] (runtime-dependent truncation, diffusion.py:264/342).
//!
//! Timestep dtype: the denoiser closure takes `t: f32` — naive/ddim/pndm feed integer
//! values (原版 long), dpm/unipc feed non-integer floats `(t_cont − 1/N)·N`
//! (dpm_solver_pytorch.py:276). The ONNX denoiser input is f32 for this reason.
//!
//! DOCUMENTED deviations (all algebraically identity or gate-quantified):
//!   - schedule + solver scalars are computed in f64 (原版: numpy f64 math stored as f32
//!     buffers / NoiseScheduleVP f32 arrays). f64 is strictly more accurate; the deviation
//!     vs the ckpt f32 buffers is quantified by the converter gate (~1e-8) and the E2E
//!     SNR gate. Tensor ops stay f32.
//!   - `numerical_clip_alpha(clipped_lambda=-5.1)` (dpm_solver_pytorch.py:112) is SKIPPED:
//!     for this repo's linear schedule λ_end ≈ −5.02 > −5.1 at full 1000 steps and even
//!     farther when truncated to betas[..t] — the clip never triggers (no-op). uni_pc.py's
//!     NoiseScheduleVP copy has no clip at all, so one implementation serves both.
//!   - broadcast scalar coefficients are factored before the elementwise op (e.g. DDIM's
//!     `√a_prev·(x/√a_t + c·ε)` becomes `(√a_prev/√a_t)·x + (√a_prev·c)·ε`) — algebraic
//!     identity, differs only at f32-rounding level (reference tests pass at rel 1e-4).
//!   - PLMS `noise_list` is a per-call local (原版: module state reset at dispatch,
//!     diffusion.py:307) — identical semantics, structurally impossible to leak state.
//!   - `SamplerMethod::Naive` names the 原版 fallback branch (`method is None or
//!     infer_speedup <= 1`, diffusion.py:382-388); there is no 'naive' method string in
//!     the original. The fallback is preserved: any method with speedup ≤ 1 runs the
//!     naive DDPM loop, exactly like the original dispatch.

use std::collections::VecDeque;

use ndarray::Array4;
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::StandardNormal;

use crate::{Result, UtaiError};

fn inference_err(msg: String) -> UtaiError {
    UtaiError::Inference(msg)
}

// ─────────────────────────────────────────────────────────────────────────────
// DiffusionSchedule — GaussianDiffusion.__init__ buffers (diffusion.py:63-113)
// ─────────────────────────────────────────────────────────────────────────────

/// Linear-β diffusion schedule + spec normalization bounds. All derived buffers are f64
/// (np.linspace semantics; 原版 computes these in numpy f64 too, then stores f32).
/// `log_one_minus_alphas_cumprod` is NOT ported (only used by q_mean_variance, never at
/// inference).
pub struct DiffusionSchedule {
    pub betas: Vec<f64>,
    pub alphas_cumprod: Vec<f64>,
    pub alphas_cumprod_prev: Vec<f64>,
    pub sqrt_alphas_cumprod: Vec<f64>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f64>,
    pub sqrt_recip_alphas_cumprod: Vec<f64>,
    pub sqrt_recipm1_alphas_cumprod: Vec<f64>,
    pub posterior_variance: Vec<f64>,
    pub posterior_log_variance_clipped: Vec<f64>,
    pub posterior_mean_coef1: Vec<f64>,
    pub posterior_mean_coef2: Vec<f64>,
    /// spec_min/spec_max: len 1 (原版 [1,1,1] scalar buffer) or len out_dims=128
    /// (sidecar `spec_min` read from the ckpt buffer). Broadcast over the M axis.
    pub spec_min: Vec<f32>,
    pub spec_max: Vec<f32>,
    pub timesteps: usize,
    /// Resolved k_step_max: yaml 0 / ≥timesteps → timesteps (unit2mel.py:87).
    pub k_step_max: usize,
}

impl DiffusionSchedule {
    /// betas = np.linspace(1e-4, max_beta, timesteps) — the ONLY schedule this repo's
    /// diffusion.py ever selects (hard-coded 'linear' at line 76).
    pub fn linear(
        timesteps: usize,
        max_beta: f64,
        spec_min: &[f32],
        spec_max: &[f32],
        k_step_max: usize,
    ) -> Self {
        assert!(timesteps >= 2, "diffusion timesteps 必须 ≥ 2");
        assert!(
            !spec_min.is_empty() && spec_min.len() == spec_max.len(),
            "spec_min/spec_max 长度必须一致且非空"
        );
        let betas = np_linspace(1e-4, max_beta, timesteps);
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let mut cum = 1.0f64;
        for &b in &betas {
            cum *= 1.0 - b;
            alphas_cumprod.push(cum);
        }
        let mut alphas_cumprod_prev = Vec::with_capacity(timesteps);
        alphas_cumprod_prev.push(1.0);
        alphas_cumprod_prev.extend_from_slice(&alphas_cumprod[..timesteps - 1]);

        let sqrt_alphas_cumprod: Vec<f64> = alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f64> =
            alphas_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();
        let sqrt_recip_alphas_cumprod: Vec<f64> =
            alphas_cumprod.iter().map(|&a| (1.0 / a).sqrt()).collect();
        let sqrt_recipm1_alphas_cumprod: Vec<f64> =
            alphas_cumprod.iter().map(|&a| (1.0 / a - 1.0).sqrt()).collect();

        let mut posterior_variance = Vec::with_capacity(timesteps);
        let mut posterior_log_variance_clipped = Vec::with_capacity(timesteps);
        let mut posterior_mean_coef1 = Vec::with_capacity(timesteps);
        let mut posterior_mean_coef2 = Vec::with_capacity(timesteps);
        for i in 0..timesteps {
            let beta = betas[i];
            let acp = alphas_cumprod[i];
            let acp_prev = alphas_cumprod_prev[i];
            let pv = beta * (1.0 - acp_prev) / (1.0 - acp);
            posterior_variance.push(pv);
            // log clipped：链首 posterior variance 为 0（diffusion.py:106）
            posterior_log_variance_clipped.push(pv.max(1e-20).ln());
            posterior_mean_coef1.push(beta * acp_prev.sqrt() / (1.0 - acp));
            posterior_mean_coef2.push((1.0 - acp_prev) * (1.0 - beta).sqrt() / (1.0 - acp));
        }

        // unit2mel.py:87 / diffusion.py:84：0 或 ≥timesteps → timesteps（可全扩散）
        let k_step_max = if k_step_max > 0 && k_step_max < timesteps {
            k_step_max
        } else {
            timesteps
        };

        Self {
            betas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            posterior_variance,
            posterior_log_variance_clipped,
            posterior_mean_coef1,
            posterior_mean_coef2,
            spec_min: spec_min.to_vec(),
            spec_max: spec_max.to_vec(),
            timesteps,
            k_step_max,
        }
    }

    /// norm_spec (diffusion.py:392): (x − min)/(max − min)·2 − 1, spec bounds broadcast
    /// over the M axis of [1,1,M,T] (len-1 → scalar, len-M → per-bin).
    pub fn norm_spec(&self, x: &Array4<f32>) -> Array4<f32> {
        let mut out = x.clone();
        for ((_, _, m, _), v) in out.indexed_iter_mut() {
            let mn = spec_at(&self.spec_min, m);
            let mx = spec_at(&self.spec_max, m);
            *v = (*v - mn) / (mx - mn) * 2.0 - 1.0;
        }
        out
    }

    /// denorm_spec (diffusion.py:395): (x + 1)/2·(max − min) + min.
    pub fn denorm_spec(&self, x: &Array4<f32>) -> Array4<f32> {
        let mut out = x.clone();
        for ((_, _, m, _), v) in out.indexed_iter_mut() {
            let mn = spec_at(&self.spec_min, m);
            let mx = spec_at(&self.spec_max, m);
            *v = (*v + 1.0) / 2.0 * (mx - mn) + mn;
        }
        out
    }

    /// q_sample (diffusion.py:203): √acp[t]·x₀ + √(1−acp[t])·noise. Shallow init calls
    /// this with `t_index = k_step − 1` (forward line 254: torch.tensor([t − 1])).
    pub fn q_sample(
        &self,
        x_start: &Array4<f32>,
        t_index: usize,
        noise: &Array4<f32>,
    ) -> Array4<f32> {
        let a = self.sqrt_alphas_cumprod[t_index] as f32;
        let b = self.sqrt_one_minus_alphas_cumprod[t_index] as f32;
        lin2(a, x_start, b, noise)
    }
}

fn spec_at(v: &[f32], m: usize) -> f32 {
    if v.len() == 1 {
        v[0]
    } else {
        v[m]
    }
}

/// np.linspace f64 semantics: step = (stop − start)/(n − 1), endpoint forced = stop.
fn np_linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
    v[n - 1] = stop;
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampler selection + noise source
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerMethod {
    /// 裸 DDPM p_sample 循环 = 原版 `method is None or infer_speedup <= 1` 分支
    /// （原版没有 'naive' 字符串；UI 里叫 naive）。
    Naive,
    Ddim,
    Pndm,
    DpmSolver,
    DpmSolverPp,
    UniPc,
}

impl SamplerMethod {
    /// wire 字符串（SovitsOptions.diffusion_method）→ 枚举。未知字符串 = None
    /// （原版对未知 method 抛 NotImplementedError —— 命令层负责报中文错误）。
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "naive" => Self::Naive,
            "ddim" => Self::Ddim,
            "pndm" => Self::Pndm,
            "dpm-solver" => Self::DpmSolver,
            "dpm-solver++" => Self::DpmSolverPp,
            "unipc" => Self::UniPc,
            _ => return None,
        })
    }
}

/// 高斯噪声源：正常推理用 per-piece seeded StdRng（sovits.rs seg_rng 同型，域分离常数
/// 由集成层负责）；`Zero` = debug_zero_noise（q_sample 初始化噪声 / naive 每步噪声 /
/// only_diffusion 初始 randn 全为 0，供 E2E 对拍与单测）。
pub enum NoiseSource<'a> {
    Rng(&'a mut StdRng),
    Zero,
}

impl NoiseSource<'_> {
    /// 按逻辑（row-major）顺序抽 N(0,1)（与 sovits.rs 的 z-noise 抽法一致）。
    pub fn randn(&mut self, dim: ndarray::Ix4) -> Array4<f32> {
        match self {
            NoiseSource::Rng(rng) => {
                Array4::from_shape_simple_fn(dim, || rng.sample(StandardNormal))
            }
            NoiseSource::Zero => Array4::zeros(dim),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────────────────────

/// 浅扩散：`x0_norm` 是 **已 norm_spec** 的 gt mel [1,1,M,T]；内部 q_sample 加噪到
/// k_step−1（forward:251-254），跑采样循环，返回 **已 denorm** 的 mel [1,1,M,T]
/// （直接喂声码器）。`progress` 每次 denoiser eval 调一次（0..=1）。
#[allow(clippy::too_many_arguments)]
pub fn sample_shallow<F>(
    sched: &DiffusionSchedule,
    x0_norm: &Array4<f32>,
    k_step: usize,
    speedup: usize,
    method: SamplerMethod,
    noise: &mut NoiseSource,
    denoise: F,
    progress: &dyn Fn(f32),
) -> Result<Array4<f32>>
where
    F: FnMut(&Array4<f32>, f32) -> Result<Array4<f32>>,
{
    if k_step == 0 {
        return Err(inference_err("浅扩散 k_step 必须 ≥ 1".into()));
    }
    if k_step > sched.k_step_max {
        // 原版 unit2mel.py:141 的 raise，中文化（contract §3.4）
        return Err(inference_err(format!(
            "浅扩散 k_step={} 超过该扩散模型的上限 k_step_max={}",
            k_step, sched.k_step_max
        )));
    }
    let init_noise = noise.randn(x0_norm.raw_dim());
    let x = sched.q_sample(x0_norm, k_step - 1, &init_noise);
    let x = sample(sched, x, k_step, speedup, method, noise, denoise, progress)?;
    Ok(sched.denorm_spec(&x))
}

/// only_diffusion：从纯噪声全步生成（forward:247-249，t = k_step_max，用户 k_step 被
/// 忽略）。返回已 denorm 的 mel [1,1,M,T]。
pub fn sample_full<F>(
    sched: &DiffusionSchedule,
    out_dims: usize,
    n_frames: usize,
    speedup: usize,
    method: SamplerMethod,
    noise: &mut NoiseSource,
    denoise: F,
    progress: &dyn Fn(f32),
) -> Result<Array4<f32>>
where
    F: FnMut(&Array4<f32>, f32) -> Result<Array4<f32>>,
{
    if sched.k_step_max != sched.timesteps {
        // 原版 unit2mel.py:144 的 raise，中文化（contract §3.4）
        return Err(inference_err(
            "该扩散模型仅支持浅扩散，无法单独推理".into(),
        ));
    }
    let x = noise.randn(ndarray::Ix4(1, 1, out_dims, n_frames));
    let t = sched.k_step_max;
    let x = sample(sched, x, t, speedup, method, noise, denoise, progress)?;
    Ok(sched.denorm_spec(&x))
}

/// 采样循环 dispatch（GaussianDiffusion.forward:256-388 的推理分支）。输入/输出都是
/// **norm 域** 的 x [1,1,M,T]。`t` = 走回的步数（浅扩散 = k_step，全扩散 = k_step_max）。
#[allow(clippy::too_many_arguments)]
pub fn sample<F>(
    sched: &DiffusionSchedule,
    x: Array4<f32>,
    t: usize,
    speedup: usize,
    method: SamplerMethod,
    noise: &mut NoiseSource,
    mut denoise: F,
    progress: &dyn Fn(f32),
) -> Result<Array4<f32>>
where
    F: FnMut(&Array4<f32>, f32) -> Result<Array4<f32>>,
{
    if t == 0 || t > sched.timesteps {
        return Err(inference_err(format!(
            "扩散步数 t={} 超出调度范围 1..={}",
            t, sched.timesteps
        )));
    }
    // 原版 dispatch：`if method is not None and infer_speedup > 1` —— speedup ≤ 1 时
    // 任何 method 都走裸 DDPM 循环。
    let effective = if speedup <= 1 {
        SamplerMethod::Naive
    } else {
        method
    };
    let total_nfe = match effective {
        SamplerMethod::Naive => t,
        SamplerMethod::Ddim => (t + speedup - 1) / speedup,
        // PLMS 首步（noise_list 空）额外 eval 一次
        SamplerMethod::Pndm => (t + speedup - 1) / speedup + 1,
        SamplerMethod::DpmSolver | SamplerMethod::DpmSolverPp | SamplerMethod::UniPc => {
            let steps = t / speedup;
            if steps < 2 {
                // 原版 sample() 的 assert steps >= order（order=2）
                return Err(inference_err(format!(
                    "扩散步数/加速倍数至少需要 2 个采样步（k_step/speedup ≥ 2），当前 {}/{} = {}",
                    t, speedup, steps
                )));
            }
            steps
        }
    };
    let mut evals = 0usize;
    let mut dn = |xx: &Array4<f32>, tt: f32| -> Result<Array4<f32>> {
        let out = denoise(xx, tt)?;
        evals += 1;
        progress((evals as f32 / total_nfe as f32).min(1.0));
        Ok(out)
    };
    match effective {
        SamplerMethod::Naive => sample_naive(sched, x, t, noise, &mut dn),
        SamplerMethod::Ddim => sample_ddim(sched, x, t, speedup, &mut dn),
        SamplerMethod::Pndm => sample_pndm(sched, x, t, speedup, &mut dn),
        SamplerMethod::DpmSolver => sample_dpm(sched, x, t, speedup, false, &mut dn),
        SamplerMethod::DpmSolverPp => sample_dpm(sched, x, t, speedup, true, &mut dn),
        SamplerMethod::UniPc => sample_unipc(sched, x, t, speedup, &mut dn),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// naive DDPM（diffusion.py p_sample:157-164 + p_mean_variance:136-143）
// ─────────────────────────────────────────────────────────────────────────────

fn sample_naive<D>(
    sched: &DiffusionSchedule,
    mut x: Array4<f32>,
    t: usize,
    noise: &mut NoiseSource,
    denoise: &mut D,
) -> Result<Array4<f32>>
where
    D: FnMut(&Array4<f32>, f32) -> Result<Array4<f32>>,
{
    for i in (0..t).rev() {
        let eps = denoise(&x, i as f32)?;
        // predict_start_from_noise + clamp ±1（diffusion.py:121,140）
        let sra = sched.sqrt_recip_alphas_cumprod[i] as f32;
        let srm = sched.sqrt_recipm1_alphas_cumprod[i] as f32;
        let mut x_recon = x.mapv(|v| sra * v);
        x_recon.zip_mut_with(&eps, |o, &e| *o = (*o - srm * e).clamp(-1.0, 1.0));
        // posterior mean（q_posterior:127-131）
        let c1 = sched.posterior_mean_coef1[i] as f32;
        let c2 = sched.posterior_mean_coef2[i] as f32;
        let mut mean = lin2(c1, &x_recon, c2, &x);
        // t > 0 时加 σ·randn（nonzero_mask，p_sample:163）
        if i > 0 {
            let sigma = (0.5 * sched.posterior_log_variance_clipped[i]).exp() as f32;
            let n = noise.randn(x.raw_dim());
            add_scaled(&mut mean, sigma, &n);
        }
        x = mean;
    }
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// DDIM（diffusion.py p_sample_ddim:146-155，interval = infer_speedup）
// ─────────────────────────────────────────────────────────────────────────────

fn sample_ddim<D>(
    sched: &DiffusionSchedule,
    mut x: Array4<f32>,
    t: usize,
    interval: usize,
    denoise: &mut D,
) -> Result<Array4<f32>>
where
    D: FnMut(&Array4<f32>, f32) -> Result<Array4<f32>>,
{
    // reversed(range(0, t, interval))
    for i in (0..t).step_by(interval).rev() {
        let a_t = sched.alphas_cumprod[i];
        let a_prev = sched.alphas_cumprod[i.saturating_sub(interval)];
        let eps = denoise(&x, i as f32)?;
        // x_prev = √a_prev·(x/√a_t + (√((1−a_prev)/a_prev) − √((1−a_t)/a_t))·ε)
        let sa_t = a_t.sqrt();
        let sa_prev = a_prev.sqrt();
        let c_eps = ((1.0 - a_prev) / a_prev).sqrt() - ((1.0 - a_t) / a_t).sqrt();
        x = lin2((sa_prev / sa_t) as f32, &x, (sa_prev * c_eps) as f32, &eps);
    }
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// PNDM / PLMS（diffusion.py p_sample_plms:167-201）
// ─────────────────────────────────────────────────────────────────────────────

fn sample_pndm<D>(
    sched: &DiffusionSchedule,
    mut x: Array4<f32>,
    t: usize,
    interval: usize,
    denoise: &mut D,
) -> Result<Array4<f32>>
where
    D: FnMut(&Array4<f32>, f32) -> Result<Array4<f32>>,
{
    // 原版每次推理重置 noise_list（diffusion.py:307）；这里是 per-call 局部变量，
    // 结构上保证重置。buffer 存 **原始** noise_pred（不是 blend 后的 n'）。
    let mut noise_list: VecDeque<Array4<f32>> = VecDeque::with_capacity(4);
    for i in (0..t).step_by(interval).rev() {
        let n = denoise(&x, i as f32)?;
        let n_prime = match noise_list.len() {
            0 => {
                // stage 0：额外一次 denoiser eval（Heun 式启动）
                let x_pred = plms_x_pred(sched, &x, &n, i, interval);
                let t_next = i.saturating_sub(interval);
                let n_next = denoise(&x_pred, t_next as f32)?;
                let mut p = n.clone();
                p.zip_mut_with(&n_next, |o, &b| *o = (*o + b) / 2.0);
                p
            }
            1 => {
                let n1 = &noise_list[noise_list.len() - 1];
                let mut p = n.clone();
                p.zip_mut_with(n1, |o, &b| *o = (3.0 * *o - b) / 2.0);
                p
            }
            2 => {
                let n1 = &noise_list[noise_list.len() - 1];
                let n2 = &noise_list[noise_list.len() - 2];
                let mut p = n.mapv(|v| 23.0 * v);
                add_scaled(&mut p, -16.0, n1);
                add_scaled(&mut p, 5.0, n2);
                p.mapv_inplace(|v| v / 12.0);
                p
            }
            _ => {
                let n1 = &noise_list[noise_list.len() - 1];
                let n2 = &noise_list[noise_list.len() - 2];
                let n3 = &noise_list[noise_list.len() - 3];
                let mut p = n.mapv(|v| 55.0 * v);
                add_scaled(&mut p, -59.0, n1);
                add_scaled(&mut p, 37.0, n2);
                add_scaled(&mut p, -9.0, n3);
                p.mapv_inplace(|v| v / 24.0);
                p
            }
        };
        x = plms_x_pred(sched, &x, &n_prime, i, interval);
        noise_list.push_back(n);
        if noise_list.len() > 4 {
            noise_list.pop_front(); // deque(maxlen=4)
        }
    }
    Ok(x)
}

/// PLMS transfer function get_x_pred（diffusion.py:173-182）。
fn plms_x_pred(
    sched: &DiffusionSchedule,
    x: &Array4<f32>,
    noise_t: &Array4<f32>,
    i: usize,
    interval: usize,
) -> Array4<f32> {
    let a_t = sched.alphas_cumprod[i];
    let a_prev = sched.alphas_cumprod[i.saturating_sub(interval)];
    let a_t_sq = a_t.sqrt();
    let a_prev_sq = a_prev.sqrt();
    let dc = a_prev - a_t;
    let cx = (dc * (1.0 / (a_t_sq * (a_t_sq + a_prev_sq)))) as f32;
    let cn = (dc
        * (1.0 / (a_t_sq * (((1.0 - a_prev) * a_t).sqrt() + ((1.0 - a_t) * a_prev).sqrt()))))
        as f32;
    // x_pred = x + (cx·x − cn·n)
    let mut out = x.clone();
    out.zip_mut_with(noise_t, |o, &nv| *o += cx * *o - cn * nv);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// NoiseScheduleVP('discrete') — dpm_solver_pytorch.py:96-165 / uni_pc.py:77-154
// ─────────────────────────────────────────────────────────────────────────────

struct NoiseScheduleVp {
    total_n: usize,
    /// t_i = (i+1)/N（离散步 → 连续时间）
    t_array: Vec<f64>,
    /// log α_{t_i} = 0.5·Σ_{j≤i} ln(1 − β_j)
    log_alpha_array: Vec<f64>,
}

impl NoiseScheduleVp {
    fn discrete(betas: &[f64]) -> Self {
        let n = betas.len();
        let mut log_alpha_array = Vec::with_capacity(n);
        let mut cum = 0.0f64;
        for &b in betas {
            cum += 0.5 * (1.0 - b).ln();
            log_alpha_array.push(cum);
        }
        // DEVIATION（no-op）：原版 dpm 的 numerical_clip_alpha(λ<−5.1 裁尾) 跳过 ——
        // linear schedule 全长 λ_end≈−5.02，截断 betas[..t] 时 λ_end 更大，永不触发；
        // uni_pc.py 的副本本来就没有 clip。见模块头 DEVIATIONS。
        let t_array = (0..n).map(|i| (i + 1) as f64 / n as f64).collect();
        Self {
            total_n: n,
            t_array,
            log_alpha_array,
        }
    }

    fn marginal_log_mean_coeff(&self, t: f64) -> f64 {
        interp_linear(t, &self.t_array, &self.log_alpha_array)
    }

    fn marginal_alpha(&self, t: f64) -> f64 {
        self.marginal_log_mean_coeff(t).exp()
    }

    fn marginal_std(&self, t: f64) -> f64 {
        (1.0 - (2.0 * self.marginal_log_mean_coeff(t)).exp()).sqrt()
    }

    fn marginal_lambda(&self, t: f64) -> f64 {
        let lm = self.marginal_log_mean_coeff(t);
        lm - 0.5 * (1.0 - (2.0 * lm).exp()).ln()
    }

    /// model_wrapper.get_model_input_time（dpm_solver_pytorch.py:276）：
    /// [1/N, 1] → [0, N−1] 的 **float**（denoiser 收到非整数步）。
    fn model_input_time(&self, t: f64) -> f64 {
        (t - 1.0 / self.total_n as f64) * self.total_n as f64
    }
}

/// interpolate_fn（dpm_solver_pytorch.py:1255-1294）的标量特化：分段线性 + 两端用最外
/// 段外推。keypoint 精确命中时两侧段给出同一值（与原版 sort/gather 语义一致）。
fn interp_linear(x: f64, xp: &[f64], yp: &[f64]) -> f64 {
    let k = xp.len();
    debug_assert!(k >= 2);
    let (i0, i1) = if x <= xp[0] {
        (0, 1)
    } else if x >= xp[k - 1] {
        (k - 2, k - 1)
    } else {
        // 第一个 > x 的下标 j → 段 (j−1, j)
        let j = xp.partition_point(|&v| v <= x);
        (j - 1, j)
    };
    yp[i0] + (x - xp[i0]) * (yp[i1] - yp[i0]) / (xp[i1] - xp[i0])
}

// ─────────────────────────────────────────────────────────────────────────────
// DPM-Solver / DPM-Solver++（dpm_solver_pytorch.py，multistep order 2, time_uniform,
// solver_type='dpmsolver', lower_order_final=True(仅 steps<10 生效), denoise_to_zero=False）
// ─────────────────────────────────────────────────────────────────────────────

fn sample_dpm<D>(
    sched: &DiffusionSchedule,
    mut x: Array4<f32>,
    t: usize,
    speedup: usize,
    plus_plus: bool,
    denoise: &mut D,
) -> Result<Array4<f32>>
where
    D: FnMut(&Array4<f32>, f32) -> Result<Array4<f32>>,
{
    let steps = t / speedup; // diffusion.py:294
    let order = 2usize;
    debug_assert!(steps >= order);
    // NoiseScheduleVP(schedule='discrete', betas=self.betas[:t])（diffusion.py:264）
    let ns = NoiseScheduleVp::discrete(&sched.betas[..t]);
    // model_fn：dpmsolver++ → x0-pred (x − σ·ε)/α（data_prediction_fn:431-440，
    // correcting_x0_fn=None 无 clamp）；dpmsolver → 原始 ε。
    let mut model_fn = |xx: &Array4<f32>, tc: f64| -> Result<Array4<f32>> {
        let eps = denoise(xx, ns.model_input_time(tc) as f32)?;
        if plus_plus {
            let alpha = ns.marginal_alpha(tc);
            let sigma = ns.marginal_std(tc);
            Ok(lin2((1.0 / alpha) as f32, xx, -(sigma / alpha) as f32, &eps))
        } else {
            Ok(eps)
        }
    };
    // timesteps = linspace(t_T=1, t_0=1/N, steps+1)（get_time_steps 'time_uniform'）
    let timesteps = np_linspace(1.0, 1.0 / ns.total_n as f64, steps + 1);

    let mut t_prev: Vec<f64> = vec![timesteps[0]];
    let mut model_prev: Vec<Array4<f32>> = vec![model_fn(&x, timesteps[0])?];
    // init 步（step=1，order-1 update 复用 model_prev[-1]，随后在新 t 处 eval）
    {
        let tc = timesteps[1];
        x = dpm_first_update(&ns, &x, t_prev[0], tc, &model_prev[0], plus_plus);
        t_prev.push(tc);
        let m = model_fn(&x, tc)?;
        model_prev.push(m);
    }
    for step in order..=steps {
        let tc = timesteps[step];
        // lower_order_final=True 默认，但 **仅 steps<10 生效**（sample:1200-1203）——
        // 与 unipc 的无条件版不同，别“统一”。
        let step_order = if steps < 10 {
            order.min(steps + 1 - step)
        } else {
            order
        };
        x = if step_order == 1 {
            dpm_first_update(
                &ns,
                &x,
                *t_prev.last().unwrap(),
                tc,
                model_prev.last().unwrap(),
                plus_plus,
            )
        } else {
            dpm_second_multistep(&ns, &x, &model_prev, &t_prev, tc, plus_plus)
        };
        // 窗口平移（sample:1209-1212）；旧 m0 只在末步残留、从不读取
        t_prev[0] = t_prev[1];
        t_prev[1] = tc;
        model_prev.swap(0, 1);
        // 末步不再 eval（sample:1214）
        if step < steps {
            model_prev[1] = model_fn(&x, tc)?;
        }
    }
    Ok(x)
}

/// dpm_solver_first_update（dpm_solver_pytorch.py:545-589，model_s 给定，无额外 eval）。
fn dpm_first_update(
    ns: &NoiseScheduleVp,
    x: &Array4<f32>,
    s: f64,
    t: f64,
    model_s: &Array4<f32>,
    plus_plus: bool,
) -> Array4<f32> {
    let h = ns.marginal_lambda(t) - ns.marginal_lambda(s);
    if plus_plus {
        // x_t = (σ_t/σ_s)·x − α_t·expm1(−h)·model_s
        let ca = (ns.marginal_std(t) / ns.marginal_std(s)) as f32;
        let cb = (ns.marginal_alpha(t) * (-h).exp_m1()) as f32;
        lin2(ca, x, -cb, model_s)
    } else {
        // x_t = e^{logα_t − logα_s}·x − σ_t·expm1(h)·model_s
        let ca = (ns.marginal_log_mean_coeff(t) - ns.marginal_log_mean_coeff(s)).exp() as f32;
        let cb = (ns.marginal_std(t) * h.exp_m1()) as f32;
        lin2(ca, x, -cb, model_s)
    }
}

/// multistep_dpm_solver_second_update（dpm_solver_pytorch.py:793-849，solver_type='dpmsolver'）。
fn dpm_second_multistep(
    ns: &NoiseScheduleVp,
    x: &Array4<f32>,
    model_prev: &[Array4<f32>],
    t_prev: &[f64],
    t: f64,
    plus_plus: bool,
) -> Array4<f32> {
    let m1 = &model_prev[model_prev.len() - 2];
    let m0 = &model_prev[model_prev.len() - 1];
    let t1 = t_prev[t_prev.len() - 2];
    let t0 = t_prev[t_prev.len() - 1];
    let lambda_1 = ns.marginal_lambda(t1);
    let lambda_0 = ns.marginal_lambda(t0);
    let lambda_t = ns.marginal_lambda(t);
    let h0 = lambda_0 - lambda_1;
    let h = lambda_t - lambda_0;
    let r0 = h0 / h;
    // D1_0 = (1/r0)·(m0 − m1)
    let inv_r0 = (1.0 / r0) as f32;
    let mut d1 = m0.clone();
    d1.zip_mut_with(m1, |o, &b| *o = inv_r0 * (*o - b));
    let sigma_t = ns.marginal_std(t);
    let (ca, cb) = if plus_plus {
        let phi_1 = (-h).exp_m1();
        (
            (sigma_t / ns.marginal_std(t0)) as f32,
            (ns.marginal_alpha(t) * phi_1) as f32,
        )
    } else {
        let phi_1 = h.exp_m1();
        (
            (ns.marginal_log_mean_coeff(t) - ns.marginal_log_mean_coeff(t0)).exp() as f32,
            (sigma_t * phi_1) as f32,
        )
    };
    // x_t = ca·x − cb·m0 − 0.5·cb·D1_0
    let mut out = lin2(ca, x, -cb, m0);
    add_scaled(&mut out, -0.5 * cb, &d1);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// UniPC（uni_pc.py，variant='bh2'，data_prediction → predict_x0=True，multistep order 2，
// lower_order_final **无条件**，末步 corrector 关闭）
// ─────────────────────────────────────────────────────────────────────────────

fn sample_unipc<D>(
    sched: &DiffusionSchedule,
    mut x: Array4<f32>,
    t: usize,
    speedup: usize,
    denoise: &mut D,
) -> Result<Array4<f32>>
where
    D: FnMut(&Array4<f32>, f32) -> Result<Array4<f32>>,
{
    let steps = t / speedup;
    let order = 2usize;
    debug_assert!(steps >= order);
    let ns = NoiseScheduleVp::discrete(&sched.betas[..t]);
    // data_prediction_fn（uni_pc.py:287-296，correcting_x0_fn=None）
    let mut model_fn = |xx: &Array4<f32>, tc: f64| -> Result<Array4<f32>> {
        let eps = denoise(xx, ns.model_input_time(tc) as f32)?;
        let alpha = ns.marginal_alpha(tc);
        let sigma = ns.marginal_std(tc);
        Ok(lin2((1.0 / alpha) as f32, xx, -(sigma / alpha) as f32, &eps))
    };
    let timesteps = np_linspace(1.0, 1.0 / ns.total_n as f64, steps + 1);

    let mut t_prev: Vec<f64> = vec![timesteps[0]];
    let mut model_prev: Vec<Array4<f32>> = vec![model_fn(&x, timesteps[0])?];
    // init 步（step=1，order-1 update **带 corrector**，corrector 的 eval 复用为
    // model_prev[-1] —— uni_pc.py sample:623-633）
    {
        let tc = timesteps[1];
        let (nx, model_x) =
            unipc_bh_update(&ns, &x, &model_prev, &t_prev, tc, 1, true, &mut model_fn)?;
        x = nx;
        t_prev.push(tc);
        let m = match model_x {
            Some(m) => m,
            None => model_fn(&x, tc)?, // 原版 fallback（corrector 开时不会走到）
        };
        model_prev.push(m);
    }
    for step in order..=steps {
        let tc = timesteps[step];
        // lower_order_final 无条件（uni_pc.py:638，无 steps<10 门）
        let step_order = order.min(steps + 1 - step);
        // 末步不跑 corrector（uni_pc.py:642-646）
        let use_corrector = step != steps;
        let (nx, model_x) = unipc_bh_update(
            &ns,
            &x,
            &model_prev,
            &t_prev,
            tc,
            step_order,
            use_corrector,
            &mut model_fn,
        )?;
        x = nx;
        t_prev[0] = t_prev[1];
        t_prev[1] = tc;
        model_prev.swap(0, 1);
        if step < steps {
            model_prev[1] = match model_x {
                Some(m) => m,
                None => model_fn(&x, tc)?,
            };
        }
    }
    Ok(x)
}

/// multistep_uni_pc_bh_update（uni_pc.py:473-590，predict_x0=True，bh2，x_t 入参恒 None）。
/// 返回 (x_t, corrector 的 model_t)。注意：model_t 是在 **predictor 的 x_t** 上求值的，
/// 主循环把它复用为 model_prev[-1]（与原版一致，别“修”成 corrected-x 处求值）。
#[allow(clippy::too_many_arguments)]
fn unipc_bh_update<MF>(
    ns: &NoiseScheduleVp,
    x: &Array4<f32>,
    model_prev: &[Array4<f32>],
    t_prev: &[f64],
    t: f64,
    order: usize,
    use_corrector: bool,
    model_fn: &mut MF,
) -> Result<(Array4<f32>, Option<Array4<f32>>)>
where
    MF: FnMut(&Array4<f32>, f64) -> Result<Array4<f32>>,
{
    let t0 = *t_prev.last().unwrap();
    let m0 = model_prev.last().unwrap();
    let lambda_0 = ns.marginal_lambda(t0);
    let lambda_t = ns.marginal_lambda(t);
    let sigma_0 = ns.marginal_std(t0);
    let sigma_t = ns.marginal_std(t);
    let alpha_t = ns.marginal_alpha(t);
    let h = lambda_t - lambda_0;

    // rks / D1s（uni_pc.py:489-499）
    let mut rks: Vec<f64> = Vec::new();
    let mut d1s: Vec<Array4<f32>> = Vec::new();
    for i in 1..order {
        let ti = t_prev[t_prev.len() - 1 - i];
        let mi = &model_prev[model_prev.len() - 1 - i];
        let rk = (ns.marginal_lambda(ti) - lambda_0) / h;
        rks.push(rk);
        let inv = (1.0 / rk) as f32;
        let mut d = mi.clone();
        d.zip_mut_with(m0, |o, &b| *o = (*o - b) * inv);
        d1s.push(d);
    }
    rks.push(1.0);

    // Vandermonde R 行 + b（uni_pc.py:502-525）；predict_x0 → hh = −h；bh2 → B_h = expm1(hh)
    let hh = -h;
    let h_phi_1 = hh.exp_m1();
    let b_h = hh.exp_m1();
    let mut h_phi_k = h_phi_1 / hh - 1.0;
    let mut r_rows: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut factorial_i = 1.0f64;
    for i in 1..=order {
        r_rows.push(rks.iter().map(|&r| r.powi(i as i32 - 1)).collect());
        b.push(h_phi_k * factorial_i / b_h);
        factorial_i *= (i + 1) as f64;
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
    }

    // x_t_ = (σ_t/σ_0)·x − α_t·h_phi_1·m0（uni_pc.py:550-553）
    let x_t_base = lin2(
        (sigma_t / sigma_0) as f32,
        x,
        -(alpha_t * h_phi_1) as f32,
        m0,
    );
    let cbh = (alpha_t * b_h) as f32;

    // predictor：order==2 用简化 rhos_p=[0.5]（uni_pc.py:533-534；本移植 order≤2，
    // 通解 solve(R[:-1,:-1], b[:-1]) 永不触达）
    let mut x_t = x_t_base.clone();
    if !d1s.is_empty() {
        if order != 2 {
            return Err(inference_err(format!("unipc: 不支持的 order={order}")));
        }
        let rhos_p = [0.5f64];
        for (k, d) in d1s.iter().enumerate() {
            add_scaled(&mut x_t, -cbh * rhos_p[k] as f32, d);
        }
    }

    let mut model_t_out = None;
    if use_corrector {
        // corrector：order==1 简化 [0.5]，否则 solve(R, b)（2×2 高斯消元）
        let rhos_c: Vec<f64> = if order == 1 {
            vec![0.5]
        } else {
            gauss_solve(&r_rows, &b)?
        };
        let model_t = model_fn(&x_t, t)?;
        // x_t = x_t_ − α_t·B_h·(Σ rhos_c[..k-1]·D1s + rhos_c[-1]·(model_t − m0))
        let mut acc = x_t_base;
        for (k, d) in d1s.iter().enumerate() {
            add_scaled(&mut acc, -cbh * rhos_c[k] as f32, d);
        }
        let mut d1t = model_t.clone();
        d1t.zip_mut_with(m0, |o, &b| *o -= b);
        add_scaled(&mut acc, -cbh * (*rhos_c.last().unwrap()) as f32, &d1t);
        x_t = acc;
        model_t_out = Some(model_t);
    }
    Ok((x_t, model_t_out))
}

// ─────────────────────────────────────────────────────────────────────────────
// small helpers
// ─────────────────────────────────────────────────────────────────────────────

/// ca·x + cb·y（elementwise，f32；f64 标量在调用处 cast）。
fn lin2(ca: f32, x: &Array4<f32>, cb: f32, y: &Array4<f32>) -> Array4<f32> {
    let mut out = x.mapv(|v| ca * v);
    out.zip_mut_with(y, |o, &yv| *o += cb * yv);
    out
}

/// out += c·y（elementwise）。
fn add_scaled(out: &mut Array4<f32>, c: f32, y: &Array4<f32>) {
    out.zip_mut_with(y, |o, &yv| *o += c * yv);
}

/// n×n 高斯消元（部分主元），n ≤ 3（torch.linalg.solve 的小规模等价）。
fn gauss_solve(rows: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = rows.len();
    debug_assert!(n >= 1 && rows.iter().all(|r| r.len() == n) && b.len() == n);
    let mut a: Vec<Vec<f64>> = rows.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        // partial pivot
        let mut piv = col;
        for r in col + 1..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-300 {
            return Err(inference_err("unipc: Vandermonde 矩阵奇异".into()));
        }
        a.swap(col, piv);
        rhs.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    let mut xsol = vec![0.0f64; n];
    for row in (0..n).rev() {
        let mut s = rhs[row];
        for c in row + 1..n {
            s -= a[row][c] * xsol[c];
        }
        xsol[row] = s / a[row][row];
    }
    Ok(xsol)
}

// ─────────────────────────────────────────────────────────────────────────────
// tests — reference vectors generated by scratchpad/gen_sampler_refs.py, which drives
// the ORIGINAL so-vits-svc modules (diffusion.py / dpm_solver_pytorch.py / uni_pc.py)
// with the same analytic stub denoiser, zero noise, in f64 (see the script header for
// why f64: the contract fixes this port at f64 scalars; the f32-buffer deviation is
// quantified by the converter gate + E2E SNR gate).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;
    use std::cell::RefCell;

    const M: usize = 4;
    const TF: usize = 8;

    fn sched() -> DiffusionSchedule {
        DiffusionSchedule::linear(50, 0.02, &[-12.0], &[2.0], 0)
    }

    fn gt4() -> Array4<f32> {
        Array4::from_shape_vec((1, 1, M, TF), refs::GT_MT.to_vec()).unwrap()
    }

    /// 与 gen_sampler_refs.py 的 stub 逐字一致：eps = 0.1·x + sin(t·0.01)·mean(cond)
    fn stub() -> impl FnMut(&Array4<f32>, f32) -> Result<Array4<f32>> {
        let cm: f64 = refs::COND.iter().map(|&v| v as f64).sum::<f64>() / refs::COND.len() as f64;
        move |x: &Array4<f32>, t: f32| {
            let s = ((t as f64) * 0.01).sin() * cm;
            Ok(x.mapv(|v| 0.1 * v + s as f32))
        }
    }

    fn assert_close_rel(got: &Array4<f32>, want: &[f64], rel: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: len");
        let mut max_rel = 0.0f64;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let g = g as f64;
            let r = (g - w).abs() / w.abs().max(1e-3);
            if r > max_rel {
                max_rel = r;
            }
            assert!(r <= rel, "{what}[{i}]: got {g} want {w} rel {r:.3e}");
        }
        eprintln!("{what}: max_rel={max_rel:.3e}");
    }

    fn assert_arr_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: len");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "{what}[{i}]: got {g} want {w} diff {:.3e}",
                (g - w).abs()
            );
        }
    }

    /// python 参照的完整浅扩散路径：norm → q_sample(k−1, 零噪) → sampler（pre-denorm）。
    fn run(method: SamplerMethod, k: usize, sp: usize) -> Array4<f32> {
        let s = sched();
        let x0 = s.norm_spec(&gt4());
        let zeros = Array4::zeros((1, 1, M, TF));
        let x = s.q_sample(&x0, k - 1, &zeros);
        sample(
            &s,
            x,
            k,
            sp,
            method,
            &mut NoiseSource::Zero,
            stub(),
            &|_p: f32| {},
        )
        .unwrap()
    }

    #[test]
    fn schedule_buffers_match_numpy_reference() {
        let s = sched();
        assert_arr_close(&s.betas, &refs::REF_BETAS, 1e-12, "betas");
        assert_arr_close(&s.alphas_cumprod, &refs::REF_ACP, 1e-12, "alphas_cumprod");
        assert_arr_close(&s.posterior_variance, &refs::REF_PVAR, 1e-12, "posterior_variance");
        assert_arr_close(
            &s.posterior_log_variance_clipped,
            &refs::REF_PLVC,
            1e-12,
            "posterior_log_variance_clipped",
        );
        assert_arr_close(&s.posterior_mean_coef1, &refs::REF_PMC1, 1e-12, "pmc1");
        assert_arr_close(&s.posterior_mean_coef2, &refs::REF_PMC2, 1e-12, "pmc2");
        // 派生 sqrt buffers 与定义严格一致
        for i in 0..s.timesteps {
            assert_eq!(s.sqrt_alphas_cumprod[i], s.alphas_cumprod[i].sqrt());
            assert_eq!(
                s.sqrt_one_minus_alphas_cumprod[i],
                (1.0 - s.alphas_cumprod[i]).sqrt()
            );
            assert_eq!(
                s.sqrt_recip_alphas_cumprod[i],
                (1.0 / s.alphas_cumprod[i]).sqrt()
            );
            assert_eq!(
                s.sqrt_recipm1_alphas_cumprod[i],
                (1.0 / s.alphas_cumprod[i] - 1.0).sqrt()
            );
        }
        assert_eq!(s.alphas_cumprod_prev[0], 1.0);
        assert_eq!(s.alphas_cumprod_prev[1..], s.alphas_cumprod[..49]);
        // k_step_max 解析：0 → timesteps；有效值透传
        assert_eq!(s.k_step_max, 50);
        assert_eq!(
            DiffusionSchedule::linear(50, 0.02, &[-12.0], &[2.0], 20).k_step_max,
            20
        );
    }

    #[test]
    fn q_sample_shallow_init_at_t_minus_1() {
        let s = sched();
        let x0 = s.norm_spec(&gt4());
        let zeros = Array4::zeros((1, 1, M, TF));
        let x = s.q_sample(&x0, 20 - 1, &zeros); // k=20 → 索引 19（off-by-one 守卫）
        // 绝对容差：norm 域值域 O(1)，f32 舍入 ~6e-8；1e-6 abs 即“逐位正确到 f32”
        for (i, (&g, &w)) in x.iter().zip(refs::REF_QSAMPLE.iter()).enumerate() {
            assert!(
                (g as f64 - w).abs() <= 1e-6,
                "q_sample[{i}]: got {g} want {w}"
            );
        }
        let a = s.sqrt_alphas_cumprod[19] as f32;
        for (g, &v) in x.iter().zip(x0.iter()) {
            assert_eq!(*g, a * v);
        }
    }

    #[test]
    fn naive_matches_original() {
        assert_close_rel(&run(SamplerMethod::Naive, 20, 1), &refs::REF_NAIVE, 1e-4, "naive");
    }

    #[test]
    fn ddim_matches_original() {
        assert_close_rel(&run(SamplerMethod::Ddim, 20, 5), &refs::REF_DDIM, 1e-4, "ddim");
    }

    #[test]
    fn pndm_matches_original() {
        assert_close_rel(&run(SamplerMethod::Pndm, 20, 5), &refs::REF_PNDM, 1e-4, "pndm");
    }

    #[test]
    fn dpm_solver_matches_original() {
        // steps=4 < 10 → lower_order_final 生效（末步降为 order-1）
        assert_close_rel(&run(SamplerMethod::DpmSolver, 20, 5), &refs::REF_DPM, 1e-4, "dpm");
    }

    #[test]
    fn dpm_solver_pp_matches_original() {
        assert_close_rel(&run(SamplerMethod::DpmSolverPp, 20, 5), &refs::REF_DPMPP, 1e-4, "dpm++");
    }

    #[test]
    fn dpm_solver_pp_steps10_not_lowered_matches_original() {
        // steps=10 → NOT-lowered 分支（末步保持 order-2）
        assert_close_rel(
            &run(SamplerMethod::DpmSolverPp, 40, 4),
            &refs::REF_DPMPP_S10,
            1e-4,
            "dpm++ s10",
        );
    }

    #[test]
    fn unipc_matches_original() {
        assert_close_rel(&run(SamplerMethod::UniPc, 20, 5), &refs::REF_UNIPC, 1e-4, "unipc");
    }

    #[test]
    fn plms_noise_list_resets_between_calls() {
        // 原版每次推理重置 noise_list；本移植是 per-call 局部（结构性重置）——
        // 两次连续调用必须逐位相同（状态泄漏则第二次不同）。
        let a = run(SamplerMethod::Pndm, 20, 5);
        let b = run(SamplerMethod::Pndm, 20, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn speedup_1_falls_back_to_naive() {
        // 原版 dispatch：infer_speedup ≤ 1 时任何 method 都走裸 DDPM 循环
        assert_close_rel(&run(SamplerMethod::Ddim, 20, 1), &refs::REF_NAIVE, 1e-4, "ddim@sp1");
    }

    #[test]
    fn sample_shallow_composes_q_sample_and_denorm() {
        let s = sched();
        let x0 = s.norm_spec(&gt4());
        let out = sample_shallow(
            &s,
            &x0,
            20,
            5,
            SamplerMethod::Ddim,
            &mut NoiseSource::Zero,
            stub(),
            &|_p: f32| {},
        )
        .unwrap();
        let manual = s.denorm_spec(&run(SamplerMethod::Ddim, 20, 5));
        assert_eq!(out, manual);
    }

    #[test]
    fn norm_denorm_roundtrip_and_per_bin_broadcast() {
        let s = sched();
        let g = gt4();
        let rt = s.denorm_spec(&s.norm_spec(&g));
        for (a, b) in rt.iter().zip(g.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
        // len-M per-bin 广播（sidecar spec_min 128 维档）
        let mn = [-12.0f32, -10.0, -8.0, -6.0];
        let mx = [2.0f32, 1.0, 0.0, -1.0];
        let s2 = DiffusionSchedule::linear(50, 0.02, &mn, &mx, 0);
        let n = s2.norm_spec(&g);
        for ((_, _, m, tt), v) in n.indexed_iter() {
            let e = (g[[0, 0, m, tt]] - mn[m]) / (mx[m] - mn[m]) * 2.0 - 1.0;
            assert_eq!(*v, e);
        }
    }

    #[test]
    fn progress_reports_once_per_denoiser_eval() {
        for (method, k, sp, expected) in [
            (SamplerMethod::Naive, 20usize, 1usize, 20usize),
            (SamplerMethod::Ddim, 20, 5, 4),
            (SamplerMethod::Pndm, 20, 5, 5), // 首步额外一次 eval
            (SamplerMethod::DpmSolver, 20, 5, 4),
            (SamplerMethod::DpmSolverPp, 40, 4, 10),
            (SamplerMethod::UniPc, 20, 5, 4),
        ] {
            let s = sched();
            let x0 = s.norm_spec(&gt4());
            let zeros = Array4::zeros((1, 1, M, TF));
            let x = s.q_sample(&x0, k - 1, &zeros);
            let calls = RefCell::new(Vec::<f32>::new());
            let progress = |p: f32| calls.borrow_mut().push(p);
            sample(&s, x, k, sp, method, &mut NoiseSource::Zero, stub(), &progress).unwrap();
            let calls = calls.into_inner();
            assert_eq!(calls.len(), expected, "{method:?}: NFE");
            assert!((calls.last().unwrap() - 1.0).abs() < 1e-6, "{method:?}: 末值=1");
            assert!(
                calls.windows(2).all(|w| w[0] <= w[1]),
                "{method:?}: 单调不减"
            );
        }
    }

    #[test]
    fn guards_reject_invalid_configs() {
        let s = sched();
        let x0 = s.norm_spec(&gt4());
        // dpm/unipc steps < 2（原版 assert steps >= order）
        assert!(sample_shallow(
            &s, &x0, 4, 4, SamplerMethod::DpmSolverPp, &mut NoiseSource::Zero, stub(), &|_p: f32| {}
        )
        .is_err());
        // k_step > k_step_max
        let s2 = DiffusionSchedule::linear(50, 0.02, &[-12.0], &[2.0], 20);
        assert!(sample_shallow(
            &s2, &x0, 21, 5, SamplerMethod::Ddim, &mut NoiseSource::Zero, stub(), &|_p: f32| {}
        )
        .is_err());
        // shallow-only 模型不能 only_diffusion；全扩散模型可以
        assert!(sample_full(
            &s2, M, TF, 10, SamplerMethod::DpmSolverPp, &mut NoiseSource::Zero, stub(), &|_p: f32| {}
        )
        .is_err());
        assert!(sample_full(
            &s, M, TF, 10, SamplerMethod::DpmSolverPp, &mut NoiseSource::Zero, stub(), &|_p: f32| {}
        )
        .is_ok());
        // wire 字符串
        assert_eq!(SamplerMethod::parse("dpm-solver++"), Some(SamplerMethod::DpmSolverPp));
        assert_eq!(SamplerMethod::parse("dpm-solver"), Some(SamplerMethod::DpmSolver));
        assert_eq!(SamplerMethod::parse("naive"), Some(SamplerMethod::Naive));
        assert_eq!(SamplerMethod::parse("euler"), None);
    }

    #[test]
    fn rng_noise_source_is_deterministic_per_seed() {
        use rand::SeedableRng;
        let s = sched();
        let x0 = s.norm_spec(&gt4());
        let go = |seed: u64| {
            let mut rng = StdRng::seed_from_u64(seed);
            sample_shallow(
                &s,
                &x0,
                20,
                1, // naive：每步都抽噪声，覆盖 randn 路径
                SamplerMethod::Naive,
                &mut NoiseSource::Rng(&mut rng),
                stub(),
                &|_p: f32| {},
            )
            .unwrap()
        };
        let a = go(7);
        assert!(a.iter().all(|v| v.is_finite()));
        assert_eq!(a, go(7));
        assert_ne!(a, go(8));
    }

    mod refs {
        #![allow(clippy::excessive_precision)]
        // ---- generated by scratchpad/gen_sampler_refs.py — do not hand-edit ----
        pub(super) const COND: [f32; 24] = [-0.6169611215591431, 0.24421754479408264, -0.12454452365636826, 0.5707171559333801, 0.5599516034126282, -0.4548147916793823, -0.44707149267196655, 0.6037443280220032, 0.9162787199020386, 0.7518652677536011, -0.2843654453754425, 0.0019902510102838278, 0.3669258654117584, 0.42540404200553894, -0.2594984769821167, 0.12239237129688263, 0.00616633053869009, -0.972463071346283, 0.545653223991394, 0.765282392501831, -0.2702280282974243, 0.2307923585176468, -0.8492375016212463, -0.26235198974609375];
        pub(super) const GT_MT: [f32; 32] = [0.1976812183856964, -7.197966575622559, -1.3742282390594482, -8.374494552612305, -10.282289505004883, -3.8645026683807373, -7.043978691101074, -4.208664417266846, -3.183462381362915, -4.182816028594971, -9.274798393249512, 0.09841154515743256, -8.788555145263672, -4.600277900695801, -4.96439790725708, -10.918830871582031, -6.233569145202637, -0.5704713463783264, -2.548868417739868, -5.694311141967773, -10.431736946105957, -10.480111122131348, -9.657268524169922, -3.5906994342803955, -1.535238265991211, -5.765918731689453, -2.5450243949890137, -0.08820848912000656, -2.901428699493408, -4.262803077697754, -3.7136754989624023, -0.05452536419034004];
        pub(super) const REF_BETAS: [f64; 50] = [0.0001, 0.0005061224489795919, 0.0009122448979591838, 0.0013183673469387756, 0.0017244897959183675, 0.002130612244897959, 0.002536734693877551, 0.002942857142857143, 0.0033489795918367348, 0.0037551020408163266, 0.004161224489795919, 0.004567346938775511, 0.004973469387755103, 0.005379591836734694, 0.005785714285714286, 0.006191836734693879, 0.00659795918367347, 0.0070040816326530616, 0.007410204081632654, 0.007816326530612245, 0.008222448979591837, 0.008628571428571428, 0.009034693877551021, 0.009440816326530613, 0.009846938775510204, 0.010253061224489796, 0.010659183673469387, 0.01106530612244898, 0.011471428571428572, 0.011877551020408163, 0.012283673469387756, 0.012689795918367348, 0.013095918367346939, 0.01350204081632653, 0.013908163265306122, 0.014314285714285715, 0.014720408163265307, 0.015126530612244898, 0.015532653061224491, 0.015938775510204083, 0.016344897959183674, 0.016751020408163265, 0.017157142857142857, 0.01756326530612245, 0.017969387755102043, 0.018375510204081635, 0.018781632653061226, 0.019187755102040818, 0.01959387755102041, 0.02];
        pub(super) const REF_ACP: [f64; 50] = [0.9999, 0.9993939281632653, 0.998482236151247, 0.9971658697746068, 0.9954462674073424, 0.9933253574008664, 0.9908055545044393, 0.9878897553011834, 0.9845813326716951, 0.98088412930003, 0.9768024502395345, 0.9723410545586446, 0.9675051460893397, 0.9623003633034387, 0.956732768344326, 0.9508088352440062, 0.9445354373575902, 0.9379198340494039, 0.9309696566668868, 0.9236928938402864, 0.9160978761478732, 0.9081932601879686, 0.8999880121005153, 0.891491390582195, 0.8827129294402376, 0.8736624197310381, 0.8643498915305172, 0.8547855953838265, 0.8449799834824948, 0.8349436906174578, 0.8246875149565874, 0.8142223986953628, 0.803559408629183, 0.7927097166955285, 0.7816845805337326, 0.7704953241095212, 0.7591533184507416, 0.7476699625398092, 0.7360566644073796, 0.7243248224706008, 0.712485807158015, 0.7005509428617844, 0.6885314902563987, 0.6764386290214057, 0.6642834410039905, 0.6520768938554192, 0.6398298251734775, 0.6275529271810673, 0.615256731969097, 0.602951597329715];
        pub(super) const REF_PVAR: [f64; 50] = [0.0, 8.35086566150923e-05, 0.00036427665694646654, 0.0007060262371262268, 0.0010732796787134508, 0.0014535966949251306, 0.0018415245877122677, 0.0022343016408137104, 0.0026303805306203873, 0.0030288272012258654, 0.00342904445175727, 0.003830632592634477, 0.0042333139526760915, 0.004636889533994358, 0.005041212905354188, 0.005446173834325988, 0.0058516876685733585, 0.006257688240975263, 0.0066641230059603235, 0.007070949629285614, 0.007478133548613841, 0.00788564619710031, 0.008293463688885967, 0.008701565832206729, 0.009109935378669422, 0.009518557445299879, 0.009927419064699858, 0.010336508831377038, 0.01074581662109973, 0.01115533336628652, 0.011565050874815465, 0.011974961682784974, 0.012385058934049243, 0.01279533628103676, 0.0132057878026138, 0.013616407935695263, 0.01402719141801755, 0.014438133240031821, 0.014849228604294808, 0.015260472891058863, 0.015671861629016216, 0.016083390470352038, 0.016495055169417957, 0.016906851564463664, 0.01731877556196414, 0.017730823123161044, 0.01814299025250202, 0.018555272987714668, 0.01896766739129531, 0.01938016954322819];
        pub(super) const REF_PLVC: [f64; 50] = [-46.051701859880914, -9.390560259444005, -7.917596932460421, -7.255858158093252, -6.837036198119699, -6.533714314601786, -6.297161470165974, -6.103826564704261, -5.940626754810012, -5.799579796690502, -5.675473642109943, -5.5647253216385995, -5.46477015230419, -5.373711496588009, -5.290108570027584, -5.212841965606132, -5.141025168989293, -5.073944452653719, -5.011016916010588, -4.951760489948464, -4.895772043718825, -4.842711109217363, -4.7922875817895365, -4.744252288802768, -4.698389661185547, -4.654511970515575, -4.61245474762873, -4.572073104111705, -4.533238751673177, -4.495837566623835, -4.459767584275513, -4.424937335457712, -4.391264457549976, -4.358674527493267, -4.327100075594661, -4.2964797475752885, -4.2667575889402265, -4.237882430880556, -4.209807360921394, -4.182489264676821, -4.155888427563649, -4.129968187310814, -4.1046946296925775, -4.080036321197186, -4.0559640733835725, -4.032450734527453, -4.009471004853693, -3.9870012722245454, -3.9650194656271367, -3.943504924197196];
        pub(super) const REF_PMC1: [f64; 50] = [1.0000000000001101, 0.8350448107787971, 0.6008631802893765, 0.46482214193008453, 0.3781610607656013, 0.3184823095513333, 0.27497630197688194, 0.24188585815349647, 0.21588372528289632, 0.19491867809825303, 0.17765927464250122, 0.16320435922768015, 0.1509225630016008, 0.14035852728110357, 0.13117564224889686, 0.12311974714707882, 0.11599536508904724, 0.10964970263564021, 0.10396160972271727, 0.09883379534545733, 0.09418723205197892, 0.08995706377601881, 0.08608956623110645, 0.08253985714167598, 0.07927014911874568, 0.07624840090592211, 0.07344726493953231, 0.07084325798191883, 0.06841610156432218, 0.06614819302691677, 0.06402417795889914, 0.062030602068842235, 0.06015562579108853, 0.058388788826199606, 0.056720814713940124, 0.055143447718956795, 0.05364931596479359, 0.05223181601847279, 0.05088501510447673, 0.04960356788551163, 0.04838264534070426, 0.04721787373893612, 0.046105282074999775, 0.04504125663107779, 0.04402250156228613, 0.04304600459534053, 0.042109007083505665, 0.04120897778636955, 0.040343589845475876, 0.03951070051099542];
        pub(super) const REF_PMC2: [f64; 50] = [0.0, 0.16495518289278022, 0.39913675054736575, 0.5351776075935236, 0.6218383269106675, 0.6815164736037208, 0.7250215717599992, 0.7581107389620364, 0.7841111656196165, 0.8050740145297678, 0.8223306650998156, 0.8367822103898511, 0.8490599565482148, 0.859619199481, 0.8687964861584001, 0.8768459144250278, 0.8839628981924434, 0.8903001678681257, 0.8959788104337529, 0.9010960537675071, 0.9057308621578934, 0.9099480284793517, 0.9138012138073057, 0.9173352371972054, 0.9205878228200234, 0.9235909487284798, 0.9263718993102191, 0.9289540946689787, 0.9313577501973453, 0.9336004055535929, 0.9356973522397292, 0.9376619817505911, 0.9395060709882544, 0.9412400177437689, 0.9428730361477898, 0.9444133198099691, 0.9458681787115114, 0.9472441546486797, 0.9485471190484585, 0.9497823562190123, 0.9509546345043083, 0.9520682673452314, 0.953127165879506, 0.9541348844179521, 0.955094659898336, 0.9560094462277624, 0.9568819442704518, 0.957714628112355, 0.9585097681315623, 0.9592694513193155];
        pub(super) const REF_QSAMPLE: [f64; 32] = [0.7136337833197913, -0.3017774921292948, 0.4978129882343361, -0.463313006933633, -0.7252503733227866, 0.15590206914702917, -0.28063518809037796, 0.10864918067384383, 0.2494078722537214, 0.11219812541008135, -0.5869233648868306, 0.7000042070725572, -0.5201629011157165, 0.05488124069516907, 0.004888113578101008, -0.8126465418417718, -0.1693671810736651, 0.6081675960091343, 0.3365366652874868, -0.09532787145369312, -0.7457692813356841, -0.752410982663935, -0.639435936371605, 0.1934948397742568, 0.47570655464520156, -0.10515948540237717, 0.33706444380709993, 0.6743815581433408, 0.2881306709325248, 0.10121602434049452, 0.1766104117746387, 0.6790062003381264];
        pub(super) const REF_NAIVE: [f64; 32] = [0.7017544352155959, -0.301550239987518, 0.4885068399301052, -0.46115979045548583, -0.7199741159044516, 0.1506724747645501, -0.2806600125499419, 0.10398297557513982, 0.24306342184564758, 0.10748960676192651, -0.583296360175104, 0.6882873624141446, -0.5173318716572244, 0.05085610288723682, 0.0014590365188390157, -0.8063282726781751, -0.1707186377202918, 0.5975457058435187, 0.32915339101298774, -0.09756208796888673, -0.7402483799757243, -0.7468108932696501, -0.6351828319442009, 0.18781703228136, 0.4666639781180415, -0.1072764810212431, 0.32967487690644054, 0.6629702085836424, 0.28132453425471543, 0.09663844367847886, 0.17113391484930762, 0.6675397118453803];
        pub(super) const REF_DDIM: [f64; 32] = [0.714826081852886, -0.3042874263051639, 0.49821839496687786, -0.4664119064686895, -0.7293043076393084, 0.15506085412265072, -0.2830680365262634, 0.10763567960781088, 0.24890758336549051, 0.11119756394753964, -0.5904729530457442, 0.7011468115908109, -0.523469077783528, 0.05367169943252259, 0.0034962952471263883, -0.8170191263068118, -0.171394341622289, 0.6089753603364427, 0.33635405168538024, -0.09708508152958573, -0.7498980284637575, -0.7565639457176092, -0.6431769876001963, 0.1927906895908957, 0.47603136038040256, -0.10695254196099335, 0.3368837545079291, 0.6754307413935333, 0.2877715670156622, 0.10017542167024118, 0.17584470024847623, 0.6800722452305077];
        pub(super) const REF_PNDM: [f64; 32] = [0.7151579349948562, -0.30392046076525236, 0.4985577110805762, -0.46603935511403033, -0.728922698625682, 0.15541199333950112, -0.28270180207631584, 0.10798845280516169, 0.2492554891999196, 0.11155021442421274, -0.5900961273086912, 0.7014791360364948, -0.5230945605888631, 0.05402633189749299, 0.0038526564486354685, -0.8166344951787756, -0.17103195476260857, 0.6093108604447484, 0.33669894465110567, -0.09672525491102683, -0.7495157099168801, -0.756181397504127, -0.6427983460055183, 0.19314052886913616, 0.47637144092316724, -0.10659237537029735, 0.33722862922334573, 0.6757639518571957, 0.2881181338356943, 0.10052845190230239, 0.17619512338152435, 0.680405295776429];
        pub(super) const REF_DPM: [f64; 32] = [0.7206223632858582, -0.3073087102562043, 0.502140545180405, -0.47083592249299505, -0.7360029191014389, 0.15601393952721473, -0.28590572625027566, 0.10817843331351573, 0.2506726486570351, 0.11177113575918107, -0.5959703690211302, 0.7068247373558659, -0.5283867633889627, 0.053747546439642725, 0.003138015041200702, -0.8244766631904544, -0.17326580913591197, 0.6138558010407378, 0.33887572057980037, -0.0983136110836391, -0.7567748207452138, -0.7634984127926581, -0.6491304089831891, 0.19407022076279193, 0.47976154411857874, -0.10826644667378185, 0.33941000649268666, 0.6808861667907989, 0.2898728909419666, 0.10065362779452397, 0.17697761147291086, 0.6855678298085071];
        pub(super) const REF_DPMPP: [f64; 32] = [0.7206654587146849, -0.3072476490108886, 0.5021874591573341, -0.4707720031767817, -0.735934365290051, 0.15606690298279008, -0.2858450390787601, 0.10823223282118488, 0.2507239577011026, 0.11182487247486497, -0.5959042626492274, 0.7068680739347288, -0.528321838219465, 0.053802297271140956, 0.0031936504082257927, -0.8244065630661949, -0.17320709064511075, 0.6139007624978556, 0.3389254880417085, -0.09825620258090055, -0.7567059038898605, -0.7634293784247362, -0.6490633734988708, 0.1941225190841161, 0.47980884922778966, -0.10820886421889575, 0.3394597646165344, 0.6809299567148428, 0.2899235148580354, 0.10070755881808807, 0.1770302085328272, 0.6856115379080973];
        pub(super) const REF_DPMPP_S10: [f64; 32] = [0.693929693227349, -0.30355860518532, 0.48191834092416674, -0.46224286415627847, -0.7195567863152157, 0.14604247551359947, -0.2827894829683394, 0.09962364556968809, 0.23789781205332183, 0.10310994804800616, -0.5836713814498093, 0.6805406919943988, -0.5180893035091868, 0.046804760990370696, -0.0023059397729998884, -0.8054103291422177, -0.1734854623508973, 0.5903250846670874, 0.3234886988273166, -0.10075301712981043, -0.7397135160403394, -0.7462379850083961, -0.6352570559859281, 0.1829716979008401, 0.46020210696266584, -0.1104110937195014, 0.3240071615529098, 0.6553703072456918, 0.27593711641973545, 0.0923216915318931, 0.16638529615189218, 0.659913320096917];
        pub(super) const REF_UNIPC: [f64; 32] = [0.7207081069546333, -0.30719394923060866, 0.502232456349159, -0.47071654527513757, -0.735876056512669, 0.15611562146706442, -0.28579156940723643, 0.1082814655967768, 0.250771658486264, 0.11187406662442735, -0.5958474593963057, 0.706910870516354, -0.5282657615745456, 0.05385211524661357, 0.003244012499244484, -0.8243473030857835, -0.17315483199309906, 0.61394455861143, 0.33897224053390385, -0.098204749758444, -0.7566473717886443, -0.7633707740365245, -0.6490059987092569, 0.19417082841593092, 0.479854087021771, -0.10815730439105072, 0.33950651136449, 0.6809730321684337, 0.28997079419173016, 0.10075687249471882, 0.17707870163149736, 0.6856545630279749];
    }
}
