"""SoVITS 4.1 shallow-diffusion model (Unit2Mel + GaussianDiffusion + WaveNet)
for ONNX export.

FAITHFUL VERBATIM PORT of the original so-vits-svc 4.1-Stable code:
  D:\\MyDev\\so-vits-svc\\so-vits-svc\\diffusion\\{unit2mel,diffusion,wavenet}.py
Every module below carries a "ported from" note; the tensor math is the
original's, line for line. diffusion_onnx.py is NOT a math reference (B=1
hardcodes, truncated constants, wrong DPM config) — diffusion.py is the truth.

Export contract (quality-path contract §2.1/2.2): TWO graphs per model —
  encoder.onnx : units[1,T,dim] f32, f0[1,T] f32 (raw Hz, in-graph
                 (1+f0/700).log()), volume[1,T] f32 (in-graph unsqueeze),
                 spk_mix[T,n_spk] f32 (ONLY when n_spk>1)
                 -> cond[1,n_hidden,T]  (ALREADY transposed, feeds denoiser)
  denoiser.onnx: x[1,1,M,T] f32, time[1] f32 (MUST be float — DPM/UniPC feed
                 non-integer t; SinusoidalPosEmb matches long input exactly at
                 integer points), cond[1,n_hidden,T] -> noise_pred[1,1,M,T]
The sampling loop itself (naive/ddim/pndm/dpm-solver/dpm-solver++/unipc) lives
Rust-side (src-tauri inference/diffusion.rs); the torch samplers kept in
GaussianDiffusion.forward below exist so verify/voice/gate1_diffusion.py can
run full-sampling parity against the original (dpm/unipc are driven through
the ORIGINAL repo's solver modules on sys.path — never shipped).

DEVIATIONS from the original (each numerically identity or gate-quantified,
also marked at the spot):
  1. spk one-hot MatMul (spk_mix @ spk_embed.weight) instead of Embedding
     gather — identical for one-hot rows (0*w exact, 1*w exact); gate (f)
     compares against the original forward's sid path at 1e-6. 0-based ids,
     the forward:161 semantics (NOT init_spkembed's buggy -1 offset).
  2. aug_shift input omitted from the export: inference always passes None so
     the original skips the term entirely (unit2mel.py:162) — identical math,
     and the layer is bias-free so None == 0 anyway.
  3. denoiser `time` input is f32 [1] (the original naive loops feed int64) —
     SinusoidalPosEmb promotes long to float internally, values identical at
     integer points (gate (b) checks int-vs-float t).
  4. f0/volume arrive as [1,T] and are unsqueezed in-graph (original takes
     [B,T,1]) — reshape only.
  5. refusal/guard messages are Chinese ValueError (original: English
     Exception) — control flow identical.
  6. the frame-wise spk_id path (spk_id.shape[1] > 1, unit2mel.py:154-159) is
     replaced by a clean raise: it requires init_spkmix, which crashes in the
     original (init_spkembed references the nonexistent self.hidden_size) —
     the path is unreachable upstream; spk mixing exports via spk_mix instead.
  7. training branch (infer=False) raises — this port is inference/export-only.
  8. spec_min/spec_max buffers are re-shaped to the checkpoint's shape before
     strict load ([1,1,1] scalar ckpts vs possible [1,1,128] vector ckpts) —
     load compatibility only, values come from the checkpoint either way.

Verified against the ORIGINAL repo by converter/verify/voice/gate1_diffusion.py.
"""

import math
import sys
from collections import deque
from functools import partial
from inspect import isfunction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Mish
from tqdm import tqdm

SUPPORTED_DIFFUSION_ENCODERS = ("vec768l12", "vec256l9")


# ---------------------------------------------------------------------------
# diffusion/diffusion.py — verbatim helpers
# ---------------------------------------------------------------------------

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


def linear_beta_schedule(timesteps, max_beta=0.02):
    """linear schedule — the ONLY schedule this repo's diffusion.py ever
    selects (the cosine_beta_schedule that exists upstream is dead code;
    unknown schedules are refused by the recompute check in
    build_from_checkpoint)."""
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


# ---------------------------------------------------------------------------
# diffusion/wavenet.py — verbatim
# ---------------------------------------------------------------------------

class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # verbatim — evaluates for FLOAT x (DPM/UniPC feed non-integer
        # timesteps); a long x promotes to float in the multiply, so integer
        # points are identical either way.
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=1
            )
            for i in range(n_layers)
        ])
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B] (long or float; float required for export)
        :param cond: [B, n_hidden, T]  (upstream docstring says [B, M, T] — wrong)
        :return: [B, 1, M, T]
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x[:, None, :, :]


# ---------------------------------------------------------------------------
# diffusion/diffusion.py — GaussianDiffusion, verbatim (inference paths)
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    def __init__(self,
                 denoise_fn,
                 out_dims=128,
                 timesteps=1000,
                 k_step=1000,
                 max_beta=0.02,
                 spec_min=-12,
                 spec_max=2):

        super().__init__()
        self.denoise_fn = denoise_fn
        self.out_dims = out_dims
        betas = linear_beta_schedule(timesteps, max_beta=max_beta)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.k_step = k_step if k_step > 0 and k_step < timesteps else timesteps

        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # spec_min buffer shape for a scalar default is [1,1,1] (the length-1
        # slice [:out_dims] keeps length 1) — NOT a 128-vector.
        self.register_buffer('spec_min', torch.FloatTensor([spec_min])[None, None, :out_dims])
        self.register_buffer('spec_max', torch.FloatTensor([spec_max])[None, None, :out_dims])

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_ddim(self, x, t, interval, cond):
        a_t = extract(self.alphas_cumprod, t, x.shape)
        a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)

        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_prev = a_prev.sqrt() * (x / a_t.sqrt() + (((1 - a_prev) / a_prev).sqrt() - ((1 - a_t) / a_t).sqrt()) * noise_pred)
        return x_prev

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from
        [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (
                    a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t - interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        else:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self,
                condition,
                gt_spec=None,
                infer=True,
                infer_speedup=10,
                method='dpm-solver',
                k_step=300,
                use_tqdm=True):
        """
            conditioning diffusion, use fastspeech2 encoder output as the condition
        """
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device

        if not infer:
            # DEVIATION 7: training (p_losses) is not ported — export-only.
            raise ValueError("移植版 GaussianDiffusion 仅支持推理（infer=True）")

        shape = (cond.shape[0], 1, self.out_dims, cond.shape[2])

        if gt_spec is None:
            t = self.k_step
            x = torch.randn(shape, device=device)
        else:
            t = k_step
            norm_spec = self.norm_spec(gt_spec)
            norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]
            x = self.q_sample(x_start=norm_spec, t=torch.tensor([t - 1], device=device).long())

        if method is not None and infer_speedup > 1:
            if method == 'dpm-solver' or method == 'dpm-solver++':
                # gate-only: drives the ORIGINAL repo's solver (so-vits-svc on
                # sys.path). Shipping inference samples Rust-side.
                DPM_Solver, NoiseScheduleVP, model_wrapper = _import_original_solver('dpm')
                # 1. Define the noise schedule.
                noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas[:t])

                # 2. Convert your discrete-time `model` to the continuous-time
                # noise prediction model.
                def my_wrapper(fn):
                    def wrapped(x, t, **kwargs):
                        ret = fn(x, t, **kwargs)
                        if use_tqdm:
                            self.bar.update(1)
                        return ret

                    return wrapped

                model_fn = model_wrapper(
                    my_wrapper(self.denoise_fn),
                    noise_schedule,
                    model_type="noise",
                    model_kwargs={"cond": cond}
                )

                # 3. Define dpm-solver and sample (multistep, order 2).
                if method == 'dpm-solver':
                    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                elif method == 'dpm-solver++':
                    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

                steps = t // infer_speedup
                if use_tqdm:
                    self.bar = tqdm(desc="sample time step", total=steps)
                x = dpm_solver.sample(
                    x,
                    steps=steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
                if use_tqdm:
                    self.bar.close()
            elif method == 'pndm':
                self.noise_list = deque(maxlen=4)
                if use_tqdm:
                    for i in tqdm(
                            reversed(range(0, t, infer_speedup)), desc='sample time step',
                            total=t // infer_speedup,
                    ):
                        x = self.p_sample_plms(
                            x, torch.full((b,), i, device=device, dtype=torch.long),
                            infer_speedup, cond=cond
                        )
                else:
                    for i in reversed(range(0, t, infer_speedup)):
                        x = self.p_sample_plms(
                            x, torch.full((b,), i, device=device, dtype=torch.long),
                            infer_speedup, cond=cond
                        )
            elif method == 'ddim':
                if use_tqdm:
                    for i in tqdm(
                            reversed(range(0, t, infer_speedup)), desc='sample time step',
                            total=t // infer_speedup,
                    ):
                        x = self.p_sample_ddim(
                            x, torch.full((b,), i, device=device, dtype=torch.long),
                            infer_speedup, cond=cond
                        )
                else:
                    for i in reversed(range(0, t, infer_speedup)):
                        x = self.p_sample_ddim(
                            x, torch.full((b,), i, device=device, dtype=torch.long),
                            infer_speedup, cond=cond
                        )
            elif method == 'unipc':
                NoiseScheduleVP, UniPC, model_wrapper = _import_original_solver('unipc')
                # 1. Define the noise schedule.
                noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas[:t])

                # 2. Convert your discrete-time `model` to the continuous-time
                # noise prediction model.
                def my_wrapper(fn):
                    def wrapped(x, t, **kwargs):
                        ret = fn(x, t, **kwargs)
                        if use_tqdm:
                            self.bar.update(1)
                        return ret

                    return wrapped

                model_fn = model_wrapper(
                    my_wrapper(self.denoise_fn),
                    noise_schedule,
                    model_type="noise",
                    model_kwargs={"cond": cond}
                )

                # 3. Define uni_pc and sample by multistep UniPC (variant bh2).
                uni_pc = UniPC(model_fn, noise_schedule, variant='bh2')

                steps = t // infer_speedup
                if use_tqdm:
                    self.bar = tqdm(desc="sample time step", total=steps)
                x = uni_pc.sample(
                    x,
                    steps=steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
                if use_tqdm:
                    self.bar.close()
            else:
                raise NotImplementedError(method)
        else:
            if use_tqdm:
                for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                    x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
            else:
                for i in reversed(range(0, t)):
                    x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
        x = x.squeeze(1).transpose(1, 2)  # [B, T, M]
        return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


def _import_original_solver(kind):
    """The original diffusion.py lazily imports its own dpm_solver_pytorch.py /
    uni_pc.py. This port does the same from the ORIGINAL repo (a `diffusion`
    package on sys.path — the gate script provides it); shipping inference
    never reaches here (samplers live in Rust)."""
    try:
        if kind == 'dpm':
            from diffusion.dpm_solver_pytorch import (
                DPM_Solver,
                NoiseScheduleVP,
                model_wrapper,
            )
            return DPM_Solver, NoiseScheduleVP, model_wrapper
        from diffusion.uni_pc import NoiseScheduleVP, UniPC, model_wrapper
        return NoiseScheduleVP, UniPC, model_wrapper
    except ImportError as e:
        raise ValueError(
            "dpm-solver/unipc 采样仅用于验证关卡（需要原版 so-vits-svc 仓库在 "
            "sys.path 上）；出货推理的采样器在 Rust 侧实现") from e


# ---------------------------------------------------------------------------
# diffusion/unit2mel.py — Unit2Mel, verbatim (inference path)
# ---------------------------------------------------------------------------

class Unit2Mel(nn.Module):
    """Parameter names match the checkpoint state_dict exactly
    (unit_embed/f0_embed/volume_embed/aug_shift_embed/spk_embed/decoder.*)."""

    def __init__(
            self,
            input_channel,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=20,
            n_chans=384,
            n_hidden=256,
            timesteps=1000,
            k_step_max=1000
    ):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)

        self.timesteps = timesteps if timesteps is not None else 1000
        self.k_step_max = k_step_max if k_step_max is not None and k_step_max > 0 and k_step_max < self.timesteps else self.timesteps

        self.n_hidden = n_hidden
        # diffusion
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden),
                                         timesteps=self.timesteps, k_step=self.k_step_max, out_dims=out_dims)
        self.input_channel = input_channel

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        """
        input:
            B x n_frames x n_unit  (f0/volume: B x n_frames x 1, f0 in raw Hz)
        return:
            B x n_frames x out_dims (denormalized mel)
        """

        if not self.training and gt_spec is not None and k_step > self.k_step_max:
            # DEVIATION 5: Chinese ValueError (original: English Exception)
            raise ValueError(f"浅扩散 k_step 超过该扩散模型的上限 k_step_max={self.k_step_max}")

        if not self.training and gt_spec is None and self.k_step_max != self.timesteps:
            raise ValueError("该扩散模型仅支持浅扩散，无法单独推理")

        x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch)
            else:
                if spk_id.shape[1] > 1:
                    # DEVIATION 6: the original's frame-wise path needs
                    # init_spkmix, which crashes upstream (self.hidden_size
                    # AttributeError) — unreachable there, refused here.
                    raise ValueError("逐帧 spk_id 混合路径未移植（原版该路径本身不可用）；请用 spk_mix_dict")
                else:
                    x = x + self.spk_embed(spk_id)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)

        return x


# ---------------------------------------------------------------------------
# export wrappers — the two shipped graph boundaries (contract §2.1/2.2)
# ---------------------------------------------------------------------------

class EncoderExport(nn.Module):
    """Condition-embedding graph: unit2mel.forward's embedding sum + the
    decoder's cond transpose (diffusion.py:236), nothing else. One graph call
    per inference; the per-step denoiser gets `cond` as-is."""

    def __init__(self, unit2mel):
        super().__init__()
        self.m = unit2mel
        self.with_spk = unit2mel.n_spk is not None and unit2mel.n_spk > 1

    def forward(self, units, f0, volume, spk_mix=None):
        # DEVIATION 4: f0/volume arrive [1,T]; unsqueeze in-graph (reshape only).
        x = (self.m.unit_embed(units)
             + self.m.f0_embed((1 + f0.unsqueeze(-1) / 700).log())
             + self.m.volume_embed(volume.unsqueeze(-1)))
        if self.with_spk:
            # DEVIATION 1: one-hot MatMul == spk_embed(sid) gather (0-based,
            # forward:161 semantics); arbitrary rows == the spk_mix_dict sum.
            # gate1_diffusion tier (f) checks both at 1e-6.
            x = x + torch.matmul(spk_mix, self.m.spk_embed.weight)
        # DEVIATION 2: aug_shift omitted (inference always None upstream).
        return x.transpose(1, 2)  # cond [1, n_hidden, T]


class DenoiserExport(nn.Module):
    """Single-step WaveNet epsilon-net: noise_pred = denoise_fn(x, time, cond).
    `time` is f32 [1] (DEVIATION 3) — DPM/UniPC feed non-integer t."""

    def __init__(self, unit2mel):
        super().__init__()
        self.m = unit2mel

    def forward(self, x, time, cond):
        return self.m.decoder.denoise_fn(x, time, cond)


# ---------------------------------------------------------------------------
# config loading / validation
# ---------------------------------------------------------------------------

def load_diffusion_config(pt_path, explicit_config=None):
    """Locate + parse the config yaml paired with the diffusion .pt (utf-8 —
    Chinese paths and speaker names must survive). Returns (dict, Path).
    Resolution order (contract §3.5): explicit --config > <stem>.yaml > the
    single *.yaml in the directory > config.yaml > Chinese refusal."""
    import yaml

    pt_path = Path(pt_path)
    if explicit_config is not None:
        path = Path(explicit_config)
        if not path.exists():
            raise ValueError(f"--config 指定的配置文件不存在: {path}")
    else:
        stem_yaml = pt_path.with_suffix(".yaml")
        candidates = sorted(pt_path.parent.glob("*.yaml"))
        config_yaml = pt_path.parent / "config.yaml"
        if stem_yaml.exists():
            path = stem_yaml
        elif len(candidates) == 1:
            path = candidates[0]
        elif config_yaml.exists():
            path = config_yaml
        elif not candidates:
            raise ValueError(
                f"未找到扩散模型的配置文件（{pt_path.parent} 下没有任何 .yaml）——"
                f"请用 --config 指定")
        else:
            names = ", ".join(p.name for p in candidates)
            raise ValueError(
                f"目录里有多个 .yaml（{names}）且没有 {stem_yaml.name} / config.yaml，"
                f"无法确定哪个是扩散配置——请用 --config 指定")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (not isinstance(cfg, dict)
            or not isinstance(cfg.get("model"), dict)
            or not isinstance(cfg.get("data"), dict)
            or not isinstance(cfg.get("vocoder"), dict)):
        raise ValueError(
            f"配置文件缺少 model/data/vocoder 段，不是 so-vits-svc 的扩散 config.yaml: {path}")
    mtype = cfg["model"].get("type")
    if mtype is not None and mtype != "Diffusion":
        raise ValueError(f"配置的 model.type={mtype} 不是 Diffusion——不是扩散模型的配置文件: {path}")
    return cfg, path


def _count_layers(sd, fmt):
    n = 0
    while fmt.format(n) in sd:
        n += 1
    return n


def _check_cfg(name, cfg_v, w_v):
    """Config value vs weight-derived truth: a mismatch means the yaml is not
    this model's pair — refuse (k_step_max from the same yaml is unverifiable,
    silently trusting a mismatched file would be dangerous)."""
    if cfg_v is not None and cfg_v != w_v:
        raise ValueError(
            f"配置的 {name}={cfg_v} 与权重推得的 {w_v} 不符 — 配置文件与模型不匹配")
    return w_v


def build_from_checkpoint(checkpoint, config, out_dims=128):
    """Build Unit2Mel from a loaded diffusion .pt dict + parsed config yaml
    dict and strict=True-load the weights. Returns (model, meta) — model in
    eval mode, meta = everything diffusion.json needs (+ exporter extras
    out_dims / schedule_check, which are NOT sidecar keys).

    Weights are the source of truth for every tensor-shaped hyperparameter
    (encoder_out_channels, n_hidden, n_chans, n_layers, timesteps, n_spk,
    use_pitch_aug, out_dims); the yaml supplies what weights cannot express
    (k_step_max, encoder name, sample_rate, block_size, sampler defaults,
    speaker map). Config/weight mismatches refuse — a wrong yaml pairing
    cannot be detected on the yaml-only values."""
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint 中没有 'model' 键，不是 so-vits-svc 的扩散模型 .pt")
    sd = checkpoint["model"]
    for key in ("unit_embed.weight", "decoder.betas",
                "decoder.denoise_fn.input_projection.weight"):
        if key not in sd:
            raise ValueError(f"权重缺少 {key} — 不是 so-vits-svc 的扩散模型 .pt")
    if config is None:
        raise ValueError("扩散模型必须带配置 yaml（k_step_max 等无法从权重推断）")

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    vocoder_cfg = config.get("vocoder", {})
    spk = dict(config.get("spk") or {})

    # --- vocoder / encoder validation (refusals BEFORE building) ---
    vocoder_type = vocoder_cfg.get("type")
    if vocoder_type != "nsf-hifigan":
        raise ValueError(
            f"暂不支持 vocoder type={vocoder_type} 的扩散模型（仅支持 nsf-hifigan；"
            f"nsf-hifigan-log10 亦未支持，排期项）")

    encoder_name = data_cfg.get("encoder")
    if encoder_name not in SUPPORTED_DIFFUSION_ENCODERS:
        raise ValueError(
            f"暂不支持 encoder={encoder_name} 的扩散模型"
            f"（仅支持 vec768l12 / vec256l9，排期项）")

    # --- weight-derived structure (truth) ---
    input_channel = sd["unit_embed.weight"].shape[1]
    n_hidden = sd["unit_embed.weight"].shape[0]
    n_chans = sd["decoder.denoise_fn.input_projection.weight"].shape[0]
    w_out_dims = sd["decoder.denoise_fn.input_projection.weight"].shape[1]
    n_layers = _count_layers(sd, "decoder.denoise_fn.residual_layers.{}.dilated_conv.weight")
    timesteps = sd["decoder.betas"].shape[0]
    has_aug = "aug_shift_embed.weight" in sd
    has_spk = "spk_embed.weight" in sd
    n_spk = sd["spk_embed.weight"].shape[0] if has_spk else 1

    if out_dims != w_out_dims:
        raise ValueError(
            f"--out-dims {out_dims} 与权重的 mel 维度 {w_out_dims} 不符"
            f"（out_dims 来自声码器 num_mels，nsf-hifigan 家族为 128）")

    expected_dim = 768 if encoder_name == "vec768l12" else 256
    if input_channel != expected_dim:
        raise ValueError(
            f"encoder={encoder_name} 期望 {expected_dim} 维特征，"
            f"但权重的 encoder_out_channels={input_channel} — 配置文件与模型不匹配")

    # --- config cross-checks (weights win detection, mismatch refuses) ---
    _check_cfg("data.encoder_out_channels", data_cfg.get("encoder_out_channels"), input_channel)
    _check_cfg("model.n_hidden", model_cfg.get("n_hidden"), n_hidden)
    _check_cfg("model.n_chans", model_cfg.get("n_chans"), n_chans)
    _check_cfg("model.n_layers", model_cfg.get("n_layers"), n_layers)
    _check_cfg("model.timesteps", model_cfg.get("timesteps"), timesteps)
    cfg_n_spk = model_cfg.get("n_spk")
    if cfg_n_spk is not None and (cfg_n_spk > 1) != has_spk:
        raise ValueError(
            f"配置的 n_spk={cfg_n_spk} 与权重（spk_embed {'存在' if has_spk else '不存在'}）"
            f"不符 — 配置文件与模型不匹配")
    if has_spk:
        _check_cfg("model.n_spk", cfg_n_spk, n_spk)
    cfg_aug = model_cfg.get("use_pitch_aug")
    if cfg_aug is not None and bool(cfg_aug) != has_aug:
        raise ValueError(
            f"配置的 use_pitch_aug={cfg_aug} 与权重（aug_shift_embed "
            f"{'存在' if has_aug else '不存在'}）不符 — 配置文件与模型不匹配")

    # --- yaml-only values ---
    # k_step_max resolution: verbatim unit2mel.py:87 semantics (0/None -> timesteps)
    k_step_max_cfg = model_cfg.get("k_step_max")
    k_step_max = (k_step_max_cfg
                  if k_step_max_cfg is not None and 0 < k_step_max_cfg < timesteps
                  else timesteps)

    sample_rate = data_cfg.get("sampling_rate", 44100)
    block_size = data_cfg.get("block_size", 512)
    if "sampling_rate" not in data_cfg or "block_size" not in data_cfg:
        print("WARNING: 配置缺少 data.sampling_rate/block_size，按 44100/512 假设",
              file=sys.stderr)
    unit_interpolate_mode = data_cfg.get("unit_interpolate_mode", "left")

    infer_cfg = config.get("infer") or {}
    infer_method = infer_cfg.get("method", "dpm-solver++")
    if infer_method not in ("ddim", "pndm", "dpm-solver", "dpm-solver++", "unipc"):
        print(f"WARNING: 配置的 infer.method={infer_method} 未知，默认档改记 dpm-solver++",
              file=sys.stderr)
        infer_method = "dpm-solver++"
    infer_speedup = int(infer_cfg.get("speedup", 10))

    # --- schedule recompute check (refuse unknown schedules) ---
    ckpt_betas = sd["decoder.betas"].detach().cpu().numpy()
    ckpt_acp = sd["decoder.alphas_cumprod"].detach().cpu().numpy()

    def schedule_diffs(max_beta):
        betas64 = np.linspace(1e-4, max_beta, timesteps)  # f64, the original recipe
        acp64 = np.cumprod(1. - betas64, axis=0)
        d_b = float(np.abs(betas64.astype(np.float32) - ckpt_betas).max())
        d_a = float(np.abs(acp64.astype(np.float32) - ckpt_acp).max())
        return d_b, d_a

    max_beta = 0.02  # the repo's only value (diffusion.py:69, never configured)
    d_betas, d_acp = schedule_diffs(max_beta)
    if max(d_betas, d_acp) > 1e-5:
        # tolerate a non-0.02 linear schedule (endpoint recoverable from betas[-1])
        alt_beta = float(ckpt_betas[-1])
        d_betas2, d_acp2 = schedule_diffs(alt_beta)
        if max(d_betas2, d_acp2) <= 1e-5:
            print(f"WARNING: max_beta 非 0.02，按 betas[-1]={alt_beta:.8g} 记录",
                  file=sys.stderr)
            max_beta, d_betas, d_acp = alt_beta, d_betas2, d_acp2
        else:
            raise ValueError(
                f"未知的扩散调度：betas 与 linear(1e-4, max_beta, {timesteps}) 重算不符"
                f"（max_beta=0.02: max_abs_diff={d_betas:.3e} / "
                f"betas[-1]={alt_beta:.6g}: {d_betas2:.3e}，阈值 1e-5）——仅支持 linear 调度")
    print(f"schedule check (linear, max_beta={max_beta:g}): "
          f"betas max_abs_diff={d_betas:.3e}, alphas_cumprod max_abs_diff={d_acp:.3e} "
          f"(tol 1e-5)")

    model = Unit2Mel(
        input_channel,
        n_spk,
        use_pitch_aug=has_aug,
        out_dims=out_dims,
        n_layers=n_layers,
        n_chans=n_chans,
        n_hidden=n_hidden,
        timesteps=timesteps,
        k_step_max=k_step_max_cfg if k_step_max_cfg is not None else timesteps,
    )
    assert model.k_step_max == k_step_max  # both apply unit2mel.py:87 verbatim

    # DEVIATION 8: some checkpoints carry [1,1,128] spec_min/spec_max vectors
    # (yaml-list configs); reshape our default [1,1,1] buffers so strict load
    # accepts them — the VALUES always come from the checkpoint.
    for name in ("spec_min", "spec_max"):
        key = f"decoder.{name}"
        if key in sd and tuple(sd[key].shape) != tuple(getattr(model.decoder, name).shape):
            setattr(model.decoder, name, torch.zeros(sd[key].shape, dtype=torch.float32))

    state_f32 = {
        k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in sd.items()
    }
    try:
        model.load_state_dict(state_f32, strict=True)
    except RuntimeError as e:
        raise ValueError(f"权重与扩散模型结构不符（strict 加载失败）: {e}") from e
    model.eval()

    meta = {
        "type": "sovits_diffusion",
        "encoder": encoder_name,
        "encoder_out_channels": int(input_channel),
        "sample_rate": int(sample_rate),
        "block_size": int(block_size),
        "n_layers": int(n_layers),
        "n_chans": int(n_chans),
        "n_hidden": int(n_hidden),
        "timesteps": int(timesteps),
        "k_step_max": int(k_step_max),
        "schedule": "linear",
        "max_beta": float(max_beta),
        "spec_min": [float(v) for v in sd["decoder.spec_min"].flatten().tolist()],
        "spec_max": [float(v) for v in sd["decoder.spec_max"].flatten().tolist()],
        "n_spk": int(n_spk),
        "speakers": {str(k): int(v) for k, v in spk.items()},
        "use_pitch_aug": bool(has_aug),
        "infer_method": infer_method,
        "infer_speedup": infer_speedup,
        "unit_interpolate_mode": unit_interpolate_mode,
        # exporter/gate extras — NOT sidecar keys (export_diffusion writes the
        # contract §2.3 schema explicitly):
        "out_dims": int(out_dims),
        "schedule_check": {"betas_max_abs_diff": d_betas,
                           "alphas_cumprod_max_abs_diff": d_acp},
    }
    return model, meta
