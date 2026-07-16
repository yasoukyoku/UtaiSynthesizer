"""SoVITS 4.0-v2 (VISinger2 architecture) synthesizer for ONNX export.

FAITHFUL VERBATIM PORT of the official so-vits-svc `4.0-v2` branch
(svc-develop-team/so-vits-svc @ cf5a8fb, later renamed Moe-SVC-V2; the local
authoritative checkout is D:\\MyDev\\TESTING\\SoVITS-4.0_v2\\src\\so-vits-svc):
  models.py (SynthesizerTrn infer() path), modules\\ddsp.py (scale_function /
  remove_above_nyquist / upsample), onnxexport\\model_onnx.py (ConviSTFT — the
  conv_transpose reformulation of torch.istft, which cannot export).
Every module below carries a "ported from" note; the tensor math is the
original's, line for line. Do NOT "simplify" any of it (same discipline as
sovits_v4.py — see its header for the bug classes hand-reconstruction causes).

4.0-v2 is a COMPLETELY different graph from 4.0/4.1 (upstream: "模型完全不通
用"): TextEncoder(pre_net Linear) → mel/energy auxiliary decoders ("aam") →
PriorDecoder → flow reverse → DDSP pair (64-harmonic sine bank + iSTFT
random-phase noise) → a downsample-conditioned HiFi-GAN. It shares 4.0's
ecosystem: ContentVec vec256l9 features, 44.1 kHz / hop 512, kmeans clusters,
config `spk` map, `emb_spk` multi-speaker table.

Shared-lineage modules are REUSED where the original 4.0-v2 code is
math-identical (diffed against 4.1-Stable 2026-07-17, see
TESTING research/s68_sovits_v2_design.md §7):
  from rvc_v2   : LayerNorm, WN (v2's n_speakers/spk_channels signature is a
                  pure rename of gin_channels — forward is line-identical),
                  MultiHeadAttention, FFN, attentions.Encoder (window_size 4),
                  ResBlock1/ResBlock2, sequence_mask, init_weights, get_padding
  from sovits_v4: Flip, ResidualCouplingLayer, ResidualCouplingBlock (v2 keeps
                  it in modules.py without share_parameter — constructed here
                  with share_parameter=False, parameter names identical), FFT
                  (v2's == 4.1's with g=None), normalize_f0 (verbatim same fn
                  in both utils.py; carries the gated torch.where rewrite),
                  load_sovits_config (same config.json discovery convention)

Export contract (OURS — the upstream onnxexport mel2ph/t_window inputs are
deliberately not copied; expansion stays Rust-side like every other export):
  inputs : c[1,T,256] f32 (ALREADY expanded to the f0 frame count),
           f0[1,T] f32 (Hz, interpolated; NO f0_to_coarse — v2 feeds
           continuous mel-scale lf0 through conv prenets),
           noise[1,inter_channels,T] f32 — caller N(0,1) ALREADY scaled by
           noice_scale (original inference default 0.4); zeros = det z_p,
           phase[1,n_fft//2+1,T] f32 — Generator_Noise's random phase,
           caller-supplied uniform*2*3.14-3.14 (upstream uses literal 3.14,
           kept verbatim); zeros = deterministic noise branch,
           sid[1] i64 — or spk_mix[1,n_spk] f32 for genuine multi-speaker
  output : audio[1,1,T*hop_size]
The main graph has NO uv input (v2's infer() only reads uv inside the
predict_f0 branch, which lives in the standalone <stem>.f0.onnx companion:
c/f0/uv/sid[|spk_mix] -> f0_pred[1,T] Hz, same mechanism as sovits_v4).

DOCUMENTED DEVIATIONS (each numerically gated by gate1_sovits_v2.py):
 1. Random phase + z_p noise become explicit graph inputs (upstream onnxexport
    keeps torch.rand in-graph = non-deterministic; upstream models.py keeps
    both in-graph). Same policy as every other voice export here.
 2. ConviSTFT normalization frames (t_window) are built IN-graph from the
    registered window buffer (broadcast against ones_like) instead of being a
    graph input (upstream onnxexport takes them as input `t_window` because
    MoeSS feeds them at runtime). Same numbers, one fewer input.
 3. ConviSTFT window = HANN. Training-time Generator_Noise runs torch.istft
    with torch.hann_window (models.py:607,621); upstream onnxexport's ConviSTFT
    ctor default 'hamming' is an upstream bug we do not reproduce — the gate
    compares against the TRAINING formulation (torch.istft, hann).
 4. ConviSTFT edge alignment follows torch.istft (= the training formulation):
    crop n_fft//2 from the front, then take length = T*hop samples (istft's
    `length` semantics — the tail keeps real OLA content). Upstream
    onnxexport's symmetric (win-hop)/2 = 768 trim is 256 samples MISALIGNED
    vs the torch.istft the model was trained with — an upstream bug (measured
    0.14 max|Δ|), not reproduced.
 5. The auto-f0 companion calls normalize_f0(random_scale=False). v2's
    models.py infer() (line 1022) omits the flag and inherits random_scale=
    True — a train-time augmentation leaking into inference (uniform 0.8-1.2
    scale jitter on the normalized contour) that the 4.0/4.1 branches fixed to
    False. We adopt the fixed semantic; the gate pins factor=1 on the original
    for comparison.

 6. Stable phase (the rvc_v2.SineGen convention): the harmonic bank's and the
    sin-condition channel's per-sample fp32 phase cumsum is reformulated as
    frac(fp64 frame cumsum) + within-frame ramp, frac'd per harmonic — a
    sin-invariant identity that keeps every argument bounded (the original
    reaches ~4e5 rad at 30 s × harmonic 64 where fp32 sin() decorrelates and
    torch-vs-ORT cumsum rounding diverges). Gated against an fp64 reference
    of the original formulation (ours must be strictly closer than the
    original's own fp32).

Unsupported (clean Chinese ValueError): checkpoints whose weights disagree
with the fixed VISinger2 template geometry in ways weights cannot express
(see build_from_checkpoint).

Verified against the ORIGINAL branch code by converter/verify/voice/
gate1_sovits_v2.py.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from .rvc_v2 import (
    LayerNorm,
    WN,
    Encoder as AttentionEncoder,   # attentions.Encoder — window_size passed explicitly (4)
    ResBlock1,
    ResBlock2,
    sequence_mask,
    init_weights,
    get_padding,
)
from .sovits_v4 import (
    FFT,
    ResidualCouplingBlock,
    normalize_f0,
    load_sovits_config,
)


# ---------------------------------------------------------------------------
# modules/ddsp.py — verbatim helpers used by the harmonic/noise generators
# ---------------------------------------------------------------------------

def upsample(signal, factor):
    # ported from modules/ddsp.py upsample — F.interpolate default mode
    # 'nearest' at an integer factor (frame -> sample rate).
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    # ported from modules/ddsp.py remove_above_nyquist — verbatim.
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa


def scale_function(x):
    # ported from modules/ddsp.py scale_function — verbatim.
    return 2 * torch.sigmoid(x) ** (math.log(10)) + 1e-7


def stable_phase_cycles(f0_frames, hop_length, sampling_rate):
    """DEVIATION 6 (header; the rvc_v2.SineGen stable-phase convention, gated):
    per-sample fundamental phase in CYCLES equal to the original
    cumsum(2*pi*f0_up/sr)/2π modulo whole cycles (sin-invariant identity).

    The original accumulates phase per-SAMPLE in fp32: at 30 s × harmonic 64
    the argument reaches ~4e5 rad where fp32 sin() decorrelates, and torch-vs-
    ORT cumsum rounding diverges (measured 0.24-0.54 max|Δ| audio at T≥137).
    Identity used instead: φ(t, j) = frac(Σ_{u<t} f0_u·hop/sr) + f0_t·j/sr,
    j = 1..hop — frame-level accumulation in fp64 (frac stays exact for hours),
    per-sample argument bounded. EPs without fp64 kernels partition the few
    frame-rate fp64 nodes back to CPU (same trade as rvc_v2).

    f0_frames: [B, T, 1] Hz (frame rate). returns φ [B, T*hop, 1] cycles;
    callers take frac(φ·h) per harmonic before sin(2π·)."""
    rad = f0_frames.double() / sampling_rate           # cycles per sample (fp64)
    cyc = rad * hop_length                             # cycles per frame
    start = torch.cumsum(cyc, 1) - cyc                 # frame-START phase (exclusive)
    start = start - torch.floor(start)                 # frac(); Floor beats fp64 Mod for EP support
    start_up = F.interpolate(start.float().transpose(2, 1),
                             scale_factor=float(hop_length),
                             mode="nearest").transpose(2, 1)
    rad_up = F.interpolate(rad.float().transpose(2, 1),
                           scale_factor=float(hop_length),
                           mode="nearest").transpose(2, 1)
    # within-frame sample index 1..hop (int64 cumsum: exact for any length;
    # matches the original's INCLUSIVE per-sample cumsum — sample j carries
    # j+1 increments within its frame)
    within = torch.cumsum(torch.ones_like(rad_up, dtype=torch.int64), 1)
    within = ((within - 1) % hop_length + 1).to(rad_up.dtype)
    return start_up + rad_up * within


# ---------------------------------------------------------------------------
# models.py ConvReluNorm — the v2 VARIANT (differs from modules/commons.py's:
# no mask argument, residual averaging (x + x_) / 2, zero-init proj)
# ---------------------------------------------------------------------------

class ConvReluNormV2(nn.Module):
    """ported from 4.0-v2 models.py ConvReluNorm (:329-365) — verbatim."""

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size,
                 n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size,
                                          padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels,
                                              kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x):
        x = self.conv_layers[0](x)
        x = self.norm_layers[0](x)
        x = self.relu_drop(x)

        for i in range(1, self.n_layers):
            x_ = self.conv_layers[i](x)
            x_ = self.norm_layers[i](x_)
            x_ = self.relu_drop(x_)
            x = (x + x_) / 2
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# models.py encoders/decoders (namespaces match the checkpoint exactly)
# ---------------------------------------------------------------------------

class TextEncoderV2(nn.Module):
    """ported from 4.0-v2 models.py TextEncoder (:74-111) — namespace
    text_encoder.{pre_net,encoder,proj}. pre_net is a LINEAR on [b,t,c_dim]
    (unlike 4.x's conv `pre`)."""

    def __init__(self, c_dim, out_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, window_size=4):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.pre_net = torch.nn.Linear(c_dim, hidden_channels)
        # 4.0-v2 attentions.Encoder == rvc_v2's port (window_size default 4
        # upstream; passed explicitly because the class is shared).
        self.encoder = AttentionEncoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths):
        x = x.transpose(1, -1)
        x = self.pre_net(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x, x_mask


class PriorDecoderV2(nn.Module):
    """ported from 4.0-v2 models.py PriorDecoder (:177-223) — namespace
    decoder.{prenet,decoder,proj,cond}."""

    def __init__(self, out_bn_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout,
                 n_speakers=0, spk_channels=0):
        super().__init__()
        self.out_bn_channels = out_bn_channels
        self.hidden_channels = hidden_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_bn_channels, 1)

        if n_speakers != 0:
            self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, x_lengths, spk_emb=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.prenet(x) * x_mask

        if (spk_emb is not None):
            x = x + self.cond(spk_emb)

        x = self.decoder(x * x_mask, x_mask)

        bn = self.proj(x) * x_mask

        return bn, x_mask


class MelDecoderV2(nn.Module):
    """ported from 4.0-v2 models.py Decoder (:226-274) — namespace
    mel_decoder.{prenet,decoder,proj,cond}. Detaches its input (the "aam"
    auxiliary mel head)."""

    def __init__(self, out_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout,
                 n_speakers=0, spk_channels=0, in_channels=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.prenet = nn.Conv1d(in_channels if in_channels is not None else hidden_channels,
                                hidden_channels, 3, padding=1)
        self.decoder = FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

        if n_speakers != 0:
            self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, x_lengths, spk_emb=None):
        x = torch.detach(x)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.prenet(x) * x_mask

        if (spk_emb is not None):
            x = x + self.cond(spk_emb)

        x = self.decoder(x * x_mask, x_mask)

        x = self.proj(x) * x_mask

        return x, x_mask


class F0DecoderV2(nn.Module):
    """ported from 4.0-v2 models.py F0Decoder (:276-326) — namespace
    f0_decoder.{prenet,decoder,proj,f0_prenet,cond}. NOTE the op order differs
    from 4.x's F0Decoder: f0_prenet is added BEFORE the mask/prenet and the
    speaker cond is added AFTER the prenet."""

    def __init__(self, out_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout,
                 n_speakers=0, spk_channels=0, in_channels=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.prenet = nn.Conv1d(in_channels if in_channels is not None else hidden_channels,
                                hidden_channels, 3, padding=1)
        self.decoder = FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)

        if n_speakers != 0:
            self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_lengths, spk_emb=None):
        x = torch.detach(x)
        x = x + self.f0_prenet(norm_f0)   # original writes `x += …` (same op)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.prenet(x) * x_mask

        if (spk_emb is not None):
            x = x + self.cond(spk_emb)

        x = self.decoder(x * x_mask, x_mask)

        x = self.proj(x) * x_mask

        return x, x_mask


class PosteriorEncoderV2(nn.Module):
    """ported from 4.0-v2 models.py PosteriorEncoder (:368-400) — namespace
    posterior_encoder.{pre,enc,proj}. Training-only (the infer() path never
    calls it); loaded for strict=True. WN cond gate: the original passes
    n_speakers/spk_channels where rvc_v2.WN takes gin_channels — pure rename
    (diffed, forward line-identical)."""

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size,
                 dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                      gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        return stats, x_mask


# ---------------------------------------------------------------------------
# models.py DSP generators
# ---------------------------------------------------------------------------

class ResBlock3(torch.nn.Module):
    """ported from 4.0-v2 models.py ResBlock3 (:403-425) — verbatim."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock3, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)  # LRELU_SLOPE
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class GeneratorHarmV2(torch.nn.Module):
    """ported from 4.0-v2 models.py Generator_Harm (:428-483) — namespace
    dec_harm.{prenet,net,postnet}. DDSP harmonic bank: 64 sine partials from
    cumsum phase, amplitudes from a ConvReluNorm stack."""

    def __init__(self, hidden_channels, kernel_size, p_dropout, n_harmonic,
                 sampling_rate, hop_length):
        super(GeneratorHarmV2, self).__init__()
        self.n_harmonic = n_harmonic
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

        self.prenet = Conv1d(hidden_channels, hidden_channels, 3, padding=1)

        self.net = ConvReluNormV2(hidden_channels,
                                  hidden_channels,
                                  hidden_channels,
                                  kernel_size,
                                  8,
                                  p_dropout)

        self.postnet = Conv1d(hidden_channels, n_harmonic + 1, 3, padding=1)

    def forward(self, f0, harm, mask):
        pitch = f0.transpose(1, 2)
        harm = self.prenet(harm)

        harm = self.net(harm) * mask

        harm = self.postnet(harm)
        harm = harm.transpose(1, 2)
        param = harm

        param = scale_function(param)
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.hop_length)

        n_harmonic = amplitudes.shape[-1]
        # DEVIATION 6: stable phase — original was
        #   pitch = upsample(pitch, hop); omega = cumsum(2π·pitch/sr, 1);
        #   omegas = omega * arange(1, n+1); sin(omegas)
        # (sin-invariant identity; frac per harmonic keeps every argument
        # bounded; gated vs an fp64 reference of the original formulation)
        phi = stable_phase_cycles(pitch, self.hop_length, self.sampling_rate)
        args = phi * torch.arange(1, n_harmonic + 1).to(phi)
        args = args - torch.floor(args)
        signal_harmonics = (torch.sin(2 * math.pi * args) * amplitudes)
        signal_harmonics = signal_harmonics.transpose(1, 2)
        return signal_harmonics


def init_istft_kernels(win_len, fft_len, win_type="hann"):
    """ported from onnxexport/model_onnx.py init_kernels (:591-608), inverse
    branch only (invers=True), numpy math verbatim. DEVIATION 3 (header): the
    window is HANN — the training graph runs torch.istft with
    torch.hann_window (models.py:607,621); upstream onnxexport's ctor default
    'hamming' is not reproduced. scipy.get_window('hann', N, fftbins=True) ==
    torch.hann_window(N) (both periodic)."""
    from scipy.signal import get_window
    window = get_window(win_type, win_len, fftbins=True)

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return (torch.from_numpy(kernel.astype(np.float32)),
            torch.from_numpy(window.astype(np.float32)))


class ConviSTFT(nn.Module):
    """ported from onnxexport/model_onnx.py ConviSTFT (:611-635): overlap-add
    inverse STFT as conv_transpose1d with a pinv(rDFT)*window kernel, divided
    by the overlap-added squared-window envelope. DEVIATION 2 (header): the
    envelope frames are built in-graph from the registered window buffer
    (upstream feeds them as the `t_window` input). DEVIATION 4 (header): edge
    alignment follows torch.istft — the TRAINING formulation — exactly:
    crop n_fft//2 from the front, then take length = T*hop samples (the tail
    keeps real OLA content — istft's `length` semantics, probe-verified).
    Upstream onnxexport trims (win-hop)/2 = 768 per side, which is 256
    samples MISALIGNED vs the torch.istft the model was trained with
    (measured 0.14 max|Δ|) — an upstream bug we do not reproduce."""

    def __init__(self, win_len, win_inc, fft_len):
        super(ConviSTFT, self).__init__()
        self.win_len = win_len
        self.stride = win_inc
        self.fft_len = fft_len
        self.trim = fft_len // 2  # torch.istft center=True crop (1024 for 2048)
        kernel, window = init_istft_kernels(win_len, fft_len, "hann")
        # persistent=False: these are derived constants, NOT checkpoint weights
        # — persistent buffers would break the strict=True state-dict load.
        self.register_buffer("weight", kernel, persistent=False)
        self.register_buffer("window_sq", (window * window).reshape(1, -1, 1),
                             persistent=False)
        self.register_buffer("enframe", torch.eye(win_len)[:, None, :],
                             persistent=False)

    def forward(self, inputs):
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)
        # in-graph t_window: every frame contributes window^2 (broadcast keeps
        # the T axis dynamic — ones_like carries the traced shape)
        t = torch.ones_like(inputs[:, :1, :]) * self.window_sq
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        # torch.istft alignment (probe-verified): crop n_fft//2 from the FRONT,
        # then take exactly length = T*hop samples — the back crop follows from
        # `length`, NOT a symmetric trim (the tail keeps real OLA content:
        # full OLA is T*hop + win - hop long, leaving hop extra past T*hop).
        end = self.trim + inputs.size(2) * self.stride
        outputs = outputs[..., self.trim:end]
        return outputs


class GeneratorNoiseV2(torch.nn.Module):
    """ported from 4.0-v2 models.py Generator_Noise (:590-624) with the
    onnxexport ConviSTFT substitution for torch.istft — namespace
    dec_noise.{istft_pre,net,istft_amplitude}. DEVIATION 1 (header): the
    random phase is an explicit `phase` input [1, fft//2+1, T] instead of the
    in-graph torch.rand; the caller supplies uniform*2*3.14-3.14 (upstream's
    literal 3.14, kept verbatim)."""

    def __init__(self, hidden_channels, kernel_size, p_dropout,
                 win_size, hop_size, n_fft):
        super(GeneratorNoiseV2, self).__init__()
        self.win_size = win_size
        self.hop_size = hop_size
        self.fft_size = n_fft
        self.istft_pre = Conv1d(hidden_channels, hidden_channels, 3, padding=1)

        self.net = ConvReluNormV2(hidden_channels,
                                  hidden_channels,
                                  hidden_channels,
                                  kernel_size,
                                  8,
                                  p_dropout)

        self.istft_amplitude = torch.nn.Conv1d(hidden_channels, self.fft_size // 2 + 1, 1, 1)
        self.istft = ConviSTFT(self.win_size, self.hop_size, self.fft_size)

    def forward(self, x, mask, phase):
        istft_x = x
        istft_x = self.istft_pre(istft_x)

        istft_x = self.net(istft_x) * mask

        amp = self.istft_amplitude(istft_x).unsqueeze(-1)
        phase = phase.unsqueeze(-1)

        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        spec = torch.cat([real, imag], 1).squeeze(3)
        istft_x = self.istft(spec)

        return istft_x  # [b, 1, T*hop]


class GeneratorV2(torch.nn.Module):
    """ported from 4.0-v2 models.py Generator (:486-580) — namespace
    dec.{conv_pre,downs,resblocks_downs,concat_pre,concat_conv,ups,resblocks,
    conv_post,cond}. The DSP condition (harm+noise+sin, n_harmonic+2 channels
    at sample rate) is downsampled level-by-level and concatenated back on the
    way up. conv_pre/concat_*/conv_post are PLAIN convs (unlike 4.x's NSF
    HiFi-GAN); downs/ups/resblocks are weight-normed."""

    def __init__(self, initial_channel, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes,
                 n_harmonic, n_speakers=0, spk_channels=0):
        super(GeneratorV2, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.upsample_rates = upsample_rates
        self.n_speakers = n_speakers

        resblock_cls = ResBlock1 if resblock == '1' else ResBlock2

        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            i = len(upsample_rates) - 1 - i
            u = upsample_rates[i]
            k = upsample_kernel_sizes[i]
            self.downs.append(weight_norm(
                Conv1d(n_harmonic + 2, n_harmonic + 2,
                       k, u, padding=k // 2)))

        self.resblocks_downs = nn.ModuleList()
        for i in range(len(self.downs)):
            self.resblocks_downs.append(ResBlock3(n_harmonic + 2, 3, (1, 3)))

        self.concat_pre = Conv1d(upsample_initial_channel + n_harmonic + 2,
                                 upsample_initial_channel, 3, 1, padding=1)
        self.concat_conv = nn.ModuleList()
        for i in range(len(upsample_rates)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.concat_conv.append(Conv1d(ch + n_harmonic + 2, ch, 3, 1, padding=1, bias=False))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock_cls(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if self.n_speakers != 0:
            self.cond = nn.Conv1d(spk_channels, upsample_initial_channel, 1)

    def forward(self, x, ddsp, g=None):

        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        se = ddsp
        res_features = [se]
        for i in range(self.num_upsamples):
            in_size = se.size(2)
            se = self.downs[i](se)
            se = self.resblocks_downs[i](se)
            up_rate = self.upsample_rates[self.num_upsamples - 1 - i]
            se = se[:, :, : in_size // up_rate]
            res_features.append(se)

        x = torch.cat([x, se], 1)
        x = self.concat_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)  # modules.LRELU_SLOPE
            in_size = x.size(2)
            x = self.ups[i](x)
            # 保证维度正确，丢掉多余通道 (original comment; no-op when k-u even)
            x = x[:, :, : in_size * self.upsample_rates[i]]

            x = torch.cat([x, res_features[self.num_upsamples - 1 - i]], 1)
            x = self.concat_conv[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.downs:
            remove_weight_norm(l)
        for l in self.resblocks_downs:
            l.remove_weight_norm()
        for l in self.resblocks:
            l.remove_weight_norm()


# ---------------------------------------------------------------------------
# top-level synthesizer — models.py SynthesizerTrn infer() path with explicit
# noise/phase inputs (parameter names match the checkpoint state_dict exactly)
# ---------------------------------------------------------------------------

class SynthesizerTrnV2(nn.Module):
    """4.0-v2 SynthesizerTrn (models.py :847-1060). Modules LR (LengthRegulator)
    and dropout hold no parameters and are dead on the infer() path — omitted
    (strict=True load is unaffected)."""

    def __init__(self,
                 c_dim,
                 prior_hidden_channels,
                 prior_filter_channels,
                 prior_n_heads,
                 prior_n_layers,
                 prior_kernel_size,
                 prior_p_dropout,
                 hidden_channels,
                 spk_channels,
                 kernel_size,
                 p_dropout,
                 acoustic_dim,
                 n_harmonic,
                 n_fft,
                 win_size,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers,
                 sampling_rate,
                 hop_length,
                 window_size=4,          # ours: from weights (emb_rel_k); v2 hardcodes 4
                 has_posterior=True,     # ours: tolerate stripped checkpoints
                 **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

        self.text_encoder = TextEncoderV2(
            c_dim,
            prior_hidden_channels,
            prior_hidden_channels,
            prior_filter_channels,
            prior_n_heads,
            prior_n_layers,
            prior_kernel_size,
            prior_p_dropout,
            window_size=window_size)

        self.decoder = PriorDecoderV2(
            hidden_channels * 2,
            prior_hidden_channels,
            prior_filter_channels,
            prior_n_heads,
            prior_n_layers,
            prior_kernel_size,
            prior_p_dropout,
            n_speakers=n_speakers,
            spk_channels=spk_channels)

        self.f0_decoder = F0DecoderV2(
            1,
            prior_hidden_channels,
            prior_filter_channels,
            prior_n_heads,
            prior_n_layers,
            prior_kernel_size,
            prior_p_dropout,
            n_speakers=n_speakers,
            spk_channels=spk_channels)

        self.mel_decoder = MelDecoderV2(
            acoustic_dim,
            prior_hidden_channels,
            prior_filter_channels,
            prior_n_heads,
            prior_n_layers,
            prior_kernel_size,
            prior_p_dropout,
            n_speakers=n_speakers,
            spk_channels=spk_channels)

        if has_posterior:
            self.posterior_encoder = PosteriorEncoderV2(
                acoustic_dim,
                hidden_channels,
                hidden_channels, 3, 1, 8,
                gin_channels=spk_channels if n_speakers != 0 else 0)

        self.dec = GeneratorV2(hidden_channels,
                               resblock,
                               resblock_kernel_sizes,
                               resblock_dilation_sizes,
                               upsample_rates,
                               upsample_initial_channel,
                               upsample_kernel_sizes,
                               n_harmonic,
                               n_speakers=n_speakers,
                               spk_channels=spk_channels)

        self.dec_harm = GeneratorHarmV2(hidden_channels, kernel_size, p_dropout,
                                        n_harmonic, sampling_rate, hop_length)

        self.dec_noise = GeneratorNoiseV2(hidden_channels, kernel_size, p_dropout,
                                          win_size, hop_length, n_fft)

        self.f0_prenet = nn.Conv1d(1, prior_hidden_channels, 3, padding=1)
        self.energy_prenet = nn.Conv1d(1, prior_hidden_channels, 3, padding=1)
        self.mel_prenet = nn.Conv1d(acoustic_dim, prior_hidden_channels, 3, padding=1)

        if n_speakers > 0:
            self.emb_spk = nn.Embedding(n_speakers, spk_channels)
        # ①c: convert.py flips this True for GENUINE multi-speaker models —
        # then `sid` is fed as a spk_mix [1, n_spk] f32 blend (matmul
        # emb_spk.weight) instead of a scalar id.
        self.export_spk_mix = False

        self.flow = ResidualCouplingBlock(prior_hidden_channels, hidden_channels,
                                          5, 1, 4, gin_channels=spk_channels,
                                          share_parameter=False)

        self.acoustic_dim = acoustic_dim

    def forward(self, c, f0, noise, phase, sid):
        """
        c: [1, T, c_dim] f32 — vec256l9 features ALREADY expanded to the f0
           frame count (repeat_expand stays Rust-side)
        f0: [1, T] f32 — continuous F0 in Hz (0 = unvoiced); v2 feeds the
           mel-scale lf0 through conv prenets (no f0_to_coarse embedding)
        noise: [1, hidden_channels, T] f32 — caller noise ALREADY scaled by
           noice_scale (original default 0.4); zeros = deterministic z_p
        phase: [1, n_fft//2+1, T] f32 — Generator_Noise phase, caller-supplied
           uniform*2*3.14-3.14; zeros = deterministic noise branch
        sid: [1] i64 — speaker id (or spk_mix [1, n_spk] f32 when
           export_spk_mix; a one-hot row == the emb_spk gather bit-for-bit)
        returns audio [1, 1, T * hop_length]
        """
        # models.py infer() :1005-1010 — g from the speaker table
        if self.export_spk_mix:
            g = torch.matmul(sid, self.emb_spk.weight).unsqueeze(-1)  # [1, spk, 1]
        else:
            g = self.emb_spk(sid).unsqueeze(-1)  # [b, h, 1]
        f0 = f0.unsqueeze(0)  # [1,T] -> [1,1,T] (infer(): len(f0.shape)==2 branch)

        c_lengths = (torch.ones(c.size(0)) * c.size(1)).to(c.device)
        c = c.transpose(1, 2)  # our contract [1,T,c_dim] -> upstream layout [b,c_dim,t]

        # Encoder
        decoder_input, x_mask = self.text_encoder(c, c_lengths)
        y_lengths = c_lengths

        LF0 = 2595. * torch.log10(1. + f0 / 700.)
        LF0 = LF0 / 500

        # (predict_f0 branch lives in the standalone .f0.onnx companion)

        # aam
        predict_mel, predict_bn_mask = self.mel_decoder(
            decoder_input + self.f0_prenet(LF0), y_lengths, spk_emb=g)
        predict_energy = predict_mel.sum(1).unsqueeze(1) / self.acoustic_dim

        decoder_input = decoder_input + \
                        self.f0_prenet(LF0) + \
                        self.energy_prenet(predict_energy) + \
                        self.mel_prenet(predict_mel)
        decoder_output, y_mask = self.decoder(decoder_input, y_lengths, spk_emb=g)

        prior_info = decoder_output

        m_p = prior_info[:, :self.hidden_channels, :]
        logs_p = prior_info[:, self.hidden_channels:, :]
        # onnxexport formulation (:1027): noise pre-scaled by the caller ==
        # infer()'s randn_like(m_p)*exp(logs_p)*noice_scale
        z_p = m_p + torch.exp(logs_p) * noise
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        prior_z = z

        noise_x = self.dec_noise(prior_z, y_mask, phase)

        harm_x = self.dec_harm(f0, prior_z, y_mask)

        # DEVIATION 6: stable phase (original: pitch = upsample(f0.T, hop);
        # sin(cumsum(2π·pitch/sr, 1)) — see stable_phase_cycles)
        phi = stable_phase_cycles(f0.transpose(1, 2), self.hop_length,
                                  self.sampling_rate)
        sin = torch.sin(2 * math.pi * (phi - torch.floor(phi))).transpose(1, 2)

        decoder_condition = torch.cat([harm_x, noise_x, sin], axis=1)

        # dsp based HiFiGAN vocoder
        o = self.dec(prior_z, decoder_condition, g=g)

        return o

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        if hasattr(self, "posterior_encoder"):
            self.posterior_encoder.enc.remove_weight_norm()


def set_deterministic(model, deterministic=True):
    """v2 has NO in-graph randomness left after noise/phase became explicit
    inputs — kept as a no-op for convert.py symmetry with the other archs."""
    _ = (model, deterministic)


class F0PredictorWrapperV2(nn.Module):
    """Standalone auto-f0 predictor export graph (<stem>.f0.onnx) for 4.0-v2.

    Wraps the predict_f0 branch of models.py infer() (:1015-1026) around the
    ALREADY-LOADED synthesizer's own modules (text_encoder / f0_decoder /
    emb_spk are the SAME nn.Module objects — shared nn.Parameters, no copies).

    inputs : c[1,T,c_dim] f32 (expanded, same layout as the main graph),
             f0[1,T] f32 (source F0 in Hz, AFTER transpose/key shift),
             uv[1,T] f32, sid[1] i64 (or spk_mix[1,n_spk] f32)
    output : f0_pred[1,T] f32 (Hz) — feed as the main graph's `f0` input
             (uv is not a main-graph input for v2; it exists only here).
    DEVIATION 5 (header): normalize_f0 runs with random_scale=False — v2's
    infer() omits the flag and inherits the train-time random scale jitter,
    which the 4.0/4.1 branches fixed; gated with factor pinned to 1."""

    def __init__(self, synth):
        super().__init__()
        if not hasattr(synth, "f0_decoder"):
            raise ValueError("该模型没有 f0_decoder 权重，无法导出自动音高预测器")
        self.text_encoder = synth.text_encoder
        self.f0_decoder = synth.f0_decoder
        self.emb_spk = synth.emb_spk
        self.export_spk_mix = getattr(synth, "export_spk_mix", False)

    def forward(self, c, f0, uv, sid):
        if self.export_spk_mix:
            g = torch.matmul(sid, self.emb_spk.weight).unsqueeze(-1)
        else:
            g = self.emb_spk(sid).unsqueeze(-1)
        f0 = f0.unsqueeze(0)  # [1,T] -> [1,1,T]

        c_lengths = (torch.ones(c.size(0)) * c.size(1)).to(c.device)
        c = c.transpose(1, 2)

        decoder_input, x_mask = self.text_encoder(c, c_lengths)
        y_lengths = c_lengths

        # models.py infer() :1018-1026 (predict_f0 branch), verbatim modulo
        # random_scale=False (DEVIATION 5)
        LF0 = 2595. * torch.log10(1. + f0 / 700.)
        LF0 = LF0 / 500
        norm_f0 = normalize_f0(LF0, x_mask, uv, random_scale=False)
        pred_lf0, predict_bn_mask = self.f0_decoder(decoder_input, norm_f0,
                                                    y_lengths, spk_emb=g)
        pred_f0 = 700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)
        return pred_f0.squeeze(1)


# ---------------------------------------------------------------------------
# checkpoint detection / config / builder
# ---------------------------------------------------------------------------

def is_v2_state_dict(sd):
    """Architecture detector for convert.py routing: a 4.0-v2 (VISinger2)
    checkpoint carries text_encoder/dec_harm/dec_noise (4.0/4.1 carry
    enc_p/emb_g instead — disjoint namespaces)."""
    return ("text_encoder.pre_net.weight" in sd
            and "dec_harm.postnet.weight" in sd
            and "dec_noise.istft_amplitude.weight" in sd)


def _count_layers(sd, fmt):
    n = 0
    while fmt.format(n) in sd:
        n += 1
    return n


def build_from_checkpoint(checkpoint, config=None):
    """Build the 4.0-v2 synthesizer from a loaded .pth checkpoint dict
    (+ parsed config.json dict, or None) and strict=True-load the weights.
    Returns (model, meta) — model ready for export (weight norm removed, eval
    mode), meta = everything the sidecar json needs.

    Weights are the source of truth for every tensor-shaped hyperparameter;
    the config supplies what weights cannot express (sampling_rate, upsample
    STRIDES, resblock dilations, n_heads, speaker map). The v2 config is the
    fixed official template (configs_template/config_template.json) — a
    missing config falls back to those template values with warnings (the
    community 底模/chika packages ship WITHOUT config.json)."""
    if "model" not in checkpoint:
        raise ValueError("checkpoint 中没有 'model' 键，不是 so-vits-svc 的 G_*.pth")
    sd = checkpoint["model"]
    if not is_v2_state_dict(sd):
        raise ValueError("权重不是 SoVITS 4.0-v2 (VISinger2) 结构")

    if config is not None:
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        spk = dict(config.get("spk") or {})
    else:
        print("WARNING: 未找到 config.json — 按 4.0-v2 官方模板默认值推断超参数"
              "（sampling_rate/上采样步幅/膨胀率），speaker 名不可恢复。",
              file=sys.stderr)
        model_cfg, data_cfg, spk = {}, {}, {}

    # --- weight-derived structure (truth) ---
    c_dim = sd["text_encoder.pre_net.weight"].shape[1]
    prior_hidden = sd["text_encoder.pre_net.weight"].shape[0]
    prior_filter = sd["text_encoder.encoder.ffn_layers.0.conv_1.weight"].shape[0]
    prior_kernel = sd["text_encoder.encoder.ffn_layers.0.conv_1.weight"].shape[2]
    emb_rel_k = sd["text_encoder.encoder.attn_layers.0.emb_rel_k"]
    window_size = (emb_rel_k.shape[1] - 1) // 2
    prior_n_layers = _count_layers(sd, "text_encoder.encoder.attn_layers.{}.conv_q.weight")
    fft_n_layers = _count_layers(sd, "decoder.decoder.self_attn_layers.{}.conv_q.weight")

    hidden_channels = sd["decoder.proj.weight"].shape[0] // 2
    spk_channels = sd["emb_spk.weight"].shape[1] if "emb_spk.weight" in sd else 0
    n_speakers = sd["emb_spk.weight"].shape[0] if "emb_spk.weight" in sd else 0
    if n_speakers == 0:
        raise ValueError("权重中没有 emb_spk（n_speakers=0）— 超出 4.0-v2 官方结构")

    acoustic_dim = sd["mel_decoder.proj.weight"].shape[0]
    n_harmonic = sd["dec_harm.postnet.weight"].shape[0] - 1
    n_fft = (sd["dec_noise.istft_amplitude.weight"].shape[0] - 1) * 2
    kernel_size = sd["dec_harm.net.conv_layers.0.weight"].shape[2]
    has_posterior = "posterior_encoder.pre.weight" in sd
    has_f0_decoder = "f0_decoder.prenet.weight" in sd

    n_flows = 0
    while f"flow.flows.{n_flows * 2}.pre.weight" in sd:
        n_flows += 1
    if n_flows != 4:
        raise ValueError(f"flow 耦合层数 {n_flows} != 4，超出 4.0-v2 官方结构")

    num_ups = _count_layers(sd, "dec.ups.{}.weight_v")
    upsample_kernel_sizes = [sd[f"dec.ups.{i}.weight_v"].shape[2] for i in range(num_ups)]
    upsample_initial_channel = sd["dec.conv_pre.weight"].shape[0]
    downs_kernel_sizes = [sd[f"dec.downs.{i}.weight_v"].shape[2] for i in range(num_ups)]

    resblock = "1" if "dec.resblocks.0.convs1.0.weight_v" in sd else "2"
    conv_key = "convs1" if resblock == "1" else "convs"
    total_resblocks = _count_layers(sd, "dec.resblocks.{}." + conv_key + ".0.weight_v")
    num_kernels = total_resblocks // num_ups
    resblock_kernel_sizes = [
        sd[f"dec.resblocks.{j}.{conv_key}.0.weight_v"].shape[2] for j in range(num_kernels)]

    # --- config-supplied (weights cannot express these; template defaults) ---
    n_heads = model_cfg.get("prior_n_heads", prior_hidden // emb_rel_k.shape[2])
    upsample_rates = model_cfg.get("upsample_rates")
    if upsample_rates is None:
        # template geometry [8,8,4,2]; kernel = 2 * stride holds for it
        upsample_rates = [k // 2 for k in upsample_kernel_sizes]
        if config is not None:
            raise ValueError("config.json 的 model 段缺少 upsample_rates")
        print(f"WARNING: 上采样步幅按 kernel//2 推断: {upsample_rates}", file=sys.stderr)
    if model_cfg.get("upsample_kernel_sizes") not in (None, upsample_kernel_sizes):
        print(f"WARNING: config 的 upsample_kernel_sizes "
              f"{model_cfg.get('upsample_kernel_sizes')} 与权重 "
              f"{upsample_kernel_sizes} 不符，以权重为准", file=sys.stderr)
    # the downs mirror the reversed rates with k == 2u (checked so the
    # `[..., : in_size // up_rate]` truncation semantics hold)
    for i, k in enumerate(downs_kernel_sizes):
        u = upsample_rates[len(upsample_rates) - 1 - i]
        if k != 2 * u:
            raise ValueError(
                f"dec.downs.{i} 的 kernel {k} != 2*stride {u} — 超出 4.0-v2 官方结构")
    resblock_dilation_sizes = model_cfg.get(
        "resblock_dilation_sizes",
        [[1, 3, 5]] * num_kernels if resblock == "1" else [[1, 3]] * num_kernels)
    sample_rate = data_cfg.get("sampling_rate", 44100)
    win_size = data_cfg.get("win_size", n_fft)
    if config is None:
        print("WARNING: sampling_rate 按 44100 假设", file=sys.stderr)
    if data_cfg.get("n_fft") not in (None, n_fft):
        raise ValueError(
            f"config 的 data.n_fft={data_cfg.get('n_fft')} 与权重的幅度谱通道数推得的 "
            f"{n_fft} 不符 — 配置文件与模型不匹配")

    hop_size = math.prod(upsample_rates)
    if config is not None and data_cfg.get("hop_length") not in (None, hop_size):
        raise ValueError(
            f"config 的 data.hop_length={data_cfg.get('hop_length')} 与解码器上采样积 "
            f"{hop_size} 不符 — 配置文件与模型不匹配")

    # cross-checks: config lies lose to weights
    for name, cfg_v, w_v in (("c_dim", data_cfg.get("c_dim"), c_dim),
                             ("acoustic_dim", data_cfg.get("acoustic_dim"), acoustic_dim),
                             ("n_speakers", data_cfg.get("n_speakers"), n_speakers),
                             ("hidden_channels", model_cfg.get("hidden_channels"), hidden_channels),
                             ("spk_channels", model_cfg.get("spk_channels"), spk_channels),
                             ("n_harmonic", model_cfg.get("n_harmonic"), n_harmonic)):
        if cfg_v is None:
            continue
        if cfg_v != w_v:
            print(f"WARNING: config 的 {name}={cfg_v} 与权重 {w_v} 不符，以权重为准",
                  file=sys.stderr)

    # 4.0-v2 uses the 4.0 ContentVec space (layer 9 + final_proj); the badge is
    # the ecosystem-distinct "4.0-v2" (完全不通用 with 4.0/4.1 checkpoints)
    if c_dim != 256:
        raise ValueError(
            f"4.0-v2 的内容特征应为 vec256l9 (256 维)，权重 c_dim={c_dim} — 超出官方结构")

    if fft_n_layers != prior_n_layers:
        raise ValueError(
            f"decoder 的 FFT 层数 {fft_n_layers} != text_encoder 层数 "
            f"{prior_n_layers} — 超出 4.0-v2 官方结构（二者共用 prior_n_layers）")

    model = SynthesizerTrnV2(
        c_dim,
        prior_hidden,
        prior_filter,
        n_heads,
        prior_n_layers,
        prior_kernel,
        float(model_cfg.get("prior_p_dropout", 0.1)),
        hidden_channels,
        spk_channels,
        kernel_size,
        float(model_cfg.get("p_dropout", 0.1)),
        acoustic_dim,
        n_harmonic,
        n_fft,
        win_size,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers,
        sample_rate,
        hop_size,
        window_size=window_size,
        has_posterior=has_posterior,
    )

    state_dict_f32 = {
        k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in sd.items()
    }
    model.load_state_dict(state_dict_f32, strict=True)

    # Fuses weight_g/weight_v -> weight; bit-identical to the pre-forward hook.
    model.remove_weight_norm()
    model.eval()

    meta = {
        "version": "4.0-v2",
        "features_dim": int(c_dim),
        "speech_encoder": "vec256l9",
        "sample_rate": int(sample_rate),
        "hop_size": int(hop_size),
        "vol_embedding": False,
        # v2's infer chain uses utils.repeat_expand_2d — the 'left' variant
        "unit_interpolate_mode": "left",
        "n_speakers": int(n_speakers),
        "speakers": {str(k): int(v) for k, v in spk.items()},
        # z_p noise rides on hidden_channels (192), same shape family as 4.x
        "inter_channels": int(hidden_channels),
        "n_fft": int(n_fft),
        # traced attention rel-pos branch is valid for T >= window_size + 2
        "min_frames": int(window_size) + 2,
        "has_f0_decoder": bool(has_f0_decoder),
    }
    return model, meta


def load_sovits_v2_config(pth_path, explicit_config=None):
    """v2 config discovery = the sovits convention (config.json next to the
    .pth > single other json > None); the section-shape check in
    load_sovits_config already matches v2 configs (model/data dicts)."""
    return load_sovits_config(pth_path, explicit_config)
