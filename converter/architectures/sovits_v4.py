"""SoVITS 4.0 / 4.1 synthesizer (SynthesizerTrn) for ONNX export.

FAITHFUL VERBATIM PORT of the original so-vits-svc 4.1-Stable code:
  D:\\MyDev\\so-vits-svc\\so-vits-svc\\{models.py, modules\\{modules,attentions,
  commons}.py, vdecoder\\hifigan\\models.py, utils.py, onnxexport\\
  model_onnx_speaker_mix.py}
Every module below carries a "ported from" note. Only device/dtype/export
ergonomics were adapted (each adaptation is commented at the spot); the tensor
math is the original's, line for line. Do NOT "simplify" any of it — the
2026-07 audit found the previous hand reconstruction of this file had the same
bug classes as the old rvc_v2.py (pre-norm vs post-norm f0_decoder, wrong flow
dilation_rate 4 vs 1, z_p missing the exp(logs_p)*noise term, missing vol
embedding, wrong NSF paddings (k-u)//2 vs (k-u+1)//2, wrong f0_to_coarse).

Shared-lineage modules are REUSED from architectures/rvc_v2.py ONLY where the
original so-vits code is math-identical to the original RVC code (diffed
2026-07-04): LayerNorm, WN, MultiHeadAttention, FFN, attentions.Encoder
(window_size is config: so-vits uses 4, RVC 10 — passed explicitly), ResBlock1/
ResBlock2 (vdecoder's variants == RVC modules' with x_mask=None; the `h` first
ctor arg of vdecoder's is unused), SineGen + SourceModuleHnNSF (so-vits'
vdecoder ONNX branch is line-identical to RVC models.py's; so-vits' extra
unused noise output is dead code). SineGen carries rvc_v2.py's ONE gated
deviation: the ONNX-stable phase reformulation (frac of fp64 frame cumsum +
per-sample ramp — sin-invariant identity; see rvc_v2.SineGen).

Where so-vits DIFFERS from RVC, the so-vits variant is ported here: Flip
(returns bare x in reverse), ResidualCouplingLayer/-Block (wn_sharing_parameter
/ share_parameter support), TextEncoder (f0 embedding, z-as-input export
variant), F0Decoder + attentions.FFT (strict=True load; exported as the
STANDALONE <stem>.f0.onnx companion graph via F0PredictorWrapper — the main
graph stays predict_f0-free, matching the official onnx export),
PosteriorEncoder (enc_q, training-only, conditional: compressed checkpoints
strip it), vdecoder Generator (weight-normed conv_pre/conv_post, (k-u+1)//2
and (stride_f0+1)//2 paddings, harmonic_num=8, unconditional cond).

Export contract (matches the official onnxexport/model_onnx_speaker_mix.py
formulation, minus mel2ph — the c-to-f0-frames expansion stays Rust-side —
and minus speaker-mix):
  inputs : c[1,T,ssl_dim] f32 (ALREADY expanded to the f0 frame count),
           f0[1,T] f32 (Hz; f0_to_coarse runs IN-graph), uv[1,T] f32,
           noise[1,inter_channels,T] f32, sid[1] i64,
           vol[1,T] f32 — present IFF the model has vol_embedding
  output : audio[1,1,T*hop_size]
  z_p = (m_p + noise * exp(logs_p)) * x_mask — the CALLER pre-multiplies its
  N(0,1) noise by noice_scale (original inference default 0.4).
SineGen noise stays in-graph (RandomNormalLike) per original semantics; the
`deterministic` flag zeroes SineGen noise + rand_ini for numerical gate builds.

Unsupported (clean Chinese ValueError, 排期项): use_depthwise_conv,
use_transformer_flow, speech_encoder outside {vec768l12, vec256l9}, vocoders
other than nsf-hifigan.

Verified against the ORIGINAL repo code by converter/verify/voice/gate1_sovits.py.
"""

import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from .rvc_v2 import (
    LayerNorm,
    WN,
    MultiHeadAttention,
    FFN,
    Encoder as AttentionEncoder,   # attentions.Encoder — window_size passed explicitly (so-vits default 4)
    ResBlock1,
    ResBlock2,
    SourceModuleHnNSF,              # identical for the used output (sine_merge); its
                                    # SineGen == so-vits vdecoder ONNX branch (diffed),
                                    # incl. the gated stable-phase deviation

    sequence_mask,
    init_weights,
)

SUPPORTED_SPEECH_ENCODERS = ("vec768l12", "vec256l9")


# ---------------------------------------------------------------------------
# utils.py — verbatim f0_to_coarse (torch ops; runs IN-graph)
# ---------------------------------------------------------------------------

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * math.log(1 + f0_min / 700)   # np.log == math.log (fp64)
f0_mel_max = 1127 * math.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    # ported from utils.py f0_to_coarse — round (not truncate), unvoiced -> 1,
    # clamp [1, 255] via the original's arithmetic masking (exports cleanly).
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse


def normalize_f0(f0, x_mask, uv, random_scale=True):
    """ported from utils.py normalize_f0 — voiced-mean removal of the lf0
    contour ([B,1,T]); the F0PredictorWrapper calls it with random_scale=False
    (models.py infer()'s predict_f0 branch)."""
    # calculate means based on x_mask
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    # DEVIATION (export ergonomics, numerically identical): the original does
    # an in-place boolean-mask assignment `uv_sum[uv_sum == 0] = 9999`, which
    # traces into an index_put graph; rewritten as torch.where. Identical for
    # every input incl. the all-unvoiced guard (uv all 0 -> uv_sum 9999 ->
    # means = 0/9999 = 0). Gated by verify/voice/gate_autof0.py (exact-0 tier).
    uv_sum = torch.where(uv_sum == 0, torch.full_like(uv_sum, 9999.0), uv_sum)
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    # normalize f0 based on means and factor
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    # The original's `if torch.isnan(f0_norm).any(): exit(0)` is dropped:
    # data-dependent control flow cannot trace, and with the 9999 guard above
    # the division can never produce NaN from finite inputs (dead branch).
    return f0_norm * x_mask


# ---------------------------------------------------------------------------
# modules/commons.py — verbatim helper (needed by FFT)
# ---------------------------------------------------------------------------

def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    # verbatim from modules/commons.py; @torch.jit.script dropped (export
    # ergonomics only — numerics identical).
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


# ---------------------------------------------------------------------------
# modules/modules.py — so-vits variants (differ from RVC's: bare-x reverse
# returns, wn_sharing_parameter)
# ---------------------------------------------------------------------------

class Flip(nn.Module):
    """ported from modules/modules.py Flip — reverse returns bare x."""

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ResidualCouplingLayer(nn.Module):
    """ported from modules/modules.py ResidualCouplingLayer — mean_only affine
    coupling with optional shared WN (wn_sharing_parameter)."""

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, p_dropout=0, gin_channels=0, mean_only=False,
                 wn_sharing_parameter=None):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                      p_dropout=p_dropout, gin_channels=gin_channels) \
            if wn_sharing_parameter is None else wn_sharing_parameter
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


# ---------------------------------------------------------------------------
# models.py — verbatim
# ---------------------------------------------------------------------------

class ResidualCouplingBlock(nn.Module):
    """ported from models.py ResidualCouplingBlock — a Flip follows EVERY
    coupling layer (8 modules for n_flows=4), share_parameter shares ONE WN
    across all couplings (flow.wn.* in the state dict)."""

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, n_flows=4, gin_channels=0, share_parameter=False):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                     p_dropout=0, gin_channels=gin_channels) if share_parameter else None

        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(channels, hidden_channels, kernel_size,
                                      dilation_rate, n_layers,
                                      gin_channels=gin_channels, mean_only=True,
                                      wn_sharing_parameter=self.wn))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        # ours (export ergonomics): fuse weight_g/weight_v — bit-identical to
        # the pre-forward hook. The shared WN must be removed exactly once.
        if self.wn is not None:
            self.wn.remove_weight_norm()
        else:
            for i in range(self.n_flows):
                self.flows[i * 2].enc.remove_weight_norm()


class TextEncoder(nn.Module):
    """ported from models.py TextEncoder (parameter namespace enc_p.*: f0_emb,
    enc_, proj). forward is the onnxexport/model_onnx_speaker_mix.py variant:
    z is an INPUT (z_p = (m + z*exp(logs)) * x_mask) — equal to models.py
    infer()'s randn_like(m)*exp(logs)*noice_scale with z = noise*noice_scale
    supplied by the caller."""

    def __init__(self, out_channels, hidden_channels, kernel_size, n_layers,
                 gin_channels=0, filter_channels=None, n_heads=None,
                 p_dropout=None, window_size=4):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)

        # so-vits attentions.Encoder defaults window_size=4 (RVC's defaults 10)
        # — passed explicitly because the class is shared with rvc_v2.py.
        self.enc_ = AttentionEncoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size)

    def forward(self, x, x_mask, f0=None, z=None):
        x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + z * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask


class FFT(nn.Module):
    """ported from modules/attentions.py FFT — post-norm transformer with
    causal self-attention + causal FFN. Used by F0Decoder (exported inside
    the standalone <stem>.f0.onnx via F0PredictorWrapper; the main graph
    never calls it). No relative-position embeddings (window_size=None), so
    no T >= window_size + 2 trace constraint here."""

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1,
                 kernel_size=1, p_dropout=0., proximal_bias=False,
                 proximal_init=True, isflow=False, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        if isflow:
            cond_layer = torch.nn.Conv1d(kwargs["gin_channels"], 2 * hidden_channels * n_layers, 1)
            self.cond_pre = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            self.cond_layer = weight_norm(cond_layer, name='weight')
            self.gin_channels = kwargs["gin_channels"]
        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads,
                                   p_dropout=p_dropout, proximal_bias=proximal_bias,
                                   proximal_init=proximal_init))
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels,
                    kernel_size, p_dropout=p_dropout, causal=True))
            self.norm_layers_1.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        """
        x: decoder input
        h: encoder output
        """
        if g is not None:
            g = self.cond_layer(g)

        self_attn_mask = subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        x = x * x_mask
        for i in range(self.n_layers):
            if g is not None:
                x = self.cond_pre(x)
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                x = fused_add_tanh_sigmoid_multiply(
                    x,
                    g_l,
                    torch.IntTensor([self.hidden_channels]))
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
        x = x * x_mask
        return x


class F0Decoder(nn.Module):
    """ported from models.py F0Decoder (parameter namespace f0_decoder.*).
    Loaded for strict=True; exported as part of the standalone auto-f0
    companion graph (F0PredictorWrapper -> <stem>.f0.onnx). The MAIN export
    graph deliberately does not call it (official onnx export parity)."""

    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads,
                 n_layers, kernel_size, p_dropout, spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if (spk_emb is not None):
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x


class PosteriorEncoder(nn.Module):
    """ported from models.py Encoder (parameter namespace enc_q.*) —
    training-only posterior encoder, loaded for strict=True but never called.
    Compressed checkpoints strip it entirely; build_from_checkpoint only
    instantiates it when the weights are present."""

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size,
                 dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


# ---------------------------------------------------------------------------
# vdecoder/hifigan/models.py — Generator (NSF HiFi-GAN), ONNX-mode semantics
# ---------------------------------------------------------------------------

class Generator(torch.nn.Module):
    """ported from vdecoder/hifigan/models.py Generator. Differences from
    RVC's GeneratorNSF that this port preserves: conv_pre AND conv_post are
    weight-normed (with bias), ups padding is (k - u + 1) // 2, noise_convs
    padding is (stride_f0 + 1) // 2, harmonic_num=8, cond is unconditional.
    forward is the OnnxExport() path only: f0 arrives at FRAME level [B, T]
    and SineGen upsamples in-graph (the non-onnx torch path upsamples f0 to
    sample level first — mathematically identical modulo whole sine cycles,
    gated in gate1_sovits.py)."""

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h

        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        # f0_upsamp exists in the original for the non-onnx path; not used here.
        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(h["upsample_rates"]))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h["sampling_rate"],
            harmonic_num=8)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h["inter_channels"], h["upsample_initial_channel"], 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            c_cur = h["upsample_initial_channel"] // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(h["upsample_initial_channel"] // (2 ** i), h["upsample_initial_channel"] // (2 ** (i + 1)),
                                k, u, padding=(k - u + 1) // 2)))
            if i + 1 < len(h["upsample_rates"]):  #
                stride_f0 = math.prod(h["upsample_rates"][i + 1:])
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0 + 1) // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h["upsample_initial_channel"] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                # rvc_v2.ResBlock1/2 == vdecoder's minus the unused `h` arg
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.cond = nn.Conv1d(h['gin_channels'], h['upsample_initial_channel'], 1)
        self.upp = math.prod(h["upsample_rates"])

    def forward(self, x, f0, g=None):
        # ONNX-mode path (original: self.onnx == True): f0 stays frame-level.
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)  # LRELU_SLOPE
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)  # default slope 0.01 — the original uses the default here
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        # verbatim (incl. conv_pre/conv_post, unlike RVC's)
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


# ---------------------------------------------------------------------------
# top-level synthesizer — models.py SynthesizerTrn, export forward per
# onnxexport/model_onnx_speaker_mix.py (minus mel2ph / speaker-mix / predict_f0)
# ---------------------------------------------------------------------------

class SynthesizerTrn(nn.Module):
    """Parameter names match the checkpoint state_dict exactly."""

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gin_channels,
                 ssl_dim,
                 n_speakers,
                 sampling_rate=44100,
                 vol_embedding=False,
                 vocoder_name="nsf-hifigan",
                 use_depthwise_conv=False,
                 use_automatic_f0_prediction=True,
                 flow_share_parameter=False,
                 n_flow_layer=4,
                 n_layers_trans_flow=3,
                 use_transformer_flow=False,
                 window_size=4,          # ours: from weights (emb_rel_k); so-vits hardcodes 4
                 has_enc_q=True,         # ours: compressed checkpoints strip enc_q
                 **kwargs):
        super().__init__()
        # Structural options are validated in build_from_checkpoint; assert
        # here too so a direct constructor call cannot silently build the
        # wrong graph.
        assert not use_depthwise_conv and not use_transformer_flow
        assert vocoder_name == "nsf-hifigan"

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        if vol_embedding:
            self.emb_vol = nn.Linear(1, hidden_channels)

        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            window_size=window_size,
        )
        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
        }
        self.dec = Generator(h=hps)

        if has_enc_q:
            self.enc_q = PosteriorEncoder(spec_channels, inter_channels,
                                          hidden_channels, 5, 1, 16,
                                          gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1,
                                          n_flow_layer, gin_channels=gin_channels,
                                          share_parameter=flow_share_parameter)
        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(
                1,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                spk_channels=gin_channels
            )
        self.emb_uv = nn.Embedding(2, hidden_channels)

    def forward(self, c, f0, uv, noise, sid, vol=None):
        """
        c: [1, T, ssl_dim] f32 — speech-encoder features ALREADY expanded to
           the f0 frame count (the mel2ph gather stays Rust-side)
        f0: [1, T] f32 — continuous F0 in Hz (0 = unvoiced); f0_to_coarse
           runs in-graph
        uv: [1, T] f32 — voiced flag (1 voiced / 0 unvoiced)
        noise: [1, inter_channels, T] f32 — caller noise ALREADY scaled by
           noice_scale (original inference default 0.4); zeros = deterministic
        sid: [1] i64 — speaker id
        vol: [1, T] f32 — frame volume (Volume_Extractor), ONLY for
           vol_embedding models (the input is omitted from the export otherwise)
        returns audio [1, 1, T * prod(upsample_rates)]
        """
        # models.py infer(): g.dim() == 1 -> unsqueeze(0); emb; transpose
        g = self.emb_g(sid.unsqueeze(0)).transpose(1, 2)  # [1, gin, 1]

        x_mask = torch.unsqueeze(torch.ones_like(f0), 1).to(c.dtype)
        # vol proj
        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0

        x = self.pre(c.transpose(1, 2)) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), z=noise)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        if hasattr(self, "enc_q"):
            self.enc_q.enc.remove_weight_norm()


def set_deterministic(model, deterministic=True):
    """Zero SineGen's in-graph randomness (rand_ini + additive noise) for
    reproducible gate builds. Shipping exports keep it False (original
    semantics: RandomNormalLike stays in the graph)."""
    model.dec.m_source.l_sin_gen.deterministic = deterministic


class F0PredictorWrapper(nn.Module):
    """Standalone auto-f0 predictor export graph (<stem>.f0.onnx).

    Wraps the predict_f0 branch of models.py infer() (:520, :523-527) around
    the ALREADY-LOADED synthesizer's own modules: pre / emb_uv / emb_vol /
    emb_g / f0_decoder are the SAME nn.Module objects (python-level weight
    sharing — same nn.Parameter tensors, no copies), so the wrapper's `x` is
    computed by the exact op sequence the main export graph uses.

    inputs : c[1,T,ssl_dim] f32 (expanded, same layout as the main graph),
             f0[1,T] f32 (source F0 in Hz, AFTER transpose/key shift),
             uv[1,T] f32, sid[1] i64,
             vol[1,T] f32 — present IFF the model has vol_embedding
    output : f0_pred[1,T] f32 (Hz) — feed as the main graph's `f0` input
             (uv stays the SOURCE uv: the original never recomputes it).
    x_mask = ones (matches the main export; the original's sequence_mask is
    all-ones for single-segment inference too). The graph contains NO
    randomness — a shipping export is already deterministic.
    """

    def __init__(self, synth):
        super().__init__()
        if not getattr(synth, "use_automatic_f0_prediction", False):
            raise ValueError(
                "该模型没有 f0_decoder 权重，无法导出自动音高预测器")
        self.vol_embedding = synth.vol_embedding
        # nn.Module attribute assignment registers the SAME module objects —
        # shared parameters with the synthesizer, not copies.
        self.pre = synth.pre
        self.emb_uv = synth.emb_uv
        self.emb_g = synth.emb_g
        if synth.vol_embedding:
            self.emb_vol = synth.emb_vol
        self.f0_decoder = synth.f0_decoder

    def forward(self, c, f0, uv, sid, vol=None):
        # models.py infer() steps :511-520, identical to SynthesizerTrn.forward
        # above (same tensors, same op order — bit-compatible `x`).
        g = self.emb_g(sid.unsqueeze(0)).transpose(1, 2)  # [1, gin, 1]
        x_mask = torch.unsqueeze(torch.ones_like(f0), 1).to(c.dtype)
        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) \
            if vol is not None and self.vol_embedding else 0
        x = self.pre(c.transpose(1, 2)) * x_mask \
            + self.emb_uv(uv.long()).transpose(1, 2) + vol

        # models.py infer() :523-527 (predict_f0 branch), verbatim
        lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
        norm_lf0 = normalize_f0(lf0, x_mask, uv, random_scale=False)
        pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
        f0_pred = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        return f0_pred


# ---------------------------------------------------------------------------
# config loading / validation
# ---------------------------------------------------------------------------

def load_sovits_config(pth_path, explicit_config=None):
    """Locate + parse the config.json adjacent to the .pth (utf-8 — Chinese
    paths and speaker names must survive). Returns (dict | None, Path | None).
    Order: explicit --config > <dir>/config.json > the single other *.json in
    the directory (some models ship a renamed config, e.g. 主配置文件.json)."""
    pth_path = Path(pth_path)
    if explicit_config is not None:
        path = Path(explicit_config)
        if not path.exists():
            raise ValueError(f"--config 指定的配置文件不存在: {path}")
    else:
        path = pth_path.parent / "config.json"
        if not path.exists():
            candidates = [p for p in sorted(pth_path.parent.glob("*.json"))]
            if len(candidates) == 1:
                path = candidates[0]
                print(f"NOTE: 未找到 config.json，使用同目录唯一的 json: {path.name}",
                      file=sys.stderr)
            else:
                return None, None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg.get("model"), dict) or not isinstance(cfg.get("data"), dict):
        raise ValueError(f"配置文件缺少 model/data 段，不是 so-vits-svc 的 config.json: {path}")
    return cfg, path


def validate_config(cfg):
    """Refuse the 排期项 options with clean Chinese errors BEFORE building."""
    model = cfg.get("model", {})
    if model.get("use_depthwise_conv"):
        raise ValueError("暂不支持 use_depthwise_conv=true 的 SoVITS 模型（排期项）")
    if model.get("use_transformer_flow"):
        raise ValueError("暂不支持 use_transformer_flow=true 的 SoVITS 模型（排期项）")
    se = model.get("speech_encoder")
    if se is not None and se not in SUPPORTED_SPEECH_ENCODERS:
        raise ValueError(
            f"暂不支持 speech_encoder={se} 的 SoVITS 模型"
            f"（仅支持 vec768l12 / vec256l9，排期项）")
    vn = model.get("vocoder_name", "nsf-hifigan")
    if vn != "nsf-hifigan":
        raise ValueError(
            f"暂不支持 vocoder_name={vn} 的 SoVITS 模型（仅支持 nsf-hifigan，排期项）")


def _count_layers(sd, fmt):
    n = 0
    while fmt.format(n) in sd:
        n += 1
    return n


def build_from_checkpoint(checkpoint, config=None):
    """Build the SoVITS 4.0/4.1 synthesizer from a loaded .pth checkpoint dict
    (+ parsed config.json dict, or None) and strict=True-load the weights.
    Returns (model, meta) — model ready for export (weight norm removed, eval
    mode), meta = everything the sidecar json needs.

    Weights are the source of truth for every tensor-shaped hyperparameter
    (ssl_dim, gin, n_speakers, vol_embedding, enc_q presence, layer counts, …);
    the config supplies what weights cannot express (sampling_rate, upsample
    STRIDES, resblock dilations, n_heads, speaker map, unit_interpolate_mode).
    Missing config -> infer what's inferable, warn, version-detect by ssl_dim."""
    if "model" not in checkpoint:
        raise ValueError("checkpoint 中没有 'model' 键，不是 so-vits-svc 的 G_*.pth")
    sd = checkpoint["model"]

    if config is not None:
        validate_config(config)
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        spk = dict(config.get("spk") or {})
    else:
        print("WARNING: 未找到 config.json — 从权重推断超参数（sampling_rate/上采样"
              "步幅/膨胀率按默认值假设），speaker 名不可恢复。", file=sys.stderr)
        model_cfg, data_cfg, spk = {}, {}, {}

    # --- weight-derived structure (truth) ---
    ssl_dim = sd["pre.weight"].shape[1]
    hidden_channels = sd["pre.weight"].shape[0]
    inter_channels = sd["enc_p.proj.weight"].shape[0] // 2
    gin_channels = sd["emb_g.weight"].shape[1]
    n_speakers = sd["emb_g.weight"].shape[0]
    filter_channels = sd["enc_p.enc_.ffn_layers.0.conv_1.weight"].shape[0]
    kernel_size = sd["enc_p.enc_.ffn_layers.0.conv_1.weight"].shape[2]
    emb_rel_k = sd["enc_p.enc_.attn_layers.0.emb_rel_k"]
    window_size = (emb_rel_k.shape[1] - 1) // 2
    n_layers = _count_layers(sd, "enc_p.enc_.attn_layers.{}.conv_q.weight")
    vol_embedding = "emb_vol.weight" in sd
    has_enc_q = "enc_q.pre.weight" in sd
    has_f0_decoder = "f0_decoder.prenet.weight" in sd
    harmonic_num = sd["dec.m_source.l_linear.weight"].shape[1] - 1
    if harmonic_num != 8:
        raise ValueError(f"dec.m_source 谐波数 {harmonic_num} != 8，不是 nsf-hifigan 的权重")

    flow_share_parameter = "flow.wn.in_layers.0.weight_v" in sd
    flow_wn_prefix = "flow.wn" if flow_share_parameter else "flow.flows.0.enc"
    n_flow_layer = _count_layers(sd, flow_wn_prefix + ".in_layers.{}.weight_v")
    # coupling layers live at even indices 0,2,4,… (Flips hold no params)
    n_flows = 0
    while f"flow.flows.{n_flows * 2}.pre.weight" in sd:
        n_flows += 1
    if n_flows != 4:
        raise ValueError(f"flow 耦合层数 {n_flows} != 4，超出原版 SynthesizerTrn 的结构")

    num_ups = _count_layers(sd, "dec.ups.{}.weight_v")
    upsample_kernel_sizes = [sd[f"dec.ups.{i}.weight_v"].shape[2] for i in range(num_ups)]
    upsample_initial_channel = sd["dec.conv_pre.weight_v"].shape[0]

    resblock = "1" if "dec.resblocks.0.convs1.0.weight_v" in sd else "2"
    conv_key = "convs1" if resblock == "1" else "convs"
    total_resblocks = _count_layers(sd, "dec.resblocks.{}." + conv_key + ".0.weight_v")
    num_kernels = total_resblocks // num_ups
    resblock_kernel_sizes = [
        sd[f"dec.resblocks.{j}.{conv_key}.0.weight_v"].shape[2] for j in range(num_kernels)]

    spec_channels = (sd["enc_q.pre.weight"].shape[1] if has_enc_q
                     else data_cfg.get("filter_length", 2048) // 2 + 1)

    # --- config-supplied (weights cannot express these) ---
    n_heads = model_cfg.get("n_heads", hidden_channels // emb_rel_k.shape[2])
    upsample_rates = model_cfg.get("upsample_rates")
    if upsample_rates is None:
        # standard HiFi-GAN geometry: kernel = 2 * stride
        upsample_rates = [k // 2 for k in upsample_kernel_sizes]
        if config is not None:
            raise ValueError("config.json 的 model 段缺少 upsample_rates")
        print(f"WARNING: 上采样步幅按 kernel//2 推断: {upsample_rates}", file=sys.stderr)
    if model_cfg.get("upsample_kernel_sizes") not in (None, upsample_kernel_sizes):
        print(f"WARNING: config 的 upsample_kernel_sizes {model_cfg.get('upsample_kernel_sizes')} "
              f"与权重 {upsample_kernel_sizes} 不符，以权重为准", file=sys.stderr)
    resblock_dilation_sizes = model_cfg.get(
        "resblock_dilation_sizes",
        [[1, 3, 5]] * num_kernels if resblock == "1" else [[1, 3]] * num_kernels)
    # the DEC's SineGen rate comes from model kwargs (models.py ctor default
    # 44100 — train.py/onnx_export.py pass **hps.model, which usually lacks
    # the key); the OUTPUT rate is data.sampling_rate.
    dec_sampling_rate = model_cfg.get("sampling_rate", 44100)
    sample_rate = data_cfg.get("sampling_rate", 44100)
    if config is None:
        print("WARNING: sampling_rate 按 44100 假设", file=sys.stderr)

    hop_size = math.prod(upsample_rates)
    if config is not None and data_cfg.get("hop_length") not in (None, hop_size):
        raise ValueError(
            f"config 的 data.hop_length={data_cfg.get('hop_length')} 与解码器上采样积 "
            f"{hop_size} 不符 — 配置文件与模型不匹配")

    # cross-checks: config lies lose to weights (mislabeled configs must not
    # silently change the graph)
    for name, cfg_v, w_v in (("ssl_dim", model_cfg.get("ssl_dim"), ssl_dim),
                             ("gin_channels", model_cfg.get("gin_channels"), gin_channels),
                             ("n_speakers", model_cfg.get("n_speakers"), n_speakers),
                             ("vol_embedding", model_cfg.get("vol_embedding"), vol_embedding)):
        if cfg_v is None:
            continue
        same = (bool(cfg_v) == bool(w_v)) if name == "vol_embedding" else (cfg_v == w_v)
        if not same:
            print(f"WARNING: config 的 {name}={cfg_v} 与权重 {w_v} 不符，以权重为准",
                  file=sys.stderr)

    speech_encoder = model_cfg.get("speech_encoder")
    if speech_encoder is None:
        speech_encoder = "vec768l12" if ssl_dim == 768 else "vec256l9"
    expected_dim = 768 if speech_encoder == "vec768l12" else 256
    if ssl_dim != expected_dim:
        raise ValueError(
            f"speech_encoder={speech_encoder} 期望 {expected_dim} 维特征，"
            f"但权重的 ssl_dim={ssl_dim} — 配置文件与模型不匹配")

    # infer_tool.py: unit_interpolate_mode defaults to 'left' when absent
    unit_interpolate_mode = data_cfg.get("unit_interpolate_mode", "left")

    is_41 = (speech_encoder == "vec768l12" or vol_embedding
             or (config is not None and "speech_encoder" in model_cfg))
    version = "4.1" if is_41 else "4.0"

    segment_size = (config.get("train", {}).get("segment_size", 10240) // hop_size
                    if config is not None else 10240 // hop_size)

    model = SynthesizerTrn(
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        float(model_cfg.get("p_dropout", 0.1)),
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        ssl_dim,
        n_speakers,
        sampling_rate=dec_sampling_rate,
        vol_embedding=vol_embedding,
        use_automatic_f0_prediction=has_f0_decoder,
        flow_share_parameter=flow_share_parameter,
        n_flow_layer=n_flow_layer,
        window_size=window_size,
        has_enc_q=has_enc_q,
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
        "version": version,
        "features_dim": int(ssl_dim),
        "speech_encoder": speech_encoder,
        "sample_rate": int(sample_rate),
        "hop_size": int(hop_size),
        "vol_embedding": bool(vol_embedding),
        "unit_interpolate_mode": unit_interpolate_mode,
        "n_speakers": int(n_speakers),
        "speakers": {str(k): int(v) for k, v in spk.items()},
        "inter_channels": int(inter_channels),
        # traced attention rel-pos branch is valid for T >= window_size + 2
        "min_frames": int(window_size) + 2,
        # weights are the truth (4.0-era configs lack the
        # use_automatic_f0_prediction key entirely — never read the config):
        # drives the <stem>.f0.onnx companion export + sidecar "auto_f0".
        "has_f0_decoder": bool(has_f0_decoder),
    }
    return model, meta
