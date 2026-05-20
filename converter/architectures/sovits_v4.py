"""
SoVITS 4.0 model architecture for ONNX export.
Reconstructed from checkpoint parameter inspection (akiko_320000.pth).

Inference path: ContentVec → pre → enc_p → flow.reverse → dec(NSF) → audio
Training-only: enc_q (posterior encoder) — loaded for strict=True but not called.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .rvc_v2 import (
    LayerNorm, FFN, Encoder, WN,
    ResidualCouplingLayer, Flip, ResidualCouplingBlock,
    ResBlock, remove_all_weight_norm,
)


class SimpleAttention(nn.Module):
    """Multi-head attention WITHOUT relative position embeddings.
    Used by f0_decoder which has no emb_rel_k/v parameters.
    """
    def __init__(self, channels, n_heads):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)

    def forward(self, x, x_mask):
        B, C, T = x.shape
        q = self.conv_q(x).view(B, self.n_heads, self.head_dim, T)
        k = self.conv_k(x).view(B, self.n_heads, self.head_dim, T)
        v = self.conv_v(x).view(B, self.n_heads, self.head_dim, T)

        scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(x_mask.unsqueeze(1).unsqueeze(2) == 0, -1e4)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1).contiguous().view(B, C, T)
        out = self.conv_o(out)
        return out


class F0DecoderTransformer(nn.Module):
    """Transformer encoder without relative position — used by f0_decoder.
    Parameter namespace: decoder.self_attn_layers, decoder.norm_layers_0, decoder.ffn_layers, decoder.norm_layers_1
    """
    def __init__(self, channels, filter_channels, n_heads, n_layers, kernel_size):
        super().__init__()
        self.n_layers = n_layers

        self.self_attn_layers = nn.ModuleList([
            SimpleAttention(channels, n_heads) for _ in range(n_layers)
        ])
        self.norm_layers_0 = nn.ModuleList([
            LayerNorm(channels) for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            FFN(channels, filter_channels, kernel_size) for _ in range(n_layers)
        ])
        self.norm_layers_1 = nn.ModuleList([
            LayerNorm(channels) for _ in range(n_layers)
        ])

    def forward(self, x, x_mask):
        for i in range(self.n_layers):
            residual = x
            x = self.norm_layers_0[i](x)
            x = self.self_attn_layers[i](x, x_mask)
            x = residual + x

            residual = x
            x = self.norm_layers_1[i](x)
            x = self.ffn_layers[i](x, x_mask)
            x = residual + x

        return x * x_mask


class F0Decoder(nn.Module):
    """F0 refinement module.
    Parameter namespace: f0_decoder.*
    """
    def __init__(self, inter_channels, filter_channels, n_heads, n_layers,
                 kernel_size, gin_channels):
        super().__init__()
        self.prenet = nn.Conv1d(inter_channels, inter_channels, 3, padding=1)
        self.decoder = F0DecoderTransformer(
            inter_channels, filter_channels, n_heads, n_layers, kernel_size,
        )
        self.proj = nn.Conv1d(inter_channels, 1, 1)
        self.f0_prenet = nn.Conv1d(1, inter_channels, 3, padding=1)
        self.cond = nn.Conv1d(gin_channels, inter_channels, 1)

    def forward(self, x, x_mask, g):
        # x: [B, inter_channels, T] — encoder output
        # g: [B, gin_channels, 1] — speaker embedding
        x = self.prenet(x) * x_mask
        x = x + self.cond(g)
        x = self.decoder(x, x_mask)
        x = self.proj(x) * x_mask
        return x


class TextEncoderSoVITS(nn.Module):
    """SoVITS content/pitch encoder.
    Parameter namespace: enc_p.*
    Note: the encoder sub-module is named `enc_` (with underscore), not `encoder`.
    """
    def __init__(self, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, window_size=4):
        super().__init__()
        self.inter_channels = inter_channels

        self.f0_emb = nn.Embedding(256, inter_channels)
        self.enc_ = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, window_size)
        self.proj = nn.Conv1d(inter_channels, inter_channels * 2, 1)

    def forward(self, x, x_mask):
        # x: [B, inter_channels, T] — already has pre(c) + emb_uv + f0_emb added
        x = self.enc_(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(self.inter_channels, dim=1)
        return m, logs, x_mask


class SourceModuleHarmonic(nn.Module):
    """Neural Source Filter with harmonic overtones.
    Parameter namespace: dec.m_source.l_linear
    """
    def __init__(self, sample_rate, upsample_factor, harmonic_num=8):
        super().__init__()
        self.sample_rate = sample_rate
        self.upsample_factor = upsample_factor
        self.harmonic_num = harmonic_num
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def forward(self, f0, upsample_factor=None):
        factor = upsample_factor or self.upsample_factor
        f0_up = F.interpolate(f0.unsqueeze(1), scale_factor=float(factor), mode="nearest")

        voiced = (f0_up > 0).float()

        harmonics = []
        for k in range(self.harmonic_num + 1):
            omega = 2 * math.pi * f0_up * (k + 1) / self.sample_rate
            phase = torch.cumsum(omega, dim=-1)
            sine = torch.sin(phase) * voiced
            harmonics.append(sine)

        # [B, harmonic_num+1, T] → transpose → linear → [B, 1, T]
        src = torch.cat(harmonics, dim=1)  # [B, H+1, T]
        src = self.l_linear(src.transpose(1, 2)).transpose(1, 2)  # [B, 1, T]

        noise = torch.randn_like(src[:, :1, :])
        src = src + noise * (1 - voiced) * 0.003
        return src


class GeneratorNSFSoVITS(nn.Module):
    """HiFi-GAN generator with harmonic NSF for SoVITS 4.0.
    Parameter namespace: dec.*
    """
    def __init__(self, inter_channels, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                 upsample_kernel_sizes, gin_channels, sample_rate, harmonic_num=8):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        total_upsample = 1
        for r in upsample_rates:
            total_upsample *= r

        self.m_source = SourceModuleHarmonic(sample_rate, total_upsample, harmonic_num)

        self.conv_pre = weight_norm(nn.Conv1d(inter_channels, upsample_initial_channel, 7, 1, 3))

        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        ch = upsample_initial_channel
        stride_product = total_upsample
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2))
            )
            stride_product //= u
            if stride_product > 1:
                self.noise_convs.append(
                    nn.Conv1d(1, ch // 2, kernel_size=stride_product * 2, stride=stride_product,
                              padding=stride_product // 2 + stride_product % 2)
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, ch // 2, kernel_size=1))
            ch = ch // 2

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            block_ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(block_ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(block_ch, 1, 7, 1, 3))
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g):
        har_source = self.m_source(f0)

        x = self.conv_pre(x)
        x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            noise = self.noise_convs[i](har_source)
            if noise.shape[-1] != x.shape[-1]:
                noise = noise[:, :, :x.shape[-1]]
            x = x + noise

            xs = None
            for j in range(self.num_kernels):
                rb_idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[rb_idx](x)
                else:
                    xs += self.resblocks[rb_idx](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class PosteriorEncoder(nn.Module):
    """Training-only posterior encoder (enc_q).
    Loaded for strict=True but not called during inference.
    Parameter namespace: enc_q.*
    """
    def __init__(self, spec_channels, inter_channels, hidden_channels,
                 kernel_size, n_layers, gin_channels):
        super().__init__()
        self.pre = nn.Conv1d(spec_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, n_layers, gin_channels)
        self.proj = nn.Conv1d(hidden_channels, inter_channels * 2, 1)

    def forward(self, x, x_mask, g):
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(stats.shape[1] // 2, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m, logs


class SynthesizerSvc(nn.Module):
    """Top-level SoVITS 4.0 inference model.
    Parameter names match checkpoint state_dict exactly.
    """
    def __init__(self, spec_channels, ssl_dim, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size,
                 resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 n_speakers, gin_channels, sample_rate,
                 window_size=4, harmonic_num=8,
                 enc_q_n_layers=16):
        super().__init__()

        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.emb_uv = nn.Embedding(2, inter_channels)

        self.pre = nn.Conv1d(ssl_dim, inter_channels, 5, padding=2)

        self.enc_p = TextEncoderSoVITS(
            inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, window_size,
        )

        self.dec = GeneratorNSFSoVITS(
            inter_channels, resblock_kernel_sizes,
            resblock_dilation_sizes, upsample_rates,
            upsample_initial_channel, upsample_kernel_sizes,
            gin_channels, sample_rate, harmonic_num,
        )

        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 4, 4, gin_channels,
        )

        self.f0_decoder = F0Decoder(
            inter_channels, filter_channels, n_heads, n_layers,
            kernel_size, gin_channels,
        )

        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels,
            5, enc_q_n_layers, gin_channels,
        )

    def forward(self, c, f0, uv, sid):
        """
        c: [B, T, ssl_dim] — ContentVec features (256-dim for SoVITS 4.0)
        f0: [B, T] — continuous F0 in Hz (0 = unvoiced)
        uv: [B, T] — voiced/unvoiced (0 or 1)
        sid: [B] — speaker ID
        Returns: audio [B, 1, L]
        """
        g = self.emb_g(sid).unsqueeze(-1)  # [B, gin_channels, 1]

        x = self.pre(c.transpose(1, 2))  # [B, inter, T]
        x = x + self.emb_uv(uv.long()).transpose(1, 2)

        # F0 coarse index for embedding
        f0_coarse = f0_to_coarse(f0)
        x = x + self.enc_p.f0_emb(f0_coarse).transpose(1, 2)

        T = x.shape[2]
        x_mask = torch.ones(1, 1, T, device=x.device, dtype=x.dtype)

        m_p, logs_p, x_mask = self.enc_p(x, x_mask)

        z_p = m_p * x_mask

        z = self.flow(z_p, x_mask, g=g, reverse=True)

        o = self.dec(z * x_mask, f0, g=g)
        return o


def f0_to_coarse(f0):
    f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - 1.0) / (1127.0 * math.log(1 + 1100.0 / 700.0) - 1.0) * 254.0 + 1.0
    f0_coarse = f0_mel.long().clamp(0, 255)
    return f0_coarse


def build_from_checkpoint(checkpoint: dict) -> SynthesizerSvc:
    """Build SoVITS 4.0 model from checkpoint and load weights.
    SoVITS 4.0 checkpoints have no 'config' key — hyperparameters are inferred from weight shapes.
    """
    state_dict = checkpoint["model"]

    # Infer architecture from parameter shapes
    ssl_dim = state_dict["pre.weight"].shape[1]  # 256
    inter_channels = state_dict["pre.weight"].shape[0]  # 192
    gin_channels = state_dict["emb_g.weight"].shape[1]  # 256
    n_speakers = state_dict["emb_g.weight"].shape[0]  # 200
    filter_channels = state_dict["enc_p.enc_.ffn_layers.0.conv_1.weight"].shape[0]  # 768
    spec_channels = state_dict["enc_q.pre.weight"].shape[1]  # 1025

    emb_rel_k = state_dict["enc_p.enc_.attn_layers.0.emb_rel_k"]
    window_size = (emb_rel_k.shape[1] - 1) // 2  # 4
    head_dim = emb_rel_k.shape[2]  # 96
    n_heads = inter_channels // head_dim  # 2

    enc_p_n_layers = 0
    while f"enc_p.enc_.attn_layers.{enc_p_n_layers}.conv_q.weight" in state_dict:
        enc_p_n_layers += 1

    # Decoder config from ups weights
    upsample_rates = []
    upsample_kernel_sizes = []
    i = 0
    while f"dec.ups.{i}.weight_v" in state_dict or f"dec.ups.{i}.weight" in state_dict:
        key = f"dec.ups.{i}.weight_v" if f"dec.ups.{i}.weight_v" in state_dict else f"dec.ups.{i}.weight"
        w = state_dict[key]
        kernel = w.shape[2]
        upsample_kernel_sizes.append(kernel)
        # stride = kernel // 2 for standard HiFi-GAN transposed conv
        stride = kernel // 2 if kernel >= 4 else kernel
        upsample_rates.append(stride)
        i += 1

    upsample_initial_channel = state_dict["dec.cond.weight"].shape[1] * 2
    # Actually read from conv_pre
    key = "dec.conv_pre.weight_v" if "dec.conv_pre.weight_v" in state_dict else "dec.conv_pre.weight"
    upsample_initial_channel = state_dict[key].shape[0]  # 512

    # Count total resblocks and extract kernel sizes
    n_stages = len(upsample_rates)
    all_resblock_kernels = []
    for j in range(100):
        key_v = f"dec.resblocks.{j}.convs1.0.weight_v"
        key_w = f"dec.resblocks.{j}.convs1.0.weight"
        if key_v in state_dict:
            all_resblock_kernels.append(state_dict[key_v].shape[2])
        elif key_w in state_dict:
            all_resblock_kernels.append(state_dict[key_w].shape[2])
        else:
            break
    num_kernels = len(all_resblock_kernels) // n_stages if n_stages > 0 else 3
    resblock_kernel_sizes = all_resblock_kernels[:num_kernels]

    resblock_dilation_sizes = [[1, 3, 5]] * len(resblock_kernel_sizes)

    # Harmonic count from source module
    harmonic_num = state_dict["dec.m_source.l_linear.weight"].shape[1] - 1  # 8

    # enc_q WN layer count
    enc_q_n_layers = 0
    while f"enc_q.enc.in_layers.{enc_q_n_layers}.weight_v" in state_dict or \
          f"enc_q.enc.in_layers.{enc_q_n_layers}.weight" in state_dict:
        enc_q_n_layers += 1

    print(f"SoVITS 4.0 config inferred:", flush=True)
    print(f"  ssl_dim={ssl_dim}, inter={inter_channels}, gin={gin_channels}, speakers={n_speakers}", flush=True)
    print(f"  filter={filter_channels}, heads={n_heads}, enc_layers={enc_p_n_layers}, window={window_size}", flush=True)
    print(f"  upsample_rates={upsample_rates}, kernels={upsample_kernel_sizes}", flush=True)
    print(f"  resblock_kernels={resblock_kernel_sizes}, harmonics={harmonic_num}", flush=True)
    print(f"  enc_q_layers={enc_q_n_layers}, spec_channels={spec_channels}", flush=True)

    # Infer sample rate from total upsample factor
    # SoVITS 4.0 uses hop_size matching total upsample, standard sample_rate=44100
    sample_rate = 44100

    model = SynthesizerSvc(
        spec_channels=spec_channels,
        ssl_dim=ssl_dim,
        inter_channels=inter_channels,
        hidden_channels=inter_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=enc_p_n_layers,
        kernel_size=3,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        n_speakers=n_speakers,
        gin_channels=gin_channels,
        sample_rate=sample_rate,
        window_size=window_size,
        harmonic_num=harmonic_num,
        enc_q_n_layers=enc_q_n_layers,
    )

    # Remove weight_norm from source state_dict (fuse weight_g + weight_v → weight)
    # to match our model which will also have weight_norm removed after loading
    clean_dict = {}
    for k, v in state_dict.items():
        if hasattr(v, "float"):
            clean_dict[k] = v.float()
        else:
            clean_dict[k] = v

    model.load_state_dict(clean_dict, strict=True)

    remove_all_weight_norm(model)
    model.eval()
    return model
