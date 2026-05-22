"""
RVC v2 model architecture for ONNX export.
Reconstructed from checkpoint parameter inspection to produce exact parameter name matches.

Inference-only path: phone → enc_p → flow.reverse → dec(NSF) → audio
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class LayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return self.gamma.unsqueeze(-1) * (x - mean) / (var + 1e-5).sqrt() + self.beta.unsqueeze(-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, n_heads, window_size=10):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.window_size = window_size

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)

        self.emb_rel_k = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.head_dim))
        self.emb_rel_v = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.head_dim))

    def forward(self, x, x_mask):
        B, C, T = x.shape
        q = self.conv_q(x).view(B, self.n_heads, self.head_dim, T)
        k = self.conv_k(x).view(B, self.n_heads, self.head_dim, T)
        v = self.conv_v(x).view(B, self.n_heads, self.head_dim, T)

        # Standard attention: scores = q^T k / sqrt(d)
        scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(self.head_dim)

        # Relative position bias
        rel_k = self._get_relative_embeddings(self.emb_rel_k, T)
        rel_scores = torch.matmul(q.transpose(-2, -1), rel_k.transpose(-2, -1))
        scores = scores + rel_scores

        scores = scores.masked_fill(x_mask.unsqueeze(1).unsqueeze(2) == 0, -1e4)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.transpose(-2, -1))

        # Relative position for values
        rel_v = self._get_relative_embeddings(self.emb_rel_v, T)
        rel_out = torch.matmul(attn, rel_v)
        out = out + rel_out

        out = out.transpose(-2, -1).contiguous().view(B, C, T)
        out = self.conv_o(out)
        return out

    def _get_relative_embeddings(self, emb, length):
        # emb: [1, 2W+1, D], returns [1, T, D] relative embeddings for each position
        W = self.window_size
        pad = max(0, length - W)
        start = max(0, W - length + 1)
        end = start + 2 * min(length, W) - 1

        if length <= W:
            return emb[:, start:end + 1, :]

        # For longer sequences, we need to handle the padding
        e = emb[:, start:end + 1, :]
        if e.shape[1] < length:
            e = F.pad(e, [0, 0, pad, pad])
        return e[:, :length, :]


class FFN(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, in_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(nn.Module):
    """Transformer encoder with relative position attention.
    Parameter namespace: encoder.attn_layers, encoder.norm_layers_1, encoder.ffn_layers, encoder.norm_layers_2
    """
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, window_size=10):
        super().__init__()
        self.n_layers = n_layers

        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(hidden_channels, n_heads, window_size)
            for _ in range(n_layers)
        ])
        self.norm_layers_1 = nn.ModuleList([
            LayerNorm(hidden_channels) for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            FFN(hidden_channels, filter_channels, kernel_size)
            for _ in range(n_layers)
        ])
        self.norm_layers_2 = nn.ModuleList([
            LayerNorm(hidden_channels) for _ in range(n_layers)
        ])

    def forward(self, x, x_mask):
        for i in range(self.n_layers):
            residual = x
            x = self.norm_layers_1[i](x)
            x = self.attn_layers[i](x, x_mask)
            x = residual + x

            residual = x
            x = self.norm_layers_2[i](x)
            x = self.ffn_layers[i](x, x_mask)
            x = residual + x

        return x * x_mask


class TextEncoder(nn.Module):
    """
    Parameter namespace: enc_p.*
    Sub-modules: emb_phone, emb_pitch, encoder, proj
    """
    def __init__(self, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, window_size=10):
        super().__init__()
        self.inter_channels = inter_channels

        self.emb_phone = nn.Linear(768, inter_channels)
        self.emb_pitch = nn.Embedding(256, inter_channels)

        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, window_size)
        self.proj = nn.Conv1d(inter_channels, inter_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        # phone: [B, T, 768], pitch: [B, T] (coarse int)
        x = self.emb_phone(phone).transpose(1, 2)  # [B, inter, T]
        x = x + self.emb_pitch(pitch.long()).transpose(1, 2)

        # Create mask from lengths
        T = x.shape[2]
        x_mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        x_mask = x_mask.unsqueeze(1).float()  # [B, 1, T]

        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(self.inter_channels, dim=1)
        return m, logs, x_mask


class WN(nn.Module):
    """WaveNet residual block used in flow coupling layers.
    Parameter namespace: enc.in_layers, enc.res_skip_layers, enc.cond_layer
    """
    def __init__(self, hidden_channels, kernel_size, n_layers, gin_channels):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        for i in range(n_layers):
            dilation = 1
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                      padding=padding, dilation=dilation))
            )
            if i < n_layers - 1:
                self.res_skip_layers.append(
                    weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels, 1))
                )
            else:
                self.res_skip_layers.append(
                    weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 1))
                )

        self.cond_layer = weight_norm(nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1))

    def forward(self, x, x_mask, g):
        # g: [B, gin_channels, 1]
        g_out = self.cond_layer(g)  # [B, 2*H*n_layers, 1]

        for i in range(self.n_layers):
            cond_offset = i * 2 * self.hidden_channels
            g_l = g_out[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]

            x_in = self.in_layers[i](x) + g_l
            t_act = torch.tanh(x_in[:, :self.hidden_channels, :])
            s_act = torch.sigmoid(x_in[:, self.hidden_channels:, :])
            acts = t_act * s_act

            res_skip = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip[:, :self.hidden_channels, :]) * x_mask
            else:
                x = res_skip * x_mask

        return x


class ResidualCouplingLayer(nn.Module):
    """One coupling layer of the normalizing flow.
    Parameter namespace: flows.{i}.pre, flows.{i}.enc, flows.{i}.post
    """
    def __init__(self, channels, hidden_channels, kernel_size, n_layers, gin_channels):
        super().__init__()
        self.half_channels = channels // 2

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, n_layers, gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g, reverse=False):
        x0, x1 = x.split(self.half_channels, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g)
        stats = self.post(h) * x_mask
        m = stats

        if not reverse:
            x1 = m + x1 * x_mask
        else:
            x1 = (x1 - m) * x_mask

        return torch.cat([x0, x1], dim=1) * x_mask


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        return x


class ResidualCouplingBlock(nn.Module):
    """Full normalizing flow: alternating coupling layers and flips.
    Parameter namespace: flow.flows.{0,2,4,6} for coupling layers
    Flips at indices 1,3,5 have no parameters.
    """
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_flows, gin_channels):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, gin_channels)
            )
            if i < n_flows - 1:
                self.flows.append(Flip())

    def forward(self, x, x_mask, g, reverse=False):
        if not reverse:
            for flow in self.flows:
                x = flow(x, x_mask, g=g, reverse=False)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=True)
        return x


class SourceModule(nn.Module):
    """Neural Source Filter — generates harmonic excitation from F0.
    Parameter namespace: dec.m_source.l_linear
    """
    def __init__(self, sample_rate, upsample_factor):
        super().__init__()
        self.sample_rate = sample_rate
        self.upsample_factor = upsample_factor
        self.l_linear = nn.Linear(1, 1)

    def forward(self, f0, upsample_factor=None):
        # f0: [B, T] — continuous F0 in Hz (0 = unvoiced)
        factor = upsample_factor or self.upsample_factor
        f0_up = F.interpolate(f0.unsqueeze(1), scale_factor=float(factor), mode="nearest")
        # f0_up: [B, 1, T*factor]

        # Generate sine excitation
        omega = 2 * math.pi * f0_up / self.sample_rate
        phase = torch.cumsum(omega, dim=-1)
        sine = torch.sin(phase)

        # Zero out unvoiced regions
        voiced = (f0_up > 0).float()
        sine = sine * voiced

        # Linear transform + noise for unvoiced
        noise = torch.randn_like(sine)
        src = self.l_linear(sine.transpose(1, 2)).transpose(1, 2) + noise * (1 - voiced) * 0.003
        return src


class ResBlock(nn.Module):
    """HiFi-GAN residual block with multiple dilations.
    Parameter namespace: dec.resblocks.{i}.convs1, dec.resblocks.{i}.convs2
    """
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.convs1.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=padding))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2))
            )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class GeneratorNSF(nn.Module):
    """HiFi-GAN generator with Neural Source Filter for pitch-controlled synthesis.
    Parameter namespace: dec.*
    """
    def __init__(self, inter_channels, resblock_type, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                 upsample_kernel_sizes, gin_channels, sample_rate):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        total_upsample = 1
        for r in upsample_rates:
            total_upsample *= r

        self.m_source = SourceModule(sample_rate, total_upsample)

        self.conv_pre = nn.Conv1d(inter_channels, upsample_initial_channel, 7, 1, 3)

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

        self.conv_post = nn.Conv1d(block_ch, 1, 7, 1, 3, bias=False)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g):
        # x: [B, inter_channels, T]
        # f0: [B, T] continuous F0
        # g: [B, gin_channels, 1] speaker embedding

        har_source = self.m_source(f0)  # [B, 1, T*upsample_factor]

        x = self.conv_pre(x)
        x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            # Add noise/harmonic source at this resolution
            noise = self.noise_convs[i](har_source)
            if noise.shape[-1] != x.shape[-1]:
                noise = noise[:, :, :x.shape[-1]]
            x = x + noise

            # Multi-receptive-field fusion
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


class SynthesizerTrnMs768NSFsid(nn.Module):
    """Top-level RVC v2 inference model.
    This wraps enc_p, flow, dec, and emb_g into a single module
    whose parameter names exactly match the checkpoint state_dict.
    """
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 n_speakers, gin_channels, sr):
        super().__init__()

        self.enc_p = TextEncoder(
            inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size,
        )

        self.dec = GeneratorNSF(
            inter_channels, resblock, resblock_kernel_sizes,
            resblock_dilation_sizes, upsample_rates,
            upsample_initial_channel, upsample_kernel_sizes,
            gin_channels, sr,
        )

        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 3, 4, gin_channels,
        )

        self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, phone, phone_lengths, pitch, pitchf, sid, noise_scale):
        """
        phone: [B, T, 768] — HuBERT features
        phone_lengths: [B] — valid lengths (int)
        pitch: [B, T] — coarse pitch (0-255, int for embedding lookup)
        pitchf: [B, T] — continuous F0 in Hz (float, 0 = unvoiced)
        sid: [B] — speaker ID (int)
        noise_scale: [1] — 0.0 = deterministic, 0.667 = original RVC default
        Returns: audio [B, 1, L]
        """
        g = self.emb_g(sid).unsqueeze(-1)  # [B, gin_channels, 1]
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask

        z = self.flow(z_p, x_mask, g=g, reverse=True)

        o = self.dec(z * x_mask, pitchf, g=g)
        return o


def build_from_checkpoint(checkpoint: dict) -> SynthesizerTrnMs768NSFsid:
    """Build RVC v2 model from a loaded checkpoint dict and load weights.
    Returns the model ready for ONNX export.
    """
    config = checkpoint["config"]
    state_dict = checkpoint["weight"]

    # Parse the config list
    # [spec_ch, seg_size, inter_ch, hidden_ch, filter_ch, n_heads, n_layers, kernel, dropout,
    #  resblock, rb_kernels, rb_dilations, up_rates, up_init_ch, up_kernels, n_spk, gin_ch, sr]
    model = SynthesizerTrnMs768NSFsid(
        spec_channels=config[0],
        segment_size=config[1],
        inter_channels=config[2],
        hidden_channels=config[3],
        filter_channels=config[4],
        n_heads=config[5],
        n_layers=config[6],
        kernel_size=config[7],
        p_dropout=config[8],
        resblock=config[9],
        resblock_kernel_sizes=config[10],
        resblock_dilation_sizes=config[11],
        upsample_rates=config[12],
        upsample_initial_channel=config[13],
        upsample_kernel_sizes=config[14],
        n_speakers=config[15],
        gin_channels=config[16],
        sr=config[17],
    )

    # Convert float16 weights to float32
    state_dict_f32 = {}
    for k, v in state_dict.items():
        if hasattr(v, "float"):
            state_dict_f32[k] = v.float()
        else:
            state_dict_f32[k] = v

    # Load weights (strict mode — any mismatch is a fatal error)
    model.load_state_dict(state_dict_f32, strict=True)

    # Remove weight normalization from all layers (fuses weight_g + weight_v → weight)
    remove_all_weight_norm(model)

    model.eval()
    return model


def remove_all_weight_norm(module: nn.Module):
    """Recursively remove weight normalization from all sub-modules."""
    for name, child in module.named_children():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except ValueError:
            pass
        remove_all_weight_norm(child)
