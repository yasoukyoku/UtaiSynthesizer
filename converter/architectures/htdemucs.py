"""HTDemucs (Hybrid Transformer Demucs) — clean architecture for ONNX export.

Dual-path U-Net: frequency branch (Conv2d) + time branch (Conv1d),
cross-domain Transformer at bottleneck. From Meta/Facebook Research.

STFT/iSTFT excluded — handled in Rust via rustfft.

Input:  cac_spec [B, 4, F, T]  (CaC: real_L, imag_L, real_R, imag_R)
        mix      [B, 2, T_audio]  (stereo waveform)
Output: freq_out [B, S, 4, F, T]  (CaC per stem)
        time_out [B, S, 2, T_audio]  (waveform per stem)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# ─── Utilities ─────────────────────────────────────────────────

class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 0, channel_last=False):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        return self.scale * x if self.channel_last else self.scale[:, None] * x


class MyGroupNorm(nn.GroupNorm):
    def forward(self, x):
        # x: (B, T, C) → transpose to (B, C, T) for GroupNorm
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def create_sin_embedding(length, dim, shift=0, device="cpu", max_period=10000):
    assert dim % 2 == 0
    pos = (shift + torch.arange(length, device=device)).view(-1, 1, 1).float()
    half = dim // 2
    adim = torch.arange(half, device=device).view(1, 1, -1).float()
    phase = pos / (max_period ** (adim / (half - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4, got {d_model}")
    pe = torch.zeros(d_model, height, width, device=device)
    d = d_model // 2
    div = torch.exp(torch.arange(0.0, d, 2, device=device) * -(math.log(max_period) / d))
    pos_w = torch.arange(0.0, width, device=device).unsqueeze(1)
    pos_h = torch.arange(0.0, height, device=device).unsqueeze(1)
    pe[0:d:2] = (pos_w * div).sin().t().unsqueeze(1).expand(-1, height, -1)
    pe[1:d:2] = (pos_w * div).cos().t().unsqueeze(1).expand(-1, height, -1)
    pe[d::2] = (pos_h * div).sin().t().unsqueeze(2).expand(-1, -1, width)
    pe[d + 1::2] = (pos_h * div).cos().t().unsqueeze(2).expand(-1, -1, width)
    return pe[None]


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            std = sub.weight.std().detach()
            scale = (std / reference) ** 0.5
            sub.weight.data /= scale
            if sub.bias is not None:
                sub.bias.data /= scale


# ─── DConv ─────────────────────────────────────────────────────

class DConv(nn.Module):
    def __init__(self, channels, compress=4, depth=2, init=1e-4, norm=True,
                 attn=False, heads=4, ndecay=4, lstm=False, gelu=True,
                 kernel=3, dilate=True):
        super().__init__()
        assert kernel % 2 == 1
        dilate = depth > 0
        hidden = int(channels / compress)
        act_cls = nn.GELU if gelu else nn.ReLU
        norm_fn = (lambda d: nn.GroupNorm(1, d)) if norm else (lambda d: nn.Identity())

        self.layers = nn.ModuleList()
        for d in range(abs(depth)):
            dilation = 2 ** d if dilate else 1
            padding = dilation * (kernel // 2)
            layer = nn.Sequential(
                nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm_fn(hidden), act_cls(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels), nn.GLU(1),
                LayerScale(channels, init),
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


# ─── Encoder / Decoder layers ─────────────────────────────────

class ScaledEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, scale=10.0, smooth=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            w = torch.cumsum(self.embedding.weight.data, dim=0)
            w = w / torch.arange(1, num_embeddings + 1).to(w).sqrt()[:, None]
            self.embedding.weight.data[:] = w
        self.embedding.weight.data /= scale
        self.scale = scale

    def forward(self, x):
        return self.embedding(x) * self.scale


class HEncLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1,
                 empty=False, freq=True, dconv=True, norm=True, context=0,
                 dconv_kw=None, pad=True, rewrite=True):
        super().__init__()
        if dconv_kw is None:
            dconv_kw = {}
        norm_fn = (lambda d: nn.GroupNorm(norm_groups, d)) if norm else (lambda d: nn.Identity())
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.pad = kernel_size // 4 if pad else 0

        if freq:
            ks = [kernel_size, 1]
            st = [stride, 1]
            pd = [self.pad, 0]
            self.conv = nn.Conv2d(chin, chout, ks, st, pd)
        else:
            self.conv = nn.Conv1d(chin, chout, kernel_size, stride, self.pad)

        if empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            if freq:
                self.rewrite = nn.Conv2d(chout, 2 * chout, 1 + 2 * context, 1, context)
            else:
                self.rewrite = nn.Conv1d(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)
        self.dconv = DConv(chout, **dconv_kw) if dconv else None

    def forward(self, x, inject=None):
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)
        if not self.freq:
            le = x.shape[-1]
            if le % self.stride != 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.dconv:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if self.rewrite:
            z = F.glu(self.norm2(self.rewrite(y)), dim=1)
        else:
            z = y
        return z


class HDecLayer(nn.Module):
    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4,
                 norm_groups=1, empty=False, freq=True, dconv=True, norm=True,
                 context=1, dconv_kw=None, pad=True, context_freq=True, rewrite=True):
        super().__init__()
        if dconv_kw is None:
            dconv_kw = {}
        norm_fn = (lambda d: nn.GroupNorm(norm_groups, d)) if norm else (lambda d: nn.Identity())
        self.pad = kernel_size // 4 if pad else 0
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride

        if freq:
            ks = [kernel_size, 1]
            st = [stride, 1]
            self.conv_tr = nn.ConvTranspose2d(chin, chout, ks, st)
        else:
            self.conv_tr = nn.ConvTranspose1d(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)

        if empty:
            return
        self.rewrite = None
        if rewrite:
            if freq and context_freq:
                self.rewrite = nn.Conv2d(chin, 2 * chin, 1 + 2 * context, 1, context)
            elif freq:
                self.rewrite = nn.Conv2d(chin, 2 * chin, [1, 1 + 2 * context], 1, [0, context])
            else:
                self.rewrite = nn.Conv1d(chin, 2 * chin, 1 + 2 * context, 1, context)
            self.norm1 = norm_fn(2 * chin)
        self.dconv = DConv(chin, **dconv_kw) if dconv else None

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)
        if not self.empty:
            x = x + skip
            if self.rewrite:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad:-self.pad, :]
        else:
            z = z[..., self.pad:self.pad + length]
        if not self.last:
            z = F.gelu(z)
        return z, y


# ─── Transformer ───────────────────────────────────────────────

class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, group_norm=0, norm_first=False, norm_out=False,
                 layer_norm_eps=1e-5, layer_scale=False, init_values=1e-4,
                 device=None, dtype=None, batch_first=False, **_kw):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
            batch_first=batch_first, norm_first=norm_first, device=device, dtype=dtype,
        )
        fk = {"device": device, "dtype": dtype}
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **fk)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **fk)
        self.norm_out = None
        if norm_first and norm_out:
            self.norm_out = MyGroupNorm(int(norm_out), d_model)
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self.gamma_1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask))
            x = x + self.gamma_2(self._ff_block(self.norm2(x)))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))
        return x


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, layer_scale=False,
                 init_values=1e-4, norm_first=False, group_norm=False,
                 norm_out=False, batch_first=False, **_kw):
        super().__init__()
        fk = {}
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first

        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps)
            self.norm3 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.norm_out = None
        if norm_first and norm_out:
            self.norm_out = MyGroupNorm(int(norm_out), d_model)
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if isinstance(activation, str):
            self.activation = F.gelu if activation == "gelu" else F.relu
        else:
            self.activation = activation

    def forward(self, q, k, mask=None):
        if self.norm_first:
            x = q + self.gamma_1(self.dropout1(
                self.cross_attn(self.norm1(q), self.norm2(k), self.norm2(k), need_weights=False)[0]))
            x = x + self.gamma_2(self.dropout2(
                self.linear2(self.dropout(self.activation(self.linear1(self.norm3(x)))))))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            ca = self.dropout1(self.cross_attn(q, k, k, need_weights=False)[0])
            x = self.norm1(q + self.gamma_1(ca))
            ff = self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
            x = self.norm2(x + self.gamma_2(ff))
        return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, emb="sin", hidden_scale=4.0, num_heads=8, num_layers=6,
                 cross_first=False, dropout=0.0, max_positions=1000, norm_in=True,
                 norm_in_group=False, group_norm=False, norm_first=False, norm_out=False,
                 max_period=10000.0, weight_decay=0.0, lr=None, layer_scale=False,
                 gelu=True, sin_random_shift=0, weight_pos_embed=1.0, **_kw):
        super().__init__()
        hidden_dim = int(dim * hidden_scale)
        self.num_layers = num_layers
        self.classic_parity = 1 if cross_first else 0
        self.max_period = max_period
        self.weight_pos_embed = weight_pos_embed

        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
            self.norm_in_t = nn.LayerNorm(dim)
        elif norm_in_group:
            self.norm_in = MyGroupNorm(int(norm_in_group), dim)
            self.norm_in_t = MyGroupNorm(int(norm_in_group), dim)
        else:
            self.norm_in = nn.Identity()
            self.norm_in_t = nn.Identity()

        activation = F.gelu if gelu else F.relu
        kw = dict(d_model=dim, nhead=num_heads, dim_feedforward=hidden_dim,
                  dropout=dropout, activation=activation, group_norm=group_norm,
                  norm_first=norm_first, norm_out=norm_out, layer_scale=layer_scale,
                  batch_first=True)

        self.layers = nn.ModuleList()
        self.layers_t = nn.ModuleList()
        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                self.layers.append(MyTransformerEncoderLayer(**kw))
                self.layers_t.append(MyTransformerEncoderLayer(**kw))
            else:
                self.layers.append(CrossTransformerEncoderLayer(**kw))
                self.layers_t.append(CrossTransformerEncoderLayer(**kw))

    def forward(self, x, xt):
        B, C, Fr, T1 = x.shape
        pos_2d = create_2d_sin_embedding(C, Fr, T1, x.device, self.max_period)
        pos_2d = pos_2d.reshape(1, C, Fr * T1).permute(0, 2, 1)  # [1, Fr*T1, C]
        x = x.reshape(B, C, Fr * T1).permute(0, 2, 1)  # [B, Fr*T1, C]
        x = self.norm_in(x) + self.weight_pos_embed * pos_2d

        B2, C2, T2 = xt.shape
        xt = xt.permute(0, 2, 1)  # [B, T2, C]
        pos_1d = create_sin_embedding(T2, C2, device=xt.device, max_period=self.max_period)
        pos_1d = pos_1d.permute(1, 0, 2)  # [1, T2, C]
        xt = self.norm_in_t(xt) + self.weight_pos_embed * pos_1d

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = x.permute(0, 2, 1).reshape(B, C, Fr, T1)  # [B, C, Fr, T1]
        xt = xt.permute(0, 2, 1)  # [B, C, T2]
        return x, xt


# ─── HTDemucs ──────────────────────────────────────────────────

class HTDemucs(nn.Module):
    """Hybrid Transformer Demucs. STFT excluded — takes CaC + waveform."""

    def __init__(
        self, *, sources=4, audio_channels=2, channels=48, channels_time=None,
        growth=2, nfft=4096, depth=4, rewrite=True, freq_emb=0.2,
        emb_scale=10, emb_smooth=True, kernel_size=8, time_stride=2,
        stride=4, context=1, context_enc=0, norm_starts=4, norm_groups=4,
        dconv_mode=1, dconv_depth=2, dconv_comp=8, dconv_init=1e-3,
        bottom_channels=0,
        t_layers=5, t_hidden_scale=4.0, t_heads=8, t_dropout=0.0,
        t_max_positions=10000, t_norm_in=True, t_norm_in_group=False,
        t_group_norm=False, t_norm_first=True, t_norm_out=True,
        t_max_period=10000.0, t_layer_scale=True, t_gelu=True,
        t_weight_pos_embed=1.0, t_cross_first=False,
        rescale=0.1,
        samplerate=44100, segment=10,
    ):
        super().__init__()
        self.sources = sources if isinstance(sources, int) else len(sources)
        self.audio_channels = audio_channels
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.samplerate = samplerate
        self.segment = segment

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin * 2  # CaC
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2
        self.freq_emb = None

        dconv_kw = {"depth": dconv_depth, "compress": dconv_comp,
                    "init": dconv_init, "gelu": True}

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size

            if not freq:
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = dict(kernel_size=ker, stride=stri, freq=freq, pad=pad,
                      norm=norm, rewrite=rewrite, norm_groups=norm_groups,
                      dconv_kw=dconv_kw)
            kwt = dict(kw, freq=0, kernel_size=kernel_size, stride=stride, pad=True)
            kw_dec = dict(kw)

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            # dconv_mode: official demucs semantics — bit 1 = encoder, bit 2 = decoder.
            # Official ckpts use dconv_mode=1 (encoder only); MSST yamls often use 3.
            self.encoder.append(
                HEncLayer(chin_z, chout_z, dconv=bool(dconv_mode & 1),
                          context=context_enc, **kw))
            if freq:
                self.tencoder.append(
                    HEncLayer(chin, chout, dconv=bool(dconv_mode & 1),
                              context=context_enc, empty=last_freq, **kwt))

            if index == 0:
                chin = audio_channels * self.sources
                chin_z = chin * 2

            self.decoder.insert(0,
                HDecLayer(chout_z, chin_z, dconv=bool(dconv_mode & 2),
                          last=index == 0, context=context, **kw_dec))
            if freq:
                self.tdecoder.insert(0,
                    HDecLayer(chout, chin, dconv=bool(dconv_mode & 2), empty=last_freq,
                              last=index == 0, context=context, **kwt))

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                freqs = 1 if freqs <= kernel_size else freqs // stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(freqs, chin_z,
                                                smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(bottom_channels, transformer_channels, 1)
            self.channel_upsampler_t = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler_t = nn.Conv1d(bottom_channels, transformer_channels, 1)
            transformer_channels = bottom_channels

        self.crosstransformer = CrossTransformerEncoder(
            dim=transformer_channels, emb="sin", hidden_scale=t_hidden_scale,
            num_heads=t_heads, num_layers=t_layers, cross_first=t_cross_first,
            dropout=t_dropout, max_positions=t_max_positions, norm_in=t_norm_in,
            norm_in_group=t_norm_in_group, group_norm=t_group_norm,
            norm_first=t_norm_first, norm_out=t_norm_out,
            max_period=t_max_period, layer_scale=t_layer_scale,
            gelu=t_gelu, weight_pos_embed=t_weight_pos_embed,
        ) if t_layers > 0 else None

    def forward(self, cac_spec: torch.Tensor, mix: torch.Tensor):
        """
        Args:
            cac_spec: [B, 4, F, T] CaC stereo spectrogram
            mix: [B, 2, T_audio] stereo waveform
        Returns:
            freq_out: [B, S, 4, F, T] CaC per-stem spectrogram
            time_out: [B, S, 2, T_audio] per-stem waveform
        """
        x = cac_spec
        B, C, Fq, T = x.shape
        S = self.sources

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved = []
        saved_t = []
        lengths = []
        lengths_t = []

        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = self.channel_upsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                b2, c2, f2, t2 = x.shape
                x = x.reshape(b2, c2, f2 * t2)
                x = self.channel_downsampler(x)
                x = x.reshape(b2, -1, f2, t2)
                xt = self.channel_downsampler_t(xt)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre_t = pre[:, :, 0]
                    xt, _ = tdec(pre_t, None, length_t)
                else:
                    skip_t = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip_t, length_t)

        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        xt = xt.view(B, S, -1, mix.shape[-1])
        xt = xt * stdt[:, None] + meant[:, None]

        return x, xt


# ─── Config detection ─────────────────────────────────────────

def _guard_unsupported(params: dict, origin: str) -> None:
    """Hard-fail on HTDemucs variants this ONNX reimplementation does not cover.
    Loading them with a mismatched graph would silently corrupt output."""
    if params.get("cac", True) is not True:
        raise RuntimeError(
            f"Unsupported HTDemucs variant ({origin}): cac=false (magnitude+Wiener "
            f"mode). This converter only supports cac=true models."
        )
    if int(params.get("num_subbands", 1) or 1) > 1:
        raise RuntimeError(
            f"Unsupported HTDemucs variant ({origin}): num_subbands="
            f"{params.get('num_subbands')} > 1 is not supported."
        )
    if params.get("multi_freqs"):
        raise RuntimeError(
            f"Unsupported HTDemucs variant ({origin}): non-empty multi_freqs "
            f"(MultiWrap band-split layers) is not supported."
        )


def _accepted_hparams(params: dict) -> dict:
    """Filter a kwargs/yaml-htdemucs mapping down to HTDemucs.__init__ params,
    sanitizing values (MSST yamls carry dconv_init as the string '1e-3')."""
    import inspect
    accepted = set(inspect.signature(HTDemucs.__init__).parameters) - {"self"}
    cfg = {k: v for k, v in params.items() if k in accepted}
    if "dconv_init" in cfg:
        cfg["dconv_init"] = float(cfg["dconv_init"])
    if "segment" in cfg:
        cfg["segment"] = float(cfg["segment"])
    return cfg


def detect_config(ckpt_path: str, yaml_path: Optional[str] = None) -> dict:
    """Detect HTDemucs config from checkpoint (+ optional MSST training yaml).

    Official demucs ckpts carry full 'kwargs'; MSST fine-tunes are raw state
    dicts, where structure is detected from the weights and remaining
    hyperparameters come from the yaml's htdemucs section.
    """
    from .msst_yaml import load_msst_yaml
    yaml_config = load_msst_yaml(yaml_path)
    training_yaml = yaml_config.get("training") or {}

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Official demucs format: {'klass', 'kwargs', 'state'}
    if isinstance(state, dict) and "kwargs" in state:
        kw = dict(state["kwargs"])
        _guard_unsupported(kw, "ckpt kwargs")
        cfg = _accepted_hparams(kw)
        sources = kw.get("sources", ["drums", "bass", "other", "vocals"])
        cfg["sources"] = list(sources) if isinstance(sources, (list, tuple)) else sources
        # Official demucs default is dconv_mode=1 (encoder only) — NOT both.
        cfg["dconv_mode"] = int(kw.get("dconv_mode", 1))
        if "segment" not in cfg:
            cfg["segment"] = float(training_yaml.get("segment", 11))
        print(f"  [kwargs] sources={cfg['sources']}, channels={cfg.get('channels', 48)}, "
              f"depth={cfg.get('depth', 4)}, dconv_mode={cfg['dconv_mode']}, "
              f"segment={cfg['segment']}")
        return cfg

    # Raw state_dict (MSST format)
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "state" in state:
        sd = state["state"]
    else:
        sd = state

    ht_yaml = yaml_config.get("htdemucs") or {}
    _guard_unsupported(ht_yaml, "yaml htdemucs section")
    cfg = _accepted_hparams(ht_yaml)

    # Structure detected from the weights overrides the yaml.
    channels = sd["encoder.0.conv.weight"].shape[0]
    chin_z = sd["encoder.0.conv.weight"].shape[1]  # audio_channels * 2 (CaC)
    audio_channels = chin_z // 2

    depth = 0
    while f"encoder.{depth}.conv.weight" in sd:
        depth += 1

    # sources: the OUTERMOST freq decoder is decoder.{depth-1} (the constructor
    # builds with decoder.insert(0, ...), so decoder.0 is the DEEPEST layer —
    # reading sources there gives channels*growth^(depth-1)-scale garbage).
    # Its conv_tr outputs sources * audio_channels * 2 (CaC) channels.
    sources = sd[f"decoder.{depth - 1}.conv_tr.weight"].shape[1] // (2 * audio_channels)
    t_key = f"tdecoder.{depth - 1}.conv_tr.weight"
    if t_key in sd:
        t_sources = sd[t_key].shape[1] // audio_channels
        if t_sources != sources:
            raise RuntimeError(
                f"Inconsistent sources: freq decoder says {sources}, "
                f"time decoder says {t_sources}"
            )

    growth = int(sd["encoder.1.conv.weight"].shape[0] / channels) if depth > 1 else 2

    t_layers = 0
    while f"crosstransformer.layers.{t_layers}.self_attn.in_proj_weight" in sd or \
          f"crosstransformer.layers.{t_layers}.cross_attn.in_proj_weight" in sd:
        t_layers += 1

    bottom_channels = 0
    if "channel_upsampler.weight" in sd:
        bottom_channels = sd["channel_upsampler.weight"].shape[0]

    # dconv per branch from actual key presence (official=encoder only,
    # MSST yamls often say 3=both; the weights are the ground truth).
    enc_dconv = any(k.startswith("encoder.") and ".dconv." in k for k in sd)
    dec_dconv = any(k.startswith("decoder.") and ".dconv." in k for k in sd)
    dconv_mode = (1 if enc_dconv else 0) | (2 if dec_dconv else 0)

    cfg.update(
        sources=sources, audio_channels=audio_channels, channels=channels,
        depth=depth, growth=growth, t_layers=t_layers,
        bottom_channels=bottom_channels, dconv_mode=dconv_mode,
    )
    cfg.setdefault("nfft", 4096)
    cfg["samplerate"] = int(training_yaml.get("samplerate", cfg.get("samplerate", 44100)))
    # MSST trains/runs with training.segment (11s for the vocal models) — 10 is wrong.
    cfg["segment"] = float(training_yaml.get("segment", 11))

    print(f"  sources={sources}, channels={channels}, depth={depth}, growth={growth}")
    print(f"  t_layers={t_layers}, bottom_channels={bottom_channels}, "
          f"dconv_mode={dconv_mode}, segment={cfg['segment']}")

    return cfg


def load_from_checkpoint(ckpt_path: str, config: Optional[dict] = None,
                         yaml_path: Optional[str] = None) -> HTDemucs:
    """Load HTDemucs from checkpoint."""
    if config is None:
        config = detect_config(ckpt_path, yaml_path)

    model = HTDemucs(**config)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "state" in state:
            sd = state["state"]
        elif "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state
    else:
        sd = state

    # strict=True: any mismatch means the detected config diverged from the
    # checkpoint (e.g. wrong dconv_mode), which would silently corrupt output.
    model.load_state_dict(sd, strict=True)

    model.eval()
    return model
