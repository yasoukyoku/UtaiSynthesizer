"""BSRoformer — clean architecture for ONNX export.

Inference-only, no external deps (no einops/beartype/rotary_embedding_torch).
STFT/iSTFT excluded — handled in Rust via rustfft.
Module hierarchy matches the original exactly so load_state_dict works.

Input:  stft_repr [B, F*S, T, 2]  (S=audio_channels, 2=real/imag)
Output: mask      [B, N, F*S, T, 2]  (N=num_stems)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ─── Norms ──────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# ─── Rotary Position Embedding ──────────────────────────────────
# Matches rotary_embedding_torch: interleaved freq layout [f0,f0,f1,f1,...].

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate_queries_or_keys(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-2]
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        freqs = torch.stack([freqs, freqs], dim=-1).flatten(-2)

        x_r = x.unflatten(-1, (-1, 2))
        x1, x2 = x_r.unbind(-1)
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)

        return x * freqs.cos() + rotated * freqs.sin()


# ─── Attention ──────────────────────────────────────────────────

class Attend(nn.Module):
    """Placeholder to match original module hierarchy (no learnable params)."""
    def __init__(self, flash: bool = True, dropout: float = 0.0):
        super().__init__()
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotary_embed: Optional[RotaryEmbedding] = None,
        flash: bool = True,
    ):
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=dropout)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_norm = self.norm(x)

        qkv = self.to_qkv(x_norm)
        qkv = qkv.view(B, T, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x_norm).permute(0, 2, 1).unsqueeze(-1)
        out = out * gates.sigmoid()

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        return self.to_out(out)


# ─── Transformer ────────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        norm_output: bool = True,
        rotary_embed: Optional[RotaryEmbedding] = None,
        flash_attn: bool = True,
        linear_attn: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim=dim, dim_head=dim_head, heads=heads,
                    dropout=attn_dropout, rotary_embed=rotary_embed, flash=flash_attn,
                ),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ─── Band Split / Merge ────────────────────────────────────────

class BandSplit(nn.Module):
    def __init__(self, dim: int, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList()
        for dim_in in dim_inputs:
            self.to_features.append(nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = x.split(list(self.dim_inputs), dim=-1)
        outs = []
        for split_input, to_feature in zip(splits, self.to_features):
            outs.append(to_feature(split_input))
        return torch.stack(outs, dim=-2)


def _build_mlp(dim_in: int, dim_out: int, dim_hidden: Optional[int] = None,
               depth: int = 1) -> nn.Sequential:
    """Matches original: depth=number of hidden layers. depth=1 → 2 linears."""
    dim_hidden = dim_hidden if dim_hidden is not None else dim_in
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)
    net = []
    for ind, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)
        net.append(nn.Linear(d_in, d_out))
        if not is_last:
            net.append(nn.Tanh())
    return nn.Sequential(*net)


class MaskEstimator(nn.Module):
    def __init__(self, dim: int, dim_inputs: Tuple[int, ...], depth: int,
                 mlp_expansion_factor: int = 4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = nn.ModuleList()
        dim_hidden = dim * mlp_expansion_factor
        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                _build_mlp(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1),
            )
            self.to_freqs.append(mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = x.unbind(dim=-2)
        outs = []
        for band_features, mlp in zip(bands, self.to_freqs):
            outs.append(mlp(band_features))
        return torch.cat(outs, dim=-1)


# ─── Default band configuration ────────────────────────────────
# 61 bands summing to 1025 freq bins (n_fft=2048 → 1025)

DEFAULT_FREQS_PER_BANDS: Tuple[int, ...] = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


# ─── BSRoformer ─────────────────────────────────────────────────

class BSRoformer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = False,
        num_stems: int = 1,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        flash_attn: bool = True,
        mlp_expansion_factor: int = 4,
        mask_estimator_depth: int = 2,
        dim_freqs_in: int = 1025,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        # ignored but accepted for config compat
        stft_normalized: bool = False,
        stft_window_fn=None,
        multi_stft_resolution_loss_weight: float = 1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size: int = 147,
        multi_stft_normalized: bool = False,
        multi_stft_window_fn=None,
        linear_transformer_depth: int = 0,
        sage_attention: bool = False,
        zero_dc: bool = True,
        use_torch_checkpoint: bool = False,
        skip_connection: bool = False,
    ):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.num_bands = len(freqs_per_bands)

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in freqs_per_bands
        )

        transformer_kwargs = dict(
            dim=dim, heads=heads, dim_head=dim_head,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            flash_attn=flash_attn, norm_output=False,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(depth=time_transformer_depth,
                            rotary_embed=time_rotary_embed, **transformer_kwargs),
                Transformer(depth=freq_transformer_depth,
                            rotary_embed=freq_rotary_embed, **transformer_kwargs),
            ]))

        self.final_norm = RMSNorm(dim)

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList()
        for _ in range(num_stems):
            self.mask_estimators.append(
                MaskEstimator(
                    dim=dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    depth=mask_estimator_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                )
            )

    def forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stft_repr: [B, F*S, T, 2]  merged-stereo spectrogram (real+imag)
        Returns:
            mask: [B, N, F*S, T, 2]  complex mask per stem
        """
        B, FS, T, C = stft_repr.shape

        x = stft_repr.permute(0, 2, 1, 3).reshape(B, T, FS * C)
        x = self.band_split(x)
        K = x.shape[2]

        for time_transformer, freq_transformer in self.layers:
            x = x.permute(0, 2, 1, 3).reshape(B * K, T, -1)
            x = time_transformer(x)
            x = x.reshape(B, K, T, -1)

            x = x.permute(0, 2, 1, 3).reshape(B * T, K, -1)
            x = freq_transformer(x)
            x = x.reshape(B, T, K, -1)

        x = self.final_norm(x)

        masks = []
        for fn in self.mask_estimators:
            masks.append(fn(x))
        mask = torch.stack(masks, dim=1)

        mask = mask.reshape(B, self.num_stems, T, FS, C)
        mask = mask.permute(0, 1, 3, 2, 4)
        return mask


# ─── Known configs for popular models ───────────────────────────

def detect_config(ckpt_path: str) -> dict:
    """Auto-detect BSRoformer config from checkpoint weights."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state

    dim = sd["final_norm.gamma"].shape[0]

    depth = 0
    while f"layers.{depth}.0.layers.0.0.norm.gamma" in sd:
        depth += 1

    t_depth = 0
    while f"layers.0.0.layers.{t_depth}.0.norm.gamma" in sd:
        t_depth += 1

    f_depth = 0
    while f"layers.0.1.layers.{f_depth}.0.norm.gamma" in sd:
        f_depth += 1

    gates_w = sd["layers.0.0.layers.0.0.to_gates.weight"]
    heads = gates_w.shape[0]
    qkv_w = sd["layers.0.0.layers.0.0.to_qkv.weight"]
    dim_head = qkv_w.shape[0] // (3 * heads)

    stems = 0
    while f"mask_estimators.{stems}.to_freqs.0.0.0.weight" in sd:
        stems += 1

    me_linears = 0
    while f"mask_estimators.0.to_freqs.0.0.{me_linears * 2}.weight" in sd:
        me_linears += 1
    me_depth = max(me_linears - 1, 1)

    freqs = []
    i = 0
    while f"band_split.to_features.{i}.1.weight" in sd:
        band_in = sd[f"band_split.to_features.{i}.1.weight"].shape[1]
        first_band_in = sd["band_split.to_features.0.1.weight"].shape[1]
        audio_ch = 2 if first_band_in == 8 else 1
        freqs.append(band_in // (2 * audio_ch))
        i += 1

    stereo = audio_ch == 2
    ffn_hidden = sd["layers.0.0.layers.0.1.net.1.weight"].shape[0]
    mlp_exp = ffn_hidden // dim

    total_freq = sum(freqs)
    n_fft = (total_freq - 1) * 2

    print(f"  dim={dim}, depth={depth}, stereo={stereo}, stems={stems}")
    print(f"  time_depth={t_depth}, freq_depth={f_depth}, bands={len(freqs)}")
    print(f"  heads={heads}, dim_head={dim_head}, mlp_exp={mlp_exp}, me_depth={me_depth}")
    print(f"  n_fft={n_fft}, freq_bins={total_freq}")

    return dict(
        dim=dim, depth=depth, stereo=stereo, num_stems=stems,
        time_transformer_depth=t_depth, freq_transformer_depth=f_depth,
        freqs_per_bands=tuple(freqs),
        dim_head=dim_head, heads=heads,
        attn_dropout=0.0, ff_dropout=0.0,
        mask_estimator_depth=me_depth, mlp_expansion_factor=mlp_exp,
        stft_n_fft=n_fft, stft_hop_length=441, stft_win_length=n_fft,
    )


def load_from_checkpoint(ckpt_path: str, config: Optional[dict] = None) -> BSRoformer:
    """Load BSRoformer from a .ckpt file, returning inference-ready model."""
    if config is None:
        config = detect_config(ckpt_path)

    model = BSRoformer(**config)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=False)
    model.eval()
    return model
