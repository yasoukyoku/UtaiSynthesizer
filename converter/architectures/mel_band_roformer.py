"""MelBandRoformer — clean architecture for ONNX export.

Shares backbone with BSRoformer (RMSNorm, RotaryEmbedding, Attention,
Transformer, BandSplit, MaskEstimator). The key difference is mel-scale
band splitting with overlapping frequency bands.

STFT/iSTFT excluded — handled in Rust via rustfft.

Input:  stft_repr [B, F*S, T, 2]  (S=audio_channels, 2=real/imag)
Output: mask      [B, N, F*S, T, 2]  (N=num_stems, averaged over overlapping bands)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .bs_roformer import (
    RMSNorm, RotaryEmbedding, Attend, Attention, FeedForward,
    Transformer, BandSplit, MaskEstimator, _build_mlp,
)


# ─── Mel Filterbank (pure numpy, matches librosa Slaney scale) ─

def hz_to_mel(freq):
    """Slaney scale: linear below 1000 Hz, logarithmic above. Matches librosa default."""
    freq = np.asarray(freq, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    mel = freq / f_sp
    log_mask = freq >= min_log_hz
    mel[log_mask] = min_log_mel + np.log(freq[log_mask] / min_log_hz) / logstep
    return mel

def mel_to_hz(mel):
    """Inverse Slaney scale."""
    mel = np.asarray(mel, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    freq = mel * f_sp
    log_mask = mel >= min_log_mel
    freq[log_mask] = min_log_hz * np.exp(logstep * (mel[log_mask] - min_log_mel))
    return freq

def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Compute mel filterbank [n_mels, n_fft//2+1]. Matches librosa.filters.mel(norm='slaney')."""
    n_freqs = n_fft // 2 + 1
    fmin, fmax = 0.0, float(sr) / 2.0

    mel_min = hz_to_mel(np.array([fmin]))[0]
    mel_max = hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fft_freqs = np.linspace(0, float(sr) / 2, n_freqs)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]

        up_slope = (fft_freqs - lower) / max(center - lower, 1e-10)
        down_slope = (upper - fft_freqs) / max(upper - center, 1e-10)
        weights[i] = np.maximum(0.0, np.minimum(up_slope, down_slope))

    # Slaney normalization
    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights.astype(np.float32)


# ─── MelBandRoformer ──────────────────────────────────────────

class MelBandRoformer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = False,
        num_stems: int = 1,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        num_bands: int = 60,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        flash_attn: bool = True,
        mlp_expansion_factor: int = 4,
        mask_estimator_depth: int = 1,
        dim_freqs_in: int = 1025,
        sample_rate: int = 44100,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 441,
        stft_win_length: int = 2048,
        linear_transformer_depth: int = 0,
        # Accepted for config compat, not used in inference-only model
        stft_normalized: bool = False,
        stft_window_fn=None,
        multi_stft_resolution_loss_weight: float = 1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size: int = 147,
        multi_stft_normalized: bool = False,
        multi_stft_window_fn=None,
        match_input_audio_length: bool = False,
        use_torch_checkpoint: bool = False,
        skip_connection: bool = False,
    ):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        freqs = dim_freqs_in  # n_fft // 2 + 1

        # Build mel filterbank
        mel_fb_np = mel_filterbank(sample_rate, stft_n_fft, num_bands)
        mel_fb = torch.from_numpy(mel_fb_np)
        mel_fb[0][0] = 1.0
        mel_fb[-1, -1] = 1.0

        freqs_per_band = mel_fb > 0  # [num_bands, freqs]
        assert freqs_per_band.any(dim=0).all(), "not all frequencies covered by mel bands"

        # Build frequency index mapping
        repeated_freq_indices = torch.arange(freqs).unsqueeze(0).expand(num_bands, -1)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = freq_indices.unsqueeze(1).expand(-1, 2)
            offsets = torch.arange(2).unsqueeze(0)
            freq_indices = freq_indices * 2 + offsets
            freq_indices = freq_indices.reshape(-1)

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = freqs_per_band.sum(dim=1)  # [num_bands]

        # Count how many bands cover each frequency (for averaging)
        num_bands_per_freq = freqs_per_band.sum(dim=0)  # [freqs]
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        # Pre-compute reconstruction matrix for ONNX-friendly matmul
        # reconstruct_matrix: [F*S, total_mel_freqs]
        fs = freqs * self.audio_channels
        total_mel = freq_indices.shape[0]
        recon = torch.zeros(fs, total_mel)
        denom_full = num_bands_per_freq.float()
        if stereo:
            denom_full = denom_full.repeat_interleave(2)
        for i in range(total_mel):
            f_idx = freq_indices[i].item()
            recon[f_idx, i] = 1.0 / max(denom_full[f_idx].item(), 1e-8)
        self.register_buffer("reconstruct_matrix", recon, persistent=False)

        # Band dimensions (with complex: *2 for real+imag, *audio_channels for stereo)
        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in num_freqs_per_band.tolist()
        )

        # Shared architecture with BSRoformer
        # MelBandRoformer uses norm_output=True (each Transformer has its own output norm)
        transformer_kwargs = dict(
            dim=dim, heads=heads, dim_head=dim_head,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            flash_attn=flash_attn, norm_output=True,
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
            mask: [B, N, F*S, T, 2]  complex mask per stem (averaged over overlapping bands)
        """
        B, FS, T, C = stft_repr.shape

        # Gather frequencies by mel band ordering
        x = stft_repr[:, self.freq_indices]  # [B, total_mel, T, 2]
        total_mel = x.shape[1]

        # Fold complex into features: [B, T, total_mel * 2]
        x = x.permute(0, 2, 1, 3).reshape(B, T, total_mel * C)
        x = self.band_split(x)  # [B, T, num_bands, dim]
        K = x.shape[2]

        # Time/freq transformers (same as BSRoformer)
        for time_transformer, freq_transformer in self.layers:
            x = x.permute(0, 2, 1, 3).reshape(B * K, T, -1)
            x = time_transformer(x)
            x = x.reshape(B, K, T, -1)

            x = x.permute(0, 2, 1, 3).reshape(B * T, K, -1)
            x = freq_transformer(x)
            x = x.reshape(B, T, K, -1)

        # Mask estimation in mel-band space
        masks = []
        for fn in self.mask_estimators:
            masks.append(fn(x))  # [B, T, total_mel * 2]
        mask = torch.stack(masks, dim=1)  # [B, N, T, total_mel * 2]
        mask = mask.reshape(B, self.num_stems, T, total_mel, C)  # [B, N, T, total_mel, 2]

        # Reconstruct full frequency mask via matmul (scatter_add + average)
        # masks shape: [B, N, T, total_mel, 2] → need [B, N, F*S, T, 2]
        mask_real = mask[..., 0].permute(0, 1, 3, 2)  # [B, N, total_mel, T]
        mask_imag = mask[..., 1].permute(0, 1, 3, 2)  # [B, N, total_mel, T]

        # reconstruct_matrix: [F*S, total_mel] @ [B, N, total_mel, T] → [B, N, F*S, T]
        avg_real = torch.matmul(self.reconstruct_matrix, mask_real)
        avg_imag = torch.matmul(self.reconstruct_matrix, mask_imag)

        # Stack real/imag → [B, N, F*S, T, 2]
        result = torch.stack([avg_real, avg_imag], dim=-1)
        return result


# ─── Config detection ─────────────────────────────────────────

def detect_config(ckpt_path: str, yaml_path: Optional[str] = None) -> dict:
    """Auto-detect MelBandRoformer config from checkpoint + optional YAML."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state

    # MelBandRoformer has no final_norm — dim comes from band_split output
    dim = sd["band_split.to_features.0.1.weight"].shape[0]

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

    num_bands_detected = 0
    while f"band_split.to_features.{num_bands_detected}.1.weight" in sd:
        num_bands_detected += 1

    first_band_in = sd["band_split.to_features.0.1.weight"].shape[1]
    audio_ch = 2 if first_band_in > 4 else 1
    stereo = audio_ch == 2

    ffn_hidden = sd["layers.0.0.layers.0.1.net.1.weight"].shape[0]
    mlp_exp = ffn_hidden // dim

    # Try to load YAML for STFT params and num_bands
    yaml_config = {}
    if yaml_path:
        try:
            import yaml
            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: could not load YAML config: {e}")

    model_yaml = yaml_config.get("model", {})
    audio_yaml = yaml_config.get("audio", {})

    num_bands = model_yaml.get("num_bands", num_bands_detected)
    sample_rate = model_yaml.get("sample_rate", audio_yaml.get("sample_rate", 44100))
    n_fft = model_yaml.get("stft_n_fft", audio_yaml.get("n_fft", 2048))
    hop_length = model_yaml.get("stft_hop_length", audio_yaml.get("hop_length", 441))
    win_length = model_yaml.get("stft_win_length", n_fft)
    dim_freqs_in = n_fft // 2 + 1

    print(f"  dim={dim}, depth={depth}, stereo={stereo}, stems={stems}")
    print(f"  time_depth={t_depth}, freq_depth={f_depth}, bands={num_bands}")
    print(f"  heads={heads}, dim_head={dim_head}, mlp_exp={mlp_exp}, me_depth={me_depth}")
    print(f"  n_fft={n_fft}, hop={hop_length}, sample_rate={sample_rate}")

    return dict(
        dim=dim, depth=depth, stereo=stereo, num_stems=stems,
        time_transformer_depth=t_depth, freq_transformer_depth=f_depth,
        num_bands=num_bands, dim_head=dim_head, heads=heads,
        attn_dropout=0.0, ff_dropout=0.0,
        mask_estimator_depth=me_depth, mlp_expansion_factor=mlp_exp,
        dim_freqs_in=dim_freqs_in, sample_rate=sample_rate,
        stft_n_fft=n_fft, stft_hop_length=hop_length, stft_win_length=win_length,
    )


def load_from_checkpoint(ckpt_path: str, config: Optional[dict] = None,
                         yaml_path: Optional[str] = None) -> MelBandRoformer:
    """Load MelBandRoformer from a .ckpt file, returning inference-ready model."""
    if config is None:
        config = detect_config(ckpt_path, yaml_path)

    model = MelBandRoformer(**config)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=False)
    model.eval()
    return model
