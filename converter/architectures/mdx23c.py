"""MDX23C (TFC-TDF-v3 UNet) — clean architecture for ONNX export.

Conv2d U-Net with TFC (Time-Frequency Convolution) and TDF (Time-Distributed
Fully-connected bottleneck) blocks. Used in UVR5/MSST for source separation.

STFT/iSTFT excluded — handled in Rust via rustfft.

Input:  stft_repr [B, 4, dim_f, T]  (CaC: real_L, imag_L, real_R, imag_R)
Output: separated [B, N, 4, dim_f, T]  (N=num_stems, direct spectrogram output)
"""

import torch
import torch.nn as nn
from typing import Optional, List


# ─── Normalization / Activation ──────────────────────────────────

def make_norm(norm_type: Optional[str]):
    if norm_type is None:
        return lambda c: nn.Identity()
    elif norm_type == "BatchNorm":
        return nn.BatchNorm2d
    elif norm_type == "InstanceNorm":
        return lambda c: nn.InstanceNorm2d(c, affine=True)
    elif norm_type.startswith("GroupNorm"):
        g = int(norm_type.replace("GroupNorm", ""))
        return lambda c: nn.GroupNorm(num_groups=g, num_channels=c)
    return lambda c: nn.Identity()


def make_act(act_type: str):
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type.startswith("elu"):
        alpha = float(act_type.replace("elu", ""))
        return nn.ELU(alpha)
    raise ValueError(f"Unknown activation: {act_type}")


# ─── Modules ─────────────────────────────────────────────────────

class Downscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act_type):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c), make_act(act_type),
            nn.Conv2d(in_c, out_c, kernel_size=scale, stride=scale, bias=False),
        )

    def forward(self, x):
        return self.conv(x)


class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act_type):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c), make_act(act_type),
            nn.ConvTranspose2d(in_c, out_c, kernel_size=scale, stride=scale, bias=False),
        )

    def forward(self, x):
        return self.conv(x)


class TFC_TDF(nn.Module):
    """Time-Frequency Convolution + Time-Distributed Fully-connected block."""

    def __init__(self, in_c, c, l, f, bn, norm, act_type):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(l):
            block = nn.Module()
            block.tfc1 = nn.Sequential(
                norm(in_c), make_act(act_type),
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(c), make_act(act_type),
                nn.Linear(f, f // bn, bias=False),
                norm(c), make_act(act_type),
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(c), make_act(act_type),
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)
            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


class TFC_TDF_net(nn.Module):
    """MDX23C core network. STFT excluded — input/output in CaC format."""

    def __init__(
        self,
        *,
        dim_f: int,
        n_fft: int = 8192,
        hop_length: int = 1024,
        num_channels: int = 128,
        num_subbands: int = 4,
        num_scales: int = 5,
        growth: int = 128,
        num_blocks_per_scale: int = 2,
        bottleneck_factor: int = 4,
        scale: List[int] = [2, 2],
        num_target_instruments: int = 1,
        norm_type: str = "InstanceNorm",
        act_type: str = "gelu",
    ):
        super().__init__()
        self.num_subbands = num_subbands
        self.num_target_instruments = num_target_instruments

        norm = make_norm(norm_type)
        dim_c = num_subbands * 2 * 2  # subbands * audio_channels * (real + imag)
        c = num_channels
        l = num_blocks_per_scale
        bn = bottleneck_factor
        f = dim_f // num_subbands

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.encoder_blocks = nn.ModuleList()
        for _ in range(num_scales):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF(c, c, l, f, bn, norm, act_type)
            block.downscale = Downscale(c, c + growth, scale, norm, act_type)
            f = f // scale[1]
            c += growth
            self.encoder_blocks.append(block)

        self.bottleneck_block = TFC_TDF(c, c, l, f, bn, norm, act_type)

        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_scales):
            block = nn.Module()
            block.upscale = Upscale(c, c - growth, scale, norm, act_type)
            f = f * scale[1]
            c -= growth
            block.tfc_tdf = TFC_TDF(2 * c, c, l, f, bn, norm, act_type)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            make_act(act_type),
            nn.Conv2d(c, num_target_instruments * dim_c, 1, 1, 0, bias=False),
        )

    def cac2cws(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        return x.reshape(b, c * k, f // k, t)

    def cws2cac(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        return x.reshape(b, c // k, f * k, t)

    def forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stft_repr: [B, 4, dim_f, T]  CaC stereo spectrogram
        Returns:
            separated: [B, N, 4, dim_f, T]  direct spectrogram per stem
        """
        x = self.cac2cws(stft_repr)
        mix = x

        first_conv_out = x = self.first_conv(x)
        x = x.transpose(-1, -2)

        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)

        x = self.bottleneck_block(x)

        for block in self.decoder_blocks:
            x = block.upscale(x)
            x = torch.cat([x, encoder_outputs.pop()], 1)
            x = block.tfc_tdf(x)

        x = x.transpose(-1, -2)
        x = x * first_conv_out
        x = self.final_conv(torch.cat([mix, x], 1))
        x = self.cws2cac(x)

        b, c, f, t = x.shape
        x = x.reshape(b, self.num_target_instruments, -1, f, t)
        return x


# ─── Config detection ─────────────────────────────────────────

def detect_config(ckpt_path: str, yaml_path: Optional[str] = None) -> dict:
    """Auto-detect MDX23C config from checkpoint weights + optional YAML."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state

    # Channel dims from first_conv
    dim_c = sd["first_conv.weight"].shape[1]
    num_subbands = dim_c // 4
    c = sd["first_conv.weight"].shape[0]

    # Count encoder stages
    num_scales = 0
    while f"encoder_blocks.{num_scales}.tfc_tdf.blocks.0.shortcut.weight" in sd:
        num_scales += 1

    # Channel growth
    growth = sd["encoder_blocks.0.downscale.conv.2.weight"].shape[0] - c

    # Blocks per scale
    l = 0
    while f"encoder_blocks.0.tfc_tdf.blocks.{l}.shortcut.weight" in sd:
        l += 1

    # Frequency dim and bottleneck factor from TDF linear
    f0 = sd["encoder_blocks.0.tfc_tdf.blocks.0.tdf.2.weight"].shape[1]
    f_bn = sd["encoder_blocks.0.tfc_tdf.blocks.0.tdf.2.weight"].shape[0]
    bn = f0 // f_bn
    dim_f = f0 * num_subbands

    # Scale from downscale kernel
    scale = list(sd["encoder_blocks.0.downscale.conv.2.weight"].shape[2:])

    # Num stems from final_conv output
    final_out = sd["final_conv.2.weight"].shape[0]
    num_stems = final_out // dim_c

    # Norm type
    has_running_mean = any("running_mean" in k for k in sd.keys())
    norm_type = "BatchNorm" if has_running_mean else "InstanceNorm"

    # YAML overrides for STFT params (not detectable from weights)
    yaml_config = {}
    if yaml_path:
        try:
            import yaml
            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: could not load YAML config: {e}")

    audio_yaml = yaml_config.get("audio", {})
    model_yaml = yaml_config.get("model", {})
    training_yaml = yaml_config.get("training", {})

    n_fft = audio_yaml.get("n_fft", dim_f * 2)
    hop_length = audio_yaml.get("hop_length", n_fft // 8)
    sample_rate = audio_yaml.get("sample_rate", 44100)

    act_type = model_yaml.get("act", "gelu")

    print(f"  dim_f={dim_f}, channels={c}, subbands={num_subbands}, stems={num_stems}")
    print(f"  scales={num_scales}, blocks={l}, growth={growth}, bn={bn}")
    print(f"  scale={scale}, norm={norm_type}, act={act_type}")
    print(f"  n_fft={n_fft}, hop={hop_length}, sr={sample_rate}")

    return dict(
        dim_f=dim_f, n_fft=n_fft, hop_length=hop_length,
        num_channels=c, num_subbands=num_subbands,
        num_scales=num_scales, growth=growth,
        num_blocks_per_scale=l, bottleneck_factor=bn,
        scale=scale, num_target_instruments=num_stems,
        norm_type=norm_type, act_type=act_type,
        sample_rate=sample_rate,
    )


def load_from_checkpoint(ckpt_path: str, config: Optional[dict] = None,
                         yaml_path: Optional[str] = None) -> TFC_TDF_net:
    """Load MDX23C from a .ckpt file, returning inference-ready model."""
    if config is None:
        config = detect_config(ckpt_path, yaml_path)

    model_config = {k: v for k, v in config.items() if k != "sample_rate"}
    model = TFC_TDF_net(**model_config)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (expected if STFT-related)")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()
    return model
