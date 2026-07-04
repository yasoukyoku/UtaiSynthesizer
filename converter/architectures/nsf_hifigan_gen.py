"""NSF-HiFiGAN standalone vocoder Generator for ONNX export.

FAITHFUL VERBATIM PORT of the original so-vits-svc 4.1-Stable code:
  D:\\MyDev\\so-vits-svc\\so-vits-svc\\vdecoder\\nsf_hifigan\\models.py
(the pretrain/nsf_hifigan checkpoint's own architecture — used by BOTH the
4.1 enhancer (modules/enhancer.py) and the diffusion vocoder
(diffusion/vocoder.py), so this ONE export serves both features).

Shared-lineage modules are REUSED from architectures/rvc_v2.py ONLY where the
original vdecoder/nsf_hifigan code is math-identical to the original RVC code
(diffed 2026-07-04): ResBlock1/ResBlock2 (vdecoder's variants == RVC modules'
with x_mask=None; the `h` first ctor arg of vdecoder's is unused, exactly as
sovits_v4.py already established for vdecoder/hifigan), SourceModuleHnNSF +
SineGen (vdecoder/nsf_hifigan's SineGen is the same %1 cycle-domain
wrap-correction scheme as RVC models.py's — the only difference is that
nsf_hifigan runs BOTH cumsums in fp64 while RVC's second cumsum is fp32;
rvc_v2's stable-phase reformulation supersedes both, see DEVIATION below).

Differences from BOTH rvc_v2.GeneratorNSF and sovits_v4.Generator that this
port preserves (do NOT "unify" — S31 lesson):
  - conv_pre input is num_mels (128 ln-mel bins), NOT a VITS latent;
  - conv_pre AND conv_post are weight-normed, WITH bias (RVC's conv_pre is
    un-normed and its conv_post is bias-free; remove_weight_norm() must strip
    conv_pre/conv_post too, like the original load_model() does);
  - ups padding is (k - u) // 2 and noise_convs padding is stride_f0 // 2 —
    the nsf_hifigan variants (vdecoder/hifigan uses the +1 forms; equal only
    because the shipped configs have even k-u / stride_f0);
  - NO speaker cond layer, NO f0_upsamp (f0 arrives at FRAME level and
    SineGen upsamples in-graph);
  - m_source harmonic_num=8 (9 harmonics), sine_amp=0.1, add_noise_std=0.003,
    voiced_threshold=0 (hard-coded in the original Generator.__init__, not in
    config.json);
  - final leaky_relu uses the torch DEFAULT slope 0.01, in-loop slope is 0.1.

DEVIATION (the one deliberate numerical deviation, gated in
converter/verify/voice/gate1_nsf_hifigan.py): SineGen phase bookkeeping uses
rvc_v2.SineGen's ONNX-stable reformulation (frac of fp64 frame-rate cumsum +
per-sample ramp — a sin-invariant identity, equal to the original phase
modulo whole cycles) instead of the original's verbatim fp64-double-cumsum +
linear-interp + %1 wrap-correction. The verbatim scheme decorrelates under
ORT (S35 measured: 1531 cycles of phase drift over 2 s, sine max_abs_diff
0.98) because the wrap detection amplifies producer/consumer fp rounding
differences into missed/spurious whole-cycle shifts. Measured cost of the
reformulation on sines: ~4.5e-7 (T=200); through the full generator the gate
line is audio max_abs_diff < 5e-4 with corr > 0.9999.

SineGen randomness (rand_ini + uv-gated noise) stays in-graph
(RandomUniformLike / RandomNormalLike) per original semantics; the
`deterministic` flag zeroes both for numerical gate builds
(set_deterministic(), same mechanism as rvc_v2/sovits_v4).

Verified against the ORIGINAL repo code by
converter/verify/voice/gate1_nsf_hifigan.py.
"""

import math

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from .rvc_v2 import (
    ResBlock1,
    ResBlock2,
    SourceModuleHnNSF,  # carries the gated stable-phase SineGen (rvc_v2.py)
    init_weights,
)

# ported from vdecoder/nsf_hifigan/models.py
LRELU_SLOPE = 0.1


class Generator(torch.nn.Module):
    """ported from vdecoder/nsf_hifigan/models.py Generator.
    h is the pretrain config.json dict (AttrDict in the original; plain dict
    access here — export ergonomics only). Parameter names match the
    checkpoint's ckpt['generator'] state_dict exactly (strict=True load)."""

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h["sampling_rate"],
            harmonic_num=8,
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(
            Conv1d(h["num_mels"], h["upsample_initial_channel"], 7, 1, padding=3)
        )
        resblock = ResBlock1 if h["resblock"] == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            c_cur = h["upsample_initial_channel"] // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h["upsample_initial_channel"] // (2 ** i),
                        h["upsample_initial_channel"] // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,  # nsf_hifigan variant — NOT (k-u+1)//2
                    )
                )
            )
            if i + 1 < len(h["upsample_rates"]):
                # math.prod == np.prod for these int lists (export ergonomics:
                # keeps numpy out of the module, same note as rvc_v2.py).
                stride_f0 = math.prod(h["upsample_rates"][i + 1:])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,  # nsf_hifigan variant — NOT (stride_f0+1)//2
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        ch = h["upsample_initial_channel"]
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(
                zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])
            ):
                # rvc_v2.ResBlock1/2 == vdecoder's minus the unused `h` arg
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = math.prod(h["upsample_rates"])

    def forward(self, x, f0):
        # x: [B, num_mels, T] ln-mel (nvSTFT semantics); f0: [B, T] Hz (frame level)
        # rvc_v2.SourceModuleHnNSF returns (sine_merge, None, None); the
        # original nsf_hifigan variant returns sine_merge only — same tensor.
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
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
        # verbatim (incl. conv_pre/conv_post, unlike RVC's GeneratorNSF)
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def set_deterministic(model, deterministic=True):
    """Zero SineGen rand_ini + additive noise for numerical gate builds
    (same mechanism as rvc_v2/sovits_v4 set_deterministic)."""
    model.m_source.l_sin_gen.deterministic = deterministic


REQUIRED_KEYS = (
    "resblock", "upsample_rates", "upsample_kernel_sizes",
    "upsample_initial_channel", "resblock_kernel_sizes",
    "resblock_dilation_sizes", "num_mels", "n_fft", "hop_size", "win_size",
    "sampling_rate", "fmin", "fmax",
)


def build_from_checkpoint(model_path, h):
    """Original load_model() semantics (vdecoder/nsf_hifigan/models.py:17-27):
    Generator(h) -> load_state_dict(ckpt['generator'], strict=True) -> eval()
    -> remove_weight_norm(). h = the config.json dict."""
    missing = [k for k in REQUIRED_KEYS if k not in h]
    if missing:
        raise ValueError(f"NSF-HiFiGAN config.json 缺少字段: {missing}")
    if h["resblock"] not in ("1", "2"):
        raise ValueError(f"NSF-HiFiGAN 不支持的 resblock 类型: {h['resblock']!r}")
    if math.prod(h["upsample_rates"]) != h["hop_size"]:
        raise ValueError(
            f"NSF-HiFiGAN 配置不一致: prod(upsample_rates)="
            f"{math.prod(h['upsample_rates'])} != hop_size={h['hop_size']}"
        )
    generator = Generator(h)
    cp_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if "generator" not in cp_dict:
        raise ValueError("该文件不是 NSF-HiFiGAN 声码器权重（缺少 'generator' 键）")
    generator.load_state_dict(cp_dict["generator"], strict=True)
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    return generator
