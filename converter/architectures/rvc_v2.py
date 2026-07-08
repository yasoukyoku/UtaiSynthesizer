"""RVC synthesizer (v1 256-dim / v2 768-dim, f0 variants only) for ONNX export.

FAITHFUL VERBATIM PORT of the original RVC 20240604 infer_pack code:
  D:\\MyDev\\RVC\\RVC20240604Nvidia\\infer\\lib\\infer_pack\\{models,models_onnx,
  attentions,modules,commons}.py
Every module below carries a "ported from" note. Only device/dtype/export
ergonomics were adapted (each adaptation is commented at the spot); the tensor
math is the original's, line for line. Do NOT "simplify" any of it — the 2026-07
audit found the previous hand reconstruction of this file had 8 classes of
numerical bugs (pre-norm vs post-norm, missing attention skew, dropped WN skip
accumulation, missing trailing Flip, missing sqrt(hidden) scaling, wrong NSF
source semantics, unbounded phase cumsum, wrong noise_conv padding).

Export contract (matches the official models_onnx.py SynthesizerTrnMsNSFsidM /
onnx_inference.py, which runs dynamic-T in production):
  inputs : phone[1,T,dim] f32, phone_lengths[1] i64, pitch[1,T] i64,
           pitchf[1,T] f32, sid[1] i64, rnd[1,inter_channels,T] f32
  output : audio[1,1,L]  (L = T * prod(upsample_rates), 10 ms per frame)
  z_p = (m_p + exp(logs_p) * rnd) * x_mask — the CALLER pre-multiplies its
  noise by noise_scale (original default 0.66666) before feeding rnd.
SineGen noise stays in-graph (RandomNormalLike) per original semantics; the
`deterministic` flag zeroes SineGen noise + rand_ini for numerical gate builds.

nof0 checkpoints are refused (build_from_checkpoint raises ValueError with a
Chinese user-facing message).

Verified against the ORIGINAL repo code by converter/verify/voice/gate1_rvc.py.
"""

import math

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

# ported from modules.py
LRELU_SLOPE = 0.1

# ported from models.py
sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


# ---------------------------------------------------------------------------
# commons.py — verbatim helpers
# ---------------------------------------------------------------------------

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    # verbatim from commons.py; the @torch.jit.script decorator is dropped
    # (export ergonomics only — tracing inlines the plain function; numerics
    # are identical).
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


# ---------------------------------------------------------------------------
# modules.py — verbatim
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """ported from modules.py LayerNorm (channel-dim layer norm via transpose)."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class WN(torch.nn.Module):
    """ported from modules.py WN — non-causal WaveNet with the skip-accumulation
    output (output += res_skip[:, hidden:]) the old reconstruction dropped."""

    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels=0, p_dropout=0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = float(p_dropout)

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(float(p_dropout))

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(
            zip(self.in_layers, self.res_skip_layers)
        ):
            x_in = in_layer(x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = res_skip_layer(acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


class ResBlock1(torch.nn.Module):
    """ported from modules.py ResBlock1 (HiFi-GAN MRF block, convs1 dilated +
    convs2 undilated, leaky_relu 0.1 before each conv)."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1,
                           dilation=dilation[0],
                           padding=get_padding(kernel_size, dilation[0]))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1,
                           dilation=dilation[1],
                           padding=get_padding(kernel_size, dilation[1]))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1,
                           dilation=dilation[2],
                           padding=get_padding(kernel_size, dilation[2]))
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1,
                           padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1,
                           padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1,
                           padding=get_padding(kernel_size, 1))
                ),
            ]
        )
        self.convs2.apply(init_weights)
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    """ported from modules.py ResBlock2 (resblock="2" variant, 2 dilated convs)."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1,
                           dilation=dilation[0],
                           padding=get_padding(kernel_size, dilation[0]))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1,
                           dilation=dilation[1],
                           padding=get_padding(kernel_size, dilation[1]))
                ),
            ]
        )
        self.convs.apply(init_weights)
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
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


class Flip(nn.Module):
    """ported from modules.py Flip."""

    def forward(self, x, x_mask, g=None, reverse=False):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x, torch.zeros([1], device=x.device)


class ResidualCouplingLayer(nn.Module):
    """ported from modules.py ResidualCouplingLayer (mean_only affine coupling)."""

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, p_dropout=0, gin_channels=0, mean_only=False):
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
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=float(p_dropout),
            gin_channels=gin_channels,
        )
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
            return x, torch.zeros([1])

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


# ---------------------------------------------------------------------------
# attentions.py — verbatim (post-norm Encoder + FULL relative-position MHA)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """ported from attentions.py MultiHeadAttention — includes the complete
    relative-position machinery: shared emb_rel_k/v of size 2*window+1 scaled
    by 1/sqrt(k_channels), query-side _matmul_with_relative_keys with the
    query/sqrt(d) scaling, the rel→abs skew and its inverse, and the
    value-side relative path."""

    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0,
                 window_size=None, heads_share=True, block_length=None,
                 proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, _ = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        # Pad first before slice to avoid using cond ops.
        # NOTE (export): the max()/if choices below get BAKED at trace time.
        # Exporting with a dummy T >= window_size + 2 bakes the generic
        # (pad_length > 0, slice_start == 0) path, which is then correct for
        # every runtime T >= window_size + 2 — hence "min_frames" in the
        # sidecar json.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                [0, 0, pad_length, pad_length, 0, 0],
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, [0, length - 1, 0, 0, 0, 0])

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1:
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    """ported from attentions.py FFN (unpadded convs + explicit same/causal
    padding helpers)."""

    def __init__(self, in_channels, out_channels, filter_channels, kernel_size,
                 p_dropout=0.0, activation=None, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.is_activation = True if activation == "gelu" else False

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def padding(self, x, x_mask):
        if self.causal:
            padding = self._causal_padding(x * x_mask)
        else:
            padding = self._same_padding(x * x_mask)
        return padding

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x, x_mask))
        if self.is_activation:
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)

        x = self.conv_2(self.padding(x, x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        x = F.pad(x, [pad_l, pad_r, 0, 0, 0, 0])
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        x = F.pad(x, [pad_l, pad_r, 0, 0, 0, 0])
        return x


class Encoder(nn.Module):
    """ported from attentions.py Encoder — POST-norm transformer:
    x = norm(x + attn(x)); x = norm(x + ffn(x))."""

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size=1, p_dropout=0.0, window_size=10, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = int(n_layers)
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        zippep = zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        )
        for attn_layers, norm_layers_1, ffn_layers, norm_layers_2 in zippep:
            y = attn_layers(x, x, attn_mask)
            y = self.drop(y)
            x = norm_layers_1(x + y)

            y = ffn_layers(x, x_mask)
            y = self.drop(y)
            x = norm_layers_2(x + y)
        x = x * x_mask
        return x


# ---------------------------------------------------------------------------
# models.py — verbatim
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """ported from models.py TextEncoder (the 20240604 unified 256/768 class;
    same parameter names as the older TextEncoder256/768). Includes the
    * sqrt(hidden_channels) embedding scale + LeakyReLU(0.1) the old
    reconstruction was missing. skip_head (realtime-only) dropped."""

    def __init__(self, in_channels, out_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, f0=True):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0 == True:  # noqa: E712 — verbatim
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    """ported from models.py ResidualCouplingBlock — a Flip follows EVERY
    coupling layer (8 modules for n_flows=4), including the trailing one the
    old reconstruction dropped."""

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class SineGen(torch.nn.Module):
    """ported from models.py SineGen (numerically identical to the
    models_onnx.py one) — sine_amp=0.1, uv-gated noise (voiced noise_std=0.003,
    unvoiced sine_amp/3), rand_ini, and the %1 cycle-domain cumsum
    wrap-correction scheme that keeps the phase argument bounded.

    `deterministic` (export/gate flag, ours): zeroes rand_ini + the additive
    noise so gate builds are reproducible; the shipping graph keeps the
    original in-graph randomness (RandomNormalLike).

    ONE deliberate numerical deviation (the ONNX-safe formulation the models.py
    /models_onnx.py duo does not provide): the PHASE bookkeeping. See the
    comment block in forward() — the sine value is the original's modulo whole
    cycles (sin-invariant identity), measured 4.5e-7 (T=200) / 1.3e-6 (T=2000)
    from verbatim fp32 torch, but it no longer explodes under a backend whose
    fp32 rounding differs from torch's (ORT: 1531 cycles of phase drift in 2 s
    of audio, sine max_abs_diff 0.98 — measured on the verbatim scheme)."""

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0, flag_for_pulse=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.deterministic = False

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def forward(self, f0, upp):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in range(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (
                    idx + 2
                )  # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
            rad_values = (
                f0_buf / self.sampling_rate
            ) % 1  ###%1意味着n_har的乘积无法后处理优化
            if self.deterministic:
                rand_ini = torch.zeros(
                    f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device
                )
            else:
                rand_ini = torch.rand(
                    f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device
                )
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
            # --- PHASE: stable reformulation of the original wrap-correction
            # scheme (the one deliberate deviation from verbatim; gated).
            # Original: tmp_over_one = cumsum(rad)*upp → linear-interp → %1;
            # wrap detection (diff < 0) builds cumsum_shift ∈ {0,-1}; phase =
            # cumsum(rad_up + cumsum_shift) over SAMPLES. The corrections keep
            # the fp32 phase bounded ONLY while producer and consumer round
            # bit-identically: under ORT the frame cumsum rounds a few ULP
            # differently, the %1-of-large-values wrap detection amplifies
            # that into missed/spurious ±1-cycle shifts, the phase grows
            # unbounded (measured: 1531 cycles over 2 s) and fp32 sin()
            # decorrelates completely (max_abs_diff 0.98 vs torch).
            # Identity used instead:
            #   phase(t, j) = frac(Σ_{t'<t} rad[t']·upp) + rad[t]·(j+1),
            #   j = 0..upp-1
            # — equal to the original phase modulo whole cycles (sin-
            # invariant), with the frame-level accumulation in fp64 so frac()
            # stays exact for hours of audio, and a per-sample argument that
            # stays bounded (< 1 + rad·upp). Frame-level fp64 ops are a
            # [B,T,dim]-sized cost; EPs without fp64 kernels partition those
            # few nodes back to CPU. Measured (converter/verify/voice):
            # 4.5e-7 @T=200 / 1.3e-6 @T=2000 vs verbatim fp32 torch, and
            # strictly closer to the fp64-exact reference at every length.
            cyc = rad_values.double() * upp  # cycles contributed per frame
            start = torch.cumsum(cyc, 1) - cyc  # frame-START phase (exclusive)
            start = start - torch.floor(start)  # frac(); Floor beats fp64 Mod for EP support
            start_up = F.interpolate(
                start.float().transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            rad_values = F.interpolate(
                rad_values.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            # within-frame sample index 1..upp (int64 cumsum: exact for any
            # audio length; fp32 ones-cumsum would break past 2**24 samples)
            within = torch.cumsum(torch.ones_like(rad_values, dtype=torch.int64), 1)
            within = ((within - 1) % upp + 1).to(rad_values.dtype)
            phase = start_up + rad_values * within
            sine_waves = torch.sin(phase * 2 * math.pi)
            sine_waves = sine_waves * self.sine_amp
            uv = self._f02uv(f0)
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            if self.deterministic:
                noise = torch.zeros_like(sine_waves)
            else:
                noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """ported from models.py SourceModuleHnNSF — l_linear merge + l_tanh."""

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0, is_half=False):
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp=1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None  # noise, uv


class GeneratorNSF(torch.nn.Module):
    """ported from models.py / models_onnx.py GeneratorNSF — NSF HiFi-GAN.
    noise_convs[i]: kernel=2*stride_f0, stride=stride_f0, padding=stride_f0//2
    (stride_f0 = prod(upsample_rates[i+1:])), NOT weight-normed; the final
    leaky_relu uses the torch DEFAULT slope (0.01) exactly as the original."""

    def __init__(self, initial_channel, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                 upsample_kernel_sizes, gin_channels, sr, is_half=False):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # math.prod == np.prod for these int lists (export ergonomics: keeps
        # numpy out of the module).
        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr, harmonic_num=0, is_half=is_half
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)

    def forward(self, x, f0, g=None):
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

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
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SynthesizerTrnMsNSFsidM(nn.Module):
    """Top-level inference/export model, ported from models_onnx.py
    SynthesizerTrnMsNSFsidM (the official ONNX-adapted variant), with:
      - the speaker_map mixing machinery dropped (we export plain sid lookup,
        g = emb_g(sid).unsqueeze(-1) as in models.py infer());
      - no enc_q (RVC checkpoints strip it, so strict=True load works);
      - z_p = (m_p + exp(logs_p) * rnd) * x_mask with rnd as a graph INPUT —
        the caller pre-multiplies its N(0,1) noise by noise_scale (original
        infer() hardcodes 0.66666).
    Parameter names match the checkpoint state_dict exactly.
    """

    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock,
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim,
                 gin_channels, sr, version="v2", is_half=False, **kwargs):
        super().__init__()
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.version = version

        self.enc_p = TextEncoder(
            256 if version == "v1" else 768,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            is_half=is_half,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        # ①c: convert.py flips this True for a GENUINE multi-speaker export (the .pth carries a
        # `speakers` name list, len > 1) — then `sid` is fed as a spk_mix [1, n_spk] f32 blend
        # (matmul emb_g.weight) instead of a scalar id. Single-speaker exports keep the scalar-id
        # gather → byte-identical. Mirrors sovits_v4.py's proven branch.
        self.export_spk_mix = False

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

    def forward(self, phone, phone_lengths, pitch, pitchf, sid, rnd):
        """
        phone: [1, T, 256|768] f32 — HuBERT/ContentVec features (10 ms frames)
        phone_lengths: [1] i64 — valid length (== T for whole-utterance runs)
        pitch: [1, T] i64 — coarse pitch 1..255 (f0_to_coarse), 0 pad
        pitchf: [1, T] f32 — continuous F0 in Hz, 0 = unvoiced
        sid: [1] i64 — speaker id
        rnd: [1, inter_channels, T] f32 — caller-supplied noise, ALREADY scaled
             by noise_scale (feed zeros for deterministic output)
        returns audio [1, 1, T * upp]
        """
        if self.export_spk_mix:
            # ①c: `sid` is a spk_mix [1, n_spk] f32 blend of the emb_g rows; a one-hot row is
            # bit-identical to emb_g(id). matmul([1,n_spk] @ [n_spk,gin]) = [1,gin] → [1,gin,1].
            g = torch.matmul(sid, self.emb_g.weight).unsqueeze(-1)  # [1, gin, 1]
        else:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, gin, 1]
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * rnd) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, pitchf, g=g)
        return o


def set_deterministic(model, deterministic=True):
    """Zero SineGen's in-graph randomness (rand_ini + additive noise) for
    reproducible gate builds. Shipping exports keep it False (original
    semantics: RandomNormalLike stays in the graph)."""
    model.dec.m_source.l_sin_gen.deterministic = deterministic


def build_from_checkpoint(checkpoint, deterministic=False):
    """Build the RVC synthesizer from a loaded .pth checkpoint dict and load
    weights strict=True. Returns the model ready for ONNX export
    (weight norm removed, eval mode)."""
    version = checkpoint.get("version", "v1")
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown RVC version tag: {version!r}")
    if checkpoint.get("f0", 1) != 1:
        raise ValueError("暂不支持无音高(nof0)的 RVC 模型")

    weights = checkpoint["weight"]

    # Cross-check the version tag against the actual emb_phone input dim —
    # a mislabeled checkpoint must fail loudly here, not as garbage audio.
    in_dim = weights["enc_p.emb_phone.weight"].shape[1]
    expected = 256 if version == "v1" else 768
    if in_dim != expected:
        raise ValueError(
            f"checkpoint says version={version} (features_dim {expected}) but "
            f"enc_p.emb_phone expects {in_dim}-dim input — mislabeled model?"
        )

    config = list(checkpoint["config"])
    # Official loader patch (infer/modules/vc/modules.py): checkpoints lie
    # about the speaker count; the truth is emb_g.weight's row count.
    config[-3] = weights["emb_g.weight"].shape[0]

    model = SynthesizerTrnMsNSFsidM(*config, version=version, is_half=False)

    state_dict_f32 = {
        k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in weights.items()
    }
    model.load_state_dict(state_dict_f32, strict=True)

    # Fuses weight_g/weight_v -> weight; bit-identical to the pre-forward hook.
    model.remove_weight_norm()
    model.eval()
    set_deterministic(model, deterministic)
    return model


def remove_all_weight_norm(module):
    """Recursively remove weight normalization from all sub-modules.
    (Generic helper; kept because architectures/sovits_v4.py imports it.)"""
    for name, child in module.named_children():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except ValueError:
            pass
        remove_all_weight_norm(child)


# NOTE: the shared classes use the ORIGINAL constructor signatures
# (e.g. Encoder(..., kernel_size, p_dropout, window_size)) — old call sites
# passing the old positional orders will fail loudly, not silently.
