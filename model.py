"""与论文对应的 AIO-FE、CSAF、PA-RoPE、DuET 和 ReSFFormer 模块。"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from torchvision.ops import DeformConv2d
except (ImportError, RuntimeError) as exc:  # 不允许静默替换为普通卷积
    raise ImportError(
        "This implementation requires torchvision.ops.DeformConv2d; install a "
        "torchvision build compatible with the installed PyTorch build."
    ) from exc


class PreActivationResidual1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.network(values)


class CausalResidual1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=self.padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=self.padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def _crop(self, values: Tensor) -> Tensor:
        return values[..., :-self.padding] if self.padding else values

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        values = self._crop(self.conv1(F.relu(self.norm1(values))))
        values = self._crop(self.conv2(self.dropout(F.relu(self.norm2(values)))))
        return residual + values


class ConvBranch(nn.Module):
    def __init__(self, input_dim: int, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=False),
            PreActivationResidual1d(channels, kernel_size=3, dropout=dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class TCNBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        dilations: Sequence[int],
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        layers = [nn.Conv1d(input_dim, channels, kernel_size=1)]
        layers.extend(CausalResidual1d(channels, kernel_size, dilation, dropout) for dilation in dilations)
        # 最后一层对应图 2 中每个分支后连接的预激活残差块。
        layers.append(PreActivationResidual1d(channels, kernel_size=3, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class AttentionPool(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, values: Tensor, return_weights: bool = False):
        weights = torch.softmax(self.score(values), dim=1)
        pooled = torch.sum(weights * values, dim=1)
        return (pooled, weights) if return_weights else pooled


class AIOFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 256,
        branch_channels: int = 64,
        d_ff: int = 1024,
        tcn_kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if branch_channels * 4 != d_model:
            raise ValueError("Four AIO-FE branches must concatenate to d_model")
        self.input_norm = nn.LayerNorm(input_dim)
        self.branches = nn.ModuleList(
            [
                ConvBranch(input_dim, branch_channels, 3, dropout),
                ConvBranch(input_dim, branch_channels, 5, dropout),
                TCNBranch(input_dim, branch_channels, (1, 2, 4), tcn_kernel_size, dropout),
                TCNBranch(input_dim, branch_channels, (2, 4, 8), tcn_kernel_size, dropout),
            ]
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.pool = AttentionPool(d_model)

    def forward(self, values: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        values = self.input_norm(values).transpose(1, 2)
        sequence = torch.cat([branch(values) for branch in self.branches], dim=1).transpose(1, 2)
        sequence = sequence + self.ffn(self.ffn_norm(sequence))
        pooled, weights = self.pool(sequence, return_weights=True)
        return sequence, pooled, weights


class DeformableConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.offset = nn.Conv2d(in_channels, 18, kernel_size=3, stride=2, padding=1)
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        groups = 8 if out_channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, out_channels)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, values: Tensor) -> Tensor:
        return F.relu(self.norm(self.conv(values, self.offset(values))))


class HeatmapFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, d_model: int = 256):
        super().__init__()
        self.layer1 = DeformableConvBlock(in_channels, hidden_channels)
        self.layer2 = DeformableConvBlock(hidden_channels, d_model)

    def forward(self, heatmap: Tensor) -> Tensor:
        values = self.layer2(self.layer1(heatmap))
        return F.adaptive_avg_pool2d(values, output_size=1).flatten(1)


class CSAF(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gamma_projection = nn.Linear(d_model, d_model)
        self.beta_projection = nn.Linear(d_model, d_model)

    def forward(self, f_sbm: Tensor, f_h: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        gamma = torch.sigmoid(self.gamma_projection(f_h))
        beta = self.beta_projection(f_h)
        fused = (1.0 + gamma) * f_sbm + beta
        return fused, gamma, beta


class PhaseAlignedRoPE(nn.Module):
    def __init__(self, max_length: int, head_dim: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("PA-RoPE requires an even attention head dimension")
        self.phase_offset = nn.Parameter(torch.zeros(max_length))
        inverse_frequency = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    @staticmethod
    def _rotate(values: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        even, odd = values[..., 0::2], values[..., 1::2]
        rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
        return rotated.flatten(-2)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
        length = query.shape[-2]
        if length > self.phase_offset.numel():
            raise ValueError(f"Sequence length {length} exceeds PA-RoPE capacity")
        positions = torch.arange(length, device=query.device, dtype=query.dtype)
        positions = positions + self.phase_offset[:length].to(dtype=query.dtype)
        angles = positions[:, None] * self.inverse_frequency[None, :].to(dtype=query.dtype)
        cosine = angles.cos()[None, None, :, :]
        sine = angles.sin()[None, None, :, :]
        return self._rotate(query, cosine, sine), self._rotate(key, cosine, sine)


class DuETAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        sequence_length: int,
        gaussian_sigmas: Sequence[float],
        rope: PhaseAlignedRoPE,
        dropout: float,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.temporal_weights = nn.Parameter(torch.zeros(n_heads, len(gaussian_sigmas)))
        self.spectral_mask_logits = nn.Parameter(torch.full((n_heads, sequence_length // 2 + 1), 2.0))
        self.mix_logits = nn.Parameter(torch.zeros(n_heads))

        distance = torch.arange(sequence_length, dtype=torch.float32)
        distance = distance[:, None] - distance[None, :]
        kernels = [torch.exp(-(distance.square()) / (2.0 * float(sigma) ** 2)) for sigma in gaussian_sigmas]
        self.register_buffer("gaussian_kernels", torch.stack(kernels, dim=0), persistent=True)

    def forward(self, values: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size, length, _ = values.shape
        qkv = self.qkv(values).view(batch_size, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.rope(query, key)

        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        kernels = self.gaussian_kernels[:, :length, :length].to(dtype=logits.dtype)
        temporal_bias = torch.einsum("hs,stq->htq", self.temporal_weights, kernels)
        temporal_logits = logits + temporal_bias.unsqueeze(0)

        spectrum = torch.fft.rfft(temporal_logits.float(), dim=-1)
        mask = torch.sigmoid(self.spectral_mask_logits[:, : spectrum.shape[-1]])
        filtered = torch.fft.irfft(spectrum * mask[None, :, None, :], n=length, dim=-1)
        filtered = filtered.to(dtype=temporal_logits.dtype)
        mix = torch.sigmoid(self.mix_logits)[None, :, None, None]
        fused_logits = (1.0 - mix) * temporal_logits + mix * filtered

        if attention_mask is not None:
            fused_logits = fused_logits.masked_fill(~attention_mask.bool(), torch.finfo(fused_logits.dtype).min)
        attention = torch.softmax(fused_logits, dim=-1)
        context = torch.matmul(self.attention_dropout(attention), value)
        context = context.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        return self.output_dropout(self.output(context)), attention


class ReSFFormerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, attention: DuETAttention, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = attention
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        attended, weights = self.attention(self.norm1(values), attention_mask)
        values = values + attended
        values = values + self.ffn(self.norm2(values))
        return values, weights


class AIOReSFFormer(nn.Module):
    def __init__(
        self,
        time_input_dim: int = 2,
        heatmap_channels: int = 3,
        sequence_length: int = 100,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        branch_channels: int = 64,
        tcn_kernel_size: int = 3,
        dcn_hidden_channels: int = 64,
        gaussian_sigmas: Sequence[float] = (1.0, 2.0, 4.0),
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.aio_fe = AIOFeatureExtractor(
            time_input_dim, d_model, branch_channels, d_ff, tcn_kernel_size, dropout
        )
        self.heatmap_encoder = HeatmapFeatureExtractor(heatmap_channels, dcn_hidden_channels, d_model)
        self.csaf = CSAF(d_model)
        self.rope = PhaseAlignedRoPE(sequence_length, d_model // n_heads)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            attention = DuETAttention(
                d_model, n_heads, sequence_length, gaussian_sigmas, self.rope, dropout
            )
            self.blocks.append(ReSFFormerBlock(d_model, d_ff, attention, dropout))
        self.output_norm = nn.LayerNorm(d_model)
        self.output_pool = AttentionPool(d_model)
        self.regressor = nn.Linear(d_model, 1)

    def encode_heatmap(self, heatmap: Tensor) -> Tensor:
        return self.heatmap_encoder(heatmap)

    def forward(
        self,
        time_series: Tensor,
        heatmap: Optional[Tensor] = None,
        *,
        heatmap_features: Optional[Tensor] = None,
        return_intermediates: bool = False,
    ):
        if heatmap_features is None:
            if heatmap is None:
                raise ValueError("Either heatmap or heatmap_features must be provided")
            heatmap_features = self.encode_heatmap(heatmap)
        sequence, f_sbm, aio_weights = self.aio_fe(time_series)
        f_fuse, gamma, beta = self.csaf(f_sbm, heatmap_features)

        # 保留时序特征，同时让其池化表征与式（5）-（6）定义的 CSAF 调制结果一致。
        values = sequence + (f_fuse - f_sbm).unsqueeze(1)
        attention_maps = []
        for block in self.blocks:
            values, attention = block(values)
            attention_maps.append(attention)
        prediction_features = self.output_pool(self.output_norm(values))
        prediction = self.regressor(prediction_features).squeeze(-1)
        if not return_intermediates:
            return prediction
        return prediction, {
            "aio_sequence": sequence,
            "f_sbm": f_sbm,
            "heatmap_features": heatmap_features,
            "f_fuse": f_fuse,
            "gamma": gamma,
            "beta": beta,
            "aio_pool_weights": aio_weights,
            "attention_maps": attention_maps,
            "prediction_features": prediction_features,
        }


def build_model(config) -> AIOReSFFormer:
    return AIOReSFFormer(
        time_input_dim=config.time_input_dim,
        heatmap_channels=config.heatmap_channels,
        sequence_length=config.sequence_length,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        branch_channels=config.branch_channels,
        tcn_kernel_size=config.tcn_kernel_size,
        dcn_hidden_channels=config.dcn_hidden_channels,
        gaussian_sigmas=config.gaussian_sigmas,
    )
