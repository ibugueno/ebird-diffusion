from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        if dimension % 2:
            raise ValueError("time_embedding_dim debe ser par")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(inputs)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(hidden))))
        return hidden + self.skip(inputs)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        if channels % heads:
            raise ValueError(f"{channels=} debe ser divisible por {heads=}")
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        sequence = self.norm(inputs).flatten(2).transpose(1, 2)
        attended, _ = self.attention(
            sequence, sequence, sequence, need_weights=False
        )
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return inputs + attended


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.nn.functional.interpolate(inputs, scale_factor=2, mode="nearest")
        return self.conv(inputs)


class DiffusionUNet(nn.Module):
    """U-Net DDPM con atención restringida a resoluciones configurables."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        channels: Sequence[int],
        attention_resolutions: Sequence[int],
        time_embedding_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.channels = tuple(int(channel) for channel in channels)
        self.attention_resolutions = {int(value) for value in attention_resolutions}
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if len(self.channels) < 2:
            raise ValueError("Se requieren al menos dos niveles de canales")
        divisor = 2 ** (len(self.channels) - 1)
        if self.image_size % divisor:
            raise ValueError(f"image_size debe ser divisible por {divisor}")

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.input_conv = nn.Conv2d(in_channels, self.channels[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_resolution = self.image_size
        for level, channel in enumerate(self.channels):
            self.down_blocks.append(ResBlock(channel, channel, time_embedding_dim, dropout))
            self.down_attentions.append(
                SpatialAttention(channel, num_heads)
                if current_resolution in self.attention_resolutions
                else nn.Identity()
            )
            if level < len(self.channels) - 1:
                self.downsamples.append(Downsample(channel, self.channels[level + 1]))
                current_resolution //= 2

        deepest = self.channels[-1]
        self.mid_block1 = ResBlock(deepest, deepest, time_embedding_dim, dropout)
        self.mid_attention = (
            SpatialAttention(deepest, num_heads)
            if current_resolution in self.attention_resolutions
            else nn.Identity()
        )
        self.mid_block2 = ResBlock(deepest, deepest, time_embedding_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(self.channels))):
            channel = self.channels[level]
            self.up_blocks.append(ResBlock(channel * 2, channel, time_embedding_dim, dropout))
            resolution = self.image_size // (2**level)
            self.up_attentions.append(
                SpatialAttention(channel, num_heads)
                if resolution in self.attention_resolutions
                else nn.Identity()
            )
            self.upsamples.append(
                Upsample(channel, self.channels[level - 1])
                if level > 0
                else nn.Identity()
            )

        self.output_norm = nn.GroupNorm(_groups(self.channels[0]), self.channels[0])
        self.output_conv = nn.Conv2d(self.channels[0], in_channels, 3, padding=1)

    def _resblock(
        self, block: nn.Module, inputs: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(block, inputs, time_embedding)
        return block(inputs, time_embedding)

    def _attention(self, block: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(block, nn.Identity):
            return inputs
        if self.gradient_checkpointing and self.training:
            return checkpoint(block, inputs)
        return block(inputs)

    def forward(
        self,
        inputs: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        controls: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        hidden = self.input_conv(inputs)
        control_skips, control_mid = controls if controls is not None else (None, None)

        skips: list[torch.Tensor] = []
        for level, (block, attention) in enumerate(
            zip(self.down_blocks, self.down_attentions)
        ):
            hidden = self._resblock(block, hidden, time_embedding)
            hidden = self._attention(attention, hidden)
            if control_skips is not None:
                hidden = hidden + control_skips[level]
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)

        hidden = self._resblock(self.mid_block1, hidden, time_embedding)
        hidden = self._attention(self.mid_attention, hidden)
        hidden = self._resblock(self.mid_block2, hidden, time_embedding)
        if control_mid is not None:
            hidden = hidden + control_mid

        for index, level in enumerate(reversed(range(len(self.channels)))):
            hidden = torch.cat((hidden, skips[level]), dim=1)
            hidden = self._resblock(self.up_blocks[index], hidden, time_embedding)
            hidden = self._attention(self.up_attentions[index], hidden)
            hidden = self.upsamples[index](hidden)

        return self.output_conv(torch.nn.functional.silu(self.output_norm(hidden)))


def unet_from_config(config: dict) -> DiffusionUNet:
    return DiffusionUNet(
        image_size=config["image_size"],
        in_channels=config.get("in_channels", 1),
        channels=config["channels"],
        attention_resolutions=config.get("attention_resolutions", []),
        time_embedding_dim=config.get("time_embedding_dim", 256),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.0),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
    )
