from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .unet import DiffusionUNet, ResBlock, SinusoidalTimeEmbedding


def _zero_conv(channels: int) -> nn.Conv2d:
    layer = nn.Conv2d(channels, channels, 1)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class ConditionEncoder(nn.Module):
    """Encoder tipo ControlNet que genera residuos inicializados en cero."""

    def __init__(self, base: DiffusionUNet, time_embedding_dim: int, dropout: float):
        super().__init__()
        self.gradient_checkpointing = base.gradient_checkpointing
        first = base.channels[0]
        self.noisy_input = nn.Conv2d(1, first, 3, padding=1)
        self.condition_input = nn.Conv2d(1, first, 3, padding=1)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.zero_skips = nn.ModuleList()
        for level, channel in enumerate(base.channels):
            self.blocks.append(ResBlock(channel, channel, time_embedding_dim, dropout))
            self.zero_skips.append(_zero_conv(channel))
            if level < len(base.channels) - 1:
                self.downsamples.append(
                    nn.Conv2d(channel, base.channels[level + 1], 3, stride=2, padding=1)
                )
        deepest = base.channels[-1]
        self.mid = ResBlock(deepest, deepest, time_embedding_dim, dropout)
        self.zero_mid = _zero_conv(deepest)

    def _block(
        self, block: nn.Module, inputs: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(block, inputs, time_embedding)
        return block(inputs, time_embedding)

    def forward(
        self, noisy: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        time_embedding = self.time_embedding(timesteps)
        hidden = self.noisy_input(noisy) + self.condition_input(condition)
        controls: list[torch.Tensor] = []
        for level, block in enumerate(self.blocks):
            hidden = self._block(block, hidden, time_embedding)
            controls.append(self.zero_skips[level](hidden))
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)
        hidden = self._block(self.mid, hidden, time_embedding)
        return controls, self.zero_mid(hidden)


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, base: DiffusionUNet, *, time_embedding_dim: int, dropout: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.control = ConditionEncoder(base, time_embedding_dim, dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(
        self, noisy: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        controls = self.control(noisy, condition, timesteps)
        return self.base(noisy, timesteps, controls=controls)


def conditional_from_config(
    base: DiffusionUNet, config: dict
) -> ConditionalDiffusionModel:
    return ConditionalDiffusionModel(
        base,
        time_embedding_dim=config.get("time_embedding_dim", 256),
        dropout=config.get("dropout", 0.0),
    )
