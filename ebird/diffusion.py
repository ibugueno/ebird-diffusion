from __future__ import annotations

import torch


class LinearNoiseScheduler:
    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float):
        self.num_timesteps = int(num_timesteps)
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        extracted = values.to(timesteps.device)[timesteps]
        return extracted.reshape(timesteps.shape[0], *([1] * (len(shape) - 1)))

    def add_noise(
        self, original: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alpha = self._extract(self.alpha_cumprod, timesteps, original.shape)
        return alpha.sqrt() * original + (1.0 - alpha).sqrt() * noise

    def sample_previous(
        self, current: torch.Tensor, predicted_noise: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_cumprod = self._extract(self.alpha_cumprod, timesteps, current.shape)
        alpha = self._extract(self.alphas, timesteps, current.shape)
        beta = self._extract(self.betas, timesteps, current.shape)
        one_minus = (1.0 - alpha_cumprod).clamp_min(1e-12)
        predicted_x0 = (current - one_minus.sqrt() * predicted_noise) / alpha_cumprod.sqrt()
        predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

        mean = (current - beta * predicted_noise / one_minus.sqrt()) / alpha.sqrt()
        previous_t = (timesteps - 1).clamp_min(0)
        previous_cumprod = self._extract(self.alpha_cumprod, previous_t, current.shape)
        variance = beta * (1.0 - previous_cumprod) / one_minus
        sample = mean + variance.clamp_min(0).sqrt() * torch.randn_like(current)
        is_first = (timesteps == 0).reshape(
            timesteps.shape[0], *([1] * (current.ndim - 1))
        )
        return torch.where(is_first, predicted_x0, sample), predicted_x0
