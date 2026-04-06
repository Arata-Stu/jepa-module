from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError(f"Expected square matrix, got shape {x.shape}")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        if knots < 2:
            raise ValueError(f"knots must be >= 2, got {knots}")
        if num_proj < 1:
            raise ValueError(f"num_proj must be >= 1, got {num_proj}")

        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-(t.square()) / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: shape (T, B, D) or (N, D)
        """
        if proj.ndim == 2:
            proj = proj.unsqueeze(0)
        if proj.ndim != 3:
            raise ValueError(f"Expected proj with 2 or 3 dims, got {proj.shape}")

        proj_f = proj.float()

        A = torch.randn(
            proj_f.size(-1),
            self.num_proj,
            device=proj_f.device,
            dtype=proj_f.dtype,
        )
        A = A.div_(A.norm(p=2, dim=0).clamp_min(1e-12))

        t = self.t.to(device=proj_f.device, dtype=proj_f.dtype)
        phi = self.phi.to(device=proj_f.device, dtype=proj_f.dtype)
        weights = self.weights.to(device=proj_f.device, dtype=proj_f.dtype)

        x_t = (proj_f @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(dim=-3) - phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = (err @ weights) * proj_f.size(-2)
        return statistic.mean().to(proj.dtype)


class VICRegRegularizer(nn.Module):
    """VICReg-style variance/covariance regularization."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _flatten_features(x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Expected at least 2 dims, got {x.shape}")
        return x.reshape(-1, x.shape[-1])

    def _std_cov_terms(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._flatten_features(x).float()
        n, d = x.shape
        if n < 2:
            zero = x.new_zeros(())
            return zero, zero

        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)
        std_loss = torch.mean(F.relu(1.0 - std))

        cov = (x.T @ x) / (n - 1)
        cov_loss = _off_diagonal(cov).pow_(2).sum().div(d)
        return std_loss, cov_loss

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std_x, cov_x = self._std_cov_terms(x)
        if y is None:
            return std_x.to(x.dtype), cov_x.to(x.dtype)

        std_y, cov_y = self._std_cov_terms(y)
        std = 0.5 * (std_x + std_y)
        cov = 0.5 * (cov_x + cov_y)
        return std.to(x.dtype), cov.to(x.dtype)
