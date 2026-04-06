from __future__ import annotations

import math

import torch


class WarmupCosineParamScheduler:
    """
    Warmup + cosine decay scheduler for any optimizer param-group scalar
    (e.g. `lr`, `weight_decay`).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        param_name: str,
        base_value: float,
        final_value: float,
        total_steps: int,
        warmup_steps: int = 0,
        start_value: float = 0.0,
    ):
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.optimizer = optimizer
        self.param_name = param_name
        self.base_value = float(base_value)
        self.final_value = float(final_value)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.start_value = float(start_value)

    def _value_at(self, step: int) -> float:
        s = max(1, min(int(step), self.total_steps))
        if self.warmup_steps > 0 and s <= self.warmup_steps:
            alpha = s / self.warmup_steps
            return self.start_value + alpha * (self.base_value - self.start_value)

        decay_span = max(1, self.total_steps - self.warmup_steps)
        decay_step = max(0, s - self.warmup_steps)
        progress = min(1.0, decay_step / decay_span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_value + (self.base_value - self.final_value) * cosine

    def step(self, step: int) -> float:
        value = self._value_at(step)
        for group in self.optimizer.param_groups:
            group[self.param_name] = value
        return value
