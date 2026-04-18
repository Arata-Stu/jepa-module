#!/usr/bin/env python3
"""
Visualize pretraining event voxels as RGB images (white background, red/blue events).

This script reuses the same Hydra config and loader build path as train_step1_pretrain.py.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torchvision.utils import save_image

from train_step1_pretrain import (
    _configure_data_loader_runtime,
    build_batch_provider,
    maybe_validate_cfg,
)
from event.representations import VoxelGrid
from jepa.utils.distributed import DistributedState


def _sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    if len(cleaned) == 0:
        return "unknown"
    return cleaned[:80]


def _safe_positive_quantile(x: torch.Tensor, q: float = 0.995) -> float:
    x = x.detach()
    if x.numel() == 0:
        return 1.0
    x_pos = x[x > 0]
    if x_pos.numel() == 0:
        return 1.0
    v = float(torch.quantile(x_pos, q).item())
    if not math.isfinite(v) or v <= 1.0e-8:
        return 1.0
    return v


def _signed_map_to_rgb(event_map: torch.Tensor) -> torch.Tensor:
    if event_map.ndim != 2:
        raise ValueError(f"Expected 2D signed map, got shape={tuple(event_map.shape)}")

    pos = torch.clamp(event_map, min=0.0)
    neg = torch.clamp(-event_map, min=0.0)

    pos_scale = _safe_positive_quantile(pos)
    neg_scale = _safe_positive_quantile(neg)
    pos = torch.clamp(pos / pos_scale, min=0.0, max=1.0)
    neg = torch.clamp(neg / neg_scale, min=0.0, max=1.0)

    # white background:
    # - positive events reduce G/B -> red
    # - negative events reduce R/G -> blue
    r = 1.0 - neg
    g = 1.0 - torch.clamp(pos + neg, min=0.0, max=1.0)
    b = 1.0 - pos
    return torch.stack([r, g, b], dim=0).to(torch.float32)


def _voxel_to_rgb(voxel: torch.Tensor) -> torch.Tensor:
    if voxel.ndim == 2:
        return _signed_map_to_rgb(voxel)
    if voxel.ndim != 3:
        raise ValueError(f"Expected voxel with shape [T,H,W] or [H,W], got {tuple(voxel.shape)}")

    pos = torch.clamp(voxel, min=0.0).sum(dim=0)
    neg = torch.clamp(-voxel, min=0.0).sum(dim=0)
    signed_sum = pos - neg
    return _signed_map_to_rgb(signed_sum)


def _extract_voxel_tensor(inputs: torch.Tensor, batch_idx: int) -> torch.Tensor:
    sample = inputs[batch_idx]
    if sample.ndim == 4:
        # [C, T, H, W] where C is expected to be 1
        if sample.shape[0] != 1:
            raise ValueError(
                "Expected channel dimension C=1 for event voxels, "
                f"got shape={tuple(sample.shape)}"
            )
        return sample[0]
    if sample.ndim == 3:
        # [T, H, W]
        return sample
    if sample.ndim == 2:
        # [H, W]
        return sample
    raise ValueError(f"Unsupported input sample shape: {tuple(sample.shape)}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_step1")
def main(cfg: DictConfig) -> None:
    maybe_validate_cfg(cfg)
    _configure_data_loader_runtime(cfg)

    viz_cfg = cfg.get("visualize", None)
    num_samples = int(viz_cfg.get("num_samples", 16)) if viz_cfg is not None else 16
    out_subdir = str(viz_cfg.get("out_subdir", "event_viz")) if viz_cfg is not None else "event_viz"

    dist_state = DistributedState(
        enabled=False,
        backend="none",
        rank=0,
        world_size=1,
        local_rank=0,
    )

    voxel_builder = VoxelGrid(
        channels=int(cfg.t_bins),
        height=int(cfg.height),
        width=int(cfg.width),
    )
    batch_provider = build_batch_provider(cfg, voxel_builder, dist_state=dist_state)

    runtime_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir = runtime_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "index.csv"

    written = 0
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "idx",
                "image_path",
                "source",
                "sample_path",
                "nonzero",
                "voxel_min",
                "voxel_max",
                "shape",
            ]
        )

        while written < num_samples:
            batch = batch_provider.next_batch()
            inputs = batch["inputs"]
            if not isinstance(inputs, torch.Tensor):
                raise TypeError("Batch must contain tensor in `inputs`.")
            if inputs.ndim not in {4, 5}:
                raise ValueError(f"Expected inputs ndim 4 or 5, got {inputs.ndim}")

            batch_size = int(inputs.shape[0])
            paths = batch.get("paths", [None] * batch_size)
            sources = batch.get("sources", [None] * batch_size)

            for bi in range(batch_size):
                if written >= num_samples:
                    break

                voxel = _extract_voxel_tensor(inputs=inputs, batch_idx=bi).detach().cpu().to(torch.float32)
                rgb = _voxel_to_rgb(voxel)

                source_name = _sanitize_name(str(sources[bi])) if bi < len(sources) else "unknown"
                image_name = f"{written:06d}_{source_name}.png"
                image_path = out_dir / image_name
                save_image(rgb, str(image_path))

                sample_path = str(paths[bi]) if bi < len(paths) else ""
                writer.writerow(
                    [
                        written,
                        str(image_path),
                        str(sources[bi]) if bi < len(sources) else "",
                        sample_path,
                        int(torch.count_nonzero(voxel).item()),
                        float(voxel.min().item()) if voxel.numel() > 0 else 0.0,
                        float(voxel.max().item()) if voxel.numel() > 0 else 0.0,
                        tuple(int(v) for v in voxel.shape),
                    ]
                )
                written += 1

    print(f"[visualize_pretrain_voxels] wrote={written} samples")
    print(f"[visualize_pretrain_voxels] out_dir={out_dir}")
    print(f"[visualize_pretrain_voxels] index={csv_path}")


if __name__ == "__main__":
    main()

