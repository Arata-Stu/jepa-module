#!/usr/bin/env python3
"""
Visualize pretraining event voxels as RGB images (white background, red/blue events).

This script reuses the same Hydra config and loader build path as train_step1_pretrain.py.
"""

from __future__ import annotations

import csv
import math
import re
import random
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torchvision.utils import save_image

from train_step1_pretrain import (
    _configure_data_loader_runtime,
    PRETRAIN_MIXED_SOURCE_NAMES,
    maybe_validate_cfg,
)
from event.data import ensure_path_exists
from event.data.pretrain_mixed import (
    PretrainMixedEventsDataset,
    PretrainMixedSourceConfig,
    PretrainMixedVoxelCollator,
)
from event.representations import VoxelGrid


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


def _build_pretrain_mixed_source_configs(cfg: DictConfig) -> tuple[list[PretrainMixedSourceConfig], dict[str, float]]:
    mixed_cfg = cfg.data.pretrain_mixed
    weights_cfg = mixed_cfg.get("weights", None)

    def _source_weight(source_name: str) -> float:
        if weights_cfg is None:
            return 1.0
        return float(weights_cfg.get(source_name, 1.0))

    source_configs: list[PretrainMixedSourceConfig] = []
    active_source_weights: dict[str, float] = {}

    for source_name in PRETRAIN_MIXED_SOURCE_NAMES:
        source_cfg = mixed_cfg[source_name]
        if not bool(source_cfg.enabled):
            continue

        source_weight = _source_weight(source_name)
        if source_weight <= 0.0:
            continue

        root_dir = source_cfg.root_dir
        if not root_dir:
            raise ValueError(
                f"data.pretrain_mixed.{source_name}.root_dir must be set "
                "when this source is enabled"
            )
        root_dir_abs = to_absolute_path(str(root_dir))
        ensure_path_exists(root_dir_abs, f"pretrain_mixed.{source_name}.root_dir")

        file_name = source_cfg.get("file_name", None)
        file_suffix = source_cfg.get("file_suffix", ".h5")
        manifest_file = source_cfg.get("manifest_file", None)
        file_name = None if file_name is None else str(file_name).strip() or None
        file_suffix = None if file_suffix is None else str(file_suffix).strip() or None

        manifest_file_path = None
        if manifest_file is not None:
            manifest_file_text = str(manifest_file).strip()
            if len(manifest_file_text) > 0:
                candidate_under_root = Path(root_dir_abs) / manifest_file_text
                if candidate_under_root.exists():
                    manifest_file_path = candidate_under_root.resolve()
                else:
                    manifest_file_path = Path(to_absolute_path(manifest_file_text)).resolve()
                ensure_path_exists(
                    str(manifest_file_path),
                    f"pretrain_mixed.{source_name}.manifest_file",
                )

        splits = tuple(str(s) for s in source_cfg.splits)
        if len(splits) == 0:
            raise ValueError(
                f"data.pretrain_mixed.{source_name}.splits must be non-empty "
                "when this source is enabled"
            )

        source_configs.append(
            PretrainMixedSourceConfig(
                name=str(source_name),
                root_dir=Path(root_dir_abs),
                splits=splits,
                sensor_height=int(source_cfg.sensor_height),
                sensor_width=int(source_cfg.sensor_width),
                recursive=bool(source_cfg.recursive),
                file_name=file_name,
                file_suffix=file_suffix,
                manifest_file=manifest_file_path,
            )
        )
        active_source_weights[source_name] = float(source_weight)

    if len(source_configs) == 0:
        raise ValueError(
            "No active source in pretrain_mixed. "
            "Enable at least one source and set its weight > 0."
        )
    return source_configs, active_source_weights


def _build_dataset_and_collator(cfg: DictConfig) -> tuple[PretrainMixedEventsDataset, PretrainMixedVoxelCollator]:
    if str(cfg.data.source) != "pretrain_mixed":
        raise ValueError("visualize_pretrain_voxels.py currently supports data.source=pretrain_mixed only.")

    mixed_cfg = cfg.data.pretrain_mixed
    source_configs, source_weights = _build_pretrain_mixed_source_configs(cfg)

    events_per_sample_default = int(
        mixed_cfg.get("events_per_sample", mixed_cfg.get("events_per_sample_min", 30000))
    )
    events_per_sample_min = int(mixed_cfg.get("events_per_sample_min", events_per_sample_default))
    events_per_sample_max = int(mixed_cfg.get("events_per_sample_max", events_per_sample_default))
    window_duration_us_min = mixed_cfg.get("window_duration_us_min", None)
    window_duration_us_max = mixed_cfg.get("window_duration_us_max", None)
    duration_sources = tuple(str(s) for s in mixed_cfg.get("duration_sources", ["dsec", "gen4"]))
    prefer_ms_to_idx = bool(mixed_cfg.get("prefer_ms_to_idx", True))
    augment_cfg = mixed_cfg.get("augment", {})
    rrc_cfg = augment_cfg.get("random_resized_crop", {})
    augment_enabled = bool(augment_cfg.get("enabled", False))
    hflip_prob = float(augment_cfg.get("hflip_prob", 0.0))
    max_shift = int(augment_cfg.get("max_shift", 0))
    time_flip_prob = float(augment_cfg.get("time_flip_prob", 0.0))
    polarity_flip_prob = float(augment_cfg.get("polarity_flip_prob", 0.0))
    rrc_enabled = bool(rrc_cfg.get("enabled", False))
    rrc_prob = float(rrc_cfg.get("prob", 0.0))
    rrc_scale_min = float(rrc_cfg.get("scale_min", 0.5))
    rrc_scale_max = float(rrc_cfg.get("scale_max", 1.0))
    rrc_aspect_min = float(rrc_cfg.get("aspect_min", 0.75))
    rrc_aspect_max = float(rrc_cfg.get("aspect_max", 4.0 / 3.0))
    rrc_attempts = int(rrc_cfg.get("attempts", 10))
    rrc_preserve_aspect = bool(rrc_cfg.get("preserve_aspect", False))

    dataset = PretrainMixedEventsDataset(
        source_configs=source_configs,
        source_weights=source_weights,
        events_per_sample_min=events_per_sample_min,
        events_per_sample_max=events_per_sample_max,
        random_slice=bool(mixed_cfg.random_slice),
        time_normalize=bool(mixed_cfg.time_normalize),
        window_duration_us_min=window_duration_us_min,
        window_duration_us_max=window_duration_us_max,
        duration_sources=duration_sources,
        prefer_ms_to_idx=prefer_ms_to_idx,
        min_events_in_window=int(mixed_cfg.get("min_events_in_window", 1)),
        min_event_rate_eps=mixed_cfg.get("min_event_rate_eps", None),
        activity_filter_sources=tuple(
            str(s) for s in mixed_cfg.get("activity_filter_sources", ["gen4"])
        ),
        max_window_attempts=int(mixed_cfg.get("max_window_attempts", 4)),
        augment_enabled=augment_enabled,
        hflip_prob=hflip_prob,
        max_shift=max_shift,
        time_flip_prob=time_flip_prob,
        polarity_flip_prob=polarity_flip_prob,
        rrc_enabled=rrc_enabled,
        rrc_prob=rrc_prob,
        rrc_scale_min=rrc_scale_min,
        rrc_scale_max=rrc_scale_max,
        rrc_aspect_min=rrc_aspect_min,
        rrc_aspect_max=rrc_aspect_max,
        rrc_attempts=rrc_attempts,
        rrc_preserve_aspect=rrc_preserve_aspect,
        drop_empty=True,
    )

    voxel_builder = VoxelGrid(
        channels=int(cfg.t_bins),
        height=int(cfg.height),
        width=int(cfg.width),
    )
    collator = PretrainMixedVoxelCollator(
        voxel_builder=voxel_builder,
        normalize_voxel=bool(cfg.normalize_voxel),
        rescale_to_voxel_grid=bool(mixed_cfg.rescale_to_voxel_grid),
        canvas_height=int(mixed_cfg.canvas_height),
        canvas_width=int(mixed_cfg.canvas_width),
        center_pad_to_canvas=bool(mixed_cfg.center_pad_to_canvas),
        debug_index_check=bool(mixed_cfg.get("debug_index_check", False)),
        debug_raise_on_oob=bool(mixed_cfg.get("debug_raise_on_oob", False)),
        debug_log_limit=int(mixed_cfg.get("debug_log_limit", 20)),
    )
    return dataset, collator


@hydra.main(version_base="1.3", config_path="../configs", config_name="visualize_pretrain_voxels")
def main(cfg: DictConfig) -> None:
    maybe_validate_cfg(cfg)
    _configure_data_loader_runtime(cfg)

    viz_cfg = cfg.get("visualize", None)
    num_samples = int(viz_cfg.get("num_samples", 16)) if viz_cfg is not None else 16
    out_subdir = str(viz_cfg.get("out_subdir", "event_viz")) if viz_cfg is not None else "event_viz"

    dataset, collator = _build_dataset_and_collator(cfg)
    weighted_sampling = bool(viz_cfg.get("weighted_sampling", True)) if viz_cfg is not None else True
    seed = int(cfg.seed) if "seed" in cfg else 42
    random.seed(seed)
    torch.manual_seed(seed)

    if int(cfg.data.get("num_workers", 0)) > 0:
        print(
            "[visualize_pretrain_voxels] note: this script reads dataset directly "
            "(no DataLoader workers). data.num_workers is ignored."
        )

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
                "sample_events",
                "sample_time_span_us",
                "sample_time_span_sec",
                "nonzero",
                "voxel_min",
                "voxel_max",
                "shape",
            ]
        )

        while written < num_samples:
            if weighted_sampling:
                idx = int(torch.multinomial(dataset.sample_weights, 1, replacement=True).item())
            else:
                idx = random.randint(0, len(dataset) - 1)

            sample = dataset[idx]
            batch = collator([sample])
            inputs = batch["inputs"]
            if not isinstance(inputs, torch.Tensor):
                raise TypeError("Batch must contain tensor in `inputs`.")

            sample_time = sample["time"].detach().cpu() if isinstance(sample.get("time"), torch.Tensor) else None
            sample_events = int(sample_time.numel()) if sample_time is not None else 0
            sample_time_span_us = 0
            if sample_time is not None and sample_time.numel() > 1:
                sample_time_span_us = int(sample_time[-1].item() - sample_time[0].item())
            sample_time_span_sec = float(sample_time_span_us) / 1.0e6

            voxel = _extract_voxel_tensor(inputs=inputs, batch_idx=0).detach().cpu().to(torch.float32)
            rgb = _voxel_to_rgb(voxel)

            sample_source = str(batch["sources"][0]) if isinstance(batch.get("sources"), list) else "unknown"
            source_name = _sanitize_name(sample_source)
            image_name = f"{written:06d}_{source_name}.png"
            image_path = out_dir / image_name
            save_image(rgb, str(image_path))

            sample_path = ""
            if isinstance(batch.get("paths"), list) and len(batch["paths"]) > 0:
                sample_path = str(batch["paths"][0])

            writer.writerow(
                [
                    written,
                    str(image_path),
                    sample_source,
                    sample_path,
                    sample_events,
                    sample_time_span_us,
                    f"{sample_time_span_sec:.6f}",
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
