#!/usr/bin/env python3
"""
Visualize encoder attention rollout maps for JEPA pretraining inputs.

- Loads a JEPA checkpoint (teacher_encoder by default).
- Samples pretraining inputs with the same pretrain_mixed dataset path.
- Computes attention rollout across encoder self-attention blocks.
- Saves panel images: [voxel RGB | rollout heatmap | overlay].
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import hydra
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from event.data import ensure_path_exists  # noqa: E402
from train_jepa_pretrain import (  # noqa: E402
    MODEL_SPECS,
    _configure_data_loader_runtime,
    maybe_validate_cfg,
)
from visualize_pretrain_voxels import (  # noqa: E402
    _build_dataset_and_collator,
    _extract_voxel_tensor,
    _sanitize_name,
    _voxel_to_rgb,
)


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _temporal_mix_enabled(cfg: DictConfig) -> bool:
    temporal_mix_cfg = cfg.get("temporal_mix", None)
    if temporal_mix_cfg is None:
        return False
    return bool(temporal_mix_cfg.get("enabled", False))


def _build_encoder(cfg: DictConfig) -> torch.nn.Module:
    spec = MODEL_SPECS[str(cfg.model_size)]
    encoder_builder: Callable[..., torch.nn.Module] = spec["builder"]  # type: ignore[assignment]
    temporal_mix = _temporal_mix_enabled(cfg)
    img_temporal_dim_size = int(cfg.temporal_mix.short_t) if temporal_mix else None

    encoder = encoder_builder(
        img_size=(int(cfg.height), int(cfg.width)),
        patch_size=int(cfg.patch_size),
        num_frames=int(cfg.t_bins),
        tubelet_size=int(cfg.tubelet_size),
        in_chans=1,
        use_rope=True,
        modality_embedding=temporal_mix,
        img_temporal_dim_size=img_temporal_dim_size,
        n_output_distillation=4,
    )
    return encoder


def _load_pretrained_encoder(encoder: torch.nn.Module, cfg: DictConfig) -> str:
    ckpt_path_raw = cfg.pretrained.checkpoint
    if ckpt_path_raw is None:
        raise ValueError("pretrained.checkpoint must be set")

    ckpt_path = to_absolute_path(str(ckpt_path_raw))
    ensure_path_exists(ckpt_path, "pretrained.checkpoint")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    preferred_key = str(cfg.pretrained.encoder_key)
    state = checkpoint.get(preferred_key, None)
    loaded_key = preferred_key
    if state is None and bool(cfg.pretrained.fallback_to_encoder):
        state = checkpoint.get("encoder", None)
        loaded_key = "encoder"

    if state is None:
        raise KeyError(
            f"Encoder state '{preferred_key}' not found. Available keys: {sorted(checkpoint.keys())}"
        )
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint key '{loaded_key}' is not a state_dict")

    state = _normalize_state_dict_keys(state)
    encoder.load_state_dict(state, strict=bool(cfg.pretrained.strict))
    return loaded_key


def _force_disable_sdpa(encoder: torch.nn.Module) -> int:
    changed = 0
    blocks = getattr(encoder, "blocks", [])
    for blk in blocks:
        attn = getattr(blk, "attn", None)
        if attn is None or not hasattr(attn, "use_sdpa"):
            continue
        if bool(attn.use_sdpa):
            changed += 1
        attn.use_sdpa = False
    return changed


def _collect_block_attentions(
    encoder: torch.nn.Module,
    inputs: torch.Tensor,
) -> list[torch.Tensor]:
    captured: list[tuple[int, torch.Tensor]] = []
    handles: list[Any] = []

    def _make_hook(layer_idx: int):
        def _hook(_module, _args, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            attn = output[1]
            if isinstance(attn, torch.Tensor):
                captured.append((layer_idx, attn.detach()))

        return _hook

    blocks = getattr(encoder, "blocks", [])
    if len(blocks) == 0:
        raise ValueError("Encoder has no transformer blocks")

    for i, blk in enumerate(blocks):
        handles.append(blk.attn.register_forward_hook(_make_hook(i)))

    prev_attn_out = bool(getattr(encoder, "attn_out", False))
    encoder.attn_out = True
    try:
        _ = encoder(inputs, masks=None, training=False)
    finally:
        encoder.attn_out = prev_attn_out
        for h in handles:
            h.remove()

    captured.sort(key=lambda x: x[0])
    return [x[1] for x in captured]


def _attention_rollout(
    attn_layers: list[torch.Tensor],
    discard_ratio: float,
) -> torch.Tensor:
    if len(attn_layers) == 0:
        raise ValueError("No attention tensors were captured from encoder forward pass")

    if not (0.0 <= discard_ratio < 1.0):
        raise ValueError("visualize.rollout.discard_ratio must be in [0, 1)")

    first = attn_layers[0]
    if first.ndim != 4:
        raise ValueError(f"Expected attention as [B,H,N,N], got {tuple(first.shape)}")
    bsz, _heads, n_tokens, _ = first.shape
    eye = torch.eye(n_tokens, device=first.device, dtype=first.dtype).unsqueeze(0).expand(bsz, -1, -1)
    rollout = eye.clone()

    for attn in attn_layers:
        if attn.ndim != 4:
            raise ValueError(f"Expected attention as [B,H,N,N], got {tuple(attn.shape)}")
        a = attn.mean(dim=1)  # [B, N, N]

        if discard_ratio > 0.0:
            flat = a.reshape(bsz, -1)
            discard_k = int(flat.shape[1] * discard_ratio)
            if discard_k > 0:
                idx = torch.topk(flat, k=discard_k, dim=1, largest=False).indices
                flat = flat.clone()
                flat.scatter_(1, idx, 0.0)
                a = flat.view_as(a)

        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        rollout = torch.bmm(a, rollout)

    return rollout


def _token_grid_shape(cfg: DictConfig, inputs: torch.Tensor) -> tuple[int, int, int]:
    if inputs.ndim != 5:
        raise ValueError(f"Expected inputs as [B,C,T,H,W], got {tuple(inputs.shape)}")
    t_in = int(inputs.shape[2])
    h_in = int(inputs.shape[3])
    w_in = int(inputs.shape[4])
    patch = int(cfg.patch_size)
    tubelet = int(cfg.tubelet_size)

    temporal_mix = _temporal_mix_enabled(cfg)
    short_t = int(cfg.temporal_mix.short_t) if temporal_mix else None
    if short_t is not None and t_in == short_t:
        t_p = t_in
    else:
        if t_in % tubelet != 0:
            raise ValueError(f"Input T={t_in} must be divisible by tubelet_size={tubelet}")
        t_p = t_in // tubelet

    if h_in % patch != 0 or w_in % patch != 0:
        raise ValueError(f"Input H/W=({h_in},{w_in}) must be divisible by patch_size={patch}")
    h_p = h_in // patch
    w_p = w_in // patch
    return t_p, h_p, w_p


def _rollout_to_spatial_map(
    cfg: DictConfig,
    rollout: torch.Tensor,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, int, int, int]:
    if rollout.ndim != 3:
        raise ValueError(f"Expected rollout as [B,N,N], got {tuple(rollout.shape)}")

    token_importance = rollout.mean(dim=1)  # [B, N]
    t_p, h_p, w_p = _token_grid_shape(cfg, inputs)
    expected_tokens = t_p * h_p * w_p
    if int(token_importance.shape[1]) != expected_tokens:
        raise ValueError(
            "Token count mismatch for rollout reshape: "
            f"N={int(token_importance.shape[1])}, expected={expected_tokens} "
            f"(t_p={t_p}, h_p={h_p}, w_p={w_p})"
        )

    token_grid = token_importance.view(int(token_importance.shape[0]), t_p, h_p, w_p)

    time_reduce = str(cfg.visualize.rollout.time_reduce)
    if time_reduce == "mean":
        map_2d = token_grid.mean(dim=1)
    elif time_reduce == "max":
        map_2d = token_grid.max(dim=1).values
    elif time_reduce == "center":
        map_2d = token_grid[:, t_p // 2]
    else:
        raise ValueError("visualize.rollout.time_reduce must be one of {mean, max, center}")

    map_2d = map_2d - map_2d.amin(dim=(1, 2), keepdim=True)
    map_2d = map_2d / map_2d.amax(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
    map_up = F.interpolate(
        map_2d.unsqueeze(1),
        size=(int(inputs.shape[3]), int(inputs.shape[4])),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return map_up, t_p, h_p, w_p


def _heatmap_to_rgb(heat: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(heat, min=0.0, max=1.0)
    r = torch.clamp(2.0 * x - 0.5, min=0.0, max=1.0)
    g = torch.clamp(1.5 - 1.5 * torch.abs(2.0 * x - 1.0), min=0.0, max=1.0)
    b = torch.clamp(1.5 * (1.0 - x) - 0.5, min=0.0, max=1.0)
    return torch.stack([r, g, b], dim=0).to(torch.float32)


def _overlay(
    base_rgb: torch.Tensor,
    heat_rgb: torch.Tensor,
    heat: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError("visualize.overlay_alpha must be in [0, 1]")
    mask = heat.unsqueeze(0).to(dtype=base_rgb.dtype)
    out = base_rgb * (1.0 - a * mask) + heat_rgb * (a * mask)
    return torch.clamp(out, min=0.0, max=1.0)


@hydra.main(version_base="1.3", config_path="../configs", config_name="visualize_attention_rollout")
def main(cfg: DictConfig) -> None:
    maybe_validate_cfg(cfg)
    _configure_data_loader_runtime(cfg)

    if cfg.pretrained.checkpoint is None:
        raise ValueError("pretrained.checkpoint must be set")
    if str(cfg.data.source) != "pretrain_mixed":
        raise ValueError("visualize_attention_rollout.py currently supports data.source=pretrain_mixed only.")

    set_seed = int(cfg.seed) if "seed" in cfg else 42
    random.seed(set_seed)
    torch.manual_seed(set_seed)

    runtime_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir = runtime_dir / str(cfg.visualize.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "index.csv"

    dataset, collator = _build_dataset_and_collator(cfg)
    weighted_sampling = bool(cfg.visualize.weighted_sampling)
    num_samples = int(cfg.visualize.num_samples)
    if num_samples < 1:
        raise ValueError("visualize.num_samples must be >= 1")

    device_name = str(cfg.device)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    encoder = _build_encoder(cfg)
    loaded_key = _load_pretrained_encoder(encoder, cfg)
    encoder = encoder.to(device)
    encoder.eval()

    changed = 0
    if bool(cfg.visualize.force_disable_sdpa):
        changed = _force_disable_sdpa(encoder)

    discard_ratio = float(cfg.visualize.rollout.discard_ratio)
    if not (0.0 <= discard_ratio < 1.0):
        raise ValueError("visualize.rollout.discard_ratio must be in [0, 1)")

    print(
        f"[attention_rollout] device={device} "
        f"checkpoint_key={loaded_key} "
        f"force_disable_sdpa={bool(cfg.visualize.force_disable_sdpa)} "
        f"sdpa_blocks_changed={changed}"
    )

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
                "checkpoint_key",
                "num_layers",
                "num_tokens",
                "time_patches",
                "height_patches",
                "width_patches",
                "rollout_min",
                "rollout_max",
            ]
        )

        with torch.no_grad():
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

                model_inputs = inputs.to(device, non_blocking=True)
                attn_layers = _collect_block_attentions(encoder=encoder, inputs=model_inputs)
                if len(attn_layers) == 0:
                    raise RuntimeError(
                        "No attention tensors were captured. "
                        "Try visualize.force_disable_sdpa=true."
                    )

                rollout = _attention_rollout(attn_layers=attn_layers, discard_ratio=discard_ratio)
                heat_up, t_p, h_p, w_p = _rollout_to_spatial_map(
                    cfg=cfg,
                    rollout=rollout,
                    inputs=model_inputs,
                )

                voxel = _extract_voxel_tensor(inputs=inputs, batch_idx=0).detach().cpu().to(torch.float32)
                base_rgb = _voxel_to_rgb(voxel)
                heat = heat_up[0].detach().cpu().to(torch.float32)
                heat_rgb = _heatmap_to_rgb(heat)
                overlay = _overlay(
                    base_rgb=base_rgb,
                    heat_rgb=heat_rgb,
                    heat=heat,
                    alpha=float(cfg.visualize.overlay_alpha),
                )
                panel = torch.cat([base_rgb, heat_rgb, overlay], dim=2)

                sample_source = str(batch["sources"][0]) if isinstance(batch.get("sources"), list) else "unknown"
                source_name = _sanitize_name(sample_source)
                image_name = f"{written:06d}_{source_name}.png"
                image_path = out_dir / image_name
                save_image(panel, str(image_path))

                sample_time = sample.get("time", None)
                sample_events = 0
                sample_time_span_us = 0
                if isinstance(sample_time, torch.Tensor):
                    sample_events = int(sample_time.numel())
                    if sample_time.numel() > 1:
                        sample_time_span_us = int(sample_time[-1].item() - sample_time[0].item())

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
                        loaded_key,
                        len(attn_layers),
                        int(rollout.shape[-1]),
                        t_p,
                        h_p,
                        w_p,
                        f"{float(heat.min().item()):.8f}",
                        f"{float(heat.max().item()):.8f}",
                    ]
                )
                written += 1

    print(f"[attention_rollout] wrote={written} samples")
    print(f"[attention_rollout] out_dir={out_dir}")
    print(f"[attention_rollout] index={csv_path}")


if __name__ == "__main__":
    main()
