#!/usr/bin/env python3
"""
Step1 prototype for event-based JEPA pretraining.

- input: event voxel grids from synthetic events (placeholder before real dataloader)
- task: masked token prediction at the same timestamp
- models: JEPA ViT encoder + predictor in `src/jepa`
- config: Hydra (`configs/train_step1.yaml`)
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jepa.models.predictor import vit_predictor
from jepa.models.vision_transformer import (  # noqa: E402
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)
from event.representations import VoxelGrid, norm_voxel_grid  # noqa: E402
from jepa.masks.multiseq_multiblock3d import _MaskGenerator  # noqa: E402
from jepa.masks.utils import apply_masks  # noqa: E402
from jepa.regularizers import SIGReg, VICRegRegularizer  # noqa: E402


MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "tiny": {"builder": vit_tiny, "embed_dim": 192, "predictor_heads": 6},
    "small": {"builder": vit_small, "embed_dim": 384, "predictor_heads": 12},
    "base": {"builder": vit_base, "embed_dim": 768, "predictor_heads": 12},
    "large": {"builder": vit_large, "embed_dim": 1024, "predictor_heads": 16},
}

COLLAPSE_STRATEGY_CHOICES = {"ema_stopgrad", "vicreg", "sigreg"}
PREDICTOR_DEPTH_CHOICES = {4, 8, 12, 20, 24, 40}
RECON_LOSS_CHOICES = {"smooth_l1", "mse"}

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_models(cfg: DictConfig) -> Tuple[torch.nn.Module, torch.nn.Module]:
    spec = MODEL_SPECS[str(cfg.model_size)]
    encoder_builder: Callable[..., torch.nn.Module] = spec["builder"]  # type: ignore[assignment]
    embed_dim: int = int(spec["embed_dim"])
    predictor_heads = (
        int(cfg.predictor_heads)
        if cfg.predictor_heads is not None
        else int(spec["predictor_heads"])
    )
    if int(cfg.predictor_embed_dim) % predictor_heads != 0:
        raise ValueError(
            "predictor-embed-dim must be divisible by predictor-heads "
            f"(got {cfg.predictor_embed_dim} and {predictor_heads})"
        )

    encoder = encoder_builder(
        img_size=(int(cfg.height), int(cfg.width)),
        patch_size=int(cfg.patch_size),
        num_frames=int(cfg.t_bins),
        tubelet_size=int(cfg.tubelet_size),
        in_chans=1,
        use_rope=True,
        modality_embedding=False,
        n_output_distillation=4,
    )

    predictor = vit_predictor(
        img_size=(int(cfg.height), int(cfg.width)),
        patch_size=int(cfg.patch_size),
        num_frames=int(cfg.t_bins),
        tubelet_size=int(cfg.tubelet_size),
        embed_dim=embed_dim,
        predictor_embed_dim=int(cfg.predictor_embed_dim),
        depth=int(cfg.predictor_depth),
        num_heads=predictor_heads,
        use_rope=True,
        use_mask_tokens=True,
        num_mask_tokens=int(cfg.num_mask_tokens),
        modality_embedding=False,
        n_output_distillation=4,
    )
    return encoder, predictor


def build_mask_generator(cfg: DictConfig) -> _MaskGenerator:
    return _MaskGenerator(
        crop_size=(int(cfg.height), int(cfg.width)),
        num_frames=int(cfg.t_bins),
        spatial_patch_size=(int(cfg.patch_size), int(cfg.patch_size)),
        temporal_patch_size=int(cfg.tubelet_size),
        spatial_pred_mask_scale=(float(cfg.spatial_mask_min), float(cfg.spatial_mask_max)),
        temporal_pred_mask_scale=(float(cfg.temporal_mask_min), float(cfg.temporal_mask_max)),
        aspect_ratio=(float(cfg.aspect_min), float(cfg.aspect_max)),
        npred=int(cfg.num_pred_blocks),
        max_context_frames_ratio=float(cfg.max_context_frames_ratio),
    )


def sample_voxel_batch(
    repr_builder: VoxelGrid,
    batch_size: int,
    num_events_min: int,
    num_events_max: int,
    normalize_voxel: bool,
) -> torch.Tensor:
    voxels = []
    for _ in range(batch_size):
        n_events = random.randint(num_events_min, num_events_max)
        x = torch.randint(0, repr_builder.width, (n_events,), dtype=torch.int64)
        y = torch.randint(0, repr_builder.height, (n_events,), dtype=torch.int64)
        pol = torch.randint(0, 2, (n_events,), dtype=torch.int64)
        time = torch.sort(torch.randint(0, 1_000_000, (n_events,), dtype=torch.int64)).values
        if time[-1] == time[0]:
            time[-1] = time[0] + 1
        voxel = repr_builder.convert(x=x, y=y, pol=pol, time=time)
        if normalize_voxel:
            voxel = norm_voxel_grid(voxel)
        voxels.append(voxel)
    # [B, 1, T, H, W]
    return torch.stack(voxels, dim=0).unsqueeze(1)


def get_collapse_strategy(cfg: DictConfig) -> str:
    # Backward compatibility for older configs/commands using `ablation`.
    if "collapse_strategy" in cfg and cfg.collapse_strategy is not None:
        return str(cfg.collapse_strategy)
    if "ablation" in cfg and cfg.ablation is not None:
        return str(cfg.ablation)
    return "ema_stopgrad"


def maybe_validate_cfg(cfg: DictConfig) -> None:
    collapse_strategy = get_collapse_strategy(cfg)
    if collapse_strategy not in COLLAPSE_STRATEGY_CHOICES:
        raise ValueError(
            "collapse_strategy must be one of "
            f"{sorted(COLLAPSE_STRATEGY_CHOICES)}"
        )
    if str(cfg.model_size) not in MODEL_SPECS:
        raise ValueError(f"model_size must be one of {list(MODEL_SPECS.keys())}")
    if int(cfg.predictor_depth) not in PREDICTOR_DEPTH_CHOICES:
        raise ValueError(
            f"predictor_depth must be one of {sorted(PREDICTOR_DEPTH_CHOICES)}"
        )
    if str(cfg.recon_loss) not in RECON_LOSS_CHOICES:
        raise ValueError(f"recon_loss must be one of {sorted(RECON_LOSS_CHOICES)}")

    if int(cfg.height) % int(cfg.patch_size) != 0 or int(cfg.width) % int(cfg.patch_size) != 0:
        raise ValueError("height/width must be divisible by patch-size")
    if int(cfg.t_bins) % int(cfg.tubelet_size) != 0:
        raise ValueError("t-bins must be divisible by tubelet-size")
    if int(cfg.num_events_min) < 1 or int(cfg.num_events_max) < int(cfg.num_events_min):
        raise ValueError("num-events range is invalid")
    if float(cfg.sigreg_weight) < 0:
        raise ValueError("sigreg-weight must be >= 0")
    if float(cfg.vicreg_std_weight) < 0 or float(cfg.vicreg_cov_weight) < 0:
        raise ValueError("vicreg weights must be >= 0")
    if float(cfg.vicreg_eps) <= 0:
        raise ValueError("vicreg-eps must be > 0")
    if not (0.0 <= float(cfg.ema_momentum) < 1.0):
        raise ValueError("ema-momentum must be in [0, 1)")
    if int(cfg.sigreg_knots) < 2:
        raise ValueError("sigreg-knots must be >= 2")
    if int(cfg.sigreg_proj) < 1:
        raise ValueError("sigreg-proj must be >= 1")


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)


@torch.no_grad()
def _update_ema(
    student: torch.nn.Module, teacher: torch.nn.Module, momentum: float
) -> None:
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.mul_(momentum).add_(p_s.detach(), alpha=1.0 - momentum)
    for b_s, b_t in zip(student.buffers(), teacher.buffers()):
        b_t.copy_(b_s)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_step1")
def main(cfg: DictConfig) -> None:
    maybe_validate_cfg(cfg)
    collapse_strategy = get_collapse_strategy(cfg)
    set_seed(int(cfg.seed))
    device = select_device(str(cfg.device))

    encoder, predictor = build_models(cfg)
    encoder.to(device)
    predictor.to(device)
    encoder.train()
    predictor.train()
    teacher_encoder: torch.nn.Module | None = None
    if collapse_strategy == "ema_stopgrad":
        teacher_encoder = copy.deepcopy(encoder).to(device)
        teacher_encoder.eval()
        _set_requires_grad(teacher_encoder, False)

    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    sigreg = SIGReg(knots=int(cfg.sigreg_knots), num_proj=int(cfg.sigreg_proj)).to(device)
    vicreg = VICRegRegularizer(eps=float(cfg.vicreg_eps)).to(device)

    sigreg_weight = float(cfg.sigreg_weight)
    vicreg_std_weight = float(cfg.vicreg_std_weight)
    vicreg_cov_weight = float(cfg.vicreg_cov_weight)
    if collapse_strategy == "sigreg" and sigreg_weight == 0.0:
        sigreg_weight = 0.01
    if (
        collapse_strategy == "vicreg"
        and vicreg_std_weight == 0.0
        and vicreg_cov_weight == 0.0
    ):
        vicreg_std_weight = 0.1
        vicreg_cov_weight = 0.01

    mask_generator = build_mask_generator(cfg)
    voxel_builder = VoxelGrid(
        channels=int(cfg.t_bins), height=int(cfg.height), width=int(cfg.width)
    )

    out_dir = Path(to_absolute_path(str(cfg.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} model={cfg.model_size} collapse_strategy={collapse_strategy} "
        f"encoder_params={count_trainable_params(encoder):,} "
        f"predictor_params={count_trainable_params(predictor):,}"
    )

    for step in range(1, int(cfg.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        inputs = sample_voxel_batch(
            repr_builder=voxel_builder,
            batch_size=int(cfg.batch_size),
            num_events_min=int(cfg.num_events_min),
            num_events_max=int(cfg.num_events_max),
            normalize_voxel=bool(cfg.normalize_voxel),
        ).to(device)

        masks_enc, masks_pred = mask_generator(int(cfg.batch_size))
        masks_enc = masks_enc.to(device=device, non_blocking=True)
        masks_pred = masks_pred.to(device=device, non_blocking=True)

        if collapse_strategy == "ema_stopgrad":
            assert teacher_encoder is not None
            with torch.no_grad():
                teacher_tokens = teacher_encoder(inputs, training=True)
                teacher_tokens = apply_masks(teacher_tokens, [masks_pred])
        else:
            teacher_tokens = encoder(inputs, training=True)
            teacher_tokens = apply_masks(teacher_tokens, [masks_pred])

        context_tokens = encoder(inputs, masks=[masks_enc], training=True)
        pred_tokens, _ = predictor(
            context_tokens,
            masks_x=[masks_enc],
            masks_y=[masks_pred],
            mod="video",
            mask_index=step,
        )

        if bool(cfg.normalize_targets):
            teacher_tokens = F.layer_norm(teacher_tokens, (teacher_tokens.shape[-1],))
            pred_tokens = F.layer_norm(pred_tokens, (pred_tokens.shape[-1],))

        if str(cfg.recon_loss) == "smooth_l1":
            recon_loss = F.smooth_l1_loss(pred_tokens, teacher_tokens)
        else:
            recon_loss = F.mse_loss(pred_tokens, teacher_tokens)

        sig_loss = pred_tokens.new_zeros(())
        if collapse_strategy == "sigreg":
            # Flatten masked tokens to increase sample count for the sketch test.
            sig_loss = sigreg(pred_tokens.reshape(1, -1, pred_tokens.shape[-1]))

        std_loss = pred_tokens.new_zeros(())
        cov_loss = pred_tokens.new_zeros(())
        if collapse_strategy == "vicreg":
            vic_y = teacher_tokens if bool(cfg.vicreg_use_target) else None
            std_loss, cov_loss = vicreg(pred_tokens, vic_y)

        loss = (
            recon_loss
            + sigreg_weight * sig_loss
            + vicreg_std_weight * std_loss
            + vicreg_cov_weight * cov_loss
        )

        loss.backward()
        if float(cfg.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
        optimizer.step()
        if teacher_encoder is not None:
            _update_ema(encoder, teacher_encoder, float(cfg.ema_momentum))

        if step % int(cfg.print_every) == 0 or step == 1:
            print(
                f"step={step:05d}/{int(cfg.steps):05d} "
                f"loss={loss.item():.6f} "
                f"recon={recon_loss.item():.6f} "
                f"sig={sig_loss.item():.6f} "
                f"std={std_loss.item():.6f} "
                f"cov={cov_loss.item():.6f} "
                f"context_tokens={masks_enc.shape[1]} pred_tokens={masks_pred.shape[1]}"
            )

        if int(cfg.save_every) > 0 and step % int(cfg.save_every) == 0:
            ckpt_path = out_dir / f"step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "encoder": encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"saved={ckpt_path}")

    print("finished")


if __name__ == "__main__":
    main()
