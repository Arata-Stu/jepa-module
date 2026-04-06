#!/usr/bin/env python3
"""
Step1 prototype for event-based JEPA pretraining.

- input: event voxel grids from synthetic events (placeholder before real dataloader)
- task: masked token prediction at the same timestamp
- models: JEPA ViT encoder + predictor in `src/jepa`
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F


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


MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "tiny": {"builder": vit_tiny, "embed_dim": 192, "predictor_heads": 6},
    "small": {"builder": vit_small, "embed_dim": 384, "predictor_heads": 12},
    "base": {"builder": vit_base, "embed_dim": 768, "predictor_heads": 12},
    "large": {"builder": vit_large, "embed_dim": 1024, "predictor_heads": 16},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step1 masked pretraining prototype")
    parser.add_argument("--model-size", default="tiny", choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--t-bins", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)

    parser.add_argument("--predictor-depth", type=int, default=4, choices=[4, 8, 12, 20, 24, 40])
    parser.add_argument("--predictor-embed-dim", type=int, default=384)
    parser.add_argument("--predictor-heads", type=int, default=None)
    parser.add_argument("--num-mask-tokens", type=int, default=2)

    parser.add_argument("--num-events-min", type=int, default=2000)
    parser.add_argument("--num-events-max", type=int, default=12000)
    parser.add_argument("--normalize-voxel", action="store_true")
    parser.add_argument("--normalize-targets", action="store_true")

    parser.add_argument("--spatial-mask-min", type=float, default=0.2)
    parser.add_argument("--spatial-mask-max", type=float, default=0.5)
    parser.add_argument("--temporal-mask-min", type=float, default=0.5)
    parser.add_argument("--temporal-mask-max", type=float, default=1.0)
    parser.add_argument("--aspect-min", type=float, default=0.5)
    parser.add_argument("--aspect-max", type=float, default=2.0)
    parser.add_argument("--num-pred-blocks", type=int, default=1)
    parser.add_argument("--max-context-frames-ratio", type=float, default=1.0)

    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--out-dir", default="checkpoints/step1_prototype")
    return parser.parse_args()


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


def build_models(
    args: argparse.Namespace,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    spec = MODEL_SPECS[args.model_size]
    encoder_builder: Callable[..., torch.nn.Module] = spec["builder"]  # type: ignore[assignment]
    embed_dim: int = int(spec["embed_dim"])
    predictor_heads = (
        args.predictor_heads
        if args.predictor_heads is not None
        else int(spec["predictor_heads"])
    )
    if args.predictor_embed_dim % predictor_heads != 0:
        raise ValueError(
            "predictor-embed-dim must be divisible by predictor-heads "
            f"(got {args.predictor_embed_dim} and {predictor_heads})"
        )

    encoder = encoder_builder(
        img_size=(args.height, args.width),
        patch_size=args.patch_size,
        num_frames=args.t_bins,
        tubelet_size=args.tubelet_size,
        in_chans=1,
        use_rope=True,
        modality_embedding=False,
        n_output_distillation=4,
    )

    predictor = vit_predictor(
        img_size=(args.height, args.width),
        patch_size=args.patch_size,
        num_frames=args.t_bins,
        tubelet_size=args.tubelet_size,
        embed_dim=embed_dim,
        predictor_embed_dim=args.predictor_embed_dim,
        depth=args.predictor_depth,
        num_heads=predictor_heads,
        use_rope=True,
        use_mask_tokens=True,
        num_mask_tokens=args.num_mask_tokens,
        modality_embedding=False,
        n_output_distillation=4,
    )
    return encoder, predictor


def build_mask_generator(args: argparse.Namespace) -> _MaskGenerator:
    return _MaskGenerator(
        crop_size=(args.height, args.width),
        num_frames=args.t_bins,
        spatial_patch_size=(args.patch_size, args.patch_size),
        temporal_patch_size=args.tubelet_size,
        spatial_pred_mask_scale=(args.spatial_mask_min, args.spatial_mask_max),
        temporal_pred_mask_scale=(args.temporal_mask_min, args.temporal_mask_max),
        aspect_ratio=(args.aspect_min, args.aspect_max),
        npred=args.num_pred_blocks,
        max_context_frames_ratio=args.max_context_frames_ratio,
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


def maybe_validate_args(args: argparse.Namespace) -> None:
    if args.height % args.patch_size != 0 or args.width % args.patch_size != 0:
        raise ValueError("height/width must be divisible by patch-size")
    if args.t_bins % args.tubelet_size != 0:
        raise ValueError("t-bins must be divisible by tubelet-size")
    if args.num_events_min < 1 or args.num_events_max < args.num_events_min:
        raise ValueError("num-events range is invalid")


def main() -> None:
    args = parse_args()
    maybe_validate_args(args)
    set_seed(args.seed)
    device = select_device(args.device)

    encoder, predictor = build_models(args)
    encoder.to(device)
    predictor.to(device)
    encoder.train()
    predictor.train()

    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    mask_generator = build_mask_generator(args)
    voxel_builder = VoxelGrid(channels=args.t_bins, height=args.height, width=args.width)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} model={args.model_size} "
        f"encoder_params={count_trainable_params(encoder):,} "
        f"predictor_params={count_trainable_params(predictor):,}"
    )

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        inputs = sample_voxel_batch(
            repr_builder=voxel_builder,
            batch_size=args.batch_size,
            num_events_min=args.num_events_min,
            num_events_max=args.num_events_max,
            normalize_voxel=args.normalize_voxel,
        ).to(device)

        masks_enc, masks_pred = mask_generator(args.batch_size)
        masks_enc = masks_enc.to(device=device, non_blocking=True)
        masks_pred = masks_pred.to(device=device, non_blocking=True)

        with torch.no_grad():
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

        if args.normalize_targets:
            teacher_tokens = F.layer_norm(teacher_tokens, (teacher_tokens.shape[-1],))
            pred_tokens = F.layer_norm(pred_tokens, (pred_tokens.shape[-1],))

        loss = F.smooth_l1_loss(pred_tokens, teacher_tokens)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == 1:
            print(
                f"step={step:05d}/{args.steps:05d} "
                f"loss={loss.item():.6f} "
                f"context_tokens={masks_enc.shape[1]} pred_tokens={masks_pred.shape[1]}"
            )

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt_path = out_dir / f"step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "args": vars(args),
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
