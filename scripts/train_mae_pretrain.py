#!/usr/bin/env python3
"""
MAE pretraining script for event voxel grids.

- encoder: JEPA ViT backbone (`src/jepa/models/vision_transformer.py`)
- decoder: lightweight transformer decoder (`src/jepa/models/mae.py`)
- data: same providers as step1 (`synthetic`, `n_imagenet`, `dsec`, `pretrain_mixed`)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from event.representations import VoxelGrid  # noqa: E402
from jepa.models.mae import MAEForwardOutput, VoxelMAE  # noqa: E402
from jepa.utils.distributed import (  # noqa: E402
    DistributedState,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    reduce_mean_scalar,
    unwrap_module,
)
from jepa.utils.schedulers import WarmupCosineParamScheduler  # noqa: E402
from train_jepa_pretrain import (  # noqa: E402
    MODEL_SPECS,
    PRETRAIN_MIXED_SOURCE_NAMES,
    _compute_source_ratios,
    _configure_data_loader_runtime,
    _extract_scalar,
    build_batch_provider,
    build_eval_batch_provider,
    maybe_validate_cfg,
    select_device,
    set_seed,
)


MAE_RECON_LOSS_CHOICES = {"mse", "smooth_l1"}


def _validate_mae_cfg(cfg: DictConfig) -> None:
    maybe_validate_cfg(cfg)

    mae_cfg = cfg.get("mae", None)
    if mae_cfg is None:
        raise ValueError("mae config block is required")

    if str(cfg.model_size) not in MODEL_SPECS:
        raise ValueError(f"model_size must be one of {list(MODEL_SPECS.keys())}")
    if not (0.0 < float(mae_cfg.mask_ratio) < 1.0):
        raise ValueError("mae.mask_ratio must be in (0, 1)")
    if str(mae_cfg.recon_loss) not in MAE_RECON_LOSS_CHOICES:
        raise ValueError(f"mae.recon_loss must be one of {sorted(MAE_RECON_LOSS_CHOICES)}")
    if int(mae_cfg.decoder_depth) < 1:
        raise ValueError("mae.decoder_depth must be >= 1")
    if int(mae_cfg.decoder_embed_dim) < 1:
        raise ValueError("mae.decoder_embed_dim must be >= 1")
    if mae_cfg.decoder_heads is not None and int(mae_cfg.decoder_heads) < 1:
        raise ValueError("mae.decoder_heads must be >= 1 when set")


def build_mae_model(cfg: DictConfig) -> VoxelMAE:
    spec = MODEL_SPECS[str(cfg.model_size)]
    encoder_builder = spec["builder"]  # type: ignore[assignment]
    encoder_embed_dim = int(spec["embed_dim"])
    default_decoder_heads = int(spec["predictor_heads"])

    decoder_heads = (
        int(cfg.mae.decoder_heads)
        if cfg.mae.decoder_heads is not None
        else default_decoder_heads
    )
    decoder_embed_dim = int(cfg.mae.decoder_embed_dim)
    if decoder_embed_dim % decoder_heads != 0:
        raise ValueError(
            "mae.decoder_embed_dim must be divisible by decoder_heads "
            f"(got {decoder_embed_dim} and {decoder_heads})"
        )

    encoder = encoder_builder(
        img_size=(int(cfg.height), int(cfg.width)),
        patch_size=int(cfg.patch_size),
        num_frames=int(cfg.t_bins),
        tubelet_size=int(cfg.tubelet_size),
        in_chans=1,
        use_rope=bool(cfg.mae.encoder_use_rope),
        modality_embedding=False,
        n_output_distillation=1,
    )
    model = VoxelMAE(
        encoder=encoder,
        encoder_embed_dim=encoder_embed_dim,
        img_size=(int(cfg.height), int(cfg.width)),
        patch_size=int(cfg.patch_size),
        num_frames=int(cfg.t_bins),
        tubelet_size=int(cfg.tubelet_size),
        in_chans=1,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=int(cfg.mae.decoder_depth),
        decoder_num_heads=decoder_heads,
        mlp_ratio=float(cfg.mae.decoder_mlp_ratio),
        use_silu=bool(cfg.mae.decoder_use_silu),
    )
    return model


def _forward_step(
    cfg: DictConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    output = model(
        inputs=inputs,
        mask_ratio=float(cfg.mae.mask_ratio),
        normalize_targets=bool(cfg.mae.normalize_targets),
        recon_loss=str(cfg.mae.recon_loss),
        loss_on_mask_only=bool(cfg.mae.loss_on_mask_only),
    )
    if not isinstance(output, MAEForwardOutput):
        raise RuntimeError(f"Unexpected MAE output type: {type(output)}")
    return {
        "loss": output.loss,
        "masked_loss": output.masked_loss,
        "visible_loss": output.visible_loss,
        "mask_ratio": float(output.mask_ratio),
        "batch_size": int(inputs.shape[0]),
    }


@torch.no_grad()
def _run_eval(
    cfg: DictConfig,
    device: torch.device,
    eval_provider: Any,
    model: torch.nn.Module,
) -> dict[str, float]:
    steps = int(cfg.eval.steps)
    if steps < 1:
        raise ValueError("eval.steps must be >= 1")

    totals = {
        "loss": 0.0,
        "masked_loss": 0.0,
        "visible_loss": 0.0,
        "mask_ratio": 0.0,
    }
    for _ in range(steps):
        batch = eval_provider.next_batch()
        inputs = batch["inputs"].to(device, non_blocking=True)
        metrics = _forward_step(cfg=cfg, model=model, inputs=inputs)
        totals["loss"] += _extract_scalar(metrics["loss"])
        totals["masked_loss"] += _extract_scalar(metrics["masked_loss"])
        totals["visible_loss"] += _extract_scalar(metrics["visible_loss"])
        totals["mask_ratio"] += float(metrics["mask_ratio"])
    return {k: v / steps for k, v in totals.items()}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_mae")
def main(cfg: DictConfig) -> None:
    _validate_mae_cfg(cfg)
    _configure_data_loader_runtime(cfg)

    dist_state = init_distributed(
        enabled=bool(cfg.distributed.enabled),
        backend=str(cfg.distributed.backend),
        port=int(cfg.distributed.port),
        timeout_sec=int(cfg.distributed.timeout_sec),
    )
    main_process = is_main_process(dist_state)

    set_seed(int(cfg.seed) + (dist_state.rank if dist_state.enabled else 0))
    device = select_device(
        str(cfg.device),
        distributed=dist_state.enabled,
        local_rank=dist_state.local_rank,
    )
    if dist_state.enabled and device.type == "cuda":
        torch.cuda.set_device(device)

    model = build_mae_model(cfg).to(device)
    model.train()

    if dist_state.enabled:
        ddp_kwargs: dict[str, Any] = {
            "broadcast_buffers": bool(cfg.distributed.broadcast_buffers),
            "find_unused_parameters": bool(cfg.distributed.find_unused_parameters),
        }
        if device.type == "cuda":
            assert device.index is not None
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        model = DDP(model, **ddp_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    lr_scheduler = None
    wd_scheduler = None
    if bool(cfg.scheduler.enabled):
        lr_scheduler = WarmupCosineParamScheduler(
            optimizer=optimizer,
            param_name="lr",
            base_value=float(cfg.lr),
            final_value=float(cfg.scheduler.final_lr),
            total_steps=int(cfg.steps),
            warmup_steps=int(cfg.scheduler.warmup_steps),
            start_value=float(cfg.scheduler.start_lr),
        )
        if bool(cfg.scheduler.update_weight_decay):
            wd_scheduler = WarmupCosineParamScheduler(
                optimizer=optimizer,
                param_name="weight_decay",
                base_value=float(cfg.weight_decay),
                final_value=float(cfg.scheduler.final_weight_decay),
                total_steps=int(cfg.steps),
                warmup_steps=0,
                start_value=float(cfg.weight_decay),
            )

    voxel_builder = VoxelGrid(
        channels=int(cfg.t_bins),
        height=int(cfg.height),
        width=int(cfg.width),
    )
    batch_provider = build_batch_provider(cfg, voxel_builder, dist_state=dist_state)
    eval_provider = build_eval_batch_provider(cfg, voxel_builder, dist_state=dist_state)
    dataset_samples = getattr(batch_provider, "num_samples", None)
    eval_dataset_samples = getattr(eval_provider, "num_samples", None)

    out_dir = Path(to_absolute_path(str(cfg.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(to_absolute_path(str(cfg.log_dir)))
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = Path(to_absolute_path(str(cfg.metrics_file)))
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    tensorboard_writer = None
    tensorboard_dir = None
    if main_process and bool(cfg.logging.tensorboard.enabled):
        if SummaryWriter is None:
            raise ModuleNotFoundError(
                "TensorBoard is not available. Install `tensorboard` (see requirements.txt)."
            )
        tensorboard_dir = log_dir / str(cfg.logging.tensorboard.subdir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir))
        tensorboard_writer.add_text("run/config", OmegaConf.to_yaml(cfg), 0)

    metrics_fp = None
    metrics_writer = None
    eval_metrics_fp = None
    eval_metrics_writer = None
    if main_process:
        metrics_fp = metrics_file.open("w", newline="", encoding="utf-8")
        metrics_writer = csv.writer(metrics_fp)
        metrics_writer.writerow(
            [
                "step",
                "batch_size",
                "loss",
                "masked_loss",
                "visible_loss",
                "mask_ratio",
                "lr",
                "weight_decay",
            ]
        )
        if eval_provider is not None:
            eval_metrics_file = log_dir / "eval_metrics.csv"
            eval_metrics_fp = eval_metrics_file.open("w", newline="", encoding="utf-8")
            eval_metrics_writer = csv.writer(eval_metrics_fp)
            eval_metrics_writer.writerow(
                ["step", "loss", "masked_loss", "visible_loss", "mask_ratio", "eval_steps"]
            )

    if main_process:
        print(
            f"device={device} model={cfg.model_size} task=mae "
            f"mask_ratio={float(cfg.mae.mask_ratio):.3f} "
            f"loss_on_mask_only={bool(cfg.mae.loss_on_mask_only)} "
            f"data_source={cfg.data.source} "
            f"rank={dist_state.rank}/{dist_state.world_size} backend={dist_state.backend} "
            f"model_params={sum(p.numel() for p in unwrap_module(model).parameters() if p.requires_grad):,} "
            f"dataset_samples={dataset_samples if dataset_samples is not None else 'n/a'} "
            f"eval_dataset_samples={eval_dataset_samples if eval_dataset_samples is not None else 'n/a'} "
            f"log_dir={log_dir} ckpt_dir={out_dir} metrics_file={metrics_file} "
            f"tensorboard_dir={tensorboard_dir if tensorboard_dir is not None else 'disabled'}"
        )

    try:
        for step in range(1, int(cfg.steps) + 1):
            step_lr = (
                float(lr_scheduler.step(step))
                if lr_scheduler is not None
                else float(optimizer.param_groups[0]["lr"])
            )
            step_wd = (
                float(wd_scheduler.step(step))
                if wd_scheduler is not None
                else float(optimizer.param_groups[0].get("weight_decay", 0.0))
            )

            optimizer.zero_grad(set_to_none=True)
            batch = batch_provider.next_batch()
            inputs = batch["inputs"].to(device, non_blocking=True)

            source_ratios = None
            if str(cfg.data.source) == "pretrain_mixed":
                source_ratios = _compute_source_ratios(
                    batch=batch,
                    dist_state=dist_state,
                    device=device,
                )

            step_metrics = _forward_step(cfg=cfg, model=model, inputs=inputs)
            loss = step_metrics["loss"]
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            if float(cfg.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

            step_loss = _extract_scalar(step_metrics["loss"])
            step_masked = _extract_scalar(step_metrics["masked_loss"])
            step_visible = _extract_scalar(step_metrics["visible_loss"])
            step_mask_ratio = float(step_metrics["mask_ratio"])
            step_batch_size = int(step_metrics["batch_size"])

            if dist_state.enabled:
                step_loss = reduce_mean_scalar(step_loss, device=device)
                step_masked = reduce_mean_scalar(step_masked, device=device)
                step_visible = reduce_mean_scalar(step_visible, device=device)
                step_mask_ratio = reduce_mean_scalar(step_mask_ratio, device=device)
                step_batch_size = int(
                    reduce_mean_scalar(float(step_batch_size), device=device)
                )

            if main_process and metrics_writer is not None and metrics_fp is not None:
                metrics_writer.writerow(
                    [
                        step,
                        step_batch_size,
                        f"{step_loss:.8f}",
                        f"{step_masked:.8f}",
                        f"{step_visible:.8f}",
                        f"{step_mask_ratio:.6f}",
                        f"{step_lr:.10f}",
                        f"{step_wd:.10f}",
                    ]
                )
                metrics_fp.flush()

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("train/loss", step_loss, step)
                tensorboard_writer.add_scalar("train/masked_loss", step_masked, step)
                tensorboard_writer.add_scalar("train/visible_loss", step_visible, step)
                tensorboard_writer.add_scalar("train/mask_ratio", step_mask_ratio, step)
                tensorboard_writer.add_scalar("train/lr", step_lr, step)
                tensorboard_writer.add_scalar("train/weight_decay", step_wd, step)
                if source_ratios is not None:
                    for source_name, ratio in source_ratios.items():
                        tensorboard_writer.add_scalar(
                            f"train/source_ratio_{source_name}",
                            float(ratio),
                            step,
                        )

            if main_process and (step % int(cfg.print_every) == 0 or step == 1):
                source_ratio_str = ""
                if source_ratios is not None:
                    source_ratio_str = " " + " ".join(
                        [
                            f"src_{name}={float(source_ratios.get(name, 0.0)):.3f}"
                            for name in PRETRAIN_MIXED_SOURCE_NAMES
                        ]
                    )
                print(
                    f"step={step:05d}/{int(cfg.steps):05d} "
                    f"loss={step_loss:.6f} "
                    f"masked={step_masked:.6f} "
                    f"visible={step_visible:.6f} "
                    f"mask_ratio={step_mask_ratio:.4f} "
                    f"lr={step_lr:.8f} "
                    f"wd={step_wd:.8f}"
                    f"{source_ratio_str}"
                )

            if eval_provider is not None and step % int(cfg.eval.every) == 0:
                model.eval()
                eval_stats = _run_eval(
                    cfg=cfg,
                    device=device,
                    eval_provider=eval_provider,
                    model=model,
                )
                model.train()

                if dist_state.enabled:
                    eval_stats = {
                        k: reduce_mean_scalar(v, device=device)
                        for k, v in eval_stats.items()
                    }

                if main_process and eval_metrics_writer is not None and eval_metrics_fp is not None:
                    eval_metrics_writer.writerow(
                        [
                            step,
                            f"{eval_stats['loss']:.8f}",
                            f"{eval_stats['masked_loss']:.8f}",
                            f"{eval_stats['visible_loss']:.8f}",
                            f"{eval_stats['mask_ratio']:.6f}",
                            int(cfg.eval.steps),
                        ]
                    )
                    eval_metrics_fp.flush()

                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("eval/loss", eval_stats["loss"], step)
                    tensorboard_writer.add_scalar("eval/masked_loss", eval_stats["masked_loss"], step)
                    tensorboard_writer.add_scalar(
                        "eval/visible_loss", eval_stats["visible_loss"], step
                    )
                    tensorboard_writer.add_scalar("eval/mask_ratio", eval_stats["mask_ratio"], step)

                if main_process:
                    print(
                        f"eval@step={step:05d} "
                        f"loss={eval_stats['loss']:.6f} "
                        f"masked={eval_stats['masked_loss']:.6f} "
                        f"visible={eval_stats['visible_loss']:.6f} "
                        f"mask_ratio={eval_stats['mask_ratio']:.4f}"
                    )

            if main_process and int(cfg.save_every) > 0 and step % int(cfg.save_every) == 0:
                ckpt_path = out_dir / f"step_{step:06d}.pt"
                torch.save(
                    {
                        "step": step,
                        "cfg": OmegaConf.to_container(cfg, resolve=True),
                        "model": unwrap_module(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"saved={ckpt_path}")
    finally:
        if metrics_fp is not None:
            metrics_fp.close()
        if eval_metrics_fp is not None:
            eval_metrics_fp.close()
        if tensorboard_writer is not None:
            tensorboard_writer.flush()
            tensorboard_writer.close()
        if dist_state.enabled:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            cleanup_distributed()

    if main_process:
        print("finished")


if __name__ == "__main__":
    main()
