#!/usr/bin/env python3
"""
N-ImageNet linear probe entrypoint.

- Loads a JEPA pretrained encoder checkpoint
- Freezes the encoder
- Trains only a linear classification head
- Uses full event files by default (no random event slicing)
"""

from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import hydra
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from event.data import (  # noqa: E402
    NImageNetEventsDataset,
    NImageNetVoxelCollator,
    ensure_path_exists,
    resolve_list_file,
)
from event.representations import VoxelGrid  # noqa: E402
from jepa.models.vision_transformer import (  # noqa: E402
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)


MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "tiny": {"builder": vit_tiny, "embed_dim": 192},
    "small": {"builder": vit_small, "embed_dim": 384},
    "base": {"builder": vit_base, "embed_dim": 768},
    "large": {"builder": vit_large, "embed_dim": 1024},
}

FEATURE_POOL_CHOICES = {"mean", "max", "mean_max"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _data_loader_prefetch_factor(cfg: DictConfig) -> int | None:
    value = cfg.data.get("prefetch_factor", None)
    if value is None:
        return None
    return int(value)


def _data_loader_persistent_workers(cfg: DictConfig) -> bool:
    return bool(cfg.data.get("persistent_workers", False))


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _resolve_n_imagenet_root_dir(cfg: DictConfig) -> str | None:
    root_dir = cfg.data.n_imagenet.root_dir
    if root_dir is None:
        return None
    root_dir_abs = to_absolute_path(str(root_dir))
    ensure_path_exists(root_dir_abs, "N-ImageNet root_dir")
    return root_dir_abs


def _resolve_n_imagenet_list_file(cfg: DictConfig, split: str) -> str:
    list_file = resolve_list_file(
        split=split,
        train_list=cfg.data.n_imagenet.train_list,
        val_list=cfg.data.n_imagenet.val_list,
    )
    list_file = to_absolute_path(str(list_file))
    ensure_path_exists(list_file, f"N-ImageNet {split} list file")
    return list_file


def _build_n_imagenet_dataset(
    cfg: DictConfig,
    split: str,
    class_names: list[str] | None = None,
) -> NImageNetEventsDataset:
    force_full_event_input = bool(cfg.data.n_imagenet.force_full_event_input)
    slice_enabled = bool(cfg.data.n_imagenet.slice.enabled)
    if force_full_event_input and slice_enabled:
        slice_enabled = False

    augment_enabled = (
        bool(cfg.data.n_imagenet.augment.enabled)
        and split == "train"
    )

    limit_samples = (
        cfg.data.n_imagenet.limit_samples_train
        if split == "train"
        else cfg.data.n_imagenet.limit_samples_val
    )

    return NImageNetEventsDataset(
        list_file=_resolve_n_imagenet_list_file(cfg=cfg, split=split),
        split=split,
        compressed=bool(cfg.data.n_imagenet.compressed),
        time_scale=float(cfg.data.n_imagenet.time_scale),
        root_dir=_resolve_n_imagenet_root_dir(cfg=cfg),
        limit_samples=limit_samples,
        limit_classes=cfg.data.n_imagenet.limit_classes,
        class_names=class_names,
        sensor_height=int(cfg.data.n_imagenet.sensor_height),
        sensor_width=int(cfg.data.n_imagenet.sensor_width),
        slice_enabled=slice_enabled,
        slice_mode=str(cfg.data.n_imagenet.slice.mode),
        slice_start=cfg.data.n_imagenet.slice.start,
        slice_end=cfg.data.n_imagenet.slice.end,
        slice_length=int(cfg.data.n_imagenet.slice.length),
        random_slice_on_train=False,
        augment_enabled=augment_enabled,
        hflip_prob=float(cfg.data.n_imagenet.augment.hflip_prob),
        max_shift=int(cfg.data.n_imagenet.augment.max_shift),
    )


def _build_loader(
    cfg: DictConfig,
    dataset: NImageNetEventsDataset,
    voxel_builder: VoxelGrid,
    split: str,
) -> DataLoader:
    collator = NImageNetVoxelCollator(
        voxel_builder=voxel_builder,
        normalize_voxel=bool(cfg.normalize_voxel),
        sensor_height=int(cfg.data.n_imagenet.sensor_height),
        sensor_width=int(cfg.data.n_imagenet.sensor_width),
        rescale_to_voxel_grid=bool(cfg.data.n_imagenet.rescale_to_voxel_grid),
    )

    loader_kwargs: dict[str, Any] = {}
    if int(cfg.data.num_workers) > 0:
        loader_kwargs["persistent_workers"] = _data_loader_persistent_workers(cfg)
        prefetch_factor = _data_loader_prefetch_factor(cfg)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=(split == "train"),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=bool(cfg.data.drop_last_train) if split == "train" else False,
        collate_fn=collator,
        **loader_kwargs,
    )


def build_encoder(cfg: DictConfig) -> tuple[nn.Module, int]:
    model_size = str(cfg.model.model_size)
    if model_size not in MODEL_SPECS:
        raise ValueError(f"model.model_size must be one of {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[model_size]
    encoder_builder: Callable[..., nn.Module] = spec["builder"]  # type: ignore[assignment]
    embed_dim = int(spec["embed_dim"])

    temporal_mix_enabled = bool(cfg.model.temporal_mix.enabled)
    img_temporal_dim_size = (
        int(cfg.model.temporal_mix.short_t) if temporal_mix_enabled else None
    )

    encoder = encoder_builder(
        img_size=(int(cfg.model.height), int(cfg.model.width)),
        patch_size=int(cfg.model.patch_size),
        num_frames=int(cfg.model.t_bins),
        tubelet_size=int(cfg.model.tubelet_size),
        in_chans=1,
        use_rope=True,
        modality_embedding=temporal_mix_enabled,
        img_temporal_dim_size=img_temporal_dim_size,
        n_output_distillation=4,
    )
    return encoder, embed_dim


def load_pretrained_encoder(encoder: nn.Module, cfg: DictConfig) -> str:
    checkpoint_path = cfg.pretrained.checkpoint
    if checkpoint_path is None:
        raise ValueError("pretrained.checkpoint must be set")

    checkpoint_path_abs = to_absolute_path(str(checkpoint_path))
    ensure_path_exists(checkpoint_path_abs, "pretrained.checkpoint")

    checkpoint = torch.load(checkpoint_path_abs, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path_abs}")

    preferred_key = str(cfg.pretrained.encoder_key)
    state = checkpoint.get(preferred_key, None)
    loaded_key = preferred_key
    if state is None and bool(cfg.pretrained.fallback_to_encoder):
        state = checkpoint.get("encoder", None)
        loaded_key = "encoder"

    if state is None:
        available_keys = sorted(checkpoint.keys())
        raise KeyError(
            f"Encoder state '{preferred_key}' not found in checkpoint. "
            f"Available keys: {available_keys}"
        )
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint key '{loaded_key}' is not a state_dict")

    state = _normalize_state_dict_keys(state)
    encoder.load_state_dict(state, strict=bool(cfg.pretrained.strict))
    return loaded_key


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def _topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = max(1, min(int(k), int(logits.shape[1])))
    pred = torch.topk(logits, k=k, dim=1).indices
    correct = pred.eq(labels.unsqueeze(1))
    return float(correct.any(dim=1).sum().item())


def _pool_features(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens to be [B, N, D], got shape={tuple(tokens.shape)}")

    if mode == "mean":
        return tokens.mean(dim=1)
    if mode == "max":
        return tokens.max(dim=1).values
    if mode == "mean_max":
        mean_f = tokens.mean(dim=1)
        max_f = tokens.max(dim=1).values
        return torch.cat([mean_f, max_f], dim=1)
    raise ValueError(f"Unknown feature pooling mode: {mode}")


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    inputs: torch.Tensor,
    feature_pooling: str,
) -> torch.Tensor:
    tokens = encoder(inputs, training=False)
    if not isinstance(tokens, torch.Tensor):
        raise ValueError("Encoder output must be a tensor for linear probe training")
    return _pool_features(tokens=tokens, mode=feature_pooling)


def build_lr_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if not bool(cfg.scheduler.enabled):
        return None

    total_steps = max(1, int(cfg.epochs) * max(1, int(steps_per_epoch)))
    warmup_steps = max(0, int(cfg.scheduler.warmup_epochs) * max(1, int(steps_per_epoch)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        cosine_steps = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(cosine_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_epoch(
    *,
    split: str,
    loader: DataLoader,
    encoder: nn.Module,
    head: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    feature_pooling: str,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    training = split == "train"
    head.train(training)

    total_loss = 0.0
    total_samples = 0.0
    correct_top1 = 0.0
    correct_top5 = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if labels.numel() == 0:
            continue
        if (labels < 0).any():
            raise ValueError("Found negative class labels; check class mapping/list files")

        features = extract_features(
            encoder=encoder,
            inputs=inputs,
            feature_pooling=feature_pooling,
        )

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                logits = head(features)
                loss = criterion(logits, labels)

        batch_size = float(labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        correct_top1 += _topk_correct(logits, labels, k=1)
        correct_top5 += _topk_correct(logits, labels, k=5)

    if total_samples <= 0:
        return {
            "loss": float("nan"),
            "top1": float("nan"),
            "top5": float("nan"),
        }

    return {
        "loss": total_loss / total_samples,
        "top1": correct_top1 / total_samples,
        "top5": correct_top5 / total_samples,
    }


def maybe_validate_cfg(cfg: DictConfig) -> None:
    if str(cfg.model.model_size) not in MODEL_SPECS:
        raise ValueError(f"model.model_size must be one of {list(MODEL_SPECS.keys())}")
    if str(cfg.model.feature_pooling) not in FEATURE_POOL_CHOICES:
        raise ValueError(
            "model.feature_pooling must be one of "
            f"{sorted(FEATURE_POOL_CHOICES)}"
        )
    if int(cfg.model.t_bins) < 1:
        raise ValueError("model.t_bins must be >= 1")
    if int(cfg.model.tubelet_size) < 1:
        raise ValueError("model.tubelet_size must be >= 1")
    if int(cfg.model.t_bins) % int(cfg.model.tubelet_size) != 0:
        raise ValueError("model.t_bins must be divisible by model.tubelet_size")
    if int(cfg.epochs) < 1:
        raise ValueError("epochs must be >= 1")
    if int(cfg.batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if float(cfg.optimizer.lr) <= 0.0:
        raise ValueError("optimizer.lr must be > 0")
    if float(cfg.optimizer.weight_decay) < 0.0:
        raise ValueError("optimizer.weight_decay must be >= 0")
    if int(cfg.logging.print_every) < 1:
        raise ValueError("logging.print_every must be >= 1")


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_n_imagenet_linear_probe",
)
def main(cfg: DictConfig) -> None:
    maybe_validate_cfg(cfg)

    set_seed(int(cfg.seed))
    device = select_device(str(cfg.device))

    voxel_builder = VoxelGrid(
        channels=int(cfg.model.t_bins),
        height=int(cfg.model.height),
        width=int(cfg.model.width),
    )

    train_dataset = _build_n_imagenet_dataset(cfg=cfg, split="train")
    val_dataset = _build_n_imagenet_dataset(
        cfg=cfg,
        split="val",
        class_names=train_dataset.class_names,
    )

    train_loader = _build_loader(
        cfg=cfg,
        dataset=train_dataset,
        voxel_builder=voxel_builder,
        split="train",
    )
    val_loader = _build_loader(
        cfg=cfg,
        dataset=val_dataset,
        voxel_builder=voxel_builder,
        split="val",
    )

    encoder, embed_dim = build_encoder(cfg)
    loaded_key = load_pretrained_encoder(encoder=encoder, cfg=cfg)
    freeze_module(encoder)
    encoder.to(device)

    feature_pooling = str(cfg.model.feature_pooling)
    head_in_dim = embed_dim * (2 if feature_pooling == "mean_max" else 1)
    inferred_num_classes = len(train_dataset.class_names)
    num_classes = (
        int(cfg.head.num_classes)
        if cfg.head.num_classes is not None
        else inferred_num_classes
    )
    if num_classes < inferred_num_classes:
        raise ValueError(
            f"head.num_classes ({num_classes}) must be >= number of train classes "
            f"({inferred_num_classes})"
        )

    head = nn.Linear(head_in_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    scheduler = build_lr_scheduler(
        cfg=cfg,
        optimizer=optimizer,
        steps_per_epoch=max(1, len(train_loader)),
    )

    out_dir = Path(to_absolute_path(str(cfg.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(to_absolute_path(str(cfg.log_dir)))
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = Path(to_absolute_path(str(cfg.metrics_file)))
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    tensorboard_writer = None
    if bool(cfg.logging.tensorboard.enabled):
        if SummaryWriter is None:
            raise ModuleNotFoundError(
                "tensorboard is not installed. Disable logging.tensorboard.enabled "
                "or install tensorboard."
            )
        tensorboard_dir = log_dir / str(cfg.logging.tensorboard.subdir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir))
        tensorboard_writer.add_text("run/config", OmegaConf.to_yaml(cfg), 0)

    best_top1 = float("-inf")
    best_epoch = -1

    with metrics_file.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "epoch",
                "split",
                "loss",
                "top1",
                "top5",
                "lr",
            ]
        )

        print(
            f"device={device} "
            f"train_samples={len(train_dataset)} val_samples={len(val_dataset)} "
            f"num_classes={num_classes} "
            f"feature_pooling={feature_pooling} "
            f"pretrained_key={loaded_key} "
            f"force_full_event_input={bool(cfg.data.n_imagenet.force_full_event_input)}"
        )

        for epoch in range(1, int(cfg.epochs) + 1):
            train_stats = run_epoch(
                split="train",
                loader=train_loader,
                encoder=encoder,
                head=head,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                feature_pooling=feature_pooling,
                grad_clip=float(cfg.optimizer.grad_clip),
                device=device,
            )
            current_lr = float(optimizer.param_groups[0]["lr"])
            writer.writerow(
                [
                    epoch,
                    "train",
                    f"{train_stats['loss']:.8f}",
                    f"{train_stats['top1']:.8f}",
                    f"{train_stats['top5']:.8f}",
                    f"{current_lr:.10f}",
                ]
            )

            do_eval = (epoch % int(cfg.eval.every_epochs) == 0) or (epoch == int(cfg.epochs))
            val_stats = None
            if do_eval:
                val_stats = run_epoch(
                    split="val",
                    loader=val_loader,
                    encoder=encoder,
                    head=head,
                    criterion=criterion,
                    optimizer=None,
                    scheduler=None,
                    feature_pooling=feature_pooling,
                    grad_clip=0.0,
                    device=device,
                )
                writer.writerow(
                    [
                        epoch,
                        "val",
                        f"{val_stats['loss']:.8f}",
                        f"{val_stats['top1']:.8f}",
                        f"{val_stats['top5']:.8f}",
                        f"{current_lr:.10f}",
                    ]
                )

            fp.flush()

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("train/loss", train_stats["loss"], epoch)
                tensorboard_writer.add_scalar("train/top1", train_stats["top1"], epoch)
                tensorboard_writer.add_scalar("train/top5", train_stats["top5"], epoch)
                tensorboard_writer.add_scalar("train/lr", current_lr, epoch)
                if val_stats is not None:
                    tensorboard_writer.add_scalar("val/loss", val_stats["loss"], epoch)
                    tensorboard_writer.add_scalar("val/top1", val_stats["top1"], epoch)
                    tensorboard_writer.add_scalar("val/top5", val_stats["top5"], epoch)

            if epoch == 1 or epoch % int(cfg.logging.print_every) == 0 or epoch == int(cfg.epochs):
                msg = (
                    f"epoch={epoch:04d}/{int(cfg.epochs):04d} "
                    f"train_loss={train_stats['loss']:.6f} "
                    f"train_top1={train_stats['top1']:.4f} "
                    f"train_top5={train_stats['top5']:.4f} "
                    f"lr={current_lr:.8f}"
                )
                if val_stats is not None:
                    msg += (
                        f" val_loss={val_stats['loss']:.6f} "
                        f"val_top1={val_stats['top1']:.4f} "
                        f"val_top5={val_stats['top5']:.4f}"
                    )
                print(msg)

            checkpoint_payload = {
                "epoch": epoch,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "train_stats": train_stats,
                "val_stats": val_stats,
                "class_names": train_dataset.class_names,
                "num_classes": num_classes,
                "feature_pooling": feature_pooling,
                "encoder_checkpoint": to_absolute_path(str(cfg.pretrained.checkpoint)),
                "encoder_key": loaded_key,
            }
            torch.save(checkpoint_payload, out_dir / "last.pt")

            if val_stats is not None and float(val_stats["top1"]) > best_top1:
                best_top1 = float(val_stats["top1"])
                best_epoch = epoch
                torch.save(checkpoint_payload, out_dir / "best.pt")

    if tensorboard_writer is not None:
        tensorboard_writer.flush()
        tensorboard_writer.close()

    print(
        f"finished best_epoch={best_epoch} "
        f"best_val_top1={best_top1:.6f} "
        f"ckpt_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
