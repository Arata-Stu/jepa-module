#!/usr/bin/env python3
"""
Step1 prototype for event-based JEPA pretraining.

- input: event voxel grids (`synthetic`, `n_imagenet`, `dsec`, `pretrain_mixed`)
- task: masked token prediction at the same timestamp
- models: JEPA ViT encoder + predictor in `src/jepa`
- config: Hydra (`configs/train_step1.yaml`)
"""

from __future__ import annotations

import csv
import copy
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # type: ignore[assignment]


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
from event.representations import VoxelGrid  # noqa: E402
from event.data import (  # noqa: E402
    DSECVoxelBatchProvider,
    NImageNetVoxelBatchProvider,
    PretrainMixedSourceConfig,
    PretrainMixedVoxelBatchProvider,
    SyntheticVoxelBatchProvider,
    ensure_path_exists,
    resolve_list_file,
)
from jepa.masks.multiseq_multiblock3d import _MaskGenerator  # noqa: E402
from jepa.masks.utils import apply_masks  # noqa: E402
from jepa.regularizers import SIGReg, VICRegRegularizer  # noqa: E402
from jepa.utils.distributed import (  # noqa: E402
    DistributedState,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    reduce_mean_scalar,
    unwrap_module,
)
from jepa.utils.schedulers import WarmupCosineParamScheduler  # noqa: E402


MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "tiny": {"builder": vit_tiny, "embed_dim": 192, "predictor_heads": 6},
    "small": {"builder": vit_small, "embed_dim": 384, "predictor_heads": 12},
    "base": {"builder": vit_base, "embed_dim": 768, "predictor_heads": 12},
    "large": {"builder": vit_large, "embed_dim": 1024, "predictor_heads": 16},
}

COLLAPSE_STRATEGY_CHOICES = {"ema_stopgrad", "vicreg", "sigreg"}
PREDICTOR_DEPTH_CHOICES = {4, 8, 12, 20, 24, 40}
RECON_LOSS_CHOICES = {"smooth_l1", "mse"}
DATA_SOURCE_CHOICES = {"synthetic", "n_imagenet", "dsec", "pretrain_mixed"}
PRETRAIN_MIXED_SOURCE_NAMES = ("dsec", "gen4", "n_imagenet")

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(
    name: str,
    distributed: bool = False,
    local_rank: int = 0,
) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            if distributed:
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cuda")
        return torch.device("cpu")

    if distributed and name == "cuda":
        return torch.device(f"cuda:{local_rank}")
    return torch.device(name)


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _temporal_mix_enabled(cfg: DictConfig) -> bool:
    temporal_mix_cfg = cfg.get("temporal_mix", None)
    if temporal_mix_cfg is None:
        return False
    return bool(temporal_mix_cfg.get("enabled", False))


def _predictor_modality(cfg: DictConfig, temporal_bins: int) -> str:
    if _temporal_mix_enabled(cfg):
        short_t = int(cfg.temporal_mix.short_t)
        if temporal_bins == short_t:
            return "image"
    return "video"


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
    temporal_mix_enabled = _temporal_mix_enabled(cfg)
    img_temporal_dim_size = int(cfg.temporal_mix.short_t) if temporal_mix_enabled else None

    encoder = encoder_builder(
        img_size=(int(cfg.height), int(cfg.width)),
        patch_size=int(cfg.patch_size),
        num_frames=int(cfg.t_bins),
        tubelet_size=int(cfg.tubelet_size),
        in_chans=1,
        use_rope=True,
        modality_embedding=temporal_mix_enabled,
        img_temporal_dim_size=img_temporal_dim_size,
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
        modality_embedding=temporal_mix_enabled,
        img_temporal_dim_size=img_temporal_dim_size,
        n_output_distillation=4,
    )
    return encoder, predictor


def _build_mask_generator(
    cfg: DictConfig,
    num_frames: int,
    temporal_patch_size: int,
) -> _MaskGenerator:
    return _MaskGenerator(
        crop_size=(int(cfg.height), int(cfg.width)),
        num_frames=int(num_frames),
        spatial_patch_size=(int(cfg.patch_size), int(cfg.patch_size)),
        temporal_patch_size=int(temporal_patch_size),
        spatial_pred_mask_scale=(float(cfg.spatial_mask_min), float(cfg.spatial_mask_max)),
        temporal_pred_mask_scale=(float(cfg.temporal_mask_min), float(cfg.temporal_mask_max)),
        aspect_ratio=(float(cfg.aspect_min), float(cfg.aspect_max)),
        npred=int(cfg.num_pred_blocks),
        max_context_frames_ratio=float(cfg.max_context_frames_ratio),
    )


def build_mask_generators(cfg: DictConfig) -> dict[int, _MaskGenerator]:
    long_t = int(cfg.t_bins)
    generators: dict[int, _MaskGenerator] = {
        long_t: _build_mask_generator(
            cfg=cfg,
            num_frames=long_t,
            temporal_patch_size=int(cfg.tubelet_size),
        )
    }

    if _temporal_mix_enabled(cfg):
        short_t = int(cfg.temporal_mix.short_t)
        if short_t not in generators:
            generators[short_t] = _build_mask_generator(
                cfg=cfg,
                num_frames=short_t,
                temporal_patch_size=1,
            )

    return generators


def _prepare_temporal_inputs(
    cfg: DictConfig,
    inputs: torch.Tensor,
    training: bool,
) -> tuple[torch.Tensor, int]:
    if inputs.ndim == 4:
        return inputs, 1
    if inputs.ndim != 5:
        raise ValueError(
            "Expected model inputs to be 4D or 5D tensor, "
            f"got shape={tuple(inputs.shape)}"
        )

    current_t = int(inputs.shape[2])
    if not _temporal_mix_enabled(cfg):
        if int(cfg.t_bins) == 1 and current_t == 1:
            return inputs.squeeze(2), 1
        return inputs, current_t

    short_t = int(cfg.temporal_mix.short_t)
    image_prob = float(cfg.temporal_mix.image_prob)
    short_mode = str(cfg.temporal_mix.short_mode)

    use_short = False
    if training and current_t != short_t and image_prob > 0.0:
        use_short = random.random() < image_prob

    if not use_short:
        return inputs, current_t

    if short_t < 1 or short_t > current_t:
        raise ValueError(
            f"temporal_mix.short_t must be in [1, current_t={current_t}], got {short_t}"
        )

    if short_t == 1:
        if short_mode == "sum":
            return inputs.sum(dim=2, keepdim=True), 1
        if short_mode == "first":
            return inputs[:, :, :1, :, :], 1
        if short_mode == "center":
            center = current_t // 2
            return inputs[:, :, center : center + 1, :, :], 1
        raise ValueError(
            f"Unknown temporal_mix.short_mode={short_mode}; use sum|first|center"
        )

    if training:
        start = random.randint(0, current_t - short_t)
    else:
        start = (current_t - short_t) // 2
    return inputs[:, :, start : start + short_t, :, :], short_t


def _data_loader_prefetch_factor(cfg: DictConfig) -> int | None:
    value = cfg.data.get("prefetch_factor", None)
    if value is None:
        return None
    return int(value)


def _data_loader_persistent_workers(cfg: DictConfig) -> bool:
    return bool(cfg.data.get("persistent_workers", False))


def _configure_data_loader_runtime(cfg: DictConfig) -> None:
    sharing_strategy_raw = cfg.data.get("sharing_strategy", None)
    if sharing_strategy_raw is None:
        return

    sharing_strategy = str(sharing_strategy_raw).strip().lower()
    if sharing_strategy in {"", "none", "null"}:
        return

    current = torch.multiprocessing.get_sharing_strategy()
    if current == sharing_strategy:
        return

    torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    print(f"[dataloader] torch.multiprocessing sharing_strategy={sharing_strategy}")


def build_batch_provider(
    cfg: DictConfig,
    voxel_builder: VoxelGrid,
    dist_state: DistributedState,
):
    prefetch_factor = _data_loader_prefetch_factor(cfg)
    persistent_workers = _data_loader_persistent_workers(cfg)

    source = str(cfg.data.source)
    if source == "synthetic":
        return SyntheticVoxelBatchProvider(
            voxel_builder=voxel_builder,
            batch_size=int(cfg.batch_size),
            num_events_min=int(cfg.data.synthetic.num_events_min),
            num_events_max=int(cfg.data.synthetic.num_events_max),
            normalize_voxel=bool(cfg.normalize_voxel),
        )

    if source == "n_imagenet":
        split = str(cfg.data.n_imagenet.split)
        list_file = resolve_list_file(
            split=split,
            train_list=cfg.data.n_imagenet.train_list,
            val_list=cfg.data.n_imagenet.val_list,
        )
        list_file = to_absolute_path(str(list_file))
        ensure_path_exists(list_file, "N-ImageNet list file")

        root_dir = cfg.data.n_imagenet.root_dir
        root_dir_abs = None
        if root_dir:
            root_dir_abs = to_absolute_path(str(root_dir))
            ensure_path_exists(root_dir_abs, "N-ImageNet root_dir")

        return NImageNetVoxelBatchProvider(
            list_file=list_file,
            split=split,
            voxel_builder=voxel_builder,
            batch_size=int(cfg.batch_size),
            normalize_voxel=bool(cfg.normalize_voxel),
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory),
            drop_last=bool(cfg.data.drop_last),
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            root_dir=root_dir_abs,
            compressed=bool(cfg.data.n_imagenet.compressed),
            time_scale=float(cfg.data.n_imagenet.time_scale),
            limit_samples=cfg.data.n_imagenet.limit_samples,
            limit_classes=cfg.data.n_imagenet.limit_classes,
            sensor_height=int(cfg.data.n_imagenet.sensor_height),
            sensor_width=int(cfg.data.n_imagenet.sensor_width),
            rescale_to_voxel_grid=bool(cfg.data.n_imagenet.rescale_to_voxel_grid),
            slice_enabled=bool(cfg.data.n_imagenet.slice.enabled),
            slice_mode=str(cfg.data.n_imagenet.slice.mode),
            slice_start=cfg.data.n_imagenet.slice.start,
            slice_end=cfg.data.n_imagenet.slice.end,
            slice_length=int(cfg.data.n_imagenet.slice.length),
            random_slice_on_train=bool(cfg.data.n_imagenet.slice.random_on_train),
            augment_enabled=bool(cfg.data.n_imagenet.augment.enabled),
            hflip_prob=float(cfg.data.n_imagenet.augment.hflip_prob),
            max_shift=int(cfg.data.n_imagenet.augment.max_shift),
            distributed=dist_state.enabled,
            rank=dist_state.rank,
            world_size=dist_state.world_size,
        )

    if source == "dsec":
        split = str(cfg.data.dsec.split)
        root_dir = cfg.data.dsec.root_dir
        if not root_dir:
            raise ValueError("data.dsec.root_dir must be set for data.source=dsec")
        root_dir_abs = to_absolute_path(str(root_dir))
        ensure_path_exists(root_dir_abs, "DSEC root_dir")

        split_config = cfg.data.dsec.split_config
        split_config_abs = None
        if split_config:
            split_config_abs = to_absolute_path(str(split_config))
            ensure_path_exists(split_config_abs, "DSEC split_config")

        downsample = bool(cfg.data.dsec.get("downsample", False))
        downsample_event_file = str(cfg.data.dsec.get("downsample_event_file", "events_2x.h5"))

        return DSECVoxelBatchProvider(
            root_dir=root_dir_abs,
            split=split,
            voxel_builder=voxel_builder,
            batch_size=int(cfg.batch_size),
            normalize_voxel=bool(cfg.normalize_voxel),
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory),
            drop_last=bool(cfg.data.drop_last),
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            split_config=split_config_abs,
            sync=str(cfg.data.dsec.sync),
            image_view=str(cfg.data.dsec.image_view),
            load_events=bool(cfg.data.dsec.load_events),
            load_rgb=bool(cfg.data.dsec.load_rgb),
            load_labels=bool(cfg.data.dsec.load_labels),
            limit_samples=cfg.data.dsec.limit_samples,
            sensor_height=int(cfg.data.dsec.sensor_height),
            sensor_width=int(cfg.data.dsec.sensor_width),
            rescale_to_voxel_grid=bool(cfg.data.dsec.rescale_to_voxel_grid),
            downsample=downsample,
            downsample_event_file=downsample_event_file,
            distributed=dist_state.enabled,
            rank=dist_state.rank,
            world_size=dist_state.world_size,
        )

    if source == "pretrain_mixed":
        mixed_cfg = cfg.data.pretrain_mixed
        weights_cfg = mixed_cfg.get("weights", None)

        def _source_weight(source_name: str) -> float:
            if weights_cfg is None:
                return 1.0
            return float(weights_cfg.get(source_name, 1.0))

        source_configs: list[PretrainMixedSourceConfig] = []
        active_source_weights: dict[str, float] = {}
        for source_name in ("dsec", "gen4", "n_imagenet"):
            source_cfg = mixed_cfg[source_name]
            if not bool(source_cfg.enabled):
                continue

            source_weight = _source_weight(source_name)
            if source_weight <= 0.0:
                print(
                    f"[pretrain_mixed] skip source={source_name} because weight={source_weight} <= 0"
                )
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

        source_weights = active_source_weights if len(active_source_weights) > 0 else None

        events_per_sample_default = int(
            mixed_cfg.get("events_per_sample", mixed_cfg.get("events_per_sample_min", 30000))
        )
        events_per_sample_min = int(mixed_cfg.get("events_per_sample_min", events_per_sample_default))
        events_per_sample_max = int(mixed_cfg.get("events_per_sample_max", events_per_sample_default))
        window_duration_us_min = mixed_cfg.get("window_duration_us_min", None)
        window_duration_us_max = mixed_cfg.get("window_duration_us_max", None)
        duration_sources = tuple(str(s) for s in mixed_cfg.get("duration_sources", ["dsec", "gen4"]))
        prefer_ms_to_idx = bool(mixed_cfg.get("prefer_ms_to_idx", True))

        return PretrainMixedVoxelBatchProvider(
            source_configs=source_configs,
            source_weights=source_weights,
            voxel_builder=voxel_builder,
            batch_size=int(cfg.batch_size),
            normalize_voxel=bool(cfg.normalize_voxel),
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory),
            drop_last=bool(cfg.data.drop_last),
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            events_per_sample_min=events_per_sample_min,
            events_per_sample_max=events_per_sample_max,
            random_slice=bool(mixed_cfg.random_slice),
            time_normalize=bool(mixed_cfg.time_normalize),
            window_duration_us_min=window_duration_us_min,
            window_duration_us_max=window_duration_us_max,
            duration_sources=duration_sources,
            prefer_ms_to_idx=prefer_ms_to_idx,
            use_source_balancing=bool(mixed_cfg.use_source_balancing),
            epoch_size=mixed_cfg.epoch_size,
            sampler_seed=int(mixed_cfg.sampler_seed),
            rescale_to_voxel_grid=bool(mixed_cfg.rescale_to_voxel_grid),
            canvas_height=int(mixed_cfg.canvas_height),
            canvas_width=int(mixed_cfg.canvas_width),
            center_pad_to_canvas=bool(mixed_cfg.center_pad_to_canvas),
            distributed=dist_state.enabled,
            rank=dist_state.rank,
            world_size=dist_state.world_size,
        )

    raise ValueError(f"Unknown data.source: {source}")


def build_eval_batch_provider(
    cfg: DictConfig,
    voxel_builder: VoxelGrid,
    dist_state: DistributedState,
):
    if not bool(cfg.eval.enabled):
        return None

    prefetch_factor = _data_loader_prefetch_factor(cfg)
    persistent_workers = _data_loader_persistent_workers(cfg)

    source = str(cfg.data.source)
    if source == "synthetic":
        return SyntheticVoxelBatchProvider(
            voxel_builder=voxel_builder,
            batch_size=int(cfg.batch_size),
            num_events_min=int(cfg.data.synthetic.num_events_min),
            num_events_max=int(cfg.data.synthetic.num_events_max),
            normalize_voxel=bool(cfg.normalize_voxel),
        )

    if source == "n_imagenet":
        split = str(cfg.eval.split)
        list_file = resolve_list_file(
            split=split,
            train_list=cfg.data.n_imagenet.train_list,
            val_list=cfg.data.n_imagenet.val_list,
        )
        list_file = to_absolute_path(str(list_file))
        ensure_path_exists(list_file, "N-ImageNet eval list file")

        root_dir = cfg.data.n_imagenet.root_dir
        root_dir_abs = None
        if root_dir:
            root_dir_abs = to_absolute_path(str(root_dir))
            ensure_path_exists(root_dir_abs, "N-ImageNet root_dir")

        return NImageNetVoxelBatchProvider(
            list_file=list_file,
            split=split,
            voxel_builder=voxel_builder,
            batch_size=int(cfg.batch_size),
            normalize_voxel=bool(cfg.normalize_voxel),
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory),
            drop_last=False,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            root_dir=root_dir_abs,
            compressed=bool(cfg.data.n_imagenet.compressed),
            time_scale=float(cfg.data.n_imagenet.time_scale),
            limit_samples=cfg.data.n_imagenet.limit_samples,
            limit_classes=cfg.data.n_imagenet.limit_classes,
            sensor_height=int(cfg.data.n_imagenet.sensor_height),
            sensor_width=int(cfg.data.n_imagenet.sensor_width),
            rescale_to_voxel_grid=bool(cfg.data.n_imagenet.rescale_to_voxel_grid),
            slice_enabled=bool(cfg.data.n_imagenet.slice.enabled),
            slice_mode=str(cfg.data.n_imagenet.slice.mode),
            slice_start=cfg.data.n_imagenet.slice.start,
            slice_end=cfg.data.n_imagenet.slice.end,
            slice_length=int(cfg.data.n_imagenet.slice.length),
            random_slice_on_train=False,
            augment_enabled=False,
            hflip_prob=0.0,
            max_shift=0,
            distributed=dist_state.enabled,
            rank=dist_state.rank,
            world_size=dist_state.world_size,
        )

    if source == "dsec":
        split = str(cfg.eval.split)
        root_dir = cfg.data.dsec.root_dir
        if not root_dir:
            raise ValueError("data.dsec.root_dir must be set for data.source=dsec")
        root_dir_abs = to_absolute_path(str(root_dir))
        ensure_path_exists(root_dir_abs, "DSEC root_dir")

        split_config = cfg.data.dsec.split_config
        split_config_abs = None
        if split_config:
            split_config_abs = to_absolute_path(str(split_config))
            ensure_path_exists(split_config_abs, "DSEC split_config")

        downsample = bool(cfg.data.dsec.get("downsample", False))
        downsample_event_file = str(cfg.data.dsec.get("downsample_event_file", "events_2x.h5"))

        return DSECVoxelBatchProvider(
            root_dir=root_dir_abs,
            split=split,
            voxel_builder=voxel_builder,
            batch_size=int(cfg.batch_size),
            normalize_voxel=bool(cfg.normalize_voxel),
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory),
            drop_last=False,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            split_config=split_config_abs,
            sync=str(cfg.data.dsec.sync),
            image_view=str(cfg.data.dsec.image_view),
            load_events=bool(cfg.data.dsec.load_events),
            load_rgb=bool(cfg.data.dsec.load_rgb),
            load_labels=bool(cfg.data.dsec.load_labels),
            limit_samples=cfg.data.dsec.limit_samples,
            sensor_height=int(cfg.data.dsec.sensor_height),
            sensor_width=int(cfg.data.dsec.sensor_width),
            rescale_to_voxel_grid=bool(cfg.data.dsec.rescale_to_voxel_grid),
            downsample=downsample,
            downsample_event_file=downsample_event_file,
            distributed=dist_state.enabled,
            rank=dist_state.rank,
            world_size=dist_state.world_size,
        )

    if source == "pretrain_mixed":
        raise ValueError(
            "data.source=pretrain_mixed is a pretraining-only loader. "
            "Set eval.enabled=false for this source."
        )

    raise ValueError(f"Unknown data.source for eval: {source}")


def _extract_scalar(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().item())
    return float(x)


def _compute_source_ratios(
    batch: dict[str, Any],
    dist_state: DistributedState,
    device: torch.device,
) -> dict[str, float] | None:
    sources = batch.get("sources", None)
    if not isinstance(sources, list) or len(sources) == 0:
        return None

    counts = torch.zeros((len(PRETRAIN_MIXED_SOURCE_NAMES),), dtype=torch.float32, device=device)
    total = 0.0
    for src in sources:
        if not isinstance(src, str):
            continue
        if src in PRETRAIN_MIXED_SOURCE_NAMES:
            idx = PRETRAIN_MIXED_SOURCE_NAMES.index(src)
            counts[idx] += 1.0
            total += 1.0

    if total <= 0.0:
        return None

    total_tensor = torch.tensor([total], dtype=torch.float32, device=device)
    if dist_state.enabled:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    denom = float(total_tensor.item())
    if denom <= 0.0:
        return None

    return {
        name: float(counts[i].item() / denom)
        for i, name in enumerate(PRETRAIN_MIXED_SOURCE_NAMES)
    }


def forward_step(
    cfg: DictConfig,
    step: int,
    inputs: torch.Tensor,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    teacher_encoder: torch.nn.Module | None,
    mask_generators: dict[int, _MaskGenerator],
    collapse_strategy: str,
    sigreg: SIGReg,
    vicreg: VICRegRegularizer,
    sigreg_weight: float,
    vicreg_std_weight: float,
    vicreg_cov_weight: float,
    device: torch.device,
    training: bool = True,
) -> dict[str, Any]:
    inputs, temporal_bins = _prepare_temporal_inputs(cfg=cfg, inputs=inputs, training=training)
    predictor_mod = _predictor_modality(cfg, temporal_bins)
    mask_generator = mask_generators.get(temporal_bins)
    if mask_generator is None:
        raise ValueError(
            f"No mask generator is configured for temporal bins={temporal_bins}. "
            f"Available={sorted(mask_generators.keys())}"
        )

    current_batch_size = int(inputs.shape[0])
    masks_enc, masks_pred = mask_generator(current_batch_size)
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
        mod=predictor_mod,
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
    return {
        "loss": loss,
        "recon_loss": recon_loss,
        "sig_loss": sig_loss,
        "std_loss": std_loss,
        "cov_loss": cov_loss,
        "batch_size": current_batch_size,
        "context_tokens": int(masks_enc.shape[1]),
        "pred_tokens": int(masks_pred.shape[1]),
    }


@torch.no_grad()
def run_eval(
    cfg: DictConfig,
    step: int,
    device: torch.device,
    eval_provider: Any,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    teacher_encoder: torch.nn.Module | None,
    mask_generators: dict[int, _MaskGenerator],
    collapse_strategy: str,
    sigreg: SIGReg,
    vicreg: VICRegRegularizer,
    sigreg_weight: float,
    vicreg_std_weight: float,
    vicreg_cov_weight: float,
) -> dict[str, float]:
    steps = int(cfg.eval.steps)
    if steps < 1:
        raise ValueError("eval.steps must be >= 1")

    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "sig_loss": 0.0,
        "std_loss": 0.0,
        "cov_loss": 0.0,
    }

    for i in range(steps):
        batch = eval_provider.next_batch()
        inputs = batch["inputs"].to(device, non_blocking=True)
        metrics = forward_step(
            cfg=cfg,
            step=step + i,
            inputs=inputs,
            encoder=encoder,
            predictor=predictor,
            teacher_encoder=teacher_encoder,
            mask_generators=mask_generators,
            collapse_strategy=collapse_strategy,
            sigreg=sigreg,
            vicreg=vicreg,
            sigreg_weight=sigreg_weight,
            vicreg_std_weight=vicreg_std_weight,
            vicreg_cov_weight=vicreg_cov_weight,
            device=device,
            training=False,
        )
        totals["loss"] += _extract_scalar(metrics["loss"])
        totals["recon_loss"] += _extract_scalar(metrics["recon_loss"])
        totals["sig_loss"] += _extract_scalar(metrics["sig_loss"])
        totals["std_loss"] += _extract_scalar(metrics["std_loss"])
        totals["cov_loss"] += _extract_scalar(metrics["cov_loss"])

    return {k: v / steps for k, v in totals.items()}


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
    if str(cfg.data.source) not in DATA_SOURCE_CHOICES:
        raise ValueError(f"data.source must be one of {sorted(DATA_SOURCE_CHOICES)}")

    if int(cfg.height) % int(cfg.patch_size) != 0 or int(cfg.width) % int(cfg.patch_size) != 0:
        raise ValueError("height/width must be divisible by patch-size")
    if int(cfg.t_bins) < 1:
        raise ValueError("t-bins must be >= 1")
    if int(cfg.tubelet_size) < 1:
        raise ValueError("tubelet-size must be >= 1")
    if int(cfg.t_bins) % int(cfg.tubelet_size) != 0:
        raise ValueError("t-bins must be divisible by tubelet-size")
    if _temporal_mix_enabled(cfg):
        short_t = int(cfg.temporal_mix.short_t)
        if short_t < 1 or short_t > int(cfg.t_bins):
            raise ValueError("temporal_mix.short_t must satisfy 1 <= short_t <= t-bins")
        if not (0.0 <= float(cfg.temporal_mix.image_prob) <= 1.0):
            raise ValueError("temporal_mix.image_prob must be in [0, 1]")
        short_mode = str(cfg.temporal_mix.short_mode)
        if short_mode not in {"sum", "first", "center"}:
            raise ValueError("temporal_mix.short_mode must be sum|first|center")
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
    if int(cfg.batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if int(cfg.data.num_workers) < 0:
        raise ValueError("data.num_workers must be >= 0")

    prefetch_factor = cfg.data.get("prefetch_factor", None)
    if prefetch_factor is not None and int(prefetch_factor) < 1:
        raise ValueError("data.prefetch_factor must be >= 1 when set")

    sharing_strategy_raw = cfg.data.get("sharing_strategy", None)
    if sharing_strategy_raw is not None:
        sharing_strategy = str(sharing_strategy_raw).strip().lower()
        if sharing_strategy not in {"", "none", "null", "file_descriptor", "file_system"}:
            raise ValueError(
                "data.sharing_strategy must be one of "
                "{file_descriptor, file_system, null}"
            )

    if str(cfg.data.source) == "synthetic":
        if int(cfg.data.synthetic.num_events_min) < 1:
            raise ValueError("data.synthetic.num_events_min must be >= 1")
        if int(cfg.data.synthetic.num_events_max) < int(cfg.data.synthetic.num_events_min):
            raise ValueError(
                "data.synthetic.num_events_max must be >= data.synthetic.num_events_min"
            )

    if str(cfg.data.source) == "n_imagenet":
        if str(cfg.data.n_imagenet.split) not in {"train", "val", "test"}:
            raise ValueError("data.n_imagenet.split must be train|val|test")
        if int(cfg.data.n_imagenet.sensor_height) < 1 or int(cfg.data.n_imagenet.sensor_width) < 1:
            raise ValueError("data.n_imagenet.sensor_height/width must be >= 1")
        if str(cfg.data.n_imagenet.slice.mode) not in {"idx", "time", "random"}:
            raise ValueError("data.n_imagenet.slice.mode must be idx|time|random")
        if int(cfg.data.n_imagenet.slice.length) < 1:
            raise ValueError("data.n_imagenet.slice.length must be >= 1")
        if not (0.0 <= float(cfg.data.n_imagenet.augment.hflip_prob) <= 1.0):
            raise ValueError("data.n_imagenet.augment.hflip_prob must be in [0, 1]")
        if int(cfg.data.n_imagenet.augment.max_shift) < 0:
            raise ValueError("data.n_imagenet.augment.max_shift must be >= 0")

    if str(cfg.data.source) == "dsec":
        if str(cfg.data.dsec.split) not in {"train", "val", "test"}:
            raise ValueError("data.dsec.split must be train|val|test")
        if str(cfg.data.dsec.sync) not in {"front", "back"}:
            raise ValueError("data.dsec.sync must be front|back")
        if str(cfg.data.dsec.image_view) not in {"distorted", "rectified"}:
            raise ValueError("data.dsec.image_view must be distorted|rectified")
        if int(cfg.data.dsec.sensor_height) < 1 or int(cfg.data.dsec.sensor_width) < 1:
            raise ValueError("data.dsec.sensor_height/width must be >= 1")
        if not bool(cfg.data.dsec.root_dir):
            raise ValueError("data.dsec.root_dir must be set for data.source=dsec")
        if not (
            bool(cfg.data.dsec.load_events)
            or bool(cfg.data.dsec.load_rgb)
            or bool(cfg.data.dsec.load_labels)
        ):
            raise ValueError(
                "At least one of data.dsec.load_events/load_rgb/load_labels must be true"
            )
        if not bool(cfg.data.dsec.load_events):
            raise ValueError(
                "Step1 pretraining requires events. Set data.dsec.load_events=true."
            )
        downsample_event_file = str(cfg.data.dsec.get("downsample_event_file", "events_2x.h5"))
        if len(downsample_event_file.strip()) == 0:
            raise ValueError("data.dsec.downsample_event_file must be non-empty")

    if str(cfg.data.source) == "pretrain_mixed":
        mixed_cfg = cfg.data.pretrain_mixed
        events_per_sample_default = int(
            mixed_cfg.get("events_per_sample", mixed_cfg.get("events_per_sample_min", 30000))
        )
        if events_per_sample_default < 1:
            raise ValueError("data.pretrain_mixed.events_per_sample (or *_min) must be >= 1")
        events_per_sample_min = int(mixed_cfg.get("events_per_sample_min", events_per_sample_default))
        events_per_sample_max = int(mixed_cfg.get("events_per_sample_max", events_per_sample_default))
        if events_per_sample_min < 1:
            raise ValueError("data.pretrain_mixed.events_per_sample_min must be >= 1")
        if events_per_sample_max < events_per_sample_min:
            raise ValueError(
                "data.pretrain_mixed.events_per_sample_max must be >= events_per_sample_min"
            )

        window_duration_us_min = mixed_cfg.get("window_duration_us_min", None)
        window_duration_us_max = mixed_cfg.get("window_duration_us_max", None)
        if (window_duration_us_min is None) != (window_duration_us_max is None):
            raise ValueError(
                "data.pretrain_mixed.window_duration_us_min and "
                "window_duration_us_max must be both set or both None"
            )
        if window_duration_us_min is not None:
            if int(window_duration_us_min) < 1:
                raise ValueError("data.pretrain_mixed.window_duration_us_min must be >= 1")
            if int(window_duration_us_max) < int(window_duration_us_min):
                raise ValueError(
                    "data.pretrain_mixed.window_duration_us_max must be >= "
                    "window_duration_us_min"
                )
        if int(mixed_cfg.canvas_height) < 1 or int(mixed_cfg.canvas_width) < 1:
            raise ValueError("data.pretrain_mixed.canvas_height/width must be >= 1")
        if mixed_cfg.epoch_size is not None and int(mixed_cfg.epoch_size) < 1:
            raise ValueError("data.pretrain_mixed.epoch_size must be >= 1 when set")

        active_sources: list[str] = []
        allowed_source_names = {"dsec", "gen4", "n_imagenet"}
        duration_sources = [str(s) for s in mixed_cfg.get("duration_sources", ["dsec", "gen4"])]
        unknown_duration_sources = [s for s in duration_sources if s not in allowed_source_names]
        if len(unknown_duration_sources) > 0:
            raise ValueError(
                "data.pretrain_mixed.duration_sources must be a subset of "
                "{dsec, gen4, n_imagenet}"
            )

        weights_cfg = mixed_cfg.get("weights", None)

        def _source_weight(source_name: str) -> float:
            if weights_cfg is None:
                return 1.0
            return float(weights_cfg.get(source_name, 1.0))

        for source_name in ("dsec", "gen4", "n_imagenet"):
            source_cfg = mixed_cfg[source_name]
            if not bool(source_cfg.enabled):
                continue

            source_weight = _source_weight(source_name)
            if source_weight <= 0.0:
                continue

            active_sources.append(source_name)
            if not bool(source_cfg.root_dir):
                raise ValueError(
                    f"data.pretrain_mixed.{source_name}.root_dir must be set when enabled"
                )
            if len(source_cfg.splits) == 0:
                raise ValueError(
                    f"data.pretrain_mixed.{source_name}.splits must be non-empty when enabled"
                )
            if int(source_cfg.sensor_height) < 1 or int(source_cfg.sensor_width) < 1:
                raise ValueError(
                    f"data.pretrain_mixed.{source_name}.sensor_height/width must be >= 1"
                )

            file_name = source_cfg.get("file_name", None)
            file_suffix = source_cfg.get("file_suffix", None)
            manifest_file = source_cfg.get("manifest_file", None)
            if file_name is not None and len(str(file_name).strip()) == 0:
                raise ValueError(
                    f"data.pretrain_mixed.{source_name}.file_name must be non-empty when set"
                )
            if file_suffix is not None and len(str(file_suffix).strip()) == 0:
                raise ValueError(
                    f"data.pretrain_mixed.{source_name}.file_suffix must be non-empty when set"
                )
            if manifest_file is not None and len(str(manifest_file).strip()) == 0:
                raise ValueError(
                    f"data.pretrain_mixed.{source_name}.manifest_file must be non-empty when set"
                )

        if len(active_sources) == 0:
            raise ValueError(
                "No active source in pretrain_mixed. "
                "Enable at least one source and set its weight > 0."
            )

        if bool(mixed_cfg.use_source_balancing):
            if weights_cfg is not None:
                total_active_weight = 0.0
                for source_name in active_sources:
                    total_active_weight += max(0.0, float(weights_cfg.get(source_name, 0.0)))
                if total_active_weight <= 0.0:
                    raise ValueError(
                        "data.pretrain_mixed.weights must include positive value(s) "
                        "for enabled sources when use_source_balancing=true"
                    )

    if bool(cfg.eval.enabled):
        if str(cfg.data.source) == "pretrain_mixed":
            raise ValueError(
                "eval.enabled is not supported with data.source=pretrain_mixed "
                "(pretraining-only loader)."
            )
        if int(cfg.eval.every) < 1:
            raise ValueError("eval.every must be >= 1")
        if int(cfg.eval.steps) < 1:
            raise ValueError("eval.steps must be >= 1")
        if (
            str(cfg.data.source) in {"n_imagenet", "dsec"}
            and str(cfg.eval.split) not in {"val", "test", "train"}
        ):
            raise ValueError("eval.split must be train|val|test")

    if bool(cfg.logging.tensorboard.enabled) and SummaryWriter is None:
        raise ModuleNotFoundError(
            "TensorBoard is not available. Install `tensorboard` (see requirements.txt)."
        )

    if bool(cfg.scheduler.enabled):
        if int(cfg.scheduler.warmup_steps) < 0:
            raise ValueError("scheduler.warmup_steps must be >= 0")
        if float(cfg.scheduler.final_lr) < 0:
            raise ValueError("scheduler.final_lr must be >= 0")
        if float(cfg.scheduler.start_lr) < 0:
            raise ValueError("scheduler.start_lr must be >= 0")
        if bool(cfg.scheduler.update_weight_decay) and float(cfg.scheduler.final_weight_decay) < 0:
            raise ValueError("scheduler.final_weight_decay must be >= 0")

    if bool(cfg.distributed.enabled):
        if str(cfg.distributed.backend) not in {"auto", "nccl", "gloo"}:
            raise ValueError("distributed.backend must be auto|nccl|gloo")
        if int(cfg.distributed.port) < 1:
            raise ValueError("distributed.port must be >= 1")
        if int(cfg.distributed.timeout_sec) < 1:
            raise ValueError("distributed.timeout_sec must be >= 1")


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
    _configure_data_loader_runtime(cfg)
    collapse_strategy = get_collapse_strategy(cfg)
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

    encoder, predictor = build_models(cfg)
    encoder.to(device)
    predictor.to(device)
    encoder.train()
    predictor.train()

    if dist_state.enabled:
        ddp_kwargs: dict[str, Any] = {
            "broadcast_buffers": bool(cfg.distributed.broadcast_buffers),
            "find_unused_parameters": bool(cfg.distributed.find_unused_parameters),
        }
        if device.type == "cuda":
            assert device.index is not None
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        encoder = DDP(encoder, **ddp_kwargs)
        predictor = DDP(predictor, **ddp_kwargs)

    teacher_encoder: torch.nn.Module | None = None
    if collapse_strategy == "ema_stopgrad":
        teacher_encoder = copy.deepcopy(unwrap_module(encoder)).to(device)
        teacher_encoder.eval()
        _set_requires_grad(teacher_encoder, False)

    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(
        params,
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

    mask_generators = build_mask_generators(cfg)
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
        tensorboard_dir = log_dir / str(cfg.logging.tensorboard.subdir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        assert SummaryWriter is not None
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
                "recon",
                "sig",
                "std",
                "cov",
                "lr",
                "weight_decay",
                "context_tokens",
                "pred_tokens",
            ]
        )
        if eval_provider is not None:
            eval_metrics_file = log_dir / "eval_metrics.csv"
            eval_metrics_fp = eval_metrics_file.open("w", newline="", encoding="utf-8")
            eval_metrics_writer = csv.writer(eval_metrics_fp)
            eval_metrics_writer.writerow(
                ["step", "loss", "recon", "sig", "std", "cov", "eval_steps"]
            )

    if main_process:
        print(
            f"device={device} model={cfg.model_size} "
            f"data_source={cfg.data.source} collapse_strategy={collapse_strategy} "
            f"rank={dist_state.rank}/{dist_state.world_size} backend={dist_state.backend} "
            f"encoder_params={count_trainable_params(unwrap_module(encoder)):,} "
            f"predictor_params={count_trainable_params(unwrap_module(predictor)):,} "
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
            step_metrics = forward_step(
                cfg=cfg,
                step=step,
                inputs=inputs,
                encoder=encoder,
                predictor=predictor,
                teacher_encoder=teacher_encoder,
                mask_generators=mask_generators,
                collapse_strategy=collapse_strategy,
                sigreg=sigreg,
                vicreg=vicreg,
                sigreg_weight=sigreg_weight,
                vicreg_std_weight=vicreg_std_weight,
                vicreg_cov_weight=vicreg_cov_weight,
                device=device,
                training=True,
            )

            loss = step_metrics["loss"]
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            if float(cfg.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
            optimizer.step()
            if teacher_encoder is not None:
                _update_ema(
                    unwrap_module(encoder),
                    teacher_encoder,
                    float(cfg.ema_momentum),
                )

            step_loss = _extract_scalar(step_metrics["loss"])
            step_recon = _extract_scalar(step_metrics["recon_loss"])
            step_sig = _extract_scalar(step_metrics["sig_loss"])
            step_std = _extract_scalar(step_metrics["std_loss"])
            step_cov = _extract_scalar(step_metrics["cov_loss"])
            step_batch_size = int(step_metrics["batch_size"])
            step_context_tokens = float(step_metrics["context_tokens"])
            step_pred_tokens = float(step_metrics["pred_tokens"])

            if dist_state.enabled:
                step_loss = reduce_mean_scalar(step_loss, device=device)
                step_recon = reduce_mean_scalar(step_recon, device=device)
                step_sig = reduce_mean_scalar(step_sig, device=device)
                step_std = reduce_mean_scalar(step_std, device=device)
                step_cov = reduce_mean_scalar(step_cov, device=device)
                step_batch_size = int(reduce_mean_scalar(float(step_batch_size), device=device))
                step_context_tokens = reduce_mean_scalar(step_context_tokens, device=device)
                step_pred_tokens = reduce_mean_scalar(step_pred_tokens, device=device)

            if main_process and metrics_writer is not None and metrics_fp is not None:
                metrics_writer.writerow(
                    [
                        step,
                        step_batch_size,
                        f"{step_loss:.8f}",
                        f"{step_recon:.8f}",
                        f"{step_sig:.8f}",
                        f"{step_std:.8f}",
                        f"{step_cov:.8f}",
                        f"{step_lr:.10f}",
                        f"{step_wd:.10f}",
                        int(round(step_context_tokens)),
                        int(round(step_pred_tokens)),
                    ]
                )
                metrics_fp.flush()

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("train/loss", step_loss, step)
                tensorboard_writer.add_scalar("train/recon", step_recon, step)
                tensorboard_writer.add_scalar("train/sig", step_sig, step)
                tensorboard_writer.add_scalar("train/std", step_std, step)
                tensorboard_writer.add_scalar("train/cov", step_cov, step)
                tensorboard_writer.add_scalar("train/lr", step_lr, step)
                tensorboard_writer.add_scalar("train/weight_decay", step_wd, step)
                tensorboard_writer.add_scalar(
                    "train/context_tokens",
                    int(round(step_context_tokens)),
                    step,
                )
                tensorboard_writer.add_scalar(
                    "train/pred_tokens",
                    int(round(step_pred_tokens)),
                    step,
                )
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
                    f"recon={step_recon:.6f} "
                    f"sig={step_sig:.6f} "
                    f"std={step_std:.6f} "
                    f"cov={step_cov:.6f} "
                    f"lr={step_lr:.8f} "
                    f"wd={step_wd:.8f} "
                    f"context_tokens={int(round(step_context_tokens))} "
                    f"pred_tokens={int(round(step_pred_tokens))}"
                    f"{source_ratio_str}"
                )

            if eval_provider is not None and step % int(cfg.eval.every) == 0:
                encoder.eval()
                predictor.eval()
                if teacher_encoder is not None:
                    teacher_encoder.eval()
                eval_stats = run_eval(
                    cfg=cfg,
                    step=step,
                    device=device,
                    eval_provider=eval_provider,
                    encoder=encoder,
                    predictor=predictor,
                    teacher_encoder=teacher_encoder,
                    mask_generators=mask_generators,
                    collapse_strategy=collapse_strategy,
                    sigreg=sigreg,
                    vicreg=vicreg,
                    sigreg_weight=sigreg_weight,
                    vicreg_std_weight=vicreg_std_weight,
                    vicreg_cov_weight=vicreg_cov_weight,
                )
                encoder.train()
                predictor.train()

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
                            f"{eval_stats['recon_loss']:.8f}",
                            f"{eval_stats['sig_loss']:.8f}",
                            f"{eval_stats['std_loss']:.8f}",
                            f"{eval_stats['cov_loss']:.8f}",
                            int(cfg.eval.steps),
                        ]
                    )
                    eval_metrics_fp.flush()

                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("eval/loss", eval_stats["loss"], step)
                    tensorboard_writer.add_scalar("eval/recon", eval_stats["recon_loss"], step)
                    tensorboard_writer.add_scalar("eval/sig", eval_stats["sig_loss"], step)
                    tensorboard_writer.add_scalar("eval/std", eval_stats["std_loss"], step)
                    tensorboard_writer.add_scalar("eval/cov", eval_stats["cov_loss"], step)

                if main_process:
                    print(
                        f"eval@step={step:05d} "
                        f"loss={eval_stats['loss']:.6f} "
                        f"recon={eval_stats['recon_loss']:.6f}"
                    )

            if main_process and int(cfg.save_every) > 0 and step % int(cfg.save_every) == 0:
                ckpt_path = out_dir / f"step_{step:06d}.pt"
                torch.save(
                    {
                        "step": step,
                        "cfg": OmegaConf.to_container(cfg, resolve=True),
                        "encoder": unwrap_module(encoder).state_dict(),
                        "predictor": unwrap_module(predictor).state_dict(),
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
