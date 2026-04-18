from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from event.data.dsec import DSECEventsDataset, DSECVoxelCollator
from event.data.n_imagenet import NImageNetEventsDataset, NImageNetVoxelCollator
from event.representations import VoxelGrid, norm_voxel_grid


class SyntheticVoxelBatchProvider:
    def __init__(
        self,
        voxel_builder: VoxelGrid,
        batch_size: int,
        num_events_min: int,
        num_events_max: int,
        normalize_voxel: bool,
    ):
        self.voxel_builder = voxel_builder
        self.batch_size = batch_size
        self.num_events_min = num_events_min
        self.num_events_max = num_events_max
        self.normalize_voxel = normalize_voxel

    def _sample_one(self) -> torch.Tensor:
        n_events = random.randint(self.num_events_min, self.num_events_max)
        x = torch.randint(0, self.voxel_builder.width, (n_events,), dtype=torch.int64)
        y = torch.randint(0, self.voxel_builder.height, (n_events,), dtype=torch.int64)
        pol = torch.randint(0, 2, (n_events,), dtype=torch.int64)
        time = torch.sort(torch.randint(0, 1_000_000, (n_events,), dtype=torch.int64)).values
        if self.voxel_builder.time_bins > 1 and time[-1] <= time[0]:
            if time.numel() == 1:
                x = torch.cat([x, x], dim=0)
                y = torch.cat([y, y], dim=0)
                pol = torch.cat([pol, pol], dim=0)
                time = torch.cat([time, time + 1], dim=0)
            else:
                time = time.clone()
                time[-1] = time[0] + 1

        voxel = self.voxel_builder.convert(x=x, y=y, pol=pol, time=time)
        if self.normalize_voxel:
            voxel = norm_voxel_grid(voxel)
        return voxel

    def next_batch(self) -> dict[str, Any]:
        voxels = [self._sample_one() for _ in range(self.batch_size)]
        batch = torch.stack(voxels, dim=0).unsqueeze(1)
        return {
            "inputs": batch,
            "labels": None,
            "paths": None,
        }


class NImageNetVoxelBatchProvider:
    def __init__(
        self,
        list_file: str,
        split: str,
        voxel_builder: VoxelGrid,
        batch_size: int,
        normalize_voxel: bool,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        root_dir: str | None,
        compressed: bool,
        time_scale: float,
        limit_samples: int | None,
        limit_classes: int | None,
        sensor_height: int,
        sensor_width: int,
        rescale_to_voxel_grid: bool,
        slice_enabled: bool,
        slice_mode: str,
        slice_start: int | None,
        slice_end: int | None,
        slice_length: int,
        random_slice_on_train: bool,
        augment_enabled: bool,
        hflip_prob: float,
        max_shift: int,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = NImageNetEventsDataset(
            list_file=list_file,
            split=split,
            compressed=compressed,
            time_scale=time_scale,
            root_dir=root_dir,
            limit_samples=limit_samples,
            limit_classes=limit_classes,
            sensor_height=sensor_height,
            sensor_width=sensor_width,
            slice_enabled=slice_enabled,
            slice_mode=slice_mode,
            slice_start=slice_start,
            slice_end=slice_end,
            slice_length=slice_length,
            random_slice_on_train=random_slice_on_train,
            augment_enabled=augment_enabled,
            hflip_prob=hflip_prob,
            max_shift=max_shift,
        )

        collator = NImageNetVoxelCollator(
            voxel_builder=voxel_builder,
            normalize_voxel=normalize_voxel,
            sensor_height=sensor_height,
            sensor_width=sensor_width,
            rescale_to_voxel_grid=rescale_to_voxel_grid,
        )

        self.sampler = None
        if distributed and world_size > 1:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=(split == "train"),
                drop_last=drop_last,
            )

        loader_kwargs: dict[str, Any] = {}
        if int(num_workers) > 0:
            loader_kwargs["persistent_workers"] = bool(persistent_workers)
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(split == "train") and self.sampler is None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collator,
            sampler=self.sampler,
            **loader_kwargs,
        )
        self._sampler_epoch = 0
        if self.sampler is not None:
            self.sampler.set_epoch(self._sampler_epoch)
        self._iter = iter(self.loader)

    def next_batch(self) -> dict[str, Any]:
        try:
            batch = next(self._iter)
        except StopIteration:
            if self.sampler is not None:
                self._sampler_epoch += 1
                self.sampler.set_epoch(self._sampler_epoch)
            self._iter = iter(self.loader)
            batch = next(self._iter)
        return batch

    @property
    def num_samples(self) -> int:
        return len(self.dataset)


class DSECVoxelBatchProvider:
    def __init__(
        self,
        root_dir: str,
        split: str,
        voxel_builder: VoxelGrid,
        batch_size: int,
        normalize_voxel: bool,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        split_config: str | None,
        sync: str,
        image_view: str,
        load_events: bool,
        load_rgb: bool,
        load_labels: bool,
        limit_samples: int | None,
        sensor_height: int,
        sensor_width: int,
        rescale_to_voxel_grid: bool,
        downsample: bool = False,
        downsample_event_file: str = "events_2x.h5",
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        if not load_events:
            raise ValueError(
                "DSECVoxelBatchProvider requires load_events=true "
                "because it builds voxel inputs from events."
            )

        self.dataset = DSECEventsDataset(
            root_dir=root_dir,
            split=split,
            sync=sync,
            split_config=split_config,
            image_view=image_view,
            load_events=load_events,
            load_rgb=load_rgb,
            load_labels=load_labels,
            limit_samples=limit_samples,
            sensor_height=sensor_height,
            sensor_width=sensor_width,
            downsample=downsample,
            downsample_event_file=downsample_event_file,
        )

        collator = DSECVoxelCollator(
            voxel_builder=voxel_builder,
            normalize_voxel=normalize_voxel,
            sensor_height=sensor_height,
            sensor_width=sensor_width,
            rescale_to_voxel_grid=rescale_to_voxel_grid,
        )

        self.sampler = None
        if distributed and world_size > 1:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=(split == "train"),
                drop_last=drop_last,
            )

        loader_kwargs: dict[str, Any] = {}
        if int(num_workers) > 0:
            loader_kwargs["persistent_workers"] = bool(persistent_workers)
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(split == "train") and self.sampler is None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collator,
            sampler=self.sampler,
            **loader_kwargs,
        )
        self._sampler_epoch = 0
        if self.sampler is not None:
            self.sampler.set_epoch(self._sampler_epoch)
        self._iter = iter(self.loader)

    def next_batch(self) -> dict[str, Any]:
        try:
            batch = next(self._iter)
        except StopIteration:
            if self.sampler is not None:
                self._sampler_epoch += 1
                self.sampler.set_epoch(self._sampler_epoch)
            self._iter = iter(self.loader)
            batch = next(self._iter)
        return batch

    @property
    def num_samples(self) -> int:
        return len(self.dataset)


def resolve_list_file(split: str, train_list: str | None, val_list: str | None) -> str:
    if split == "train":
        if not train_list:
            raise ValueError("data.n_imagenet.train_list must be set for split=train")
        return train_list

    if split in {"val", "test"}:
        if not val_list:
            raise ValueError("data.n_imagenet.val_list must be set for split=val/test")
        return val_list

    raise ValueError(f"Unknown split: {split}")


def ensure_path_exists(path: str, name: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
