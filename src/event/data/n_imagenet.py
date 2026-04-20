from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from event.representations import VoxelGrid, norm_voxel_grid

try:
    import hdf5plugin  # noqa: F401
    _HAS_HDF5PLUGIN = True
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment.
    _HAS_HDF5PLUGIN = False


def _open_h5_for_read(path: Path):
    try:
        return h5py.File(str(path), "r")
    except OSError as exc:
        if not _HAS_HDF5PLUGIN:
            raise ModuleNotFoundError(
                "Failed to open H5 file. This dataset may use Blosc-compressed HDF5; "
                "install hdf5plugin (`pip install hdf5plugin`)."
            ) from exc
        raise


def _load_n_imagenet_npz(
    event_path: Path,
    compressed: bool,
    time_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(event_path) as npz:
        if compressed and "event_data" in npz:
            event = npz["event_data"]
            x = np.asarray(event["x"])
            y = np.asarray(event["y"])
            t = np.asarray(event["t"])
            p = np.asarray(event["p"])
        else:
            x = np.asarray(npz["x_pos"])
            y = np.asarray(npz["y_pos"])
            t = np.asarray(npz["timestamp"])
            p = np.asarray(npz["polarity"])

    x_t = torch.from_numpy(np.rint(x).astype(np.int64))
    y_t = torch.from_numpy(np.rint(y).astype(np.int64))

    if np.issubdtype(t.dtype, np.floating) and t.size > 0:
        # If timestamps look like seconds, convert to microseconds.
        if float(np.nanmax(t)) < 1.0e4:
            t = t * time_scale
    t_t = torch.from_numpy(np.rint(t).astype(np.int64))

    p_t = torch.from_numpy((np.asarray(p) > 0).astype(np.int64))

    if t_t.numel() > 1:
        order = torch.argsort(t_t)
        x_t = x_t[order]
        y_t = y_t[order]
        t_t = t_t[order]
        p_t = p_t[order]

    return x_t, y_t, p_t, t_t


def _load_n_imagenet_h5(
    event_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with _open_h5_for_read(event_path) as h5f:
        if "events" in h5f:
            events = h5f["events"]
            x = np.asarray(events["x"])
            y = np.asarray(events["y"])
            t = np.asarray(events["t"])
            p = np.asarray(events["p"])
        else:
            x = np.asarray(h5f["x"])
            y = np.asarray(h5f["y"])
            t = np.asarray(h5f["t"])
            p = np.asarray(h5f["p"])

    x_t = torch.from_numpy(np.rint(x).astype(np.int64))
    y_t = torch.from_numpy(np.rint(y).astype(np.int64))
    t_t = torch.from_numpy(np.rint(t).astype(np.int64))
    p_t = torch.from_numpy((np.asarray(p) > 0).astype(np.int64))

    if t_t.numel() > 1:
        order = torch.argsort(t_t)
        x_t = x_t[order]
        y_t = y_t[order]
        t_t = t_t[order]
        p_t = p_t[order]

    return x_t, y_t, p_t, t_t


def _load_n_imagenet_events(
    event_path: Path,
    compressed: bool,
    time_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    suffix = event_path.suffix.lower()
    if suffix == ".npz":
        return _load_n_imagenet_npz(
            event_path=event_path,
            compressed=compressed,
            time_scale=time_scale,
        )
    if suffix in {".h5", ".hdf5"}:
        return _load_n_imagenet_h5(event_path=event_path)
    raise ValueError(f"Unsupported event file extension for N-ImageNet: {event_path}")


class NImageNetEventsDataset(Dataset):
    def __init__(
        self,
        list_file: str,
        split: str = "train",
        compressed: bool = True,
        time_scale: float = 1_000_000.0,
        root_dir: str | None = None,
        limit_samples: int | None = None,
        limit_classes: int | None = None,
        class_names: list[str] | tuple[str, ...] | None = None,
        sensor_height: int = 480,
        sensor_width: int = 640,
        slice_enabled: bool = False,
        slice_mode: str = "random",
        slice_start: int | None = None,
        slice_end: int | None = None,
        slice_length: int = 30_000,
        random_slice_on_train: bool = True,
        augment_enabled: bool = False,
        hflip_prob: float = 0.5,
        max_shift: int = 0,
    ):
        super().__init__()
        self.split = split
        self.compressed = compressed
        self.time_scale = time_scale
        self.root_dir = Path(root_dir) if root_dir else None
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width

        self.slice_enabled = slice_enabled
        self.slice_mode = slice_mode
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.slice_length = slice_length
        self.random_slice_on_train = random_slice_on_train

        self.augment_enabled = augment_enabled
        self.hflip_prob = hflip_prob
        self.max_shift = max_shift

        self.list_file = Path(list_file)
        if not self.list_file.exists():
            raise FileNotFoundError(f"list file not found: {self.list_file}")

        self.sample_paths = self._read_list_file(self.list_file)

        if class_names is None:
            resolved_class_names = sorted({p.parent.name for p in self.sample_paths})
        else:
            resolved_class_names = [str(name) for name in class_names]
            if len(resolved_class_names) == 0:
                raise ValueError("class_names must not be empty when provided")
            if len(set(resolved_class_names)) != len(resolved_class_names):
                raise ValueError("class_names contains duplicated class names")

        if limit_classes is not None:
            resolved_class_names = resolved_class_names[:limit_classes]
            allowed = set(resolved_class_names)
            self.sample_paths = [p for p in self.sample_paths if p.parent.name in allowed]

        self.class_names = list(resolved_class_names)
        self.label_map = {name: idx for idx, name in enumerate(self.class_names)}
        if limit_samples is not None:
            self.sample_paths = self.sample_paths[:limit_samples]

    def _read_list_file(self, list_file: Path) -> list[Path]:
        paths: list[Path] = []
        for line in list_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            raw = line.split()[0]
            p = Path(raw)
            if not p.is_absolute():
                if self.root_dir is not None:
                    p = self.root_dir / p
                else:
                    p = list_file.parent / p
            paths.append(p)
        return paths

    def _slice_events(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.slice_enabled or t.numel() == 0:
            return x, y, p, t

        if self.slice_mode == "idx":
            start = self.slice_start if self.slice_start is not None else 0
            end = self.slice_end if self.slice_end is not None else t.numel()
            return x[start:end], y[start:end], p[start:end], t[start:end]

        if self.slice_mode == "time":
            start = self.slice_start if self.slice_start is not None else int(t[0].item())
            end = self.slice_end if self.slice_end is not None else int(t[-1].item())
            mask = (t >= start) & (t <= end)
            return x[mask], y[mask], p[mask], t[mask]

        if self.slice_mode == "random":
            if self.slice_length <= 0 or t.numel() <= self.slice_length:
                return x, y, p, t
            if self.random_slice_on_train and self.split != "train":
                return x, y, p, t
            start = random.randint(0, t.numel() - self.slice_length)
            end = start + self.slice_length
            return x[start:end], y[start:end], p[start:end], t[start:end]

        raise ValueError(f"Unknown slice_mode: {self.slice_mode}")

    def _augment_events(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.augment_enabled or self.split != "train" or x.numel() == 0:
            return x, y, p, t

        x = x.clone()
        y = y.clone()

        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            x = self.sensor_width - 1 - x

        if self.max_shift > 0:
            shift_x = random.randint(-self.max_shift, self.max_shift)
            shift_y = random.randint(-self.max_shift, self.max_shift)
            x = x + shift_x
            y = y + shift_y

        return x, y, p, t

    def _filter_valid_coords(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.numel() == 0:
            return x, y, p, t

        mask = (
            (x >= 0)
            & (x < self.sensor_width)
            & (y >= 0)
            & (y < self.sensor_height)
        )
        return x[mask], y[mask], p[mask], t[mask]

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        event_path = self.sample_paths[idx]
        x, y, p, t = _load_n_imagenet_events(
            event_path=event_path,
            compressed=self.compressed,
            time_scale=self.time_scale,
        )

        x, y, p, t = self._slice_events(x, y, p, t)
        x, y, p, t = self._augment_events(x, y, p, t)
        x, y, p, t = self._filter_valid_coords(x, y, p, t)

        label = self.label_map.get(event_path.parent.name, -1)
        return {
            "x": x,
            "y": y,
            "pol": p,
            "time": t,
            "label": label,
            "path": str(event_path),
        }


class NImageNetVoxelCollator:
    def __init__(
        self,
        voxel_builder: VoxelGrid,
        normalize_voxel: bool,
        sensor_height: int,
        sensor_width: int,
        rescale_to_voxel_grid: bool = True,
    ):
        self.voxel_builder = voxel_builder
        self.normalize_voxel = normalize_voxel
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width
        self.rescale_to_voxel_grid = rescale_to_voxel_grid

    def _to_voxel(
        self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros(
                (self.voxel_builder.nb_channels, self.voxel_builder.height, self.voxel_builder.width),
                dtype=torch.float32,
            )

        if self.rescale_to_voxel_grid:
            x = torch.floor(x.float() * (self.voxel_builder.width / self.sensor_width)).long()
            y = torch.floor(y.float() * (self.voxel_builder.height / self.sensor_height)).long()
        else:
            x = x.long()
            y = y.long()

        mask = (
            (x >= 0)
            & (x < self.voxel_builder.width)
            & (y >= 0)
            & (y < self.voxel_builder.height)
        )
        x = x[mask]
        y = y[mask]
        pol = pol[mask].long()
        time = time[mask].long()

        if x.numel() == 0:
            return torch.zeros(
                (self.voxel_builder.nb_channels, self.voxel_builder.height, self.voxel_builder.width),
                dtype=torch.float32,
            )
        if self.voxel_builder.time_bins > 1:
            if time.numel() == 1:
                x = torch.cat([x, x], dim=0)
                y = torch.cat([y, y], dim=0)
                pol = torch.cat([pol, pol], dim=0)
                time = torch.cat([time, time + 1], dim=0)
            elif time[-1] <= time[0]:
                time = time.clone()
                time[-1] = time[0] + 1

        voxel = self.voxel_builder.convert(x=x, y=y, pol=pol, time=time)
        if self.normalize_voxel:
            voxel = norm_voxel_grid(voxel)
        return voxel

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        voxels = []
        labels = []
        paths = []
        for sample in samples:
            voxels.append(
                self._to_voxel(
                    x=sample["x"],
                    y=sample["y"],
                    pol=sample["pol"],
                    time=sample["time"],
                )
            )
            labels.append(int(sample["label"]))
            paths.append(sample["path"])

        batch = torch.stack(voxels, dim=0).unsqueeze(1)
        return {
            "inputs": batch,
            "labels": torch.tensor(labels, dtype=torch.long),
            "paths": paths,
        }
