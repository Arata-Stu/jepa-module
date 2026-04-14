from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from event.representations import VoxelGrid, norm_voxel_grid

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment.
    h5py = None  # type: ignore[assignment]


def _load_split_config(path: str | Path | None) -> dict[str, list[str]] | None:
    if path is None:
        return None

    cfg_obj = OmegaConf.load(str(path))
    cfg = OmegaConf.to_container(cfg_obj, resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid DSEC split config format: {path}")

    split_cfg: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        values = cfg.get(split, [])
        if values is None:
            split_cfg[split] = []
        elif isinstance(values, list):
            split_cfg[split] = [str(v) for v in values]
        else:
            raise ValueError(f"DSEC split config '{split}' must be a list: {path}")
    return split_cfg


def _compute_img_idx_to_track_idx(
    t_track: np.ndarray,
    t_image: np.ndarray,
) -> np.ndarray:
    if t_image.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if t_track.size == 0:
        zeros = np.zeros((t_image.shape[0] + 1,), dtype=np.int64)
        return np.stack([zeros[:-1], zeros[1:]], axis=-1)

    unique_t, counts = np.unique(t_track.astype(np.int64), return_counts=True)
    image_t = t_image.astype(np.int64)
    count_per_image = np.zeros((image_t.shape[0],), dtype=np.int64)

    pos = np.searchsorted(unique_t, image_t)
    valid_pos = pos < unique_t.shape[0]
    valid_match = np.zeros_like(valid_pos, dtype=bool)
    valid_match[valid_pos] = unique_t[pos[valid_pos]] == image_t[valid_pos]
    count_per_image[valid_match] = counts[pos[valid_match]]

    idx = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(count_per_image, dtype=np.int64)]
    )
    return np.stack([idx[:-1], idx[1:]], axis=-1)


def _extract_events_by_time_window(
    event_file: Path,
    t_min_us: int,
    t_max_us: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if h5py is None:
        raise ModuleNotFoundError(
            "h5py is required to load DSEC events. Please install h5py."
        )

    with h5py.File(str(event_file), "r") as h5f:
        ms_to_idx = np.asarray(h5f["ms_to_idx"], dtype=np.int64)
        t_offset = int(h5f["t_offset"][()])

        events = h5f["events"]
        t = events["t"]
        if t.shape[0] == 0 or ms_to_idx.shape[0] == 0:
            empty = torch.empty((0,), dtype=torch.int64)
            return empty, empty, empty, empty

        t_ev_start_us = int(t_min_us) - t_offset
        t_ev_end_us = int(t_max_us) - t_offset

        start_ms = max(0, t_ev_start_us // 1000)
        end_ms = max(0, math.floor(t_ev_end_us / 1000))
        start_ms = min(start_ms, ms_to_idx.shape[0] - 1)
        end_ms = min(end_ms, ms_to_idx.shape[0] - 1)

        ev_start_idx = int(ms_to_idx[start_ms])
        ev_end_idx = int(ms_to_idx[end_ms])
        if ev_end_idx < ev_start_idx:
            ev_end_idx = ev_start_idx

        x = np.asarray(events["x"][ev_start_idx:ev_end_idx], dtype=np.int64)
        y = np.asarray(events["y"][ev_start_idx:ev_end_idx], dtype=np.int64)
        p = np.asarray(events["p"][ev_start_idx:ev_end_idx])
        t_us = np.asarray(events["t"][ev_start_idx:ev_end_idx], dtype=np.int64) + t_offset

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    p_t = torch.from_numpy((p > 0).astype(np.int64))
    t_t = torch.from_numpy(t_us)
    return x_t, y_t, p_t, t_t


@dataclass
class _DSECSequenceInfo:
    name: str
    root: Path
    timestamps: np.ndarray
    image_files: list[Path] | None
    event_file: Path
    tracks: np.ndarray | None
    img_idx_to_track_idx: np.ndarray | None
    num_samples: int


class DSECEventsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        sync: str = "back",
        split_config: str | None = None,
        image_view: str = "distorted",
        load_events: bool = True,
        load_rgb: bool = False,
        load_labels: bool = False,
        limit_samples: int | None = None,
        sensor_height: int = 480,
        sensor_width: int = 640,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train|val|test")
        if sync not in {"front", "back"}:
            raise ValueError("sync must be one of front|back")
        if image_view not in {"distorted", "rectified"}:
            raise ValueError("image_view must be one of distorted|rectified")
        if not (load_events or load_rgb or load_labels):
            raise ValueError("At least one of load_events/load_rgb/load_labels must be true.")

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"DSEC root_dir not found: {self.root_dir}")

        self.split = split
        self.sync = sync
        self.image_view = image_view
        self.load_events = load_events
        self.load_rgb = load_rgb
        self.load_labels = load_labels
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width

        self._split_cfg = _load_split_config(split_config)
        self._sequences = self._build_sequence_infos()
        self._index = self._build_global_index(limit_samples=limit_samples)

    def _resolve_split_root(self) -> Path:
        if self.split in {"train", "val"}:
            split_root = self.root_dir / "train"
        else:
            split_root = self.root_dir / "test"

        if not split_root.exists():
            raise FileNotFoundError(f"DSEC split directory not found: {split_root}")
        return split_root

    def _select_sequence_dirs(self, split_root: Path) -> list[Path]:
        dirs = [p for p in split_root.iterdir() if p.is_dir()]
        if self._split_cfg is None:
            if self.split == "val":
                raise ValueError(
                    "split='val' requires data.dsec.split_config (official DSEC stores val under train)."
                )
            return sorted(dirs, key=lambda p: p.name)

        wanted = set(self._split_cfg.get(self.split, []))
        return sorted([p for p in dirs if p.name in wanted], key=lambda p: p.name)

    def _load_image_timestamps(self, seq_dir: Path) -> np.ndarray:
        ts_file = seq_dir / "images" / "timestamps.txt"
        if not ts_file.exists():
            raise FileNotFoundError(f"DSEC image timestamp file not found: {ts_file}")
        t = np.genfromtxt(ts_file, dtype=np.int64)
        if t.ndim == 0:
            t = np.asarray([int(t)], dtype=np.int64)
        return t.astype(np.int64)

    def _load_image_files(self, seq_dir: Path) -> list[Path]:
        img_dir = seq_dir / "images" / "left" / self.image_view
        if not img_dir.exists():
            raise FileNotFoundError(f"DSEC image directory not found: {img_dir}")
        return sorted(img_dir.glob("*.png"))

    def _load_tracks(self, seq_dir: Path) -> np.ndarray:
        tracks_file = seq_dir / "object_detections" / "left" / "tracks.npy"
        if not tracks_file.exists():
            raise FileNotFoundError(f"DSEC tracks file not found: {tracks_file}")
        return np.load(tracks_file)

    def _build_sequence_infos(self) -> list[_DSECSequenceInfo]:
        split_root = self._resolve_split_root()
        sequence_dirs = self._select_sequence_dirs(split_root)
        if len(sequence_dirs) == 0:
            raise FileNotFoundError(
                f"No DSEC sequences found for split='{self.split}' under: {split_root}"
            )

        infos: list[_DSECSequenceInfo] = []
        for seq_dir in sequence_dirs:
            timestamps = self._load_image_timestamps(seq_dir)
            image_files: list[Path] | None = None
            if self.load_rgb:
                image_files = self._load_image_files(seq_dir)
                if len(image_files) != int(timestamps.shape[0]):
                    raise ValueError(
                        "Number of image files and timestamps must match for DSEC sequence "
                        f"{seq_dir.name}: {len(image_files)} vs {int(timestamps.shape[0])}"
                    )

            event_file = seq_dir / "events" / "left" / "events.h5"
            if self.load_events and not event_file.exists():
                raise FileNotFoundError(f"DSEC event file not found: {event_file}")

            tracks = None
            img_idx_to_track_idx = None
            if self.load_labels:
                tracks = self._load_tracks(seq_dir)
                t_track = (
                    np.asarray(tracks["t"], dtype=np.int64)
                    if tracks.size > 0
                    else np.empty((0,), dtype=np.int64)
                )
                img_idx_to_track_idx = _compute_img_idx_to_track_idx(
                    t_track=t_track,
                    t_image=timestamps,
                )

            num_samples = max(0, int(timestamps.shape[0]) - 1)
            infos.append(
                _DSECSequenceInfo(
                    name=seq_dir.name,
                    root=seq_dir,
                    timestamps=timestamps,
                    image_files=image_files,
                    event_file=event_file,
                    tracks=tracks,
                    img_idx_to_track_idx=img_idx_to_track_idx,
                    num_samples=num_samples,
                )
            )

        return infos

    def _build_global_index(
        self,
        limit_samples: int | None = None,
    ) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for seq_idx, seq in enumerate(self._sequences):
            for local_idx in range(seq.num_samples):
                pairs.append((seq_idx, local_idx))

        if limit_samples is not None:
            pairs = pairs[: int(limit_samples)]
        return pairs

    @staticmethod
    def _to_window_indices(local_idx: int, sync: str) -> tuple[int, int, int]:
        start_idx = local_idx
        end_idx = local_idx + 1
        anchor_idx = start_idx if sync == "back" else end_idx
        return start_idx, end_idx, anchor_idx

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_idx, local_idx = self._index[idx]
        seq = self._sequences[seq_idx]
        start_idx, end_idx, anchor_idx = self._to_window_indices(local_idx, self.sync)

        t_start = int(seq.timestamps[start_idx])
        t_end = int(seq.timestamps[end_idx])
        t_anchor = int(seq.timestamps[anchor_idx])

        sample: dict[str, Any] = {
            "sequence": seq.name,
            "sample_index": int(local_idx),
            "path": f"{seq.name}:{anchor_idx}",
            "time_start_us": t_start,
            "time_end_us": t_end,
            "time_anchor_us": t_anchor,
        }

        if self.load_events:
            x, y, p, t = _extract_events_by_time_window(
                event_file=seq.event_file,
                t_min_us=t_start,
                t_max_us=t_end,
            )
            if x.numel() > 0:
                mask = (
                    (x >= 0)
                    & (x < self.sensor_width)
                    & (y >= 0)
                    & (y < self.sensor_height)
                )
                x = x[mask]
                y = y[mask]
                p = p[mask]
                t = t[mask]

            sample["x"] = x
            sample["y"] = y
            sample["pol"] = p
            sample["time"] = t

        if self.load_rgb:
            if seq.image_files is None:
                raise ValueError("image_files are not initialized although load_rgb is enabled.")
            img_path = seq.image_files[anchor_idx]
            with Image.open(img_path) as im:
                image = np.asarray(im.convert("RGB"), dtype=np.uint8)
            sample["image"] = image
            sample["image_path"] = str(img_path)

        if self.load_labels:
            assert seq.tracks is not None
            assert seq.img_idx_to_track_idx is not None
            i0, i1 = seq.img_idx_to_track_idx[anchor_idx]
            tracks = seq.tracks[int(i0) : int(i1)]
            sample["tracks"] = tracks
            sample["tracks_path"] = str(
                seq.root / "object_detections" / "left" / "tracks.npy"
            )

        return sample


class DSECVoxelCollator:
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
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pol: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros(
                (
                    self.voxel_builder.nb_channels,
                    self.voxel_builder.height,
                    self.voxel_builder.width,
                ),
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
                (
                    self.voxel_builder.nb_channels,
                    self.voxel_builder.height,
                    self.voxel_builder.width,
                ),
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
        voxels: list[torch.Tensor] = []
        paths: list[str] = []

        has_images = all("image" in s for s in samples)
        has_tracks = all("tracks" in s for s in samples)
        image_tensors: list[torch.Tensor] = []
        tracks_list: list[np.ndarray] = []

        for sample in samples:
            if not {"x", "y", "pol", "time"}.issubset(sample.keys()):
                raise ValueError(
                    "DSECVoxelCollator requires event fields x/y/pol/time. "
                    "Set data.dsec.load_events=true when using voxel provider."
                )
            voxels.append(
                self._to_voxel(
                    x=sample["x"],
                    y=sample["y"],
                    pol=sample["pol"],
                    time=sample["time"],
                )
            )
            paths.append(str(sample["path"]))

            if has_images:
                image_np = np.asarray(sample["image"], dtype=np.uint8)
                image_tensors.append(torch.from_numpy(image_np).permute(2, 0, 1).contiguous())
            if has_tracks:
                tracks_list.append(sample["tracks"])

        batch = torch.stack(voxels, dim=0).unsqueeze(1)
        output: dict[str, Any] = {
            "inputs": batch,
            "labels": tracks_list if has_tracks else None,
            "paths": paths,
        }
        if has_images:
            output["images"] = torch.stack(image_tensors, dim=0)
        if has_tracks:
            output["tracks"] = tracks_list
        return output
