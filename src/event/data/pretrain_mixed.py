from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

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


@dataclass(frozen=True)
class PretrainMixedSourceConfig:
    name: str
    root_dir: Path
    splits: tuple[str, ...]
    sensor_height: int
    sensor_width: int
    recursive: bool = True
    file_name: str | None = None
    file_suffix: str | None = ".h5"
    manifest_file: Path | None = None


@dataclass(frozen=True)
class _EventFileRecord:
    source: str
    path: Path
    sensor_height: int
    sensor_width: int


def _discover_h5_files(
    root_dir: Path,
    splits: Sequence[str],
    recursive: bool,
    file_name: str | None,
    file_suffix: str | None,
    manifest_file: Path | None = None,
) -> list[Path]:
    if manifest_file is not None:
        files: list[Path] = []
        missing_count = 0
        split_dirs = [(root_dir / split).resolve() for split in splits]

        entries: list[str] = []
        if manifest_file.suffix.lower() == ".npy":
            loaded = np.load(str(manifest_file), allow_pickle=True)
            if isinstance(loaded, np.ndarray):
                loaded_obj = loaded.tolist()
                if isinstance(loaded_obj, (list, tuple)):
                    entries = [str(v) for v in loaded_obj]
                else:
                    entries = [str(loaded_obj)]
            else:
                entries = [str(loaded)]
        else:
            with manifest_file.open("r", encoding="utf-8") as f:
                for line in f:
                    item = line.strip()
                    if len(item) == 0 or item.startswith("#"):
                        continue
                    if "," in item:
                        item = item.split(",", 1)[0].strip()
                    elif "\t" in item:
                        item = item.split("\t", 1)[0].strip()
                    elif (" " in item) and (not item.lower().endswith((".h5", ".hdf5"))):
                        first = item.split(" ", 1)[0].strip()
                        if first.lower().endswith((".h5", ".hdf5")):
                            item = first
                    if len(item) == 0:
                        continue
                    entries.append(item)

        for item in entries:
            p = Path(item).expanduser()
            if not p.is_absolute():
                p = (root_dir / p).resolve()
            if not p.is_file():
                missing_count += 1
                continue

            p_resolved = p.resolve()
            if len(split_dirs) > 0 and not any(sd == p_resolved or sd in p_resolved.parents for sd in split_dirs):
                continue
            if file_name is not None and p.name != file_name:
                continue
            if file_suffix is not None and not p.name.endswith(file_suffix):
                continue
            files.append(p_resolved)

        files = sorted(set(files))
        if missing_count > 0:
            print(
                f"[pretrain_mixed][WARN] manifest={manifest_file} "
                f"missing_entries={missing_count}"
            )
        return files

    files: list[Path] = []
    for split in splits:
        split_dir = root_dir / split
        if not split_dir.exists():
            print(f"[WARN] missing split directory: {split_dir}")
            continue

        iterator = split_dir.rglob("*.h5") if recursive else split_dir.glob("*.h5")
        for p in iterator:
            if not p.is_file():
                continue
            if file_name is not None and p.name != file_name:
                continue
            if file_suffix is not None and not p.name.endswith(file_suffix):
                continue
            files.append(p)
    files = sorted(set(files))
    return files


class PretrainMixedEventsDataset(Dataset):
    def __init__(
        self,
        source_configs: Sequence[PretrainMixedSourceConfig],
        events_per_sample_min: int,
        events_per_sample_max: int,
        random_slice: bool,
        time_normalize: bool,
        source_weights: dict[str, float] | None = None,
        drop_empty: bool = True,
        window_duration_us_min: int | None = None,
        window_duration_us_max: int | None = None,
        duration_sources: Sequence[str] | None = None,
        prefer_ms_to_idx: bool = True,
        min_events_in_window: int = 1,
        min_event_rate_eps: float | None = None,
        activity_filter_sources: Sequence[str] | None = None,
        max_window_attempts: int = 4,
    ):
        super().__init__()
        self.events_per_sample_min = int(events_per_sample_min)
        self.events_per_sample_max = int(events_per_sample_max)
        self.random_slice = bool(random_slice)
        self.time_normalize = bool(time_normalize)
        self.drop_empty = bool(drop_empty)
        self.window_duration_us_min = (
            None if window_duration_us_min is None else int(window_duration_us_min)
        )
        self.window_duration_us_max = (
            None if window_duration_us_max is None else int(window_duration_us_max)
        )
        self.duration_sources = set(duration_sources or [])
        self.prefer_ms_to_idx = bool(prefer_ms_to_idx)
        self.min_events_in_window = int(min_events_in_window)
        self.min_event_rate_eps = (
            None if min_event_rate_eps is None else float(min_event_rate_eps)
        )
        self.activity_filter_sources = set(activity_filter_sources or [])
        self.max_window_attempts = max(1, int(max_window_attempts))
        self._max_read_retry = 8
        self._warn_budget = 20
        self._warned_read_errors = 0

        if self.events_per_sample_min < 1:
            raise ValueError("events_per_sample_min must be >= 1")
        if self.events_per_sample_max < self.events_per_sample_min:
            raise ValueError("events_per_sample_max must be >= events_per_sample_min")
        if self.min_events_in_window < 1:
            raise ValueError("min_events_in_window must be >= 1")
        if self.min_event_rate_eps is not None and self.min_event_rate_eps <= 0.0:
            raise ValueError("min_event_rate_eps must be > 0 when set")
        if (self.window_duration_us_min is None) != (self.window_duration_us_max is None):
            raise ValueError(
                "window_duration_us_min and window_duration_us_max must be both set or both None"
            )
        if self.window_duration_us_min is not None:
            if self.window_duration_us_min < 1:
                raise ValueError("window_duration_us_min must be >= 1")
            if self.window_duration_us_max is None or self.window_duration_us_max < self.window_duration_us_min:
                raise ValueError("window_duration_us_max must be >= window_duration_us_min")

        self.records = self._build_records(source_configs=source_configs)
        if len(self.records) == 0:
            raise FileNotFoundError("No downsampled .h5 files found for pretrain_mixed sources")

        self.source_counts = Counter(r.source for r in self.records)
        self.sample_weights = self._build_sample_weights(source_weights=source_weights)

    @staticmethod
    def _build_records(source_configs: Sequence[PretrainMixedSourceConfig]) -> list[_EventFileRecord]:
        records: list[_EventFileRecord] = []
        for cfg in source_configs:
            files = _discover_h5_files(
                root_dir=cfg.root_dir,
                splits=cfg.splits,
                recursive=cfg.recursive,
                file_name=cfg.file_name,
                file_suffix=cfg.file_suffix,
                manifest_file=cfg.manifest_file,
            )
            mode = "manifest" if cfg.manifest_file is not None else "scan"
            print(
                f"[pretrain_mixed] source={cfg.name} files={len(files)} "
                f"root={cfg.root_dir} mode={mode}"
            )
            for p in files:
                records.append(
                    _EventFileRecord(
                        source=str(cfg.name),
                        path=p,
                        sensor_height=int(cfg.sensor_height),
                        sensor_width=int(cfg.sensor_width),
                    )
                )
        return records

    def _build_sample_weights(self, source_weights: dict[str, float] | None) -> torch.Tensor:
        enabled_sources = sorted(self.source_counts.keys())
        if len(enabled_sources) == 0:
            raise ValueError("No enabled sources")

        if source_weights is None:
            raw = {name: 1.0 for name in enabled_sources}
        else:
            raw = {name: max(0.0, float(source_weights.get(name, 0.0))) for name in enabled_sources}
            if sum(raw.values()) <= 0.0:
                raw = {name: 1.0 for name in enabled_sources}

        total = float(sum(raw.values()))
        source_prob = {name: val / total for name, val in raw.items()}

        weights: list[float] = []
        for r in self.records:
            count = int(self.source_counts[r.source])
            w = source_prob[r.source] / float(count)
            weights.append(w)
        return torch.tensor(weights, dtype=torch.double)

    @staticmethod
    def _resolve_sensor_size(
        h5f: h5py.File,
        events_group: h5py.Group,
        fallback_height: int,
        fallback_width: int,
    ) -> tuple[int, int]:
        if "height" in events_group and "width" in events_group:
            try:
                h = int(np.asarray(events_group["height"][()]).item())
                w = int(np.asarray(events_group["width"][()]).item())
                if h > 0 and w > 0:
                    return h, w
            except Exception:
                pass

        if "height" in h5f and "width" in h5f:
            try:
                h = int(np.asarray(h5f["height"][()]).item())
                w = int(np.asarray(h5f["width"][()]).item())
                if h > 0 and w > 0:
                    return h, w
            except Exception:
                pass

        return int(fallback_height), int(fallback_width)

    def _sample_range_by_event_count(self, total_events: int) -> tuple[int, int]:
        if total_events <= 0:
            return 0, 0

        if self.events_per_sample_max == self.events_per_sample_min:
            target_events = self.events_per_sample_min
        else:
            target_events = random.randint(self.events_per_sample_min, self.events_per_sample_max)

        if total_events <= target_events:
            return 0, total_events

        if self.random_slice:
            start_idx = random.randint(0, total_events - target_events)
        else:
            start_idx = (total_events - target_events) // 2
        return start_idx, start_idx + target_events

    def _sample_range_by_duration_us(
        self,
        h5f: h5py.File,
        events_group: h5py.Group,
        total_events: int,
    ) -> tuple[int, int]:
        if (
            self.window_duration_us_min is None
            or self.window_duration_us_max is None
            or total_events <= 0
        ):
            return self._sample_range_by_event_count(total_events=total_events)

        duration_us = (
            self.window_duration_us_min
            if self.window_duration_us_min == self.window_duration_us_max
            else random.randint(self.window_duration_us_min, self.window_duration_us_max)
        )

        t_dataset = events_group["t"]
        t_offset = int(np.asarray(h5f["t_offset"][()]).item()) if "t_offset" in h5f else 0
        t_first_abs = int(t_dataset[0]) + t_offset
        t_last_abs = int(t_dataset[total_events - 1]) + t_offset

        if t_last_abs <= t_first_abs:
            return self._sample_range_by_event_count(total_events=total_events)

        available_span_us = t_last_abs - t_first_abs
        if available_span_us <= duration_us:
            target_start_abs = t_first_abs
        else:
            if self.random_slice:
                target_start_abs = random.randint(t_first_abs, t_last_abs - duration_us)
            else:
                target_start_abs = t_first_abs + (available_span_us - duration_us) // 2
        target_end_abs = target_start_abs + duration_us

        target_start_raw = target_start_abs - t_offset
        target_end_raw = target_end_abs - t_offset

        coarse_start = 0
        coarse_end = total_events
        if self.prefer_ms_to_idx and "ms_to_idx" in h5f:
            ms_to_idx = np.asarray(h5f["ms_to_idx"], dtype=np.int64)
            if ms_to_idx.size > 0:
                start_ms = int(np.clip(target_start_raw // 1000, 0, ms_to_idx.size - 1))
                end_ms = int(np.clip(math.ceil(target_end_raw / 1000) + 1, 0, ms_to_idx.size - 1))
                coarse_start = int(ms_to_idx[start_ms])
                if end_ms >= ms_to_idx.size - 1:
                    coarse_end = total_events
                else:
                    coarse_end = int(ms_to_idx[end_ms])
                if coarse_end <= coarse_start:
                    coarse_start = int(np.clip(coarse_start, 0, total_events - 1))
                    coarse_end = min(total_events, coarse_start + 1)

        t_slice = np.asarray(t_dataset[coarse_start:coarse_end], dtype=np.int64)
        if t_slice.size == 0:
            idx = int(np.clip(coarse_start, 0, max(0, total_events - 1)))
            return idx, min(total_events, idx + 1)

        local_start = int(np.searchsorted(t_slice, target_start_raw, side="left"))
        local_end = int(np.searchsorted(t_slice, target_end_raw, side="left"))
        start_idx = int(np.clip(coarse_start + local_start, 0, total_events - 1))
        end_idx = int(np.clip(coarse_start + max(local_end, local_start + 1), start_idx + 1, total_events))
        return start_idx, end_idx

    def _resolve_slice_range(
        self,
        record: _EventFileRecord,
        h5f: h5py.File,
        events_group: h5py.Group,
        total_events: int,
    ) -> tuple[int, int]:
        if (
            self.window_duration_us_min is not None
            and self.window_duration_us_max is not None
            and record.source in self.duration_sources
        ):
            return self._sample_range_by_duration_us(
                h5f=h5f,
                events_group=events_group,
                total_events=total_events,
            )
        return self._sample_range_by_event_count(total_events=total_events)

    def _activity_filter_enabled(self, source: str) -> bool:
        if self.min_events_in_window <= 1 and self.min_event_rate_eps is None:
            return False
        if len(self.activity_filter_sources) == 0:
            return True
        return source in self.activity_filter_sources

    def _passes_activity_filter(
        self,
        record: _EventFileRecord,
        x: np.ndarray,
        t: np.ndarray,
    ) -> bool:
        if not self._activity_filter_enabled(record.source):
            return True

        count = int(x.size)
        if count < self.min_events_in_window:
            return False

        if self.min_event_rate_eps is not None:
            if count <= 1:
                return False
            span_us = int(t[-1]) - int(t[0])
            if span_us <= 0:
                return False
            event_rate_eps = float(count) * 1_000_000.0 / float(span_us)
            if event_rate_eps < self.min_event_rate_eps:
                return False

        return True

    def _read_event_window(self, record: _EventFileRecord) -> dict[str, Any]:
        with _open_h5_for_read(record.path) as h5f:
            events = h5f["events"] if "events" in h5f else h5f
            total_events = int(events["t"].shape[0])
            sensor_h, sensor_w = self._resolve_sensor_size(
                h5f=h5f,
                events_group=events,
                fallback_height=record.sensor_height,
                fallback_width=record.sensor_width,
            )
            t_offset = int(np.asarray(h5f["t_offset"][()]).item()) if "t_offset" in h5f else 0

            if total_events == 0:
                x = np.zeros((0,), dtype=np.int64)
                y = np.zeros((0,), dtype=np.int64)
                p = np.zeros((0,), dtype=np.int64)
                t = np.zeros((0,), dtype=np.int64)
            else:
                accepted = False
                x = np.zeros((0,), dtype=np.int64)
                y = np.zeros((0,), dtype=np.int64)
                p = np.zeros((0,), dtype=np.int64)
                t = np.zeros((0,), dtype=np.int64)
                for _ in range(self.max_window_attempts):
                    start_idx, end_idx = self._resolve_slice_range(
                        record=record,
                        h5f=h5f,
                        events_group=events,
                        total_events=total_events,
                    )

                    x = np.asarray(events["x"][start_idx:end_idx], dtype=np.int64)
                    y = np.asarray(events["y"][start_idx:end_idx], dtype=np.int64)
                    p_raw = np.asarray(events["p"][start_idx:end_idx])
                    p = (p_raw > 0).astype(np.int64, copy=False)
                    t = np.asarray(events["t"][start_idx:end_idx], dtype=np.int64)
                    if t_offset != 0:
                        t = t + t_offset

                    if t.size > 1 and np.any(t[1:] < t[:-1]):
                        order = np.argsort(t, kind="stable")
                        x = x[order]
                        y = y[order]
                        p = p[order]
                        t = t[order]

                    if x.size > 0:
                        mask = (
                            (x >= 0)
                            & (x < sensor_w)
                            & (y >= 0)
                            & (y < sensor_h)
                        )
                        x = x[mask]
                        y = y[mask]
                        p = p[mask]
                        t = t[mask]

                    if self._passes_activity_filter(record=record, x=x, t=t):
                        accepted = True
                        break

                if not accepted:
                    x = np.zeros((0,), dtype=np.int64)
                    y = np.zeros((0,), dtype=np.int64)
                    p = np.zeros((0,), dtype=np.int64)
                    t = np.zeros((0,), dtype=np.int64)

        if t.size > 1 and np.any(t[1:] < t[:-1]):
            order = np.argsort(t, kind="stable")
            x = x[order]
            y = y[order]
            p = p[order]
            t = t[order]

        if self.time_normalize and t.size > 0:
            t = t - int(t[0])

        if x.size > 0:
            mask = (
                (x >= 0)
                & (x < sensor_w)
                & (y >= 0)
                & (y < sensor_h)
            )
            x = x[mask]
            y = y[mask]
            p = p[mask]
            t = t[mask]

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "pol": torch.from_numpy(p),
            "time": torch.from_numpy(t),
            "path": str(record.path),
            "source": str(record.source),
            "sensor_height": int(sensor_h),
            "sensor_width": int(sensor_w),
        }

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _sample_alternative_index(total: int, exclude_idx: int) -> int:
        if total <= 1:
            return int(exclude_idx)
        r = random.randint(0, total - 2)
        return r + 1 if r >= exclude_idx else r

    @staticmethod
    def _build_empty_sample(record: _EventFileRecord) -> dict[str, Any]:
        empty = torch.empty((0,), dtype=torch.int64)
        return {
            "x": empty,
            "y": empty,
            "pol": empty,
            "time": empty,
            "path": str(record.path),
            "source": str(record.source),
            "sensor_height": int(record.sensor_height),
            "sensor_width": int(record.sensor_width),
        }

    def _warn_read_error(self, record: _EventFileRecord, exc: Exception) -> None:
        if self._warned_read_errors >= self._warn_budget:
            return
        print(
            f"[pretrain_mixed][WARN] failed to read source={record.source} "
            f"path={record.path}: {type(exc).__name__}: {exc}"
        )
        self._warned_read_errors += 1

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if len(self.records) == 0:
            raise RuntimeError("PretrainMixedEventsDataset has no records")

        total = len(self.records)
        current_idx = int(idx) % total
        max_attempts = 1 if total <= 1 else min(self._max_read_retry, total)
        last_sample: dict[str, Any] | None = None
        last_record = self.records[current_idx]

        for _ in range(max_attempts):
            record = self.records[current_idx]
            last_record = record
            try:
                sample = self._read_event_window(record=record)
                last_sample = sample
                if (not self.drop_empty) or sample["x"].numel() > 0 or total <= 1:
                    return sample
            except Exception as exc:
                self._warn_read_error(record=record, exc=exc)

            current_idx = self._sample_alternative_index(total=total, exclude_idx=current_idx)

        if last_sample is not None:
            return last_sample
        return self._build_empty_sample(record=last_record)


class _DistributedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        weights: torch.Tensor,
        num_replicas: int,
        rank: int,
        num_samples: int,
        replacement: bool = True,
        seed: int = 0,
    ):
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas - 1}]")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if weights.numel() == 0:
            raise ValueError("weights must be non-empty")

        self.weights = weights.double()
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = int(num_samples)
        self.total_size = int(self.num_samples * self.num_replicas)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=g,
        ).tolist()
        rank_indices = global_indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples


class PretrainMixedVoxelCollator:
    def __init__(
        self,
        voxel_builder: VoxelGrid,
        normalize_voxel: bool,
        rescale_to_voxel_grid: bool = True,
        canvas_height: int = 240,
        canvas_width: int = 320,
        center_pad_to_canvas: bool = True,
        debug_index_check: bool = False,
        debug_raise_on_oob: bool = False,
        debug_log_limit: int = 20,
    ):
        self.voxel_builder = voxel_builder
        self.normalize_voxel = normalize_voxel
        self.rescale_to_voxel_grid = rescale_to_voxel_grid
        self.canvas_height = int(canvas_height)
        self.canvas_width = int(canvas_width)
        self.center_pad_to_canvas = bool(center_pad_to_canvas)
        self.debug_index_check = bool(debug_index_check)
        self.debug_raise_on_oob = bool(debug_raise_on_oob)
        self.debug_log_limit = max(1, int(debug_log_limit))
        self._debug_logged = 0

    def _empty_voxel(self) -> torch.Tensor:
        return torch.zeros(
            (
                self.voxel_builder.nb_channels,
                self.voxel_builder.height,
                self.voxel_builder.width,
            ),
            dtype=torch.float32,
        )

    def _align_to_canvas(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sensor_h: int,
        sensor_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if not self.center_pad_to_canvas:
            return x, y, int(sensor_h), int(sensor_w)

        if sensor_h == self.canvas_height and sensor_w == self.canvas_width:
            return x, y, int(sensor_h), int(sensor_w)

        if sensor_h <= self.canvas_height and sensor_w <= self.canvas_width:
            pad_y = (self.canvas_height - sensor_h) // 2
            pad_x = (self.canvas_width - sensor_w) // 2
            return x + int(pad_x), y + int(pad_y), self.canvas_height, self.canvas_width

        x_scaled = torch.floor(x.float() * (self.canvas_width / float(sensor_w))).long()
        y_scaled = torch.floor(y.float() * (self.canvas_height / float(sensor_h))).long()
        return x_scaled, y_scaled, self.canvas_height, self.canvas_width

    @staticmethod
    def _minmax(t: torch.Tensor) -> str:
        if t.numel() == 0:
            return "empty"
        return f"{int(t.min().item())}..{int(t.max().item())}"

    def _debug_log_oob(
        self,
        *,
        sample: dict[str, Any],
        x: torch.Tensor,
        y: torch.Tensor,
        time: torch.Tensor,
        t0_center: int | None,
        t1_center: int | None,
        oob_count: int,
        extra: str = "",
    ) -> None:
        if self._debug_logged >= self.debug_log_limit:
            return
        msg = (
            "[pretrain_mixed][debug_oob] "
            f"source={sample.get('source')} path={sample.get('path')} "
            f"events={int(x.numel())} "
            f"x={self._minmax(x)} y={self._minmax(y)} t={self._minmax(time)} "
            f"t0={t0_center} t1={t1_center} "
            f"bins={self.voxel_builder.time_bins} h={self.voxel_builder.height} w={self.voxel_builder.width} "
            f"oob_count={oob_count}"
        )
        if len(extra) > 0:
            msg += f" {extra}"
        print(msg)
        self._debug_logged += 1

    def _debug_check_flat_indices(
        self,
        *,
        sample: dict[str, Any],
        x: torch.Tensor,
        y: torch.Tensor,
        time: torch.Tensor,
        t0_center: int | None,
        t1_center: int | None,
    ) -> None:
        if not self.debug_index_check:
            return
        if x.numel() == 0:
            return

        ch = int(self.voxel_builder.time_bins)
        ht = int(self.voxel_builder.height)
        wd = int(self.voxel_builder.width)
        total = ch * ht * wd

        if ch == 1:
            idx = wd * y.long() + x.long()
            bad = (idx < 0) | (idx >= total)
            bad_count = int(bad.sum().item())
            if bad_count > 0:
                self._debug_log_oob(
                    sample=sample,
                    x=x,
                    y=y,
                    time=time,
                    t0_center=t0_center,
                    t1_center=t1_center,
                    oob_count=bad_count,
                    extra=f"idx={self._minmax(idx)}",
                )
                if self.debug_raise_on_oob:
                    raise RuntimeError(
                        "debug_index_check caught OOB flat indices in PretrainMixedVoxelCollator (ch=1)"
                    )
            return

        if t0_center is None or t1_center is None:
            return

        if t1_center <= t0_center:
            t1_center = t0_center + 1

        t_norm = (
            (time.to(torch.float32) - float(t0_center))
            / float(t1_center - t0_center)
            * float(ch - 1)
        )
        t0 = torch.floor(t_norm).to(torch.int64)
        bad_total = 0
        for offset in (0, 1):
            tlim = t0 + int(offset)
            mask = (
                (x >= 0)
                & (x < wd)
                & (y >= 0)
                & (y < ht)
                & (tlim >= 0)
                & (tlim < ch)
            )
            if not bool(torch.any(mask)):
                continue
            idx = ht * wd * tlim + wd * y.long() + x.long()
            bad = mask & ((idx < 0) | (idx >= total))
            bad_total += int(bad.sum().item())

        if bad_total > 0:
            self._debug_log_oob(
                sample=sample,
                x=x,
                y=y,
                time=time,
                t0_center=t0_center,
                t1_center=t1_center,
                oob_count=bad_total,
                extra=f"t_norm={float(t_norm.min().item()):.4f}..{float(t_norm.max().item()):.4f}",
            )
            if self.debug_raise_on_oob:
                raise RuntimeError(
                    "debug_index_check caught OOB flat indices in PretrainMixedVoxelCollator (ch>1)"
                )

    def _to_voxel(self, sample: dict[str, Any]) -> torch.Tensor:
        x = sample["x"].long()
        y = sample["y"].long()
        pol = sample["pol"].long()
        time = sample["time"].long()
        sensor_h = int(sample["sensor_height"])
        sensor_w = int(sample["sensor_width"])

        if x.numel() == 0:
            return self._empty_voxel()

        if time.numel() > 1 and torch.any(time[1:] < time[:-1]):
            order = torch.argsort(time)
            x = x[order]
            y = y[order]
            pol = pol[order]
            time = time[order]

        x, y, aligned_h, aligned_w = self._align_to_canvas(
            x=x,
            y=y,
            sensor_h=sensor_h,
            sensor_w=sensor_w,
        )
        mask = (
            (x >= 0)
            & (x < aligned_w)
            & (y >= 0)
            & (y < aligned_h)
        )
        x = x[mask]
        y = y[mask]
        pol = pol[mask]
        time = time[mask]

        if x.numel() == 0:
            return self._empty_voxel()

        if self.rescale_to_voxel_grid:
            x = torch.floor(x.float() * (self.voxel_builder.width / float(aligned_w))).long()
            y = torch.floor(y.float() * (self.voxel_builder.height / float(aligned_h))).long()

        mask = (
            (x >= 0)
            & (x < self.voxel_builder.width)
            & (y >= 0)
            & (y < self.voxel_builder.height)
        )
        x = x[mask]
        y = y[mask]
        pol = pol[mask]
        time = time[mask]

        if x.numel() == 0:
            return self._empty_voxel()

        if self.voxel_builder.time_bins > 1:
            if time.numel() == 1:
                x = torch.cat([x, x], dim=0)
                y = torch.cat([y, y], dim=0)
                pol = torch.cat([pol, pol], dim=0)
                time = torch.cat([time, time + 1], dim=0)
            elif int(time[-1]) <= int(time[0]):
                time = time.clone()
                time[-1] = time[0] + 1

        t0_center = int(torch.min(time).item()) if time.numel() > 0 else None
        t1_center = int(torch.max(time).item()) if time.numel() > 0 else None
        if t0_center is not None and t1_center is not None and t1_center <= t0_center:
            t1_center = t0_center + 1

        self._debug_check_flat_indices(
            sample=sample,
            x=x,
            y=y,
            time=time,
            t0_center=t0_center,
            t1_center=t1_center,
        )

        voxel = self.voxel_builder.convert(
            x=x,
            y=y,
            pol=pol,
            time=time,
            t0_center=t0_center,
            t1_center=t1_center,
        )
        if self.normalize_voxel:
            voxel = norm_voxel_grid(voxel)
        return voxel

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        voxels: list[torch.Tensor] = []
        paths: list[str] = []
        sources: list[str] = []
        for sample in samples:
            voxels.append(self._to_voxel(sample))
            paths.append(str(sample["path"]))
            sources.append(str(sample["source"]))

        batch = torch.stack(voxels, dim=0).unsqueeze(1)
        return {
            "inputs": batch,
            "labels": None,
            "paths": paths,
            "sources": sources,
        }


class PretrainMixedVoxelBatchProvider:
    def __init__(
        self,
        source_configs: Sequence[PretrainMixedSourceConfig],
        source_weights: dict[str, float] | None,
        voxel_builder: VoxelGrid,
        batch_size: int,
        normalize_voxel: bool,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        events_per_sample_min: int = 30_000,
        events_per_sample_max: int = 30_000,
        random_slice: bool = True,
        time_normalize: bool = True,
        window_duration_us_min: int | None = None,
        window_duration_us_max: int | None = None,
        duration_sources: Sequence[str] | None = None,
        prefer_ms_to_idx: bool = True,
        min_events_in_window: int = 1,
        min_event_rate_eps: float | None = None,
        activity_filter_sources: Sequence[str] | None = None,
        max_window_attempts: int = 4,
        use_source_balancing: bool = True,
        epoch_size: int | None = None,
        sampler_seed: int = 42,
        rescale_to_voxel_grid: bool = True,
        canvas_height: int = 240,
        canvas_width: int = 320,
        center_pad_to_canvas: bool = True,
        debug_index_check: bool = False,
        debug_raise_on_oob: bool = False,
        debug_log_limit: int = 20,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = PretrainMixedEventsDataset(
            source_configs=source_configs,
            events_per_sample_min=events_per_sample_min,
            events_per_sample_max=events_per_sample_max,
            random_slice=random_slice,
            time_normalize=time_normalize,
            source_weights=source_weights,
            drop_empty=True,
            window_duration_us_min=window_duration_us_min,
            window_duration_us_max=window_duration_us_max,
            duration_sources=duration_sources,
            prefer_ms_to_idx=prefer_ms_to_idx,
            min_events_in_window=min_events_in_window,
            min_event_rate_eps=min_event_rate_eps,
            activity_filter_sources=activity_filter_sources,
            max_window_attempts=max_window_attempts,
        )
        self._dataset_size = len(self.dataset)
        self._epoch_size = int(epoch_size) if epoch_size is not None else self._dataset_size
        if self._epoch_size < 1:
            raise ValueError("epoch_size must be >= 1")

        collator = PretrainMixedVoxelCollator(
            voxel_builder=voxel_builder,
            normalize_voxel=normalize_voxel,
            rescale_to_voxel_grid=rescale_to_voxel_grid,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            center_pad_to_canvas=center_pad_to_canvas,
            debug_index_check=debug_index_check,
            debug_raise_on_oob=debug_raise_on_oob,
            debug_log_limit=debug_log_limit,
        )

        self.sampler = None
        if distributed and world_size > 1 and use_source_balancing:
            per_rank_samples = int(math.ceil(self._epoch_size / float(world_size)))
            self.sampler = _DistributedWeightedSampler(
                weights=self.dataset.sample_weights,
                num_replicas=world_size,
                rank=rank,
                num_samples=per_rank_samples,
                replacement=True,
                seed=int(sampler_seed),
            )
        elif distributed and world_size > 1:
            from torch.utils.data.distributed import DistributedSampler

            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=drop_last,
            )
        elif use_source_balancing:
            self.sampler = WeightedRandomSampler(
                weights=self.dataset.sample_weights,
                num_samples=self._epoch_size,
                replacement=True,
            )

        loader_kwargs: dict[str, Any] = {}
        if int(num_workers) > 0:
            loader_kwargs["persistent_workers"] = bool(persistent_workers)
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=self.sampler is None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collator,
            sampler=self.sampler,
            **loader_kwargs,
        )
        self._sampler_epoch = 0
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self._sampler_epoch)
        self._iter = iter(self.loader)

    def next_batch(self) -> dict[str, Any]:
        try:
            batch = next(self._iter)
        except StopIteration:
            if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
                self._sampler_epoch += 1
                self.sampler.set_epoch(self._sampler_epoch)
            self._iter = iter(self.loader)
            batch = next(self._iter)
        return batch

    @property
    def num_samples(self) -> int:
        if self.sampler is not None:
            return len(self.sampler)
        return self._dataset_size
