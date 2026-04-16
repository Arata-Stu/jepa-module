#!/usr/bin/env python3
"""Benchmark N-ImageNet event loading speed for NPZ vs H5.

This script measures end-to-end loading time over all files, using the same
loader logic as training (`_load_n_imagenet_events`).
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from event.data.n_imagenet import _load_n_imagenet_events  # noqa: E402


@dataclass
class BenchResult:
    label: str
    elapsed_sec: float
    files_ok: int
    files_failed: int
    total_events: int
    checksum: int


@dataclass
class BenchAggregate:
    label: str
    trial_results: list[BenchResult]

    @property
    def median_elapsed(self) -> float:
        return statistics.median(r.elapsed_sec for r in self.trial_results)

    @property
    def best_elapsed(self) -> float:
        return min(r.elapsed_sec for r in self.trial_results)

    @property
    def worst_elapsed(self) -> float:
        return max(r.elapsed_sec for r in self.trial_results)

    @property
    def files_ok(self) -> int:
        return self.trial_results[-1].files_ok

    @property
    def files_failed(self) -> int:
        return self.trial_results[-1].files_failed

    @property
    def total_events(self) -> int:
        return self.trial_results[-1].total_events


def _read_list_file(list_file: Path, root_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    for line in list_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = line.split()[0]
        p = Path(raw)
        if not p.is_absolute():
            base = root_dir if root_dir is not None else list_file.parent
            p = base / p
        paths.append(p)
    return paths


def _collect_paths_from_lists(list_files: Iterable[Path], root_dir: Path | None) -> list[Path]:
    resolved: list[Path] = []
    for list_file in list_files:
        if not list_file.exists():
            raise FileNotFoundError(f"List file not found: {list_file}")
        resolved.extend(_read_list_file(list_file=list_file, root_dir=root_dir))

    unique: dict[str, Path] = {}
    for p in resolved:
        unique[str(p)] = p
    return list(unique.values())


def _collect_paths_from_root(
    *,
    dataset_root: Path,
    splits: list[str],
    recursive: bool,
    suffix: str,
) -> list[Path]:
    collected: list[Path] = []
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        if recursive:
            matches = split_dir.rglob(f"*{suffix}")
        else:
            matches = split_dir.glob(f"*{suffix}")
        collected.extend([p for p in matches if p.is_file()])

    unique: dict[str, Path] = {}
    for p in collected:
        unique[str(p)] = p
    return list(unique.values())


def _limit_paths(
    *,
    paths: list[Path],
    max_files: int | None,
    sample_mode: str,
    seed: int,
) -> list[Path]:
    if max_files is None:
        return paths
    if max_files < 1:
        raise ValueError("--max_files must be >= 1")
    if len(paths) <= max_files:
        return paths

    if sample_mode == "head":
        return paths[:max_files]
    if sample_mode == "random":
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(paths)), k=max_files))
        return [paths[i] for i in indices]
    raise ValueError(f"Unsupported sample_mode: {sample_mode}")


def _normalize_h5_suffix(h5_suffix: str) -> str:
    suffix = h5_suffix
    if not suffix.endswith(".h5"):
        suffix = f"{suffix}.h5"
    return suffix


def _derive_h5_paths(npz_paths: list[Path], npz_root: Path, h5_root: Path, h5_suffix: str) -> list[Path]:
    normalized_suffix = _normalize_h5_suffix(h5_suffix)
    h5_paths: list[Path] = []
    for npz_path in npz_paths:
        rel = npz_path.relative_to(npz_root)
        h5_path = h5_root / rel.parent / f"{npz_path.stem}{normalized_suffix}"
        h5_paths.append(h5_path)
    return h5_paths


def _benchmark_once(
    *,
    label: str,
    paths: list[Path],
    compressed: bool,
    time_scale: float,
    skip_missing: bool,
    skip_errors: bool,
    show_progress: bool,
) -> BenchResult:
    total_events = 0
    files_ok = 0
    files_failed = 0
    checksum = 0

    t0 = perf_counter()
    iterator = tqdm.tqdm(paths, desc=label, leave=False) if show_progress else paths
    for path in iterator:
        if not path.exists():
            if skip_missing:
                files_failed += 1
                continue
            raise FileNotFoundError(f"Missing file: {path}")

        try:
            x, y, pol, t = _load_n_imagenet_events(
                event_path=path,
                compressed=compressed,
                time_scale=time_scale,
            )
        except Exception:
            if skip_errors:
                files_failed += 1
                continue
            raise

        n_events = int(t.numel())
        total_events += n_events
        files_ok += 1

        # Lightweight checksum to avoid accidental dead-code elimination in future edits.
        local = n_events
        if n_events > 0:
            local ^= int(x[0].item())
            local ^= int(y[-1].item())
            local ^= int(pol[0].item())
            local ^= int(t[-1].item() & 0xFFFFFFFF)
        checksum ^= local

    elapsed = perf_counter() - t0
    return BenchResult(
        label=label,
        elapsed_sec=elapsed,
        files_ok=files_ok,
        files_failed=files_failed,
        total_events=total_events,
        checksum=checksum,
    )


def _run_trials(
    *,
    label: str,
    paths: list[Path],
    trials: int,
    compressed: bool,
    time_scale: float,
    skip_missing: bool,
    skip_errors: bool,
    show_progress: bool,
) -> BenchAggregate:
    if trials < 1:
        raise ValueError("trials must be >= 1")

    results: list[BenchResult] = []
    for i in range(trials):
        trial_label = f"{label} (trial {i + 1}/{trials})" if trials > 1 else label
        result = _benchmark_once(
            label=trial_label,
            paths=paths,
            compressed=compressed,
            time_scale=time_scale,
            skip_missing=skip_missing,
            skip_errors=skip_errors,
            show_progress=show_progress,
        )
        results.append(result)
    return BenchAggregate(label=label, trial_results=results)


def _format_events_per_sec(events: int, elapsed_sec: float) -> str:
    if elapsed_sec <= 0.0:
        return "inf"
    return f"{events / elapsed_sec:,.0f}"


def _print_summary(summary: BenchAggregate) -> None:
    median = summary.median_elapsed
    files_per_sec = summary.files_ok / median if median > 0 else float("inf")
    events_per_sec = summary.total_events / median if median > 0 else float("inf")

    print(f"[{summary.label}] files_ok={summary.files_ok} files_failed={summary.files_failed}")
    print(f"[{summary.label}] total_events={summary.total_events:,}")
    print(
        f"[{summary.label}] elapsed_sec median={summary.median_elapsed:.4f} "
        f"best={summary.best_elapsed:.4f} worst={summary.worst_elapsed:.4f}"
    )
    print(
        f"[{summary.label}] throughput files_per_sec={files_per_sec:.2f} "
        f"events_per_sec={events_per_sec:,.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser("Benchmark N-ImageNet loading speed (npz vs h5)")

    parser.add_argument(
        "--npz_list_files",
        type=Path,
        nargs="+",
        default=None,
        help="List file(s) for NPZ dataset (list mode).",
    )
    parser.add_argument(
        "--npz_dataset_root",
        type=Path,
        default=None,
        help="NPZ dataset root (root mode). Use with --splits; recursive is enabled by default.",
    )
    parser.add_argument(
        "--npz_root",
        type=Path,
        default=None,
        help="Base dir for relative NPZ list paths and H5 path derivation.",
    )

    parser.add_argument(
        "--h5_list_files",
        type=Path,
        nargs="+",
        default=None,
        help="Optional list file(s) for H5 dataset.",
    )
    parser.add_argument(
        "--h5_dataset_root",
        type=Path,
        default=None,
        help="H5 dataset root for NPZ->H5 path derivation in root mode.",
    )
    parser.add_argument(
        "--h5_root",
        type=Path,
        default=None,
        help="H5 root dir (for relative H5 list paths, or NPZ->H5 path derivation in list mode).",
    )
    parser.add_argument(
        "--h5_suffix",
        type=str,
        default="_2x.h5",
        help="Suffix used when deriving H5 paths from NPZ paths.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training", "validation"],
        help="Split directories for root mode.",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Recursively search NPZ files under split directories in root mode (default: enabled).",
    )
    parser.add_argument(
        "--no_recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive search in root mode.",
    )

    parser.add_argument(
        "--compressed",
        dest="compressed",
        action="store_true",
        default=True,
        help="Use compressed NPZ key format (event_data) when available.",
    )
    parser.add_argument(
        "--uncompressed",
        dest="compressed",
        action="store_false",
        help="Use legacy NPZ keys x_pos/y_pos/timestamp/polarity.",
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        default=1_000_000.0,
        help="Scale factor for second-based timestamps in NPZ.",
    )

    parser.add_argument("--trials", type=int, default=1, help="Number of repeated full-dataset trials.")
    parser.add_argument("--skip_missing", action="store_true", help="Skip missing files instead of failing.")
    parser.add_argument("--skip_errors", action="store_true", help="Skip read errors instead of failing.")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Limit number of files per dataset for quick benchmarking.",
    )
    parser.add_argument(
        "--sample_mode",
        choices=["head", "random"],
        default="head",
        help="How to select files when --max_files is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --sample_mode random.",
    )

    args = parser.parse_args()

    npz_list_mode = args.npz_list_files is not None and len(args.npz_list_files) > 0
    npz_root_mode = args.npz_dataset_root is not None
    if int(npz_list_mode) + int(npz_root_mode) != 1:
        raise ValueError(
            "Choose exactly one NPZ input mode: --npz_list_files or --npz_dataset_root"
        )

    if npz_list_mode:
        npz_paths = _collect_paths_from_lists(
            list_files=args.npz_list_files,
            root_dir=args.npz_root,
        )
        npz_base_for_derivation = args.npz_root
    else:
        npz_paths = _collect_paths_from_root(
            dataset_root=args.npz_dataset_root,
            splits=[str(s) for s in args.splits],
            recursive=bool(args.recursive),
            suffix=".npz",
        )
        npz_base_for_derivation = args.npz_dataset_root

    if len(npz_paths) == 0:
        raise FileNotFoundError("No NPZ files found")

    npz_paths = _limit_paths(
        paths=npz_paths,
        max_files=args.max_files,
        sample_mode=str(args.sample_mode),
        seed=int(args.seed),
    )

    h5_list_mode = args.h5_list_files is not None and len(args.h5_list_files) > 0
    h5_root_mode = args.h5_dataset_root is not None
    if h5_list_mode and h5_root_mode:
        raise ValueError("Use either --h5_list_files or --h5_dataset_root, not both")

    if h5_list_mode:
        h5_root_for_lists = args.h5_root if args.h5_root is not None else args.h5_dataset_root
        h5_paths = _collect_paths_from_lists(
            list_files=args.h5_list_files,
            root_dir=h5_root_for_lists,
        )
        h5_paths = _limit_paths(
            paths=h5_paths,
            max_files=args.max_files,
            sample_mode=str(args.sample_mode),
            seed=int(args.seed),
        )
    else:
        if npz_base_for_derivation is None:
            raise ValueError(
                "--npz_root is required for H5 path derivation when using --npz_list_files"
            )
        h5_base = args.h5_dataset_root if h5_root_mode else args.h5_root
        if h5_base is None:
            raise ValueError(
                "Specify --h5_list_files, --h5_dataset_root, or --h5_root for H5 inputs"
            )
        h5_paths = _derive_h5_paths(
            npz_paths=npz_paths,
            npz_root=npz_base_for_derivation,
            h5_root=h5_base,
            h5_suffix=args.h5_suffix,
        )

    if len(h5_paths) == 0:
        raise FileNotFoundError("No H5 paths found")

    show_progress = not bool(args.no_progress)

    print(f"[INFO] npz files: {len(npz_paths)}")
    print(f"[INFO] h5 files:  {len(h5_paths)}")
    if len(npz_paths) != len(h5_paths):
        print("[WARN] NPZ/H5 file counts differ. Throughput will still be measured independently.")

    npz_summary = _run_trials(
        label="NPZ",
        paths=npz_paths,
        trials=int(args.trials),
        compressed=bool(args.compressed),
        time_scale=float(args.time_scale),
        skip_missing=bool(args.skip_missing),
        skip_errors=bool(args.skip_errors),
        show_progress=show_progress,
    )
    h5_summary = _run_trials(
        label="H5",
        paths=h5_paths,
        trials=int(args.trials),
        compressed=bool(args.compressed),
        time_scale=float(args.time_scale),
        skip_missing=bool(args.skip_missing),
        skip_errors=bool(args.skip_errors),
        show_progress=show_progress,
    )

    print("\n=== Summary ===")
    _print_summary(npz_summary)
    _print_summary(h5_summary)

    npz_t = npz_summary.median_elapsed
    h5_t = h5_summary.median_elapsed
    if h5_t > 0:
        print(f"[SPEEDUP] npz_over_h5_time_ratio={npz_t / h5_t:.3f}x")
    if npz_t > 0:
        print(f"[SPEEDUP] h5_over_npz_time_ratio={h5_t / npz_t:.3f}x")

    print(
        "[CHECKSUM] "
        f"npz={npz_summary.trial_results[-1].checksum} "
        f"h5={h5_summary.trial_results[-1].checksum}"
    )


if __name__ == "__main__":
    main()
