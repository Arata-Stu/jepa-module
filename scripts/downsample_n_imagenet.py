import argparse
from pathlib import Path
import weakref
import os
import multiprocessing as mp

import h5py
import numba
import numpy as np
import tqdm

try:
    import hdf5plugin  # noqa: F401

    _HAS_BLOSC = True
except ImportError:
    _HAS_BLOSC = False


def _compression_opts():
    compression_level = 1  # {0, ..., 9}
    shuffle = 2  # {0: none, 1: byte, 2: bit}
    compressor_type = 5  # BLOSC_ZSTD
    return (0, 0, 0, 0, compression_level, shuffle, compressor_type)


if _HAS_BLOSC:
    H5_COMPRESSION_FLAGS = dict(
        compression=32001,
        compression_opts=_compression_opts(),
        chunks=True,
    )
else:
    H5_COMPRESSION_FLAGS = dict(
        compression="gzip",
        compression_opts=1,
        chunks=True,
    )


def create_ms_to_idx(t_us: np.ndarray) -> np.ndarray:
    if t_us.size == 0:
        return np.zeros((0,), dtype="uint64")
    t_ms = t_us // 1000
    x, counts = np.unique(t_ms, return_counts=True)
    ms_to_idx = np.zeros(shape=(int(t_ms[-1]) + 2,), dtype="uint64")
    ms_to_idx[x + 1] = counts
    ms_to_idx = ms_to_idx[:-1].cumsum()
    return ms_to_idx


def _load_npz_events(
    input_path: Path,
    compressed: bool,
    time_scale: float,
) -> dict[str, np.ndarray]:
    with np.load(input_path) as npz:
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

    x = np.rint(x).astype(np.int64, copy=False)
    y = np.rint(y).astype(np.int64, copy=False)

    if np.issubdtype(t.dtype, np.floating) and t.size > 0:
        # If timestamps look like seconds, convert to microseconds.
        if float(np.nanmax(t)) < 1.0e4:
            t = t * time_scale
    t = np.rint(t).astype(np.int64, copy=False)

    p = (np.asarray(p) > 0).astype(np.int8, copy=False)

    if t.size > 1:
        order = np.argsort(t, kind="stable")
        x = x[order]
        y = y[order]
        t = t[order]
        p = p[order]

    return {
        "x": x,
        "y": y,
        "p": p,
        "t": t,
    }


def _resolve_input_resolution(
    events: dict[str, np.ndarray],
    input_height: int | None,
    input_width: int | None,
) -> tuple[int, int]:
    resolved_height = input_height
    resolved_width = input_width

    if resolved_height is None:
        if events["y"].size > 0:
            resolved_height = int(events["y"].max()) + 1
        else:
            resolved_height = 480
    if resolved_width is None:
        if events["x"].size > 0:
            resolved_width = int(events["x"].max()) + 1
        else:
            resolved_width = 640

    return int(resolved_height), int(resolved_width)


class H5Writer:
    def __init__(self, outfile: Path, out_height: int, out_width: int):
        assert not outfile.exists(), f"Output already exists: {outfile}"

        self.h5f = h5py.File(str(outfile), "w")
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
        self.num_events = 0

        events = self.h5f.create_group("events")
        shape = (2**16,)
        maxshape = (None,)

        events.create_dataset("x", shape=shape, dtype="u2", maxshape=maxshape, **H5_COMPRESSION_FLAGS)
        events.create_dataset("y", shape=shape, dtype="u2", maxshape=maxshape, **H5_COMPRESSION_FLAGS)
        events.create_dataset("p", shape=shape, dtype="u1", maxshape=maxshape, **H5_COMPRESSION_FLAGS)
        events.create_dataset("t", shape=shape, dtype="u8", maxshape=maxshape, **H5_COMPRESSION_FLAGS)
        events.create_dataset("height", data=np.uint16(out_height), dtype="u2")
        events.create_dataset("width", data=np.uint16(out_width), dtype="u2")

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def close(self):
        self._finalizer()

    def add_data(self, events):
        size = len(events["t"])
        if size == 0:
            return

        start = self.num_events
        end = start + size
        self.num_events = end

        self.h5f["events/x"].resize(end, axis=0)
        self.h5f["events/y"].resize(end, axis=0)
        self.h5f["events/p"].resize(end, axis=0)
        self.h5f["events/t"].resize(end, axis=0)

        self.h5f["events/x"][start:end] = events["x"].astype("uint16", copy=False)
        self.h5f["events/y"][start:end] = events["y"].astype("uint16", copy=False)
        self.h5f["events/p"][start:end] = events["p"].astype("uint8", copy=False)
        self.h5f["events/t"][start:end] = events["t"].astype("uint64", copy=False)

    def create_ms_to_idx(self):
        t_us = self.h5f["events/t"][()]
        self.h5f.create_dataset("ms_to_idx", data=create_ms_to_idx(t_us), dtype="u8", **H5_COMPRESSION_FLAGS)


def _scale_factors(input_height: int, input_width: int, output_height: int, output_width: int):
    if input_width % output_width != 0 or input_height % output_height != 0:
        raise ValueError(
            "Input resolution must be divisible by output resolution "
            f"({input_width}x{input_height} -> {output_width}x{output_height})."
        )
    fx = input_width // output_width
    fy = input_height // output_height
    return fx, fy


def downsample_events(events, input_height, input_width, output_height, output_width, change_map=None):
    if change_map is None:
        change_map = np.zeros((output_height, output_width), dtype="float32")

    fx, fy = _scale_factors(input_height, input_width, output_height, output_width)
    mask = np.zeros(shape=(len(events["t"]),), dtype="bool")
    mask, change_map = _filter_events_resize(events["x"], events["y"], events["p"], mask, change_map, fx, fy)

    events = {k: v[mask] for k, v in events.items()}
    events["x"] = (events["x"] / fx).astype("uint16")
    events["y"] = (events["y"] / fy).astype("uint16")
    return events, change_map


@numba.jit(nopython=True, cache=True)
def _filter_events_resize(x, y, p, mask, change_map, fx, fy):
    for i in range(len(x)):
        x_l = x[i] // fx
        y_l = y[i] // fy
        change_map[y_l, x_l] += p[i] * 1.0 / (fx * fy)

        if np.abs(change_map[y_l, x_l]) >= 1:
            mask[i] = True
            change_map[y_l, x_l] -= p[i]

    return mask, change_map


def _process_chunk(
    events: dict[str, np.ndarray],
    start_idx: int,
    end_idx: int,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    change_map,
):
    chunk_events = {
        "x": events["x"][start_idx:end_idx],
        "y": events["y"][start_idx:end_idx],
        "p": events["p"][start_idx:end_idx],
        "t": events["t"][start_idx:end_idx],
    }
    chunk_events["p"] = 2 * chunk_events["p"].astype("int8") - 1
    downsampled_events, change_map = downsample_events(
        chunk_events,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        change_map=change_map,
    )
    downsampled_events["p"] = ((downsampled_events["p"] + 1) // 2).astype("uint8")
    return downsampled_events, change_map


def _tmp_output_path(output_path: Path, tmp_suffix: str) -> Path:
    return output_path.with_name(f"{output_path.name}{tmp_suffix}")


def _cleanup_tmp_file(tmp_path: Path, context: str, strict: bool = True) -> bool:
    if not tmp_path.exists():
        return True

    try:
        tmp_path.unlink()
        print(f"[RESUME] removed stale tmp: {tmp_path}")
        return True
    except Exception as exc:
        msg = f"[FAILED] could not remove tmp file ({context}): {tmp_path} ({exc})"
        if strict:
            raise RuntimeError(msg) from exc
        print(msg)
        return False


def process_single_file(
    input_path: Path,
    output_path: Path,
    input_height: int | None,
    input_width: int | None,
    output_height: int,
    output_width: int,
    num_events_per_chunk: int,
    write_ms_to_idx: bool,
    show_progress: bool,
    tmp_suffix: str,
    compressed: bool,
    time_scale: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = _tmp_output_path(output_path=output_path, tmp_suffix=tmp_suffix)
    _cleanup_tmp_file(tmp_path=tmp_output_path, context=f"start processing {input_path}", strict=True)

    events = _load_npz_events(
        input_path=input_path,
        compressed=compressed,
        time_scale=time_scale,
    )
    resolved_input_height, resolved_input_width = _resolve_input_resolution(
        events=events,
        input_height=input_height,
        input_width=input_width,
    )
    _scale_factors(resolved_input_height, resolved_input_width, output_height, output_width)

    num_events = int(len(events["t"]))
    num_events_per_chunk = max(1, int(num_events_per_chunk))
    num_full_chunks = num_events // num_events_per_chunk
    has_remainder = (num_events % num_events_per_chunk) > 0
    total_steps = num_full_chunks + (1 if has_remainder else 0)

    writer = None
    pbar = tqdm.tqdm(total=total_steps, leave=False, desc=input_path.name) if show_progress else None
    try:
        writer = H5Writer(tmp_output_path, out_height=output_height, out_width=output_width)
        change_map = None

        for i in range(num_full_chunks):
            start_idx = i * num_events_per_chunk
            end_idx = (i + 1) * num_events_per_chunk
            downsampled_events, change_map = _process_chunk(
                events=events,
                start_idx=start_idx,
                end_idx=end_idx,
                input_height=resolved_input_height,
                input_width=resolved_input_width,
                output_height=output_height,
                output_width=output_width,
                change_map=change_map,
            )
            writer.add_data(downsampled_events)
            if pbar is not None:
                pbar.update(1)

        if has_remainder:
            start_idx = num_full_chunks * num_events_per_chunk
            downsampled_events, change_map = _process_chunk(
                events=events,
                start_idx=start_idx,
                end_idx=num_events,
                input_height=resolved_input_height,
                input_width=resolved_input_width,
                output_height=output_height,
                output_width=output_width,
                change_map=change_map,
            )
            writer.add_data(downsampled_events)
            if pbar is not None:
                pbar.update(1)

        if write_ms_to_idx:
            writer.create_ms_to_idx()
        writer.close()
        writer = None
        os.replace(tmp_output_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        _cleanup_tmp_file(tmp_path=tmp_output_path, context=f"exception cleanup for {input_path}", strict=False)
        raise
    finally:
        if pbar is not None:
            pbar.close()


def _process_file_with_retry(
    input_path: Path,
    output_path: Path,
    input_height: int | None,
    input_width: int | None,
    output_height: int,
    output_width: int,
    num_events_per_chunk: int,
    write_ms_to_idx: bool,
    tmp_suffix: str,
    compressed: bool,
    time_scale: float,
) -> tuple[bool, str | None]:
    stale_tmp_path = _tmp_output_path(output_path=output_path, tmp_suffix=tmp_suffix)
    if not _cleanup_tmp_file(tmp_path=stale_tmp_path, context=f"resume prep for {input_path}", strict=False):
        return False, f"could not remove stale tmp file: {stale_tmp_path}"

    for attempt in (1, 2):
        try:
            process_single_file(
                input_path=input_path,
                output_path=output_path,
                input_height=input_height,
                input_width=input_width,
                output_height=output_height,
                output_width=output_width,
                num_events_per_chunk=num_events_per_chunk,
                write_ms_to_idx=write_ms_to_idx,
                show_progress=False,
                tmp_suffix=tmp_suffix,
                compressed=compressed,
                time_scale=time_scale,
            )
            return True, None
        except Exception as exc:
            if attempt == 1:
                cleanup_ok = _cleanup_tmp_file(
                    tmp_path=stale_tmp_path,
                    context=f"retry prep for {input_path}",
                    strict=False,
                )
                if not cleanup_ok:
                    return False, f"retry cleanup failed for {stale_tmp_path}: {exc}"
                continue
            return False, str(exc)

    return False, "unknown failure"


def _worker_process_file(job: dict) -> tuple[str, bool, str | None]:
    input_path = Path(job["input_path"])
    output_path = Path(job["output_path"])
    ok, err = _process_file_with_retry(
        input_path=input_path,
        output_path=output_path,
        input_height=job["input_height"],
        input_width=job["input_width"],
        output_height=job["output_height"],
        output_width=job["output_width"],
        num_events_per_chunk=job["num_events_per_chunk"],
        write_ms_to_idx=job["write_ms_to_idx"],
        tmp_suffix=job["tmp_suffix"],
        compressed=job["compressed"],
        time_scale=job["time_scale"],
    )
    return str(input_path), ok, err


def _find_npz_files(dataset_root: Path, splits: list[str], recursive: bool) -> list[Path]:
    input_files: list[Path] = []
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] missing split directory: {split_dir}")
            continue

        files = split_dir.rglob("*.npz") if recursive else split_dir.glob("*.npz")
        input_files.extend(sorted([p for p in files if p.is_file()]))
    return input_files


def _normalized_output_suffix(output_suffix: str) -> str:
    suffix = output_suffix
    if not suffix.endswith(".h5"):
        suffix = f"{suffix}.h5"
    return suffix


def _build_output_path_from_root(
    input_path: Path,
    dataset_root: Path,
    output_root: Path | None,
    normalized_suffix: str,
) -> Path:
    output_name = f"{input_path.stem}{normalized_suffix}"
    if output_root is None:
        return input_path.with_name(output_name)

    relative_input = input_path.relative_to(dataset_root)
    output_dir = output_root / relative_input.parent
    return output_dir / output_name


def _read_list_file_inputs(list_file: Path, root_dir: Path | None) -> list[tuple[Path, Path | None]]:
    inputs: list[tuple[Path, Path | None]] = []
    for line in list_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = line.split()[0]
        raw_path = Path(raw)

        if raw_path.is_absolute():
            resolved = raw_path
            if root_dir is not None:
                try:
                    rel_parent = resolved.relative_to(root_dir).parent
                except ValueError:
                    rel_parent = None
            else:
                rel_parent = None
        else:
            base = root_dir if root_dir is not None else list_file.parent
            resolved = base / raw_path
            rel_parent = raw_path.parent

        inputs.append((resolved, rel_parent))

    return inputs


def _collect_inputs_from_list_files(
    list_files: list[Path],
    root_dir: Path | None,
) -> list[tuple[Path, Path | None]]:
    inputs: list[tuple[Path, Path | None]] = []
    for list_file in list_files:
        if not list_file.exists():
            raise FileNotFoundError(f"List file not found: {list_file}")
        entries = _read_list_file_inputs(list_file=list_file, root_dir=root_dir)
        inputs.extend(entries)

    unique_inputs: dict[tuple[str, str | None], tuple[Path, Path | None]] = {}
    for input_path, rel_parent in inputs:
        key = (str(input_path), str(rel_parent) if rel_parent is not None else None)
        unique_inputs[key] = (input_path, rel_parent)

    return list(unique_inputs.values())


def _build_output_path_from_list(
    input_path: Path,
    rel_parent: Path | None,
    output_root: Path | None,
    normalized_suffix: str,
) -> Path:
    output_name = f"{input_path.stem}{normalized_suffix}"
    if output_root is None:
        return input_path.with_name(output_name)

    if rel_parent is None:
        return output_root / output_name

    return output_root / rel_parent / output_name


def _run_jobs(jobs: list[dict], num_processes: int) -> tuple[int, int]:
    if int(num_processes) < 1:
        raise ValueError("num_processes must be >= 1")

    num_done = 0
    num_failed = 0

    if len(jobs) == 0:
        return num_done, num_failed

    if int(num_processes) == 1:
        iterator = (_worker_process_file(job) for job in jobs)
        for input_name, success, err in tqdm.tqdm(iterator, total=len(jobs), desc="n_imagenet workers"):
            if success:
                num_done += 1
            else:
                num_failed += 1
                print(f"[FAILED] {input_name}: {err}")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(num_processes)) as pool:
            for input_name, success, err in tqdm.tqdm(
                pool.imap_unordered(_worker_process_file, jobs),
                total=len(jobs),
                desc="n_imagenet workers",
            ):
                if success:
                    num_done += 1
                else:
                    num_failed += 1
                    print(f"[FAILED] {input_name}: {err}")

    return num_done, num_failed


def process_dataset_root(
    dataset_root: Path,
    splits: list[str],
    output_suffix: str,
    overwrite: bool,
    output_root: Path | None,
    input_height: int | None,
    input_width: int | None,
    output_height: int,
    output_width: int,
    num_events_per_chunk: int,
    write_ms_to_idx: bool,
    recursive: bool,
    tmp_suffix: str,
    num_processes: int,
    compressed: bool,
    time_scale: float,
) -> None:
    normalized_suffix = _normalized_output_suffix(output_suffix)
    input_files = _find_npz_files(dataset_root=dataset_root, splits=splits, recursive=recursive)
    if len(input_files) == 0:
        raise FileNotFoundError(f"No .npz files found under {dataset_root} for splits={splits}")
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    num_skipped = 0

    for input_path in tqdm.tqdm(input_files, desc="n_imagenet sequences"):
        output_path = _build_output_path_from_root(
            input_path=input_path,
            dataset_root=dataset_root,
            output_root=output_root,
            normalized_suffix=normalized_suffix,
        )
        if output_path.exists():
            if overwrite:
                output_path.unlink()
            else:
                num_skipped += 1
                continue

        jobs.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width,
                "num_events_per_chunk": num_events_per_chunk,
                "write_ms_to_idx": write_ms_to_idx,
                "tmp_suffix": tmp_suffix,
                "compressed": compressed,
                "time_scale": time_scale,
            }
        )

    num_done, num_failed = _run_jobs(jobs=jobs, num_processes=num_processes)

    print(f"[SUMMARY] done={num_done}, skipped={num_skipped}, failed={num_failed}")
    if num_failed > 0:
        raise RuntimeError(f"{num_failed} files failed while processing {dataset_root}")


def process_list_files(
    list_files: list[Path],
    root_dir: Path | None,
    output_suffix: str,
    overwrite: bool,
    output_root: Path | None,
    input_height: int | None,
    input_width: int | None,
    output_height: int,
    output_width: int,
    num_events_per_chunk: int,
    write_ms_to_idx: bool,
    tmp_suffix: str,
    num_processes: int,
    compressed: bool,
    time_scale: float,
) -> None:
    normalized_suffix = _normalized_output_suffix(output_suffix)
    inputs_with_rel = _collect_inputs_from_list_files(list_files=list_files, root_dir=root_dir)
    if len(inputs_with_rel) == 0:
        raise FileNotFoundError("No input paths found from list_files")
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    num_skipped = 0
    num_missing = 0

    for input_path, rel_parent in tqdm.tqdm(inputs_with_rel, desc="n_imagenet list entries"):
        if input_path.suffix.lower() != ".npz":
            num_missing += 1
            print(f"[WARN] skip non-npz input file: {input_path}")
            continue
        if not input_path.exists():
            num_missing += 1
            print(f"[WARN] missing input file: {input_path}")
            continue

        output_path = _build_output_path_from_list(
            input_path=input_path,
            rel_parent=rel_parent,
            output_root=output_root,
            normalized_suffix=normalized_suffix,
        )
        if output_path.exists():
            if overwrite:
                output_path.unlink()
            else:
                num_skipped += 1
                continue

        jobs.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width,
                "num_events_per_chunk": num_events_per_chunk,
                "write_ms_to_idx": write_ms_to_idx,
                "tmp_suffix": tmp_suffix,
                "compressed": compressed,
                "time_scale": time_scale,
            }
        )

    num_done, num_failed = _run_jobs(jobs=jobs, num_processes=num_processes)

    print(
        f"[SUMMARY] done={num_done}, skipped={num_skipped}, missing={num_missing}, failed={num_failed}"
    )
    if num_failed > 0:
        raise RuntimeError(f"{num_failed} files failed while processing list files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Downsample N-ImageNet npz events and save as H5")
    parser.add_argument("--input_path", type=Path, help="Input events .npz")
    parser.add_argument("--output_path", type=Path, help="Output events .h5")

    parser.add_argument(
        "--dataset_root",
        type=Path,
        help="Dataset root with train/test/val split dirs (root mode)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "val"],
        help="Splits for --dataset_root mode",
    )

    parser.add_argument(
        "--list_files",
        type=Path,
        nargs="+",
        default=None,
        help="List file(s) containing npz paths (list mode)",
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=None,
        help="Base directory for relative paths in --list_files",
    )

    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_2x.h5",
        help="Output suffix for root/list modes (e.g. _2x.h5)",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional output root dir for root/list modes (preserves relative paths)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--recursive", action="store_true", help="Recursively search .npz under split dirs")
    parser.add_argument(
        "--tmp_suffix",
        type=str,
        default=".tmp",
        help="Temporary suffix used while writing (renamed on success)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Parallel workers for root/list modes (spawn).",
    )

    parser.add_argument(
        "--compressed",
        dest="compressed",
        action="store_true",
        default=True,
        help="Read compressed npz format if available (event_data).",
    )
    parser.add_argument(
        "--uncompressed",
        dest="compressed",
        action="store_false",
        help="Read legacy npz keys x_pos/y_pos/timestamp/polarity.",
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        default=1_000_000.0,
        help="Scale factor when timestamps look like seconds.",
    )

    parser.add_argument("--input_height", type=int, default=480, help="Input event height")
    parser.add_argument("--input_width", type=int, default=640, help="Input event width")
    parser.add_argument("--output_height", type=int, default=240, help="Output event height")
    parser.add_argument("--output_width", type=int, default=320, help="Output event width")

    parser.add_argument(
        "--num_events_per_chunk",
        type=int,
        default=100000,
        help="Number of events processed per chunk.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Deprecated alias for --num_events_per_chunk.",
    )
    parser.add_argument(
        "--write_ms_to_idx",
        action="store_true",
        help="Also write ms_to_idx index to output H5",
    )

    args = parser.parse_args()

    num_events_per_chunk = int(args.num_events_per_chunk)
    if args.chunk_size is not None:
        num_events_per_chunk = int(args.chunk_size)

    is_single_mode = args.input_path is not None or args.output_path is not None
    is_root_mode = args.dataset_root is not None
    is_list_mode = args.list_files is not None and len(args.list_files) > 0

    enabled_modes = int(is_single_mode) + int(is_root_mode) + int(is_list_mode)
    if enabled_modes != 1:
        parser.error(
            "Choose exactly one mode: single (--input_path/--output_path), "
            "root (--dataset_root), or list (--list_files)."
        )

    if is_single_mode:
        if args.input_path is None or args.output_path is None:
            parser.error("Single-file mode requires both --input_path and --output_path.")
        if args.input_path.suffix.lower() != ".npz":
            parser.error("Single-file mode requires --input_path to be a .npz file.")
        process_single_file(
            input_path=args.input_path,
            output_path=args.output_path,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            num_events_per_chunk=num_events_per_chunk,
            write_ms_to_idx=args.write_ms_to_idx,
            show_progress=True,
            tmp_suffix=args.tmp_suffix,
            compressed=bool(args.compressed),
            time_scale=float(args.time_scale),
        )
    elif is_root_mode:
        process_dataset_root(
            dataset_root=args.dataset_root,
            splits=args.splits,
            output_suffix=args.output_suffix,
            overwrite=args.overwrite,
            output_root=args.output_root,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            num_events_per_chunk=num_events_per_chunk,
            write_ms_to_idx=args.write_ms_to_idx,
            recursive=args.recursive,
            tmp_suffix=args.tmp_suffix,
            num_processes=args.num_processes,
            compressed=bool(args.compressed),
            time_scale=float(args.time_scale),
        )
    else:
        process_list_files(
            list_files=args.list_files,
            root_dir=args.root_dir,
            output_suffix=args.output_suffix,
            overwrite=args.overwrite,
            output_root=args.output_root,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            num_events_per_chunk=num_events_per_chunk,
            write_ms_to_idx=args.write_ms_to_idx,
            tmp_suffix=args.tmp_suffix,
            num_processes=args.num_processes,
            compressed=bool(args.compressed),
            time_scale=float(args.time_scale),
        )
