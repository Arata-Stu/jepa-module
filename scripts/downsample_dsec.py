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
    # From https://github.com/Blosc/c-blosc/blob/7435f28dd08606bd51ab42b49b0e654547becac4/blosc/blosc.h#L66-L71
    # define BLOSC_BLOSCLZ   0
    # define BLOSC_LZ4       1
    # define BLOSC_LZ4HC     2
    # define BLOSC_SNAPPY    3
    # define BLOSC_ZLIB      4
    # define BLOSC_ZSTD      5
    compressor_type = 5
    compression_opts = (0, 0, 0, 0, compression_level, shuffle, compressor_type)
    return compression_opts


if _HAS_BLOSC:
    H5_COMPRESSION_FLAGS = dict(
        compression=32001,
        compression_opts=_compression_opts(),  # Blosc
        chunks=True,
    )
else:
    H5_COMPRESSION_FLAGS = dict(
        compression="gzip",
        compression_opts=1,
        chunks=True,
    )


def _extract_from_h5_by_index(filehandle, ev_start_idx: int, ev_end_idx: int):
    events = filehandle["events"]
    x = events["x"]
    y = events["y"]
    p = events["p"]
    t = events["t"]

    x_new = x[ev_start_idx:ev_end_idx]
    y_new = y[ev_start_idx:ev_end_idx]
    p_new = p[ev_start_idx:ev_end_idx]
    t_new = t[ev_start_idx:ev_end_idx].astype("int64") + filehandle["t_offset"][()]

    return {
        "p": p_new,
        "t": t_new,
        "x": x_new,
        "y": y_new,
    }


def get_num_events(h5file: Path):
    with h5py.File(str(h5file), "r") as h5f:
        return len(h5f["events/t"])


def extract_from_h5_by_index(h5file: Path, ev_start_idx: int, ev_end_idx: int):
    with h5py.File(str(h5file), "r") as h5f:
        return _extract_from_h5_by_index(h5f, ev_start_idx, ev_end_idx)


def create_ms_to_idx(t_us):
    if len(t_us) == 0:
        return np.zeros(shape=(0,), dtype="uint64")

    t_ms = t_us // 1000
    x, counts = np.unique(t_ms, return_counts=True)
    ms_to_idx = np.zeros(shape=(t_ms[-1] + 2,), dtype="uint64")
    ms_to_idx[x + 1] = counts
    ms_to_idx = ms_to_idx[:-1].cumsum()
    return ms_to_idx


class H5Writer:
    def __init__(self, outfile: Path):
        assert not outfile.exists()

        self.h5f = h5py.File(str(outfile), "a")
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        self.t_offset = None
        self.num_events = 0

        # create hdf5 datasets
        shape = (2**16,)
        maxshape = (None,)

        self.h5f.create_dataset("events/x", shape=shape, dtype="u2", maxshape=maxshape, **H5_COMPRESSION_FLAGS)
        self.h5f.create_dataset("events/y", shape=shape, dtype="u2", maxshape=maxshape, **H5_COMPRESSION_FLAGS)
        self.h5f.create_dataset("events/p", shape=shape, dtype="u1", maxshape=maxshape, **H5_COMPRESSION_FLAGS)
        self.h5f.create_dataset("events/t", shape=shape, dtype="u4", maxshape=maxshape, **H5_COMPRESSION_FLAGS)

    def create_ms_to_idx(self):
        if self.t_offset is None:
            self.t_offset = 0
            self.h5f.create_dataset("t_offset", data=self.t_offset, dtype="i8")

        t_us = self.h5f["events/t"][()]
        self.h5f.create_dataset("ms_to_idx", data=create_ms_to_idx(t_us), dtype="u8", **H5_COMPRESSION_FLAGS)

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def close(self):
        self._finalizer()

    def add_data(self, events):
        size = len(events["t"])
        if size == 0:
            return

        if self.t_offset is None:
            self.t_offset = events["t"][0]
            self.h5f.create_dataset("t_offset", data=self.t_offset, dtype="i8")

        events["t"] -= self.t_offset
        self.num_events += size

        self.h5f["events/x"].resize(self.num_events, axis=0)
        self.h5f["events/y"].resize(self.num_events, axis=0)
        self.h5f["events/p"].resize(self.num_events, axis=0)
        self.h5f["events/t"].resize(self.num_events, axis=0)

        self.h5f["events/x"][self.num_events - size:self.num_events] = events["x"]
        self.h5f["events/y"][self.num_events - size:self.num_events] = events["y"]
        self.h5f["events/p"][self.num_events - size:self.num_events] = events["p"]
        self.h5f["events/t"][self.num_events - size:self.num_events] = events["t"]


def downsample_events(events, input_height, input_width, output_height, output_width, change_map=None):
    # this subsamples events if they were generated with cv2.INTER_AREA
    if change_map is None:
        change_map = np.zeros((output_height, output_width), dtype="float32")

    fx = int(input_width / output_width)
    fy = int(input_height / output_height)

    mask = np.zeros(shape=(len(events["t"]),), dtype="bool")
    mask, change_map = _filter_events_resize(events["x"], events["y"], events["p"], mask, change_map, fx, fy)

    events = {k: v[mask] for k, v in events.items()}
    events["x"] = (events["x"] / fx).astype("uint16")
    events["y"] = (events["y"] / fy).astype("uint16")

    return events, change_map


@numba.jit(nopython=True, cache=True)
def _filter_events_resize(x, y, p, mask, change_map, fx, fy):
    # iterates through x,y,p of events, and increments cells of size fx x fy by 1/(fx*fy)
    # if one of these cells reaches +-1, then reset the cell, and pass through that event.
    # for memory reasons, this only returns the True/False for every event, indicating if
    # the event was skipped or passed through.
    for i in range(len(x)):
        x_l = x[i] // fx
        y_l = y[i] // fy
        change_map[y_l, x_l] += p[i] * 1.0 / (fx * fy)

        if np.abs(change_map[y_l, x_l]) >= 1:
            mask[i] = True
            change_map[y_l, x_l] -= p[i]

    return mask, change_map


def _process_chunk(input_path, start_idx, end_idx, input_height, input_width, output_height, output_width, change_map):
    events = extract_from_h5_by_index(input_path, start_idx, end_idx)
    events["p"] = 2 * events["p"].astype("int8") - 1
    downsampled_events, change_map = downsample_events(
        events,
        change_map=change_map,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
    )
    downsampled_events["p"] = ((downsampled_events["p"] + 1) // 2).astype("uint8")
    return downsampled_events, change_map


def _downsample_one_chunk(input_path, start_idx, end_idx, input_height, input_width, output_height, output_width, change_map):
    # Backward-compatible alias.
    return _process_chunk(input_path, start_idx, end_idx, input_height, input_width, output_height, output_width, change_map)


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


def _process_file_with_retry(
    input_path: Path,
    output_path: Path,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    num_events_per_chunk: int,
    tmp_suffix: str,
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
                show_pbar=False,
                tmp_suffix=tmp_suffix,
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
        tmp_suffix=job["tmp_suffix"],
    )
    return str(input_path), ok, err


def process_single_file(
    input_path,
    output_path,
    input_height,
    input_width,
    output_height,
    output_width,
    num_events_per_chunk=100000,
    show_pbar=True,
    tmp_suffix=".tmp",
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = _tmp_output_path(output_path=output_path, tmp_suffix=tmp_suffix)
    _cleanup_tmp_file(tmp_path=tmp_output_path, context=f"start processing {input_path}", strict=True)

    num_events = get_num_events(input_path)
    num_iterations = num_events // num_events_per_chunk
    has_remainder = (num_events % num_events_per_chunk) > 0
    total_steps = num_iterations + (1 if has_remainder else 0)

    writer = None
    pbar = tqdm.tqdm(total=total_steps, desc=input_path.name, leave=False) if show_pbar else None
    try:
        writer = H5Writer(tmp_output_path)
        change_map = None

        for i in range(num_iterations):
            start_idx = i * num_events_per_chunk
            end_idx = (i + 1) * num_events_per_chunk
            downsampled_events, change_map = _process_chunk(
                input_path=input_path,
                start_idx=start_idx,
                end_idx=end_idx,
                input_height=input_height,
                input_width=input_width,
                output_height=output_height,
                output_width=output_width,
                change_map=change_map,
            )
            writer.add_data(downsampled_events)
            if pbar is not None:
                pbar.update(1)

        if has_remainder:
            start_idx = num_iterations * num_events_per_chunk
            downsampled_events, change_map = _process_chunk(
                input_path=input_path,
                start_idx=start_idx,
                end_idx=num_events,
                input_height=input_height,
                input_width=input_width,
                output_height=output_height,
                output_width=output_width,
                change_map=change_map,
            )
            writer.add_data(downsampled_events)
            if pbar is not None:
                pbar.update(1)

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


def find_dsec_event_files(dsec_root: Path, splits):
    event_files = []
    for split in splits:
        split_dir = dsec_root / split
        if not split_dir.exists():
            print(f"[WARN] missing split directory: {split_dir}")
            continue

        for sequence_dir in sorted(split_dir.iterdir()):
            if not sequence_dir.is_dir():
                continue

            input_file = sequence_dir / "events/left/events.h5"
            if input_file.exists():
                event_files.append(input_file)
            else:
                print(f"[WARN] missing events file: {input_file}")

    return event_files


def _build_output_path(input_path: Path, dsec_root: Path, output_root: Path | None, output_name: str) -> Path:
    if output_root is None:
        return input_path.with_name(output_name)

    rel_input = input_path.relative_to(dsec_root)
    return output_root / rel_input.parent / output_name


def process_dataset_root(
    dataset_root,
    splits,
    output_name,
    overwrite,
    output_root,
    input_height,
    input_width,
    output_height,
    output_width,
    num_events_per_chunk,
    tmp_suffix,
    num_processes,
):
    if int(num_processes) < 1:
        raise ValueError("num_processes must be >= 1")

    input_files = find_dsec_event_files(dsec_root=dataset_root, splits=splits)
    if len(input_files) == 0:
        raise FileNotFoundError(f"No events.h5 found under root={dataset_root}, splits={splits}")
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    num_done = 0
    num_skipped = 0
    num_failed = 0

    for input_path in tqdm.tqdm(input_files, desc="DSEC sequences"):
        output_path = _build_output_path(
            input_path=input_path,
            dsec_root=dataset_root,
            output_root=output_root,
            output_name=output_name,
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
                "tmp_suffix": tmp_suffix,
            }
        )

    if len(jobs) > 0:
        if int(num_processes) == 1:
            iterator = (_worker_process_file(job) for job in jobs)
            for input_name, success, err in tqdm.tqdm(iterator, total=len(jobs), desc="DSEC workers"):
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
                    desc="DSEC workers",
                ):
                    if success:
                        num_done += 1
                    else:
                        num_failed += 1
                        print(f"[FAILED] {input_name}: {err}")

    print(f"[SUMMARY] done={num_done}, skipped={num_skipped}, failed={num_failed}")

    if num_failed > 0:
        raise RuntimeError(f"{num_failed} sequences failed while processing {dataset_root}")


def process_dsec_root(
    dsec_root,
    splits,
    output_name,
    overwrite,
    output_root,
    input_height,
    input_width,
    output_height,
    output_width,
    num_events_per_chunk,
    tmp_suffix,
    num_processes,
):
    # Backward-compatible wrapper.
    process_dataset_root(
        dataset_root=dsec_root,
        splits=splits,
        output_name=output_name,
        overwrite=overwrite,
        output_root=output_root,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        num_events_per_chunk=num_events_per_chunk,
        tmp_suffix=tmp_suffix,
        num_processes=num_processes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Downsample events""")
    parser.add_argument("--input_path", type=Path, help="Path to input events.h5.")
    parser.add_argument("--output_path", type=Path, help="Path where output events.h5 will be written.")
    parser.add_argument("--dsec_root", type=Path, help="Path to DSEC root (contains train/test splits).")
    parser.add_argument("--dataset_root", type=Path, help="Alias of --dsec_root for naming consistency.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Split names for --dsec_root mode.")
    parser.add_argument("--output_name", type=str, default="events_2x.h5",
                        help="Output filename per sequence in --dsec_root mode.")
    parser.add_argument("--output_root", type=Path, default=None,
                        help="Optional output root for --dsec_root mode (preserves relative split paths).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files in --dsec_root mode.")
    parser.add_argument("--tmp_suffix", type=str, default=".tmp",
                        help="Temporary suffix used while writing (renamed on success).")
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Parallel workers for --dsec_root mode (spawn).")
    parser.add_argument("--input_height", type=int, default=480, help="Height of the input events resolution.")
    parser.add_argument("--input_width", type=int, default=640, help="Width of the input events resolution")
    parser.add_argument("--output_height", type=int, default=240, help="Height of the output events resolution.")
    parser.add_argument("--output_width", type=int, default=320, help="Width of the output events resolution.")
    parser.add_argument("--num_events_per_chunk", type=int, default=100000, help="Number of events loaded per chunk.")
    args = parser.parse_args()

    is_single_mode = args.input_path is not None or args.output_path is not None
    root_dir = args.dataset_root if args.dataset_root is not None else args.dsec_root
    is_root_mode = root_dir is not None

    if is_single_mode and is_root_mode:
        parser.error(
            "Use either single-file mode (--input_path/--output_path) "
            "or root mode (--dataset_root/--dsec_root), not both."
        )

    if is_root_mode:
        process_dataset_root(
            dataset_root=root_dir,
            splits=args.splits,
            output_name=args.output_name,
            overwrite=args.overwrite,
            output_root=args.output_root,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            num_events_per_chunk=args.num_events_per_chunk,
            tmp_suffix=args.tmp_suffix,
            num_processes=args.num_processes,
        )
    else:
        if args.input_path is None or args.output_path is None:
            parser.error("Single-file mode requires both --input_path and --output_path.")

        process_single_file(
            input_path=args.input_path,
            output_path=args.output_path,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            num_events_per_chunk=args.num_events_per_chunk,
            show_pbar=True,
            tmp_suffix=args.tmp_suffix,
        )
