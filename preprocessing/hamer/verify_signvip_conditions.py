#!/usr/bin/env python3
"""Verify generated SignVIP HaMeR sidecar HDF5 shards.

The verifier never modifies either dataset. It checks shard/sample coverage,
frame counts, image shapes and dtypes, copied metadata, stale temporary keys,
and optionally reads representative or all rendered frames for pixel checks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


@dataclass
class VerificationStats:
    source_shards: int = 0
    output_shards: int = 0
    source_samples: int = 0
    valid_samples: int = 0
    missing_samples: int = 0
    extra_samples: int = 0
    inspected_frames: int = 0
    black_frames: int = 0
    errors: int = 0
    warnings: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/rwth"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/hamer"),
    )
    parser.add_argument("--splits", nargs="+", default=("train", "dev", "test"))
    parser.add_argument(
        "--pixel-scan",
        choices=("none", "sample", "all"),
        default="sample",
        help="Read no frames, first/middle/last frame, or every rendered frame.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="Verify at most this many source samples globally (for a smoke check).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report missing output shards/samples as warnings instead of errors.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optionally write the summary and messages to a JSON file.",
    )
    return parser.parse_args()


def numeric_keys(group: h5py.Group) -> list[str]:
    return sorted((key for key in group.keys() if key.isdigit()), key=int)


def discover_shards(root: Path, splits: list[str]) -> list[tuple[Path, Path]]:
    if root.is_file() and root.suffix.lower() in (".h5", ".hdf5"):
        return [(root, Path(root.name))]
    shards: list[tuple[Path, Path]] = []
    for split in splits:
        split_root = root / split
        for path in sorted((*split_root.glob("*.hdf5"), *split_root.glob("*.h5"))):
            shards.append((path, path.relative_to(root)))
    return shards


def frame_count(node: h5py.Group | h5py.Dataset) -> int:
    return int(node.shape[0]) if isinstance(node, h5py.Dataset) else len(numeric_keys(node))


def source_frame_shape(node: h5py.Group | h5py.Dataset) -> tuple[int, int, int] | None:
    if isinstance(node, h5py.Dataset):
        if len(node.shape) != 4 or node.shape[0] == 0:
            return None
        shape = tuple(int(value) for value in node.shape[1:])
    else:
        keys = numeric_keys(node)
        if not keys:
            return None
        shape = tuple(int(value) for value in node[keys[0]].shape)
    if len(shape) != 3:
        return None
    if shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
        shape = (shape[1], shape[2], shape[0])
    return shape


def scalar_value(dataset: h5py.Dataset):
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return value


def selected_frame_indices(length: int, mode: str) -> list[int]:
    if length <= 0 or mode == "none":
        return []
    if mode == "all":
        return list(range(length))
    return sorted({0, length // 2, length - 1})


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_root}")
    if args.limit_samples is not None and args.limit_samples <= 0:
        raise ValueError("--limit-samples must be positive")

    shards = discover_shards(input_root, list(args.splits))
    if not shards:
        raise FileNotFoundError(f"No HDF5 shards found under {input_root}")

    stats = VerificationStats(source_shards=len(shards))
    messages: list[dict[str, str]] = []

    def report(level: str, location: str, message: str) -> None:
        messages.append({"level": level, "location": location, "message": message})
        if level == "ERROR":
            stats.errors += 1
        else:
            stats.warnings += 1
        tqdm.write(f"{level} {location}: {message}")

    checked = 0
    for source_path, relative_path in tqdm(shards, desc="Verify HDF5 shards"):
        if args.limit_samples is not None and checked >= args.limit_samples:
            break
        output_path = output_root / relative_path
        if not output_path.exists():
            try:
                with h5py.File(source_path, "r") as source:
                    if "video" not in source:
                        report("ERROR", str(source_path), "missing source group 'video'")
                        continue
                    missing_keys = numeric_keys(source["video"])
                    if args.limit_samples is not None:
                        missing_keys = missing_keys[: args.limit_samples - checked]
                    stats.source_samples += len(missing_keys)
                    stats.missing_samples += len(missing_keys)
                    checked += len(missing_keys)
            except Exception as error:
                report("ERROR", str(source_path), f"cannot open source HDF5: {error}")
                continue
            level = "WARNING" if args.allow_incomplete else "ERROR"
            report(level, str(output_path), f"output shard is missing ({len(missing_keys)} samples)")
            continue
        stats.output_shards += 1

        try:
            source_file = h5py.File(source_path, "r")
        except Exception as error:
            report("ERROR", str(source_path), f"cannot open source HDF5: {error}")
            continue
        try:
            output_file = h5py.File(output_path, "r")
        except Exception as error:
            source_file.close()
            report("ERROR", str(output_path), f"cannot open output HDF5: {error}")
            continue

        with source_file as source, output_file as output:
            if "video" not in source:
                report("ERROR", str(source_path), "missing source group 'video'")
                continue
            if "hamer_video" not in output:
                report("ERROR", str(output_path), "missing output group 'hamer_video'")
                continue

            hamer_group = output["hamer_video"]
            temporary_keys = sorted(key for key in hamer_group if key.startswith("__tmp__"))
            for key in temporary_keys:
                report("WARNING", f"{output_path}:hamer_video/{key}", "unfinished temporary sample")

            source_keys = numeric_keys(source["video"])
            remaining = None if args.limit_samples is None else args.limit_samples - checked
            if remaining is not None:
                source_keys = source_keys[:remaining]
            output_keys = set(numeric_keys(hamer_group))
            stats.source_samples += len(source_keys)

            for extra_key in sorted(output_keys - set(numeric_keys(source["video"])), key=int):
                stats.extra_samples += 1
                report("WARNING", f"{output_path}:hamer_video/{extra_key}", "sample is absent from source")

            for sample_key in tqdm(source_keys, desc=relative_path.as_posix(), leave=False):
                checked += 1
                location = f"{output_path}:hamer_video/{sample_key}"
                if sample_key not in output_keys:
                    stats.missing_samples += 1
                    level = "WARNING" if args.allow_incomplete else "ERROR"
                    report(level, location, "sample is missing")
                    continue

                source_node = source["video"][sample_key]
                rendered = hamer_group[sample_key]
                expected_length = frame_count(source_node)
                if "len" in source and sample_key in source["len"]:
                    expected_length = min(expected_length, int(source["len"][sample_key][()]))

                sample_valid = True
                if not isinstance(rendered, h5py.Dataset) or rendered.ndim != 4:
                    report("ERROR", location, f"expected a 4D dataset, got {type(rendered).__name__} {rendered.shape}")
                    continue
                if int(rendered.shape[0]) != expected_length:
                    report("ERROR", location, f"has {rendered.shape[0]} frames; expected {expected_length}")
                    sample_valid = False
                expected_shape = source_frame_shape(source_node)
                if expected_shape is None:
                    report("ERROR", f"{source_path}:video/{sample_key}", "cannot determine source HWC shape")
                    sample_valid = False
                elif tuple(rendered.shape[1:]) != expected_shape:
                    report("ERROR", location, f"frame shape {rendered.shape[1:]}; expected {expected_shape}")
                    sample_valid = False
                if rendered.dtype != np.uint8:
                    report("ERROR", location, f"dtype is {rendered.dtype}; expected uint8")
                    sample_valid = False

                if "len" not in output or sample_key not in output["len"]:
                    report("ERROR", f"{output_path}:len/{sample_key}", "length metadata is missing")
                    sample_valid = False
                elif int(output["len"][sample_key][()]) != expected_length:
                    report("ERROR", f"{output_path}:len/{sample_key}", "length metadata does not match source")
                    sample_valid = False

                if "text_key" in source and sample_key in source["text_key"]:
                    if "text_key" not in output or sample_key not in output["text_key"]:
                        report("ERROR", f"{output_path}:text_key/{sample_key}", "text_key metadata is missing")
                        sample_valid = False
                    elif scalar_value(source["text_key"][sample_key]) != scalar_value(output["text_key"][sample_key]):
                        report("ERROR", f"{output_path}:text_key/{sample_key}", "text_key differs from source")
                        sample_valid = False

                readable_length = min(int(rendered.shape[0]), expected_length)
                for frame_index in selected_frame_indices(readable_length, args.pixel_scan):
                    try:
                        frame = np.asarray(rendered[frame_index])
                    except Exception as error:
                        report("ERROR", f"{location}[{frame_index}]", f"cannot read frame: {error}")
                        sample_valid = False
                        continue
                    stats.inspected_frames += 1
                    if not np.any(frame):
                        stats.black_frames += 1

                if sample_valid:
                    stats.valid_samples += 1

    if output_root.is_dir() and args.limit_samples is None:
        expected_relatives = {relative for _, relative in shards}
        for split in args.splits:
            for extra_path in sorted((*((output_root / split).glob("*.hdf5")), *((output_root / split).glob("*.h5")))):
                relative = extra_path.relative_to(output_root)
                if relative not in expected_relatives:
                    report("WARNING", str(extra_path), "output shard is absent from source")

    black_percent = (
        100.0 * stats.black_frames / stats.inspected_frames
        if stats.inspected_frames
        else 0.0
    )
    summary = asdict(stats)
    summary["black_frame_percent"] = round(black_percent, 4)
    summary["status"] = (
        "FAIL" if stats.errors else "PASS_WITH_WARNINGS" if stats.warnings else "PASS"
    )

    print("\nVerification summary")
    print(json.dumps(summary, indent=2))
    if args.pixel_scan != "none":
        print("Note: black frames are valid when no hand was detected; inspect the percentage above.")

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps({"summary": summary, "messages": messages}, indent=2),
            encoding="utf-8",
        )
        print(f"JSON report: {args.report_json}")

    if stats.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
