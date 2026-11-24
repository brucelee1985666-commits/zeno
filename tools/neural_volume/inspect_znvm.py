#!/usr/bin/env python3
"""Inspect and validate a ZNVM v1 portable neural-volume model container."""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

MAGIC = b"ZNVM0001"
VERSION = 1
FLAG_FLOATGRID_AFFINE = 3
FLAG_PARTIAL_DOMAINS = 4


class ModelError(ValueError):
    pass


def read_exact(stream, size, label):
    data = stream.read(size)
    if len(data) != size:
        raise ModelError(f"truncated {label}: expected {size}, got {len(data)} bytes")
    return data


def u16(stream, label):
    return struct.unpack("<H", read_exact(stream, 2, label))[0]


def u32(stream, label):
    return struct.unpack("<I", read_exact(stream, 4, label))[0]


def u64(stream, label):
    return struct.unpack("<Q", read_exact(stream, 8, label))[0]


def i32(stream, label):
    return struct.unpack("<i", read_exact(stream, 4, label))[0]


def f32(stream, label):
    return struct.unpack("<f", read_exact(stream, 4, label))[0]


def f64(stream, label):
    return struct.unpack("<d", read_exact(stream, 8, label))[0]


def read_string(stream, label):
    size = u32(stream, f"{label} length")
    return read_exact(stream, size, label).decode("utf-8")


def skip_params(stream, label):
    count = u64(stream, f"{label} parameter count")
    if count > 100_000_000:
        raise ModelError(f"{label} parameter count is implausibly large: {count}")
    read_exact(stream, count * 4, f"{label} parameters")
    return count


def inspect_znvm(path: str | Path) -> dict:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        if read_exact(stream, 8, "magic") != MAGIC:
            raise ModelError("invalid ZNVM magic")
        version = u32(stream, "version")
        flags = u32(stream, "flags")
        if version != VERSION or flags not in (FLAG_FLOATGRID_AFFINE, FLAG_FLOATGRID_AFFINE | FLAG_PARTIAL_DOMAINS):
            raise ModelError(f"unsupported ZNVM version/flags: {version}/{flags}")

        source_hash = read_exact(stream, 32, "source dataset SHA-256").hex()
        grid_name = read_string(stream, "grid name")
        grid_class = u32(stream, "grid class")
        background = f32(stream, "background")
        voxel_size = [f32(stream, f"voxel size[{i}]") for i in range(3)]
        affine = [f64(stream, f"affine matrix[{i}]") for i in range(16)]
        if any(not math.isfinite(x) for x in affine):
            raise ModelError("affine transform contains non-finite values")

        domain_size = u32(stream, "domain size")
        steps = u32(stream, "training steps")
        batch_size = u32(stream, "batch size")
        learning_rate = f32(stream, "learning rate")
        value_tolerance = f32(stream, "value tolerance")
        config_text = read_string(stream, "model config")
        try:
            model_config = json.loads(config_text)
        except json.JSONDecodeError as exc:
            raise ModelError(f"invalid model config JSON: {exc}") from exc

        tile_count = u64(stream, "tile count")
        domain_count = u64(stream, "domain count")
        total_leaf_count = u64(stream, "total source leaf count")
        total_active_voxels = u64(stream, "total source active voxels")

        for index in range(tile_count):
            for axis in range(3):
                i32(stream, f"tile[{index}] coordinate[{axis}]")
            level = u32(stream, f"tile[{index}] level")
            f32(stream, f"tile[{index}] value")
            if level == 0:
                raise ModelError(f"tile[{index}] has invalid leaf level")

        encoded_leaf_count = 0
        occupancy_corrections = 0
        value_corrections = 0
        for domain_index in range(domain_count):
            for axis in range(3):
                i32(stream, f"domain[{domain_index}] key[{axis}]")
            for axis in range(3):
                i32(stream, f"domain[{domain_index}] origin[{axis}]")
            skip_params(stream, f"domain[{domain_index}] classifier")

            has_regressor = u32(stream, f"domain[{domain_index}] regressor flag")
            if has_regressor not in (0, 1):
                raise ModelError(f"domain[{domain_index}] has invalid regressor flag")
            if has_regressor:
                f32(stream, f"domain[{domain_index}] value center")
                scale = f32(stream, f"domain[{domain_index}] value scale")
                if not math.isfinite(scale) or scale <= 0:
                    raise ModelError(f"domain[{domain_index}] has invalid value scale")
                skip_params(stream, f"domain[{domain_index}] regressor")

            for metric_index in range(5):
                f32(stream, f"domain[{domain_index}] metric[{metric_index}]")
            domain_occ_corrections = u32(stream, f"domain[{domain_index}] occupancy corrections")
            domain_value_corrections = u32(stream, f"domain[{domain_index}] value corrections")
            occupancy_corrections += domain_occ_corrections
            value_corrections += domain_value_corrections

            leaf_count = u32(stream, f"domain[{domain_index}] leaf count")
            encoded_leaf_count += leaf_count
            observed_domain_occ_corrections = 0
            observed_domain_value_corrections = 0
            for leaf_index in range(leaf_count):
                for axis in range(3):
                    i32(stream, f"leaf[{leaf_index}] origin[{axis}]")
                occ_count = u32(stream, f"leaf[{leaf_index}] occupancy correction count")
                if occ_count > 512:
                    raise ModelError("occupancy correction count exceeds leaf capacity")
                observed_domain_occ_corrections += occ_count
                seen_occ = set()
                for _ in range(occ_count):
                    offset = u16(stream, "occupancy correction offset")
                    if offset >= 512 or offset in seen_occ:
                        raise ModelError("invalid or duplicate occupancy correction offset")
                    seen_occ.add(offset)

                val_count = u32(stream, f"leaf[{leaf_index}] value correction count")
                if val_count > 512:
                    raise ModelError("value correction count exceeds leaf capacity")
                observed_domain_value_corrections += val_count
                seen_values = set()
                for _ in range(val_count):
                    offset = u16(stream, "value correction offset")
                    f32(stream, "exact value correction")
                    if offset >= 512 or offset in seen_values:
                        raise ModelError("invalid or duplicate value correction offset")
                    seen_values.add(offset)

        if encoded_leaf_count > total_leaf_count:
            raise ModelError("encoded leaf count exceeds source leaf count")
        partial = bool(flags & FLAG_PARTIAL_DOMAINS)
        if not partial and encoded_leaf_count != total_leaf_count:
            raise ModelError("full checkpoint does not contain every source leaf")
        if any(not math.isfinite(v) or v <= 0 for v in voxel_size):
            raise ModelError(f"invalid voxel size: {voxel_size}")
        if stream.read(1):
            raise ModelError("unexpected trailing bytes")

    return {
        "path": str(path),
        "file_size_bytes": size,
        "format": "ZNVM",
        "version": version,
        "partial": partial,
        "source_dataset_sha256": source_hash,
        "grid_name": grid_name,
        "grid_class": grid_class,
        "background": background,
        "voxel_size": voxel_size,
        "domain_size_voxels": domain_size,
        "source_leaf_blocks": total_leaf_count,
        "encoded_leaf_blocks": encoded_leaf_count,
        "active_tiles": tile_count,
        "source_active_voxels": total_active_voxels,
        "domains": domain_count,
        "occupancy_corrections": occupancy_corrections,
        "value_corrections": value_corrections,
        "model_config": model_config,
        "validation": "ok",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    try:
        result = inspect_znvm(args.model)
    except (OSError, UnicodeDecodeError, ModelError) as exc:
        parser.exit(2, f"ZNVM validation failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
