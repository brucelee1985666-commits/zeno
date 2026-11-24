#!/usr/bin/env python3
"""Inspect and validate ZNVDS v1 neural-volume training datasets.

This parser intentionally uses only the Python standard library. ZNVDS is a
training-data interchange format, not the final compressed neural-volume file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import BinaryIO

MAGIC = b"ZNVDS001"
VERSION = 1
HEADER_FLAGS_FLOAT_INDEX_SPACE_AFFINE = 3


class DatasetError(ValueError):
    pass


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise DatasetError(f"truncated {label}: expected {size} bytes, got {len(data)}")
    return data


def _u16(stream: BinaryIO, label: str) -> int:
    return struct.unpack("<H", _read_exact(stream, 2, label))[0]


def _u32(stream: BinaryIO, label: str) -> int:
    return struct.unpack("<I", _read_exact(stream, 4, label))[0]


def _u64(stream: BinaryIO, label: str) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8, label))[0]


def _i32(stream: BinaryIO, label: str) -> int:
    return struct.unpack("<i", _read_exact(stream, 4, label))[0]


def _f32(stream: BinaryIO, label: str) -> float:
    return struct.unpack("<f", _read_exact(stream, 4, label))[0]


def inspect_dataset(path: str | Path, domain_size: int = 128) -> dict:
    if domain_size <= 0 or domain_size % 8:
        raise DatasetError("domain size must be a positive multiple of 8")
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        magic = _read_exact(stream, 8, "magic")
        if magic != MAGIC:
            raise DatasetError(f"invalid magic: {magic!r}")

        version = _u32(stream, "version")
        if version != VERSION:
            raise DatasetError(f"unsupported ZNVDS version: {version}")

        flags = _u32(stream, "flags")
        if flags != HEADER_FLAGS_FLOAT_INDEX_SPACE_AFFINE:
            raise DatasetError(f"unsupported flags: 0x{flags:08x}")

        leaf_count = _u64(stream, "leaf count")
        tile_count = _u64(stream, "tile count")
        active_voxel_count = _u64(stream, "active voxel count")

        name_size = _u32(stream, "grid-name size")
        name = _read_exact(stream, name_size, "grid name").decode("utf-8")
        grid_class = _u32(stream, "grid class")
        background = _f32(stream, "background")
        if not math.isfinite(background):
            raise DatasetError("background is not finite")
        voxel_size = [_f32(stream, f"voxel size[{i}]") for i in range(3)]
        affine_transform = [
            struct.unpack("<d", _read_exact(stream, 8, f"affine matrix[{row},{column}]"))[0]
            for row in range(4)
            for column in range(4)
        ]
        if any(not math.isfinite(v) for v in affine_transform):
            raise DatasetError("affine transform contains non-finite values")

        for index in range(tile_count):
            coord = [_i32(stream, f"tile[{index}] coordinate") for _ in range(3)]
            level = _u32(stream, f"tile[{index}] level")
            value = _f32(stream, f"tile[{index}] value")
            if level == 0:
                raise DatasetError(f"tile[{index}] has leaf level 0")
            if not math.isfinite(value):
                raise DatasetError(f"tile[{index}] has a non-finite value")

        observed_active = 0
        seen_leaf_origins = set()
        domain_active_voxels = {}
        negative_domains = set()
        boundary_active_voxels = 0
        active_digest = hashlib.sha256()
        for block_index in range(leaf_count):
            origin = tuple(_i32(stream, f"leaf[{block_index}] origin") for _ in range(3))
            if any(value % 8 for value in origin):
                raise DatasetError(f"leaf[{block_index}] origin is not aligned to the 8-voxel leaf grid")
            if origin in seen_leaf_origins:
                raise DatasetError(f"duplicate leaf origin: {origin}")
            seen_leaf_origins.add(origin)
            domain_key = tuple(value // domain_size for value in origin)
            domain_origin = tuple(value * domain_size for value in domain_key)
            local_origin = tuple(origin[axis] - domain_origin[axis] for axis in range(3))
            if any(value < 0 or value + 8 > domain_size for value in local_origin):
                raise DatasetError(f"leaf[{block_index}] crosses its assigned neural domain")
            domain_active_voxels.setdefault(domain_key, 0)
            if any(value < 0 for value in domain_key):
                negative_domains.add(domain_key)
            active_count = _u32(stream, f"leaf[{block_index}] active count")
            if active_count > 512:
                raise DatasetError(f"leaf[{block_index}] active count exceeds 512")

            occupancy = _read_exact(stream, 64, f"leaf[{block_index}] occupancy")
            bit_count = sum(byte.bit_count() for byte in occupancy)
            if bit_count != active_count:
                raise DatasetError(
                    f"leaf[{block_index}] occupancy has {bit_count} bits, header says {active_count}"
                )

            seen = set()
            for value_index in range(active_count):
                offset = _u16(stream, f"leaf[{block_index}] value[{value_index}] offset")
                value = _f32(stream, f"leaf[{block_index}] value[{value_index}]")
                if not math.isfinite(value):
                    raise DatasetError(f"leaf[{block_index}] has a non-finite active value")
                if offset >= 512:
                    raise DatasetError(f"leaf[{block_index}] has invalid voxel offset {offset}")
                if offset in seen:
                    raise DatasetError(f"leaf[{block_index}] repeats voxel offset {offset}")
                seen.add(offset)
                if not (occupancy[offset >> 3] & (1 << (offset & 7))):
                    raise DatasetError(
                        f"leaf[{block_index}] value offset {offset} is not active in occupancy mask"
                    )
                x = origin[0] + (offset >> 6)
                y = origin[1] + ((offset >> 3) & 7)
                z = origin[2] + (offset & 7)
                local = (x - domain_origin[0], y - domain_origin[1], z - domain_origin[2])
                if any(value == 0 or value == domain_size - 1 for value in local):
                    boundary_active_voxels += 1
                active_digest.update(struct.pack("<iiif", x, y, z, value))
            domain_active_voxels[domain_key] += active_count
            observed_active += active_count

        if observed_active != active_voxel_count:
            raise DatasetError(
                f"active voxel count mismatch: header={active_voxel_count}, observed={observed_active}"
            )

        trailing = stream.read(1)
        if trailing:
            raise DatasetError("unexpected trailing bytes after final leaf block")

    if any(not math.isfinite(v) or v <= 0 for v in voxel_size):
        raise DatasetError(f"invalid voxel size: {voxel_size}")

    return {
        "path": str(path),
        "file_size_bytes": file_size,
        "format": "ZNVDS",
        "version": version,
        "grid_name": name,
        "grid_class": grid_class,
        "background": background,
        "voxel_size": voxel_size,
        "index_to_world_affine_row_major": affine_transform,
        "leaf_blocks": leaf_count,
        "active_tiles": tile_count,
        "active_voxels": active_voxel_count,
        "cpu_reference_audit": {
            "backend": "python-standard-library",
            "domain_size_voxels": domain_size,
            "domain_count": len(domain_active_voxels),
            "negative_domain_count": len(negative_domains),
            "active_voxels_on_domain_boundary_planes": boundary_active_voxels,
            "active_voxels_by_domain": [
                {"key": list(key), "active_voxels": count}
                for key, count in sorted(domain_active_voxels.items())
            ],
            "active_coordinate_value_sha256": active_digest.hexdigest(),
            "validation": "ok",
        },
        "validation": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="path to a .znvds dataset")
    parser.add_argument("--domain-size", type=int, default=128,
                        help="neural domain size in voxels (positive multiple of 8)")
    args = parser.parse_args()
    try:
        result = inspect_dataset(args.dataset, domain_size=args.domain_size)
    except (OSError, UnicodeDecodeError, DatasetError) as exc:
        parser.exit(2, f"ZNVDS validation failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
