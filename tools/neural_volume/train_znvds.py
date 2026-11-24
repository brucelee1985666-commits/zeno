#!/usr/bin/env python3
"""Train a first ZNVDS -> neural checkpoint prototype using tiny-cuda-nn.

The checkpoint is a research/interchange artifact (.znvm), not yet a C++ runtime
codec. It stores per-domain occupancy/value models plus sparse exact corrections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tinycudann as tcnn

MAGIC = b"ZNVDS001"
DOMAIN_DEFAULT = 128


class DatasetError(ValueError):
    pass


def read_exact(stream, size, label):
    data = stream.read(size)
    if len(data) != size:
        raise DatasetError(f"truncated {label}")
    return data


def read_u16(stream):
    return struct.unpack("<H", read_exact(stream, 2, "uint16"))[0]


def read_u32(stream):
    return struct.unpack("<I", read_exact(stream, 4, "uint32"))[0]


def read_u64(stream):
    return struct.unpack("<Q", read_exact(stream, 8, "uint64"))[0]


def read_i32(stream):
    return struct.unpack("<i", read_exact(stream, 4, "int32"))[0]


def read_f32(stream):
    return struct.unpack("<f", read_exact(stream, 4, "float32"))[0]


def load_znvds(path: Path, domain_size: int):
    domains = defaultdict(list)
    tiles = []
    with path.open("rb") as stream:
        if read_exact(stream, 8, "magic") != MAGIC:
            raise DatasetError("invalid ZNVDS magic")
        version, flags = read_u32(stream), read_u32(stream)
        if version != 1 or flags != 3:
            raise DatasetError(f"unsupported ZNVDS version/flags: {version}/{flags}")
        leaf_count, tile_count, active_count = read_u64(stream), read_u64(stream), read_u64(stream)
        name = read_exact(stream, read_u32(stream), "grid name").decode("utf-8")
        grid_class = read_u32(stream)
        background = read_f32(stream)
        voxel_size = [read_f32(stream) for _ in range(3)]
        affine = [struct.unpack("<d", read_exact(stream, 8, "affine matrix"))[0]
                  for _ in range(16)]

        for _ in range(tile_count):
            coord = [read_i32(stream) for _ in range(3)]
            level = read_u32(stream)
            value = read_f32(stream)
            tiles.append({"coord": coord, "level": level, "value": value})

        observed_active = 0
        for _ in range(leaf_count):
            origin = [read_i32(stream) for _ in range(3)]
            count = read_u32(stream)
            if count > 512:
                raise DatasetError("leaf active count exceeds 512")
            mask = read_exact(stream, 64, "occupancy mask")
            if sum(byte.bit_count() for byte in mask) != count:
                raise DatasetError("leaf occupancy mask/count mismatch")
            values = []
            seen = set()
            for _ in range(count):
                offset, value = read_u16(stream), read_f32(stream)
                if offset >= 512 or offset in seen:
                    raise DatasetError("invalid or duplicate active voxel offset")
                if not (mask[offset >> 3] & (1 << (offset & 7))):
                    raise DatasetError("active value is not set in occupancy mask")
                seen.add(offset)
                values.append((offset, value))
            observed_active += count
            key = tuple(math.floor(v / domain_size) for v in origin)
            domains[key].append({"origin": origin, "mask": mask, "values": values})

        if observed_active != active_count:
            raise DatasetError("global active voxel count mismatch")
        if stream.read(1):
            raise DatasetError("unexpected trailing bytes")

    metadata = {
        "grid_name": name,
        "grid_class": grid_class,
        "background": background,
        "voxel_size": voxel_size,
        "index_to_world_affine_row_major": affine,
        "leaf_blocks": leaf_count,
        "active_tiles": tile_count,
        "active_voxels": active_count,
    }
    return metadata, tiles, domains


def make_model(seed: int, output_dims: int = 1):
    encoding = {
        "otype": "HashGrid",
        "n_levels": 12,
        "n_features_per_level": 2,
        "log2_hashmap_size": 15,
        "base_resolution": 8,
        "per_level_scale": 1.5,
    }
    network = {
        "otype": "CutlassMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2,
    }
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=output_dims,
        encoding_config=encoding,
        network_config=network,
        seed=seed,
    ).cuda()
    if tcnn.supports_jit_fusion():
        model.jit_fusion = True
    return model, {
        "backend": "tiny-cuda-nn",
        "upstream_commit": "3daa6e5b8b8e0fa17971a1a20acfbb3754494b38",
        "param_precision": "float16" if model.params.dtype == torch.float16 else "float32",
        "encoding": encoding,
        "network": network,
    }


def train_model(model, inputs, targets, steps, batch_size, learning_rate, kind):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    count = inputs.shape[0]
    if count == 0:
        raise ValueError("cannot train a model without samples")
    batch = min(batch_size, count)
    for _ in range(steps):
        indices = torch.randint(count, (batch,), device="cuda")
        prediction = model(inputs[indices]).float()
        if kind == "classifier":
            loss = F.binary_cross_entropy_with_logits(prediction, targets[indices])
        else:
            loss = F.mse_loss(prediction, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def infer(model, inputs, batch_size=65536):
    outputs = []
    for start in range(0, len(inputs), batch_size):
        outputs.append(model(inputs[start:start + batch_size]).float())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 1), device="cuda")


def encode_domain(key, leaves, args):
    domain_origin = [int(v * args.domain_size) for v in key]
    coords = []
    occupancy = []
    value_coords = []
    values = []
    leaf_slices = []

    for leaf in leaves:
        begin = len(coords)
        ox, oy, oz = leaf["origin"]
        mask = leaf["mask"]
        value_map = dict(leaf["values"])
        for offset in range(512):
            lx, ly, lz = offset >> 6, (offset >> 3) & 7, offset & 7
            gx, gy, gz = ox + lx, oy + ly, oz + lz
            coords.append(((gx - domain_origin[0]) / args.domain_size,
                           (gy - domain_origin[1]) / args.domain_size,
                           (gz - domain_origin[2]) / args.domain_size))
            active = 1.0 if (mask[offset >> 3] & (1 << (offset & 7))) else 0.0
            occupancy.append(active)
            if active:
                value_coords.append(coords[-1])
                values.append(value_map[offset])
        leaf_slices.append((begin, len(coords)))

    x_occ = torch.as_tensor(np.asarray(coords, dtype=np.float32), device="cuda")
    y_occ = torch.as_tensor(np.asarray(occupancy, dtype=np.float32).reshape(-1, 1), device="cuda")

    domain_hash = (
        key[0] * 73856093 ^ key[1] * 19349663 ^ key[2] * 83492791
    ) & 0x7fffffff
    domain_seed = (args.seed + domain_hash) & 0x7fffffff
    classifier, classifier_config = make_model(domain_seed)
    train_model(classifier, x_occ, y_occ, args.steps, args.batch_size, args.learning_rate, "classifier")
    logits = infer(classifier, x_occ)
    predicted = (logits[:, 0] >= 0.0).to(torch.uint8).cpu().numpy()
    target = np.asarray(occupancy, dtype=np.uint8)

    occupancy_corrections = []
    for leaf_index, (begin, end) in enumerate(leaf_slices):
        mismatch = np.flatnonzero(predicted[begin:end] != target[begin:end]).astype(np.uint16)
        occupancy_corrections.append(mismatch.tolist())

    classifier_accuracy = float((predicted == target).mean())
    true_positive = int(np.logical_and(predicted == 1, target == 1).sum())
    false_positive = int(np.logical_and(predicted == 1, target == 0).sum())
    false_negative = int(np.logical_and(predicted == 0, target == 1).sum())
    occupancy_iou = true_positive / max(1, true_positive + false_positive + false_negative)
    occupancy_precision = true_positive / max(1, true_positive + false_positive)
    occupancy_recall = true_positive / max(1, true_positive + false_negative)

    regressor_payload = None
    value_corrections = [[] for _ in leaves]
    value_rmse = None
    if len(values) >= args.min_value_samples:
        vmin, vmax = float(min(values)), float(max(values))
        center = (vmin + vmax) * 0.5
        scale = max((vmax - vmin) * 0.5, 1.0e-8)
        x_val = torch.as_tensor(np.asarray(value_coords, dtype=np.float32), device="cuda")
        y_val_np = ((np.asarray(values, dtype=np.float32) - center) / scale).reshape(-1, 1)
        y_val = torch.as_tensor(y_val_np, device="cuda")
        regressor, regressor_config = make_model(domain_seed + 100003)
        train_model(regressor, x_val, y_val, args.steps, args.batch_size, args.learning_rate, "regressor")
        pred_norm = infer(regressor, x_val)[:, 0].cpu().numpy()
        pred_values = pred_norm * scale + center
        errors = np.abs(pred_values - np.asarray(values, dtype=np.float32))
        value_rmse = float(np.sqrt(np.mean(errors * errors)))

        value_cursor = 0
        for leaf_index, leaf in enumerate(leaves):
            corrections = []
            for offset, exact_value in leaf["values"]:
                error = abs(float(pred_values[value_cursor]) - exact_value)
                if error > args.value_tolerance:
                    corrections.append((int(offset), float(exact_value)))
                value_cursor += 1
            value_corrections[leaf_index] = corrections

        regressor_payload = {
            "config": regressor_config,
            "params": regressor.params.detach().float().cpu().contiguous(),
            "center": center,
            "scale": scale,
        }
    else:
        # Tiny domains are stored explicitly rather than paying for a model
        # whose parameter payload would exceed the source samples.
        for i, leaf in enumerate(leaves):
            value_corrections[i] = [(int(offset), float(value)) for offset, value in leaf["values"]]

    leaf_payload = []
    for index, leaf in enumerate(leaves):
        leaf_payload.append({
            "origin": leaf["origin"],
            "occupancy_corrections": occupancy_corrections[index],
            "value_corrections": value_corrections[index],
        })

    return {
        "domain_key": list(key),
        "domain_origin": domain_origin,
        "classifier": {
            "config": classifier_config,
            "params": classifier.params.detach().float().cpu().contiguous(),
        },
        "regressor": regressor_payload,
        "leaves": leaf_payload,
        "metrics": {
            "classifier_accuracy": classifier_accuracy,
            "occupancy_iou": float(occupancy_iou),
            "occupancy_precision": float(occupancy_precision),
            "occupancy_recall": float(occupancy_recall),
            "occupancy_corrections": sum(len(x) for x in occupancy_corrections),
            "value_rmse_before_corrections": value_rmse,
            "value_corrections": sum(len(x) for x in value_corrections),
        },
    }


def write_u16(stream, value):
    stream.write(struct.pack("<H", int(value)))


def write_u32(stream, value):
    stream.write(struct.pack("<I", int(value)))


def write_u64(stream, value):
    stream.write(struct.pack("<Q", int(value)))


def write_i32(stream, value):
    stream.write(struct.pack("<i", int(value)))


def write_f32(stream, value):
    stream.write(struct.pack("<f", float(value)))


def write_f64(stream, value):
    stream.write(struct.pack("<d", float(value)))


def write_string(stream, value):
    encoded = value.encode("utf-8")
    write_u32(stream, len(encoded))
    stream.write(encoded)


def write_params(stream, params):
    array = params.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    write_u64(stream, array.size)
    array.tofile(stream)


def write_znvm(path, metadata, tiles, domains, args, source_hash, partial, model_config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"ZNVM0001")
        write_u32(stream, 1)
        write_u32(stream, 7 if partial else 3)  # FloatGrid + affine + optional partial-domain flag
        stream.write(bytes.fromhex(source_hash))
        write_string(stream, metadata["grid_name"])
        write_u32(stream, metadata["grid_class"])
        write_f32(stream, metadata["background"])
        for value in metadata["voxel_size"]:
            write_f32(stream, value)
        for value in metadata["index_to_world_affine_row_major"]:
            write_f64(stream, value)
        write_u32(stream, args.domain_size)
        write_u32(stream, args.steps)
        write_u32(stream, args.batch_size)
        write_f32(stream, args.learning_rate)
        write_f32(stream, args.value_tolerance)
        write_string(stream, json.dumps(model_config, sort_keys=True, separators=(",", ":")))
        write_u64(stream, len(tiles))
        write_u64(stream, len(domains))
        write_u64(stream, metadata["leaf_blocks"])
        write_u64(stream, metadata["active_voxels"])

        for tile in tiles:
            for value in tile["coord"]:
                write_i32(stream, value)
            write_u32(stream, tile["level"])
            write_f32(stream, tile["value"])

        for domain in domains:
            for value in domain["domain_key"]:
                write_i32(stream, value)
            for value in domain["domain_origin"]:
                write_i32(stream, value)
            write_params(stream, domain["classifier"]["params"])

            regressor = domain["regressor"]
            write_u32(stream, 1 if regressor is not None else 0)
            if regressor is not None:
                write_f32(stream, regressor["center"])
                write_f32(stream, regressor["scale"])
                write_params(stream, regressor["params"])

            metrics = domain["metrics"]
            for key in (
                "classifier_accuracy", "occupancy_iou", "occupancy_precision",
                "occupancy_recall", "value_rmse_before_corrections"
            ):
                value = metrics[key]
                write_f32(stream, -1.0 if value is None else value)
            write_u32(stream, metrics["occupancy_corrections"])
            write_u32(stream, metrics["value_corrections"])

            write_u32(stream, len(domain["leaves"]))
            for leaf in domain["leaves"]:
                for value in leaf["origin"]:
                    write_i32(stream, value)
                occupancy = leaf["occupancy_corrections"]
                write_u32(stream, len(occupancy))
                for offset in occupancy:
                    write_u16(stream, offset)
                value_corrections = leaf["value_corrections"]
                write_u32(stream, len(value_corrections))
                for offset, value in value_corrections:
                    write_u16(stream, offset)
                    write_f32(stream, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--domain-size", type=int, default=DOMAIN_DEFAULT)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--value-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--min-value-samples", type=int, default=64)
    parser.add_argument("--max-domains", type=int, default=0, help="0 means all domains")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.domain_size <= 0 or args.domain_size % 8:
        parser.error("--domain-size must be a positive multiple of 8")
    if args.steps <= 0 or args.batch_size <= 0 or args.value_tolerance < 0:
        parser.error("steps/batch-size must be positive and value-tolerance non-negative")
    if not torch.cuda.is_available():
        parser.error("tiny-cuda-nn training requires a CUDA-enabled PyTorch installation and NVIDIA GPU")

    torch.manual_seed(args.seed)
    metadata, tiles, domains = load_znvds(args.dataset, args.domain_size)
    if not domains:
        parser.error("dataset contains no leaf blocks; nothing to train")

    ordered_keys = sorted(domains)
    if args.max_domains:
        ordered_keys = ordered_keys[:args.max_domains]

    encoded = []
    for index, key in enumerate(ordered_keys):
        payload = encode_domain(key, domains[key], args)
        encoded.append(payload)
        torch.cuda.empty_cache()
        print(json.dumps({
            "domain": index + 1,
            "domain_count": len(ordered_keys),
            "key": list(key),
            "metrics": payload["metrics"],
        }))

    partial = len(ordered_keys) != len(domains)
    model_config = encoded[0]["classifier"]["config"]
    source_hash = sha256_file(args.dataset)
    write_znvm(args.output, metadata, tiles, encoded, args, source_hash, partial, model_config)
    print(json.dumps({
        "output": str(args.output),
        "domains_encoded": len(encoded),
        "partial_checkpoint": partial,
        "checkpoint_bytes": args.output.stat().st_size,
        "source_dataset_bytes": args.dataset.stat().st_size,
        "status": "checkpoint_written",
    }, indent=2))


if __name__ == "__main__":
    main()
