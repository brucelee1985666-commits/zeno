import struct
import tempfile
import unittest
from pathlib import Path

from inspect_znvds import DatasetError, inspect_dataset


def make_dataset(*, occupancy_override=None, trailing=b"", origin=(0, 0, 0), values=(0.25, 0.75)):
    occupancy = bytearray(64)
    occupancy[0] = 0b00000011
    if occupancy_override is not None:
        occupancy = bytearray(occupancy_override)
    chunks = [
        b"ZNVDS001",
        struct.pack("<IIQQQ", 1, 3, 1, 0, 2),
        struct.pack("<I", 4),
        b"test",
        struct.pack("<Iffff", 2, 0.0, 1.0, 1.0, 1.0),
        struct.pack("<16d", 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0),
        struct.pack("<iiiI", *origin, 2),
        bytes(occupancy),
        struct.pack("<Hf", 0, values[0]),
        struct.pack("<Hf", 1, values[1]),
        trailing,
    ]
    return b"".join(chunks)


class ZNVDSInspectionTests(unittest.TestCase):
    def test_valid_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid.znvds"
            path.write_bytes(make_dataset())
            result = inspect_dataset(path)
            self.assertEqual(result["validation"], "ok")
            self.assertEqual(result["active_voxels"], 2)
            self.assertEqual(result["leaf_blocks"], 1)

    def test_rejects_occupancy_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.znvds"
            path.write_bytes(make_dataset(occupancy_override=bytes(64)))
            with self.assertRaises(DatasetError):
                inspect_dataset(path)

    def test_rejects_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trailing.znvds"
            path.write_bytes(make_dataset(trailing=b"x"))
            with self.assertRaises(DatasetError):
                inspect_dataset(path)

    def test_cpu_audit_handles_negative_domain_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "negative.znvds"
            path.write_bytes(make_dataset(origin=(-8, 0, 0)))
            result = inspect_dataset(path, domain_size=8)
            audit = result["cpu_reference_audit"]
            self.assertEqual(audit["domain_count"], 1)
            self.assertEqual(audit["negative_domain_count"], 1)
            self.assertEqual(audit["active_voxels_on_domain_boundary_planes"], 2)
            self.assertEqual(audit["active_voxels_by_domain"][0]["key"], [-1, 0, 0])
            self.assertEqual(len(audit["active_coordinate_value_sha256"]), 64)

    def test_rejects_non_finite_active_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "non-finite.znvds"
            path.write_bytes(make_dataset(values=(float("nan"), 0.75)))
            with self.assertRaises(DatasetError):
                inspect_dataset(path)

    def test_rejects_unaligned_leaf_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unaligned.znvds"
            path.write_bytes(make_dataset(origin=(1, 0, 0)))
            with self.assertRaises(DatasetError):
                inspect_dataset(path)


if __name__ == "__main__":
    unittest.main()
