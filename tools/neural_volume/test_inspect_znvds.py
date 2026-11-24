import struct
import tempfile
import unittest
from pathlib import Path

from inspect_znvds import DatasetError, inspect_dataset


def make_dataset(*, occupancy_override=None, trailing=b""):
    occupancy = bytearray(64)
    occupancy[0] = 0b00000011
    if occupancy_override is not None:
        occupancy = bytearray(occupancy_override)
    chunks = [
        b"ZNVDS001",
        struct.pack("<IIQQQ", 1, 1, 1, 0, 2),
        struct.pack("<I", 4),
        b"test",
        struct.pack("<ffff", 0.0, 1.0, 1.0, 1.0),
        struct.pack("<iiiI", 0, 0, 0, 2),
        bytes(occupancy),
        struct.pack("<Hf", 0, 0.25),
        struct.pack("<Hf", 1, 0.75),
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


if __name__ == "__main__":
    unittest.main()
