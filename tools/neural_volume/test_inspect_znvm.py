import json
import struct
import tempfile
import unittest
from pathlib import Path

from inspect_znvm import ModelError, inspect_znvm


def make_model(*, partial=False, trailing=b""):
    flags = 7 if partial else 3
    config = json.dumps({"encoding": {"otype": "HashGrid"}, "network": {"otype": "CutlassMLP"}}).encode()
    chunks = [
        b"ZNVM0001",
        struct.pack("<II", 1, flags),
        bytes.fromhex("00" * 32),
        struct.pack("<I", 4), b"test",
        struct.pack("<Iffff", 2, 0.0, 1.0, 1.0, 1.0),
        struct.pack("<16d", 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0),
        struct.pack("<IIIff", 128, 2000, 32768, 0.01, 0.001),
        struct.pack("<I", len(config)), config,
        struct.pack("<QQQQ", 0, 1, 1, 2),
        struct.pack("<iii", 0, 0, 0),
        struct.pack("<iii", 0, 0, 0),
        struct.pack("<Qf", 1, 0.5),
        struct.pack("<I", 0),  # no regressor
        struct.pack("<fffffII", 1.0, 1.0, 1.0, 1.0, -1.0, 0, 2),
        struct.pack("<I", 1),  # one leaf
        struct.pack("<iii", 0, 0, 0),
        struct.pack("<IHH", 1, 7, 2),
        struct.pack("<I", 1),
        struct.pack("<Hf", 3, 0.75),
        trailing,
    ]
    return b"".join(chunks)


class ZNVMInspectionTests(unittest.TestCase):
    def test_valid_full_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid.znvm"
            path.write_bytes(make_model())
            result = inspect_znvm(path)
            self.assertEqual(result["validation"], "ok")
            self.assertFalse(result["partial"])
            self.assertEqual(result["encoded_leaf_blocks"], 1)

    def test_valid_partial_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.znvm"
            path.write_bytes(make_model(partial=True))
            self.assertTrue(inspect_znvm(path)["partial"])

    def test_rejects_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.znvm"
            path.write_bytes(make_model(trailing=b"x"))
            with self.assertRaises(ModelError):
                inspect_znvm(path)


if __name__ == "__main__":
    unittest.main()
