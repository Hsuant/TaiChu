"""Tests for GPT-2 style vocabulary export."""

import json
from pathlib import Path
import tempfile
import unittest

from tokenizer.export_vocab_merges import MERGES_HEADER, export_vocab_merges


class TestTokenizerExport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.tokenizer_json = self.root / "tokenizer.json"
        self.tokenizer_json.write_text(
            json.dumps(
                {
                    "model": {
                        "type": "BPE",
                        "vocab": {
                            "<unk>": 0,
                            "a": 1,
                            "b": 2,
                            "ab": 3,
                        },
                        "merges": [["a", "b"]],
                    }
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_writes_standard_files(self):
        vocab_path, merges_path = export_vocab_merges(
            self.tokenizer_json,
            self.root,
        )
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.assertEqual(
            vocab,
            {"<unk>": 0, "a": 1, "b": 2, "ab": 3},
        )
        self.assertEqual(
            merges_path.read_text(encoding="utf-8").splitlines(),
            [MERGES_HEADER, "a b"],
        )

    def test_existing_files_require_overwrite(self):
        export_vocab_merges(self.tokenizer_json, self.root)
        with self.assertRaises(FileExistsError):
            export_vocab_merges(self.tokenizer_json, self.root)
        export_vocab_merges(self.tokenizer_json, self.root, overwrite=True)

    def test_non_contiguous_vocab_is_rejected(self):
        self.tokenizer_json.write_text(
            json.dumps(
                {
                    "model": {
                        "type": "BPE",
                        "vocab": {"a": 0, "b": 2},
                        "merges": [["a", "b"]],
                    }
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            export_vocab_merges(self.tokenizer_json, self.root)

    def test_merge_token_starting_with_hash_is_preserved(self):
        self.tokenizer_json.write_text(
            json.dumps(
                {
                    "model": {
                        "type": "BPE",
                        "vocab": {"a": 0, "#": 1, "a#": 2},
                        "merges": [["a", "#"]],
                    }
                }
            ),
            encoding="utf-8",
        )
        _, merges_path = export_vocab_merges(
            self.tokenizer_json,
            self.root,
            overwrite=True,
        )
        self.assertEqual(
            merges_path.read_text(encoding="utf-8").splitlines(),
            [MERGES_HEADER, "a #"],
        )


if __name__ == "__main__":
    unittest.main()
