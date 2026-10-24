from __future__ import annotations

import contextlib
import gzip
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from phasis.stages import indexing


class ReferenceCompressionProgressTests(unittest.TestCase):
    def test_progress_tracks_bytes_without_changing_archive_contents(self):
        content = b">1\n" + b"ACGT" * (600 * 1024) + b"\n"
        expected = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=expected, mtime=0) as dst:
            dst.write(content)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "ref.clean.fa"
            source.write_bytes(content)
            messages = io.StringIO()
            with mock.patch.object(indexing, "tqdm") as progress, contextlib.redirect_stdout(messages):
                archive = Path(indexing._archive_file_to_gz(str(source)))
            progress.assert_called_once_with(total=len(content), desc="Compressing cleaned reference",
                                             unit="B", unit_scale=True)
            updates = progress.return_value.__enter__.return_value.update.call_args_list
            self.assertEqual(sum(call.args[0] for call in updates), len(content))
            self.assertTrue(all(call.args[0] <= 1024 * 1024 for call in updates))
            self.assertEqual(archive.read_bytes(), expected.getvalue())
            self.assertEqual(gzip.decompress(archive.read_bytes()), content)
            self.assertFalse(source.exists())
            self.assertFalse(Path(str(archive) + ".tmp").exists())
            self.assertIn("gzip level 9, single worker", messages.getvalue())
            self.assertIn("compression complete", messages.getvalue())

    def test_failed_compression_preserves_source_and_existing_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "ref.clean.fa"
            source.write_bytes(b">1\nACGT\n")
            archive = Path(str(source) + ".gz")
            archive.write_bytes(b"previous archive")
            messages = io.StringIO()
            with mock.patch.object(indexing.gzip, "GzipFile", side_effect=OSError("disk full")):
                with contextlib.redirect_stdout(messages), self.assertRaisesRegex(OSError, "disk full"):
                    indexing._archive_file_to_gz(str(source))
            self.assertEqual(source.read_bytes(), b">1\nACGT\n")
            self.assertEqual(archive.read_bytes(), b"previous archive")
            self.assertFalse(Path(str(archive) + ".tmp").exists())
            self.assertNotIn("compression complete", messages.getvalue())

    def test_existing_archive_is_not_recompressed(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "ref.clean.fa"
            archive = Path(str(source) + ".gz")
            archive.write_bytes(gzip.compress(b">1\nACGT\n"))
            with mock.patch.object(indexing, "tqdm") as progress:
                self.assertEqual(indexing._archive_file_to_gz(str(source)), str(archive))
                progress.assert_not_called()


if __name__ == "__main__":
    unittest.main()
