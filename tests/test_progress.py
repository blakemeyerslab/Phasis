from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

from phasis import progress


class ProgressOutputTests(unittest.TestCase):
    def test_real_bar_writes_to_redirected_stdout(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            with progress.tqdm(total=1, desc="stdout-check", mininterval=0) as bar:
                bar.update(1)

        self.assertIn("stdout-check", stream.getvalue())

    def test_tqdm_defaults_to_stdout(self):
        marker = object()
        with mock.patch.object(progress, "_tqdm", return_value=marker) as raw_tqdm:
            self.assertIs(progress.tqdm(total=1), marker)

        self.assertIs(raw_tqdm.call_args.kwargs["file"], sys.stdout)

    def test_tqdm_preserves_an_explicit_stream(self):
        stream = io.StringIO()
        with mock.patch.object(progress, "_tqdm") as raw_tqdm:
            progress.tqdm(total=1, file=stream)

        self.assertIs(raw_tqdm.call_args.kwargs["file"], stream)

    def test_all_tqdm_imports_use_the_stdout_wrapper(self):
        repo_root = Path(__file__).resolve().parents[1]
        direct_import = "from tqdm import tqdm"
        for path in list((repo_root / "phasis").rglob("*.py")) + list(
            (repo_root / "support_scripts").rglob("*.py")
        ):
            if path == repo_root / "phasis" / "progress.py":
                continue
            self.assertNotIn(direct_import, path.read_text(encoding="utf-8"), path)


if __name__ == "__main__":
    unittest.main()
