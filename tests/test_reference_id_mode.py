from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from phasis import runtime as rt
from phasis.stages import indexing


def _write_reference(path: str) -> str:
    seq = "A" * 220
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f">chr01\n{seq}\n>Mt\n{seq}\n")
    return path


def _read_headers(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line[1:].strip() for line in handle if line.startswith(">")]


class ReferenceIdModeTests(unittest.TestCase):
    def test_numeric_reference_id_mode_renumbers_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = _write_reference(os.path.join(tmpdir, "ref.fa"))
            with mock.patch.object(rt, "reference", ref, create=True):
                with mock.patch.object(rt, "outdir", tmpdir, create=True):
                    with mock.patch.object(rt, "cores", 1, create=True):
                        with mock.patch.object(rt, "runtype", "G", create=True):
                            with mock.patch.object(rt, "reference_id_mode", "numeric", create=True):
                                with mock.patch.object(rt, "mindepth", 1, create=True):
                                    with mock.patch.object(rt, "clustbuffer", 300, create=True):
                                        with mock.patch.object(rt, "maxhits", 25, create=True):
                                            with mock.patch.object(rt, "mismat", 0, create=True):
                                                with mock.patch.object(rt, "memFile", os.path.join(tmpdir, "phasis.mem"), create=True):
                                                    with mock.patch.object(rt, "run_dir", tmpdir, create=True):
                                                        cwd = os.getcwd()
                                                        try:
                                                            os.chdir(tmpdir)
                                                            clean, _summary = indexing.refClean(ref)
                                                        finally:
                                                            os.chdir(cwd)

            self.assertEqual(_read_headers(clean), ["1", "2"])

    def test_preserve_reference_id_mode_keeps_headers_even_with_legacy_g_runtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = _write_reference(os.path.join(tmpdir, "ref.fa"))
            with mock.patch.object(rt, "reference", ref, create=True):
                with mock.patch.object(rt, "outdir", tmpdir, create=True):
                    with mock.patch.object(rt, "cores", 1, create=True):
                        with mock.patch.object(rt, "runtype", "G", create=True):
                            with mock.patch.object(rt, "reference_id_mode", "preserve", create=True):
                                with mock.patch.object(rt, "mindepth", 1, create=True):
                                    with mock.patch.object(rt, "clustbuffer", 300, create=True):
                                        with mock.patch.object(rt, "maxhits", 25, create=True):
                                            with mock.patch.object(rt, "mismat", 0, create=True):
                                                with mock.patch.object(rt, "memFile", os.path.join(tmpdir, "phasis.mem"), create=True):
                                                    with mock.patch.object(rt, "run_dir", tmpdir, create=True):
                                                        cwd = os.getcwd()
                                                        try:
                                                            os.chdir(tmpdir)
                                                            clean, _summary = indexing.refClean(ref)
                                                        finally:
                                                            os.chdir(cwd)

            self.assertEqual(_read_headers(clean), ["chr01", "Mt"])


if __name__ == "__main__":
    unittest.main()
