from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from phasis import cache
from phasis.stages import feature_assembly


class MemfileDefaultTests(unittest.TestCase):
    def test_default_memfile_path_prefers_run_dir(self):
        with tempfile.TemporaryDirectory() as run_dir, tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(cache.rt, "run_dir", run_dir):
                with mock.patch.object(cache.rt, "outdir", outdir):
                    resolved = cache.default_memfile_path()

        self.assertEqual(resolved, os.path.join(run_dir, "phasis.mem"))

    def test_feature_assembly_get_memfile_ignores_outdir_fallback(self):
        with tempfile.TemporaryDirectory() as run_dir, tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(feature_assembly.rt, "memFile", None):
                with mock.patch.object(feature_assembly.rt, "run_dir", run_dir):
                    with mock.patch.object(feature_assembly.rt, "outdir", outdir):
                        resolved = feature_assembly._get_memfile()

        self.assertEqual(resolved, os.path.join(run_dir, "phasis.mem"))

    def test_feature_assembly_get_memfile_preserves_explicit_override(self):
        with tempfile.TemporaryDirectory() as run_dir:
            explicit = os.path.join(run_dir, "custom.mem")
            with mock.patch.object(feature_assembly.rt, "memFile", explicit):
                with mock.patch.object(feature_assembly.rt, "run_dir", run_dir):
                    resolved = feature_assembly._get_memfile()

        self.assertEqual(resolved, explicit)


if __name__ == "__main__":
    unittest.main()
