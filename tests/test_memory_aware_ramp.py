from __future__ import annotations

import gzip
import os
import tempfile
import unittest
from unittest import mock

from phasis import cache, parallel
from phasis import runtime as rt


def _double(value):
    return value * 2


class MemorySupportedWorkersTests(unittest.TestCase):
    def test_half_of_the_available_memory_bounds_the_worker_count(self):
        with mock.patch.object(parallel, "_available_memory_bytes", return_value=16 * 10**9), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("Phasis_MEMORY_AWARE_RAMP", None)
            self.assertEqual(parallel.memory_supported_workers(1 * 10**9), 8)
            self.assertEqual(parallel.memory_supported_workers(3 * 10**9), 2)

    def test_unknown_values_and_the_switch_give_no_answer(self):
        with mock.patch.object(parallel, "_available_memory_bytes", return_value=None):
            self.assertIsNone(parallel.memory_supported_workers(10**9))
        with mock.patch.object(parallel, "_available_memory_bytes", return_value=16 * 10**9):
            self.assertIsNone(parallel.memory_supported_workers(None))
            self.assertIsNone(parallel.memory_supported_workers(0))
            with mock.patch.dict(os.environ, {"Phasis_MEMORY_AWARE_RAMP": "0"}):
                self.assertIsNone(parallel.memory_supported_workers(10**9))


class MemoryAwareGrowthTests(unittest.TestCase):
    def _run(self, supported, **kwargs):
        slices = []

        class _FakePool:
            _pool = []

            def __init__(self, nworkers):
                self.nworkers = nworkers

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def imap_unordered(self, func, iterable, chunksize=1):
                items = list(iterable)
                slices.append((self.nworkers, len(items)))
                return [func(item) for item in items]

        def fake_make_pool(nworkers, **_kwargs):
            return _FakePool(nworkers)

        with mock.patch.object(parallel, "make_pool", side_effect=fake_make_pool), \
                mock.patch.object(parallel, "_pool_peak_rss_bytes", return_value=10**9), \
                mock.patch.object(parallel, "memory_supported_workers", return_value=supported), \
                mock.patch.object(rt, "ncores", 12, create=True):
            results = parallel.run_parallel_with_progress(
                _double, list(range(24)), unit="lib-chr", show_progress=False,
                initial_worker_cap=2, max_worker_cap=8, initial_chunk_size=2, max_chunk_size=8,
                adaptive_recovery=True, **kwargs,
            )
        return results, slices

    def test_results_are_the_same_with_and_without_the_jump(self):
        stepwise, stepwise_slices = self._run(None, memory_aware_growth=True)
        jumped, jumped_slices = self._run(64, memory_aware_growth=True)
        self.assertEqual(sorted(stepwise), [v * 2 for v in range(24)])
        self.assertEqual(sorted(jumped), sorted(stepwise))
        # Stepwise: 2, 2, 4, 4, 8, ...; with enough memory: 2, then 8 at once.
        self.assertEqual(stepwise_slices[:5], [(2, 2), (2, 2), (4, 4), (4, 4), (8, 8)])
        self.assertEqual(jumped_slices[:2], [(2, 2), (8, 8)])

    def test_too_little_memory_and_callers_that_did_not_opt_in_keep_the_steps(self):
        _res, small = self._run(3, memory_aware_growth=True)
        self.assertEqual(small[:3], [(2, 2), (2, 2), (4, 4)])
        _res, not_opted = self._run(64)
        self.assertEqual(not_opted[:3], [(2, 2), (2, 2), (4, 4)])


class IntermediateGzipLevelTests(unittest.TestCase):
    def test_level_changes_the_bytes_and_not_the_content(self):
        text = "".join(f"lib{n % 7}\tcluster{n}\t{n * 31 % 977}\n" for n in range(20000))
        sizes = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            for level in ("9", "4"):
                path = os.path.join(tmpdir, f"table_{level}.tab")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text)
                with mock.patch.dict(os.environ, {"Phasis_INTERMEDIATE_GZIP_LEVEL": level}):
                    gz_path = cache.gzip_text_artifact(path)
                self.assertFalse(os.path.exists(path))
                with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), text)
                sizes[level] = os.path.getsize(gz_path)
        self.assertLessEqual(sizes["9"], sizes["4"])

    def test_default_and_invalid_values(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("Phasis_INTERMEDIATE_GZIP_LEVEL", None)
            os.environ.pop("PHASIS_INTERMEDIATE_GZIP_LEVEL", None)
            self.assertEqual(cache.intermediate_gzip_level(), 4)
        for bad in ("0", "12", "fast"):
            with mock.patch.dict(os.environ, {"Phasis_INTERMEDIATE_GZIP_LEVEL": bad}):
                self.assertEqual(cache.intermediate_gzip_level(), 4)


if __name__ == "__main__":
    unittest.main()
