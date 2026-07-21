from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from phasis import parallel
from phasis import runtime as rt
from phasis.stages import indexing


class _FakePool:
    def __init__(self, nworkers, worker_counts, *, fail_once_at=None):
        self.nworkers = nworkers
        self.worker_counts = worker_counts
        self.fail_once_at = fail_once_at

    def __enter__(self):
        self.worker_counts.append(self.nworkers)
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def imap_unordered(self, func, iterable, chunksize=1):
        if self.fail_once_at is not None and self.nworkers == self.fail_once_at[0]:
            self.fail_once_at[0] = None
            raise MemoryError("simulated worker-memory failure")
        for item in iterable:
            yield func(item)


class ParallelCpuTests(unittest.TestCase):
    def test_scheduler_cpu_limit_prefers_first_available_variable(self):
        env = {
            "SLURM_CPUS_PER_TASK": "12",
            "PBS_NP": "8",
        }
        self.assertEqual(parallel._scheduler_cpu_limit(env=env), 12)

    def test_core_reserve_respects_scheduler_visible_cpu_limit(self):
        with mock.patch.object(parallel.multiprocessing, "cpu_count", return_value=64):
            with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "8"}, clear=True):
                self.assertEqual(parallel.coreReserve(0), 8)

    def test_core_reserve_caps_explicit_request_to_visible_cpus(self):
        with mock.patch.object(parallel.multiprocessing, "cpu_count", return_value=64):
            with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "8"}, clear=True):
                self.assertEqual(parallel.coreReserve(20), 8)

    def test_core_reserve_honors_an_explicit_request_without_scaling(self):
        with mock.patch.object(parallel.multiprocessing, "cpu_count", return_value=64):
            with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "8"}, clear=True):
                self.assertEqual(parallel.coreReserve(4), 4)

    def test_core_reserve_never_returns_zero_for_a_single_visible_cpu(self):
        with mock.patch.object(parallel.multiprocessing, "cpu_count", return_value=1):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(parallel.coreReserve(0), 1)

    def test_index_stage_uses_resolved_core_count_for_cores_zero(self):
        original_ncores = indexing.ncores
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch.multiple(
                    rt,
                    create=True,
                    cores=0,
                    ncores=8,
                    reference=os.path.join(tmpdir, "reference.fa"),
                    outdir=tmpdir,
                    runtype="G",
                    reference_id_mode="numeric",
                    mindepth=1,
                    clustbuffer=300,
                    maxhits=25,
                    mismat=0,
                    memFile=os.path.join(tmpdir, "phasis.mem"),
                ):
                    indexing.sync_from_runtime()
                    self.assertEqual(indexing.ncores, 8)
        finally:
            indexing.ncores = original_ncores

    def test_index_builder_never_passes_zero_threads_to_hisat2(self):
        original_cwd = os.getcwd()
        original_ncores = indexing.ncores
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                reference = os.path.join(tmpdir, "reference.fa")
                clean_reference = os.path.join(tmpdir, "reference.clean.fa")
                summary = os.path.join(tmpdir, "reference.summ.txt")
                for path in (reference, clean_reference, summary):
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write("test\n")

                os.chdir(tmpdir)
                with mock.patch.multiple(
                    rt,
                    create=True,
                    cores=0,
                    ncores=8,
                    reference=reference,
                    outdir=tmpdir,
                    runtype="G",
                    reference_id_mode="numeric",
                    mindepth=1,
                    clustbuffer=300,
                    maxhits=25,
                    mismat=0,
                    memFile=os.path.join(tmpdir, "phasis.mem"),
                ):
                    with mock.patch.object(indexing, "refClean", return_value=(clean_reference, summary)):
                        with mock.patch.object(indexing.subprocess, "call", return_value=1) as run_hisat2:
                            with self.assertRaises(SystemExit):
                                indexing.indexBuilder(reference, 8)

                self.assertEqual(run_hisat2.call_args.args[0][0:3], ["hisat2-build", "-p", "8"])
        finally:
            os.chdir(original_cwd)
            indexing.ncores = original_ncores


class ParallelPycacheTests(unittest.TestCase):
    def test_worker_pycache_prefix_uses_configured_preferred_env(self):
        original_prefix = parallel.sys.pycache_prefix
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                configured = os.path.join(tmpdir, "worker-pycache")
                env = {"Phasis_PYCACHE_PREFIX": configured}

                resolved = parallel._ensure_worker_pycache_prefix(env=env)

                self.assertEqual(resolved, os.path.abspath(configured))
                self.assertEqual(env["PYTHONPYCACHEPREFIX"], os.path.abspath(configured))
                self.assertTrue(os.path.isdir(configured))
        finally:
            parallel.sys.pycache_prefix = original_prefix

    def test_worker_pycache_prefix_can_be_disabled(self):
        original_prefix = parallel.sys.pycache_prefix
        try:
            env = {"Phasis_PYCACHE_PREFIX": "off"}

            resolved = parallel._ensure_worker_pycache_prefix(env=env)

            self.assertIsNone(resolved)
            self.assertNotIn("PYTHONPYCACHEPREFIX", env)
        finally:
            parallel.sys.pycache_prefix = original_prefix

    def test_worker_pycache_prefix_defaults_to_scratch(self):
        original_prefix = parallel.sys.pycache_prefix
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                env = {"TMPDIR": tmpdir}

                resolved = parallel._ensure_worker_pycache_prefix(env=env)

                self.assertTrue(resolved.startswith(os.path.abspath(tmpdir)))
                self.assertEqual(env["PYTHONPYCACHEPREFIX"], resolved)
                self.assertTrue(os.path.isdir(resolved))
        finally:
            parallel.sys.pycache_prefix = original_prefix


class LibraryWorkerCapTests(unittest.TestCase):
    def setUp(self):
        self.original_format = getattr(rt, "libformat", None)
        self.original_cap = getattr(rt, "parallel_lib_worker_cap", None)

    def tearDown(self):
        rt.libformat = self.original_format
        rt.parallel_lib_worker_cap = self.original_cap

    def test_fastq_uses_an_adaptive_eight_worker_default_cap(self):
        rt.libformat = "Q"
        rt.parallel_lib_worker_cap = None
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                parallel.resolve_library_worker_cap(8),
                (8, "adaptive FASTQ default (starts at 1)"),
            )

    def test_explicit_cap_is_honored_and_capped_by_cores(self):
        rt.libformat = "Q"
        rt.parallel_lib_worker_cap = None
        with mock.patch.dict(os.environ, {"PHASIS_LIB_WORKER_CAP": "12"}, clear=True):
            self.assertEqual(parallel.resolve_library_worker_cap(4), (4, "PHASIS_LIB_WORKER_CAP"))


class AdaptiveParallelTests(unittest.TestCase):
    def setUp(self):
        self.original_ncores = getattr(rt, "ncores", None)

    def tearDown(self):
        rt.ncores = self.original_ncores

    def test_conservative_library_start_grows_after_successful_batches(self):
        worker_counts = []
        rt.ncores = 4

        def fake_make_pool(nworkers, **_kwargs):
            return _FakePool(nworkers, worker_counts)

        with mock.patch.object(parallel, "make_pool", side_effect=fake_make_pool):
            results = parallel.run_parallel_with_progress(
                lambda value: value * 2,
                list(range(10)),
                unit="lib",
                initial_worker_cap=1,
                max_worker_cap=4,
                initial_chunk_size=1,
                max_chunk_size=4,
                recovery_progress_fraction=0.0,
            )

        self.assertEqual(results, [value * 2 for value in range(10)])
        self.assertEqual(worker_counts, [1, 1, 2, 2, 4])

    def test_explicit_fastq_growth_steps_follow_the_eight_worker_ramp(self):
        worker_counts = []
        rt.ncores = 8

        def fake_make_pool(nworkers, **_kwargs):
            return _FakePool(nworkers, worker_counts)

        with mock.patch.object(parallel, "make_pool", side_effect=fake_make_pool):
            results = parallel.run_parallel_with_progress(
                lambda value: value,
                list(range(21)),
                unit="lib",
                initial_worker_cap=1,
                max_worker_cap=8,
                initial_chunk_size=1,
                max_chunk_size=8,
                recovery_success_slices=1,
                recovery_progress_fraction=0.0,
                recovery_growth_steps=parallel.FASTQ_DYNAMIC_LIBRARY_WORKER_STEPS,
            )

        self.assertEqual(results, list(range(21)))
        self.assertEqual(worker_counts, [1, 2, 4, 6, 8])

    def test_worker_memory_failure_reduces_parallelism_and_retries(self):
        worker_counts = []
        fail_once_at = [4]
        rt.ncores = 4

        def fake_make_pool(nworkers, **_kwargs):
            return _FakePool(nworkers, worker_counts, fail_once_at=fail_once_at)

        with mock.patch.object(parallel, "make_pool", side_effect=fake_make_pool):
            results = parallel.run_parallel_with_progress(
                lambda value: value,
                list(range(10)),
                unit="lib",
                initial_worker_cap=1,
                max_worker_cap=4,
                initial_chunk_size=1,
                max_chunk_size=4,
                recovery_progress_fraction=0.0,
            )

        self.assertEqual(results, list(range(10)))
        self.assertEqual(worker_counts, [1, 1, 2, 2, 4, 2])

    def test_worker_task_failure_retries_serially_then_reduces_parallelism(self):
        worker_counts = []
        attempts = {}
        rt.ncores = 4

        def fake_make_pool(nworkers, **_kwargs):
            return _FakePool(nworkers, worker_counts)

        def intermittently_failing_task(value):
            attempts[value] = attempts.get(value, 0) + 1
            if value == 6 and attempts[value] == 1:
                raise MemoryError("simulated task-memory failure")
            return value

        with mock.patch.object(parallel, "make_pool", side_effect=fake_make_pool):
            results = parallel.run_parallel_with_progress(
                intermittently_failing_task,
                list(range(12)),
                unit="lib",
                initial_worker_cap=1,
                max_worker_cap=4,
                initial_chunk_size=1,
                max_chunk_size=4,
                recovery_progress_fraction=0.0,
            )

        self.assertEqual(results, list(range(12)))
        self.assertEqual(attempts[6], 2)
        self.assertEqual(worker_counts, [1, 1, 2, 2, 4, 2])


if __name__ == "__main__":
    unittest.main()
