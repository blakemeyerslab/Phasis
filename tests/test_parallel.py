from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from phasis import parallel


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
                self.assertEqual(parallel.coreReserve(0), 7)

    def test_core_reserve_caps_explicit_request_to_visible_cpus(self):
        with mock.patch.object(parallel.multiprocessing, "cpu_count", return_value=64):
            with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "8"}, clear=True):
                self.assertEqual(parallel.coreReserve(20), 8)


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


if __name__ == "__main__":
    unittest.main()
