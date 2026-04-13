from __future__ import annotations

import os
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


if __name__ == "__main__":
    unittest.main()
