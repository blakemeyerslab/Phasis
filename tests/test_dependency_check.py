from __future__ import annotations

import unittest
from unittest import mock

from phasis.stages import dependency_check


class DependencyCheckTests(unittest.TestCase):
    def _run_check(self, version_info):
        with (
            mock.patch.object(dependency_check.sys, "version_info", version_info),
            mock.patch.object(dependency_check, "_has_executable", return_value=True),
            mock.patch.object(dependency_check, "validate_runtime_samtools"),
        ):
            return dependency_check.checkDependency()

    def test_accepts_supported_python_range(self):
        self.assertIsNone(self._run_check((3, 10, 0)))
        self.assertIsNone(self._run_check((3, 12, 0)))

    def test_rejects_python_313_and_newer(self):
        with self.assertRaises(SystemExit):
            self._run_check((3, 13, 0))


if __name__ == "__main__":
    unittest.main()
