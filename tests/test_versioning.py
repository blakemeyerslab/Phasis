from __future__ import annotations

import argparse
import pathlib
import tomllib
import unittest

from phasis import __version__
from phasis.cli import build_parser


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class VersioningTests(unittest.TestCase):
    def test_package_version_is_2_7(self):
        self.assertEqual(__version__, "2.7")

    def test_cli_version_reports_2_7(self):
        parser = build_parser()
        version_actions = [action for action in parser._actions if isinstance(action, argparse._VersionAction)]
        self.assertEqual(len(version_actions), 1)
        self.assertIn("2.7", version_actions[0].version)

    def test_pyproject_and_readme_versions_match(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(pyproject["project"]["version"], "2.7")
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("**Version:** v2.7", readme_text)


if __name__ == "__main__":
    unittest.main()
