from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from phasis import cli
from phasis import runtime as rt
from phasis.config import Phase2Config
from phasis.stages import phase2_pipeline


def _touch(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# test\n")
    return path


class CliReferenceIdModeTests(unittest.TestCase):
    def test_classifier_defaults_to_gmm(self):
        parser = cli.build_parser()
        args = parser.parse_args([])
        cli._validate_args(args)

        self.assertEqual(args.classifier, "GMM")
        self.assertEqual(args.classifier_aliases, ["GMM"])
        self.assertEqual(args.requested_classifier, "GMM")

    def test_legacy_knn_classifier_is_accepted_as_alias_only(self):
        parser = cli.build_parser()
        args = parser.parse_args(["-classifier", "KNN"])

        captured = io.StringIO()
        with redirect_stdout(captured):
            cli._validate_args(args)

        self.assertEqual(args.classifier, "GMM")
        self.assertEqual(args.classifier_aliases, ["GMM", "KNN"])
        self.assertEqual(args.requested_classifier, "KNN")
        self.assertIn("deprecated and ignored", captured.getvalue())

    def test_invalid_classifier_still_errors(self):
        parser = cli.build_parser()
        args = parser.parse_args(["-classifier", "SVM"])

        with self.assertRaises(SystemExit):
            cli._validate_args(args)

    def test_runtype_compatibility_maps_to_reference_id_mode(self):
        parser = cli.build_parser()

        args = parser.parse_args(["-runtype", "G"])
        cli._validate_args(args)
        self.assertEqual(cli._reference_id_mode_from_args(args), "numeric")

        args = parser.parse_args(["-runtype", "T"])
        cli._validate_args(args)
        self.assertEqual(cli._reference_id_mode_from_args(args), "preserve")

        args = parser.parse_args(["-runtype", "S"])
        cli._validate_args(args)
        self.assertEqual(cli._reference_id_mode_from_args(args), "preserve")

    def test_reference_id_mode_overrides_runtype_alias(self):
        parser = cli.build_parser()
        args = parser.parse_args(["-runtype", "G", "--reference_id_mode", "preserve"])
        cli._validate_args(args)

        self.assertEqual(cli._reference_id_mode_from_args(args), "preserve")

    def test_class_cluster_files_alias_populates_compatible_field(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            ["-steps", "class", "-class_cluster_files", "a.candidate.clusters", "b.candidate.clusters"]
        )
        cli._validate_args(args)

        self.assertEqual(
            args.class_cluster_file,
            ["a.candidate.clusters", "b.candidate.clusters"],
        )

    def test_class_cluster_files_warns_when_not_class_mode(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            ["-steps", "cfind", "-class_cluster_file", "a.candidate.clusters"]
        )

        captured = io.StringIO()
        with redirect_stdout(captured):
            cli._validate_args(args)

        self.assertIn("only used with -steps class", captured.getvalue())

    def test_configure_runtime_stores_reference_id_mode(self):
        parser = cli.build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = _touch(os.path.join(tmpdir, "ref.fa"))
            lib = _touch(os.path.join(tmpdir, "lib.tag"))
            args = parser.parse_args(
                [
                    "-libs",
                    lib,
                    "-reference",
                    ref,
                    "-runtype",
                    "T",
                    "--outdir",
                    os.path.join(tmpdir, "results"),
                ]
            )

            with mock.patch.object(rt, "reference_id_mode", None, create=True):
                cli.configure_runtime(args)
                self.assertEqual(rt.reference_id_mode, "preserve")


class ClassClusterInferenceTests(unittest.TestCase):
    def _cfg(self, *, tmpdir, phase="21", libs=None, concat=False, class_files=None):
        return Phase2Config(
            phase=phase,
            outdir=os.path.join(tmpdir, "results"),
            concat_libs=concat,
            libs=libs or [],
            steps="class",
            class_cluster_file=class_files,
        )

    def test_explicit_class_cluster_file_is_used_even_when_cross_phase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            explicit = _touch(os.path.join(tmpdir, "manual.21-PHAS.candidate.clusters"))
            cfg = self._cfg(tmpdir=tmpdir, phase="24", class_files=[explicit])

            self.assertEqual(phase2_pipeline.infer_class_cluster_files(cfg), [explicit])

    def test_missing_explicit_class_cluster_file_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "missing.21-PHAS.candidate.clusters")
            cfg = self._cfg(tmpdir=tmpdir, class_files=[missing])

            with self.assertRaisesRegex(FileNotFoundError, "Explicit -class_cluster_file"):
                phase2_pipeline.infer_class_cluster_files(cfg)

    def test_infers_non_concat_cluster_files_from_library_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_a = _touch(os.path.join(tmpdir, "libA.tag.gz"))
            lib_b = _touch(os.path.join(tmpdir, "libB.fas"))
            expected_a = _touch(os.path.join(tmpdir, "libA.21-PHAS.candidate.clusters"))
            expected_b = _touch(os.path.join(tmpdir, "libB.21-PHAS.candidate.clusters"))
            cfg = self._cfg(tmpdir=tmpdir, libs=[lib_a, lib_b])

            with mock.patch.object(rt, "run_dir", tmpdir, create=True):
                resolved = phase2_pipeline.infer_class_cluster_files(cfg)

            self.assertEqual(resolved, [expected_a, expected_b])

    def test_infers_concat_cluster_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_a = _touch(os.path.join(tmpdir, "libA.tag"))
            lib_b = _touch(os.path.join(tmpdir, "libB.tag"))
            expected = _touch(os.path.join(tmpdir, "ALL_LIBS.24-PHAS.candidate.clusters"))
            cfg = self._cfg(tmpdir=tmpdir, phase="24", libs=[lib_a, lib_b], concat=True)

            with mock.patch.object(rt, "run_dir", tmpdir, create=True):
                resolved = phase2_pipeline.infer_class_cluster_files(cfg)

            self.assertEqual(resolved, [expected])

    def test_missing_inferred_cluster_file_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_a = _touch(os.path.join(tmpdir, "libA.fastq"))
            cfg = self._cfg(tmpdir=tmpdir, libs=[lib_a])

            with mock.patch.object(rt, "run_dir", tmpdir, create=True):
                with self.assertRaisesRegex(FileNotFoundError, "Run -steps cfind first"):
                    phase2_pipeline.infer_class_cluster_files(cfg)


if __name__ == "__main__":
    unittest.main()
