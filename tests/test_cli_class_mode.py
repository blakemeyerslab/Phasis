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
    def test_main_without_args_prints_help_and_exits_before_run_path(self):
        captured = io.StringIO()
        with (
            mock.patch.object(cli, "require_dependencies") as deps,
            mock.patch.object(cli, "run_pipeline") as pipeline,
            redirect_stdout(captured),
        ):
            result = cli.main([])

        self.assertEqual(result, 0)
        self.assertIn("usage:", captured.getvalue())
        self.assertIn("-libs", captured.getvalue())
        deps.assert_not_called()
        pipeline.assert_not_called()

    def test_classifier_defaults_to_gmm(self):
        parser = cli.build_parser()
        args = parser.parse_args([])
        captured = io.StringIO()
        with redirect_stdout(captured):
            cli._validate_args(args)

        self.assertEqual(args.classifier, "GMM")
        self.assertEqual(args.requested_classifier, "GMM")
        self.assertEqual(captured.getvalue(), "")

    def test_legacy_knn_classifier_is_accepted_but_ignored(self):
        parser = cli.build_parser()
        args = parser.parse_args(["-classifier", "KNN"])

        captured = io.StringIO()
        with redirect_stdout(captured):
            cli._validate_args(args)

        self.assertEqual(args.classifier, "GMM")
        self.assertEqual(args.requested_classifier, "KNN")
        self.assertIn("deprecated and ignored", captured.getvalue())

    def test_explicit_gmm_classifier_is_also_deprecated(self):
        parser = cli.build_parser()
        args = parser.parse_args(["-classifier", "GMM"])

        captured = io.StringIO()
        with redirect_stdout(captured):
            cli._validate_args(args)

        self.assertEqual(args.classifier, "GMM")
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

    def test_pool_libraries_preferred_option_sets_concat_runtime_flag(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--pool_libraries"])
        cli._validate_args(args)

        self.assertTrue(args.concat_libs)

    def test_concat_libs_legacy_alias_sets_same_runtime_flag(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--concat_libs"])
        cli._validate_args(args)

        self.assertTrue(args.concat_libs)

    def test_concat_libs_legacy_alias_is_hidden_from_main_help(self):
        help_text = cli.build_parser().format_help()

        self.assertIn("--pool_libraries", help_text)
        self.assertNotIn("--concat_libs", help_text)

    def test_compress_intermediates_is_default_with_visible_opt_out(self):
        parser = cli.build_parser()

        default_args = parser.parse_args([])
        disabled_args = parser.parse_args(["--no_compress_intermediates"])
        enabled_args = parser.parse_args(["--compress_intermediates"])

        self.assertTrue(default_args.compress_intermediates)
        self.assertFalse(disabled_args.compress_intermediates)
        self.assertTrue(enabled_args.compress_intermediates)
        help_text = parser.format_help()
        self.assertIn("--compress_intermediates", help_text)
        self.assertIn("--no_compress_intermediates", help_text)

    def test_fastq_chunk_size_option_is_validated_and_stored_in_runtime(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--fastq-chunk-unique-tags", "100000"])
        cli._validate_args(args)
        self.assertEqual(args.fastq_chunk_unique_tags, 100000)

        original_chunk_size = getattr(rt, "fastq_chunk_unique_tags", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ref = _touch(os.path.join(tmpdir, "ref.fa"))
                lib = _touch(os.path.join(tmpdir, "lib.fastq"))
                args = parser.parse_args(
                    [
                        "-libs",
                        lib,
                        "-reference",
                        ref,
                        "--fastq-chunk-unique-tags",
                        "100000",
                        "--outdir",
                        os.path.join(tmpdir, "results"),
                    ]
                )
                cli.configure_runtime(args)
                self.assertEqual(rt.fastq_chunk_unique_tags, 100000)
        finally:
            rt.fastq_chunk_unique_tags = original_chunk_size

        invalid_args = parser.parse_args(["--fastq-chunk-unique-tags", "0"])
        with self.assertRaises(SystemExit):
            cli._validate_args(invalid_args)

    def test_negative_cores_are_rejected(self):
        args = cli.build_parser().parse_args(["-cores", "-1"])
        with self.assertRaises(SystemExit):
            cli._validate_args(args)

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

    def test_configure_runtime_stores_compress_intermediates_flag(self):
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
                    "--no_compress_intermediates",
                    "--outdir",
                    os.path.join(tmpdir, "results"),
                ]
            )

            with mock.patch.object(rt, "compress_intermediates", True, create=True):
                cli.configure_runtime(args)
                self.assertFalse(rt.compress_intermediates)


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
