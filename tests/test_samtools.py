from __future__ import annotations

from io import StringIO
import subprocess
import tempfile
import unittest
from unittest import mock

from phasis import runtime as rt
from phasis.samtools import SamtoolsError, validate_samtools
from phasis.stages import mapping, sam_parsing


class SamtoolsValidationTests(unittest.TestCase):
    def _completed_process(self, output: str, returncode: int = 0):
        return subprocess.CompletedProcess(
            args=["samtools", "--version"], returncode=returncode, stdout=output, stderr=""
        )

    @mock.patch("phasis.samtools.subprocess.run")
    @mock.patch("phasis.samtools.shutil.which")
    def test_accepts_supported_version_and_preserves_path_with_spaces(self, which, run):
        executable = "/opt/conda environments/phasis/bin/samtools"
        which.return_value = executable
        run.return_value = self._completed_process("samtools 1.18\n")

        info = validate_samtools(executable)

        self.assertEqual(info.path, executable)
        self.assertEqual(info.version, (1, 18))
        self.assertEqual(run.call_args.args[0], [executable, "--version"])

    @mock.patch("phasis.samtools.subprocess.run")
    @mock.patch("phasis.samtools.shutil.which", return_value="/conda/bin/samtools")
    def test_rejects_old_version_with_actionable_message(self, _which, run):
        run.return_value = self._completed_process("samtools 1.9\n")

        with self.assertRaisesRegex(SamtoolsError, "requires samtools >= 1.10"):
            validate_samtools()

    @mock.patch("phasis.samtools.subprocess.run")
    @mock.patch("phasis.samtools.shutil.which", return_value="/conda/bin/samtools")
    def test_rejects_unparseable_version_output(self, _which, run):
        run.return_value = self._completed_process("unexpected executable output\n")

        with self.assertRaisesRegex(SamtoolsError, "Raw 'samtools --version' output"):
            validate_samtools()

    @mock.patch("phasis.samtools.shutil.which", return_value=None)
    def test_rejects_missing_executable_with_path_guidance(self, _which):
        with self.assertRaisesRegex(SamtoolsError, "which samtools"):
            validate_samtools()


class SamtoolsDownstreamPathTests(unittest.TestCase):
    def setUp(self):
        self.original_path = getattr(rt, "samtools_path", None)
        rt.samtools_path = "/tools with spaces/bin/samtools"

    def tearDown(self):
        rt.samtools_path = self.original_path

    @mock.patch("phasis.stages.mapping.subprocess.call", return_value=0)
    def test_mapper_reuses_validated_path(self, call):
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = f"{tmpdir}/sample.fas"
            output = mapping.mapper((fasta, "reference-index", 2, 10, "preserve"))

        self.assertTrue(output.endswith(".bam"))
        commands = [entry.args[0] for entry in call.call_args_list]
        self.assertEqual(commands[1][0], rt.samtools_path)
        self.assertEqual(commands[2][0], rt.samtools_path)

    @mock.patch("phasis.stages.sam_parsing.subprocess.Popen")
    def test_bam_parser_reuses_validated_path(self, popen):
        process = mock.Mock()
        process.stdout = StringIO("read\t0\n")
        process.stderr = StringIO("")
        process.wait.return_value = 0
        popen.return_value = process

        self.assertEqual(list(sam_parsing._iter_alignment_lines("sample.bam")), ["read\t0\n"])
        self.assertEqual(popen.call_args.args[0], [rt.samtools_path, "view", "sample.bam"])


if __name__ == "__main__":
    unittest.main()
