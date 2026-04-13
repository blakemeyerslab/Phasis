from __future__ import annotations

import configparser
import os
import tempfile
import unittest
from unittest import mock

from phasis import libprep
from phasis import runtime as rt
from phasis.cache import compute_md5_str
from phasis.stages import library_processing
from phasis.stages import mapping


def _serial_parallel_runner(func, data, **_kwargs):
    return [func(item) for item in data]


def _serial_ppbalance(func, rawinputs, **_kwargs):
    return [func(item) for item in rawinputs]


def _load_memcfg(mem_path):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(mem_path)
    return cfg


def _write_text(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _fake_mapper(aninput):
    alib, _geno_index, _nspread, _maxhits_local, _runtype_local = aninput
    bam_path = mapping._bam_output_for_fas(alib)
    with open(bam_path, "wb") as handle:
        handle.write(b"BAM\n")
    return bam_path


class LibraryProcessingCacheTests(unittest.TestCase):
    def test_library_processing_compat_lookup_accepts_legacy_plain_fasta_key(self):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg["FASTAS"] = {"/tmp/libA.fas": "legacy-fp"}

        self.assertEqual(
            library_processing._compat_fasta_fp(cfg, "/tmp/libA.fas.gz"),
            "legacy-fp",
        )

    def test_fresh_processed_library_is_archived_to_canonical_gz_and_reused_without_materialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_path = _write_text(
                os.path.join(tmpdir, "libA.tag"),
                "AAAA\t3\nCCCC\t2\n",
            )
            mem_path = os.path.join(tmpdir, "phasis.mem")

            runtime_patch = mock.patch.multiple(
                rt,
                create=True,
                run_dir=tmpdir,
                outdir=tmpdir,
                memFile=mem_path,
                mindepth=1,
                libformat="T",
                concat_libs=False,
            )
            with runtime_patch:
                with mock.patch.object(
                    library_processing,
                    "run_parallel_with_progress",
                    side_effect=_serial_parallel_runner,
                ):
                    outputs = library_processing.libraryprocess([lib_path])

                expected_fas = library_processing._fas_output_for_input(lib_path)
                expected_sum = library_processing._sum_output_for_fas(expected_fas)
                expected_gz = f"{expected_fas}.gz"

                self.assertEqual(outputs, [expected_fas])
                self.assertFalse(os.path.exists(expected_fas))
                self.assertTrue(os.path.isfile(expected_gz))
                self.assertTrue(os.path.isfile(expected_sum))

                cfg = _load_memcfg(mem_path)
                self.assertEqual(cfg["FASTAS"].get(expected_gz), compute_md5_str(expected_gz))
                self.assertIsNone(cfg["FASTAS"].get(expected_fas))
                self.assertEqual(
                    cfg[library_processing.LIBRARY_PROCESSING_SECTION].get(expected_fas),
                    compute_md5_str(expected_gz),
                )

                with mock.patch.object(
                    library_processing,
                    "_parallel_materialize_fas",
                    side_effect=AssertionError("cache hit should not materialize processed libraries"),
                ):
                    with mock.patch.object(
                        library_processing,
                        "_process_input_libraries",
                        side_effect=AssertionError("cache hit should not reprocess libraries"),
                    ):
                        with mock.patch.object(
                            library_processing,
                            "run_parallel_with_progress",
                            side_effect=_serial_parallel_runner,
                        ):
                            outputs = library_processing.libraryprocess([lib_path])

                self.assertEqual(outputs, [expected_fas])
                self.assertFalse(os.path.exists(expected_fas))
                self.assertTrue(os.path.isfile(expected_gz))

    def test_plain_only_processed_library_still_upgrades_legacy_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_path = _write_text(
                os.path.join(tmpdir, "legacy.tag"),
                "AAAA\t4\n",
            )
            mem_path = os.path.join(tmpdir, "phasis.mem")

            runtime_patch = mock.patch.multiple(
                rt,
                create=True,
                run_dir=tmpdir,
                outdir=tmpdir,
                memFile=mem_path,
                mindepth=1,
                libformat="T",
                concat_libs=False,
            )
            with runtime_patch:
                expected_fas = library_processing._fas_output_for_input(lib_path)
                os.makedirs(os.path.dirname(expected_fas), exist_ok=True)
                _write_text(expected_fas, ">seq_1|4\nAAAA\n")
                _write_text(library_processing._sum_output_for_fas(expected_fas), "legacy summary\n")

                cfg = configparser.ConfigParser()
                cfg.optionxform = str
                cfg["ADVANCED"] = {"mindepth": "1"}
                cfg["LIBRARIES"] = {lib_path: compute_md5_str(lib_path)}
                cfg["FASTAS"] = {expected_fas: compute_md5_str(expected_fas)}
                with open(mem_path, "w", encoding="utf-8") as handle:
                    cfg.write(handle)

                with mock.patch.object(
                    library_processing,
                    "_process_input_libraries",
                    side_effect=AssertionError("legacy cache hit should not reprocess libraries"),
                ):
                    with mock.patch.object(
                        library_processing,
                        "run_parallel_with_progress",
                        side_effect=_serial_parallel_runner,
                    ):
                        outputs = library_processing.libraryprocess([lib_path])

                self.assertEqual(outputs, [expected_fas])
                self.assertTrue(os.path.isfile(expected_fas))
                self.assertFalse(os.path.exists(f"{expected_fas}.gz"))

                cfg = _load_memcfg(mem_path)
                self.assertEqual(
                    cfg[library_processing.LIBRARY_PROCESSING_SECTION].get(expected_fas),
                    compute_md5_str(expected_fas),
                )

    def test_concat_libs_merges_directly_from_gz_only_processed_libraries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_a = _write_text(
                os.path.join(tmpdir, "libA.tag"),
                "AAAA\t3\nCCCC\t2\n",
            )
            lib_b = _write_text(
                os.path.join(tmpdir, "libB.tag"),
                "AAAA\t5\nGGGG\t4\n",
            )
            mem_path = os.path.join(tmpdir, "phasis.mem")

            runtime_patch = mock.patch.multiple(
                rt,
                create=True,
                run_dir=tmpdir,
                outdir=tmpdir,
                memFile=mem_path,
                mindepth=1,
                libformat="T",
                concat_libs=True,
            )
            with runtime_patch:
                with mock.patch.object(
                    library_processing,
                    "run_parallel_with_progress",
                    side_effect=_serial_parallel_runner,
                ):
                    outputs = library_processing.libraryprocess([lib_a, lib_b])

                merged_path = os.path.join(tmpdir, "processed_libraries", "ALL_LIBS.fas")
                merged_gz = f"{merged_path}.gz"

                self.assertEqual(outputs, [merged_path])
                self.assertFalse(os.path.exists(merged_path))
                self.assertTrue(os.path.isfile(merged_gz))

                for lib_path in (lib_a, lib_b):
                    expected_fas = library_processing._fas_output_for_input(lib_path)
                    self.assertFalse(os.path.exists(expected_fas))
                    self.assertTrue(os.path.isfile(f"{expected_fas}.gz"))

                merged_counts = dict(libprep.fas_records(merged_gz))
                self.assertEqual(merged_counts, {"AAAA": 8, "CCCC": 2, "GGGG": 4})


class MappingCacheTests(unittest.TestCase):
    def test_mapping_compat_lookup_accepts_legacy_plain_fasta_key(self):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg["FASTAS"] = {"/tmp/libB.fas": "legacy-fp"}

        self.assertEqual(mapping._compat_fasta_fp(cfg, "/tmp/libB.fas.gz"), "legacy-fp")

    def test_mapprocess_cleans_only_materialized_plain_fastas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "phasis.mem")
            gz_only_fas = os.path.join(tmpdir, "libA.fas")
            plain_fas = os.path.join(tmpdir, "libB.fas")

            _write_text(gz_only_fas, ">seq_1|3\nAAAA\n")
            library_processing._archive_fas_to_gz(gz_only_fas)
            _write_text(plain_fas, ">seq_1|5\nCCCC\n")

            runtime_patch = mock.patch.multiple(
                rt,
                create=True,
                run_dir=tmpdir,
                outdir=tmpdir,
                memFile=mem_path,
                mismat=0,
                runtype="G",
                maxhits=25,
                clustbuffer=150,
                phase=21,
                reference=os.path.join(tmpdir, "ref.fa"),
            )
            with runtime_patch:
                _write_text(rt.reference, ">chr1\nAAAA\n")
                with mock.patch.object(
                    mapping,
                    "run_parallel_with_progress",
                    side_effect=_serial_parallel_runner,
                ):
                    with mock.patch.object(mapping, "PPBalance", side_effect=_serial_ppbalance):
                        with mock.patch.object(mapping, "optimize", return_value=(1, 1)):
                            with mock.patch.object(mapping, "mapper", side_effect=_fake_mapper):
                                outputs = mapping.mapprocess(
                                    [gz_only_fas, plain_fas],
                                    genoIndex=os.path.join(tmpdir, "index", "ref"),
                                    ncores_local=2,
                                )

            gz_only_bam = mapping._bam_output_for_fas(gz_only_fas)
            plain_bam = mapping._bam_output_for_fas(plain_fas)

            self.assertEqual(sorted(outputs), sorted([gz_only_bam, plain_bam]))
            self.assertTrue(os.path.isfile(gz_only_bam))
            self.assertTrue(os.path.isfile(plain_bam))
            self.assertFalse(os.path.exists(gz_only_fas))
            self.assertTrue(os.path.isfile(f"{gz_only_fas}.gz"))
            self.assertTrue(os.path.isfile(plain_fas))
            self.assertFalse(os.path.exists(f"{plain_fas}.gz"))

            cfg = _load_memcfg(mem_path)
            self.assertEqual(cfg["FASTAS"].get(f"{gz_only_fas}.gz"), compute_md5_str(f"{gz_only_fas}.gz"))
            self.assertEqual(cfg["FASTAS"].get(plain_fas), compute_md5_str(plain_fas))


if __name__ == "__main__":
    unittest.main()
