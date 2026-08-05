from __future__ import annotations

import configparser
import os
import tempfile
import unittest
from unittest import mock

from phasis import libprep
from phasis import runtime as rt
from phasis.cache import compute_md5_str, sig_key
from phasis.stages import cluster_build
from phasis.stages import library_processing
from phasis.stages import mapping
from phasis.stages import sam_parsing


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
    def test_mindepth_metadata_updates_after_cache_evaluation(self):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg["ADVANCED"] = {"mindepth": "2", "maxhits": "25"}

        original_mindepth = library_processing.mindepth
        try:
            library_processing.mindepth = 1
            self.assertTrue(library_processing._record_mindepth_metadata(cfg))
        finally:
            library_processing.mindepth = original_mindepth

        self.assertEqual(cfg["ADVANCED"]["mindepth"], "1")
        self.assertEqual(cfg["ADVANCED"]["maxhits"], "25")

    def test_fastq_processing_requests_a_conservative_adaptive_worker_ramp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            libraries = [
                _write_text(os.path.join(tmpdir, f"lib{index}.fastq"), "@r\n" + "A" * 21 + "\n+\n" + "I" * 21 + "\n")
                for index in range(4)
            ]
            captured = {}

            def fake_runner(_func, jobs, **kwargs):
                captured.update(kwargs)
                outputs = []
                for _lib, output in jobs:
                    os.makedirs(os.path.dirname(output), exist_ok=True)
                    _write_text(output, ">seq_1|1\n" + "A" * 21 + "\n")
                    outputs.append(output)
                return outputs

            with mock.patch.multiple(
                rt,
                create=True,
                ncores=8,
                libformat="Q",
                parallel_lib_worker_cap=None,
                run_dir=tmpdir,
            ):
                with mock.patch.object(library_processing, "libformat", "Q"):
                    with mock.patch.dict(os.environ, {"PHASIS_LIB_WORKER_CAP": ""}, clear=False):
                        with mock.patch.object(
                            library_processing,
                            "run_parallel_with_progress",
                            side_effect=fake_runner,
                        ):
                            outputs = library_processing._process_input_libraries(libraries)

            self.assertEqual(len(outputs), 4)
            self.assertEqual(captured["initial_worker_cap"], 1)
            self.assertEqual(captured["max_worker_cap"], 8)
            self.assertEqual(captured["initial_chunk_size"], 1)
            self.assertEqual(captured["max_chunk_size"], 8)
            self.assertEqual(captured["recovery_success_slices"], 1)
            self.assertEqual(captured["recovery_progress_fraction"], 0.0)
            self.assertEqual(captured["recovery_growth_steps"], (1, 2, 4, 6, 8))

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
    def test_cluster_build_modern_signature_miss_cannot_fall_back_to_legacy_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lclust_path = _write_text(os.path.join(tmpdir, "libA.chr1.lclust"), "cluster\n")
            current_fp = compute_md5_str(lclust_path)

            modern_miss = cluster_build.inspect_cluster_cache_entry(
                (
                    "libA.chr1",
                    lclust_path,
                    "new-input-signature",
                    current_fp,
                    "old-input-signature",
                    current_fp,
                )
            )
            legacy_only = cluster_build.inspect_cluster_cache_entry(
                (
                    "libA.chr1",
                    lclust_path,
                    "new-input-signature",
                    "",
                    "",
                    current_fp,
                )
            )

            self.assertEqual(modern_miss[2], "rebuild")
            self.assertEqual(legacy_only[2], "rebuild")

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
                reference_id_mode="numeric",
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

    def test_maxhits_change_remaps_and_reparses_despite_matching_legacy_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "phasis.mem")
            fas_path = _write_text(os.path.join(tmpdir, "libA.fas"), ">seq_1|3\nAAAA\n")
            reference_path = _write_text(os.path.join(tmpdir, "ref.fa"), ">chr1\nAAAA\n")
            geno_index = os.path.join(tmpdir, "index", "ref")

            def fake_parser(aninput):
                alignment_path, _maxhits_local, _mismat_local = aninput
                stem = alignment_path.rpartition(".")[0]
                dict_path = f"{stem}_21.dict"
                count_path = f"{stem}_21.count"
                with open(dict_path, "wb") as handle:
                    handle.write(b"DICT\n")
                with open(count_path, "wb") as handle:
                    handle.write(b"COUNT\n")
                return dict_path, count_path

            runtime_patch = mock.patch.multiple(
                rt,
                create=True,
                run_dir=tmpdir,
                outdir=tmpdir,
                memFile=mem_path,
                mismat=0,
                runtype="G",
                reference_id_mode="numeric",
                maxhits=25,
                clustbuffer=150,
                phase=21,
                norm=False,
                norm_factor=1_000_000.0,
                reference=reference_path,
            )
            with runtime_patch:
                with mock.patch.object(
                    mapping,
                    "run_parallel_with_progress",
                    side_effect=_serial_parallel_runner,
                ):
                    with mock.patch.object(mapping, "PPBalance", side_effect=_serial_ppbalance):
                        with mock.patch.object(mapping, "optimize", return_value=(1, 1)):
                            with mock.patch.object(mapping, "mapper", side_effect=_fake_mapper) as mapper_mock:
                                with mock.patch.object(
                                    sam_parsing,
                                    "run_parallel_with_progress",
                                    side_effect=_serial_parallel_runner,
                                ):
                                    with mock.patch.object(
                                        sam_parsing,
                                        "samparser_streaming",
                                        side_effect=fake_parser,
                                    ) as parser_mock:
                                        first_maps = mapping.mapprocess(
                                            [fas_path],
                                            genoIndex=geno_index,
                                            ncores_local=1,
                                        )
                                        sam_parsing.parserprocess([fas_path])

                                        expected_bam = mapping._bam_output_for_fas(fas_path)
                                        dict_path, count_path = sam_parsing._parser_output_paths_for_lib(
                                            fas_path,
                                            "21",
                                        )
                                        first_cfg = _load_memcfg(mem_path)
                                        first_mapping_sig = first_cfg[mapping.MAPPING_SECTION][sig_key(expected_bam)]
                                        first_dict_sig = first_cfg[sam_parsing.SAM_PARSING_SECTION][sig_key(dict_path)]
                                        first_count_sig = first_cfg[sam_parsing.SAM_PARSING_SECTION][sig_key(count_path)]
                                        first_mapping_fp = first_cfg[mapping.MAPPING_SECTION][expected_bam]
                                        first_dict_fp = first_cfg[sam_parsing.SAM_PARSING_SECTION][dict_path]
                                        first_count_fp = first_cfg[sam_parsing.SAM_PARSING_SECTION][count_path]

                                        rt.maxhits = 50

                                        second_maps = mapping.mapprocess(
                                            [fas_path],
                                            genoIndex=geno_index,
                                            ncores_local=1,
                                        )
                                        sam_parsing.parserprocess([fas_path])

                                        second_cfg = _load_memcfg(mem_path)
                                        second_mapping_sig = second_cfg[mapping.MAPPING_SECTION][sig_key(expected_bam)]
                                        second_dict_sig = second_cfg[sam_parsing.SAM_PARSING_SECTION][sig_key(dict_path)]
                                        second_count_sig = second_cfg[sam_parsing.SAM_PARSING_SECTION][sig_key(count_path)]
                                        second_mapping_fp = second_cfg[mapping.MAPPING_SECTION][expected_bam]
                                        second_dict_fp = second_cfg[sam_parsing.SAM_PARSING_SECTION][dict_path]
                                        second_count_fp = second_cfg[sam_parsing.SAM_PARSING_SECTION][count_path]

                                        third_maps = mapping.mapprocess(
                                            [fas_path],
                                            genoIndex=geno_index,
                                            ncores_local=1,
                                        )
                                        sam_parsing.parserprocess([fas_path])

                                        third_hit_cfg = _load_memcfg(mem_path)
                                        third_mapping_sig = third_hit_cfg[mapping.MAPPING_SECTION][sig_key(expected_bam)]
                                        third_dict_sig = third_hit_cfg[sam_parsing.SAM_PARSING_SECTION][sig_key(dict_path)]
                                        third_count_sig = third_hit_cfg[sam_parsing.SAM_PARSING_SECTION][sig_key(count_path)]

                                        rt.mismat = 1
                                        sam_parsing.parserprocess([fas_path])
                                        sam_parsing.parserprocess([fas_path])

            self.assertEqual(first_maps, [expected_bam])
            self.assertEqual(second_maps, [expected_bam])
            self.assertEqual(third_maps, [])
            self.assertEqual(mapper_mock.call_count, 2)
            self.assertEqual(parser_mock.call_count, 3)
            self.assertNotEqual(first_mapping_sig, second_mapping_sig)
            self.assertNotEqual(first_dict_sig, second_dict_sig)
            self.assertNotEqual(first_count_sig, second_count_sig)
            self.assertEqual(first_mapping_fp, first_cfg["MAPS"][expected_bam])
            self.assertEqual(first_dict_fp, first_cfg["PARSED"][dict_path])
            self.assertEqual(first_count_fp, first_cfg["COUNTERS"][count_path])
            self.assertEqual(second_mapping_fp, first_mapping_fp)
            self.assertEqual(second_dict_fp, first_dict_fp)
            self.assertEqual(second_count_fp, first_count_fp)
            self.assertEqual(
                [call.args[0][3] for call in mapper_mock.call_args_list],
                [25, 50],
            )
            self.assertEqual(
                [call.args[0][1] for call in parser_mock.call_args_list],
                [25, 50, 50],
            )
            self.assertEqual(third_mapping_sig, second_mapping_sig)
            self.assertEqual(third_dict_sig, second_dict_sig)
            self.assertEqual(third_count_sig, second_count_sig)

            final_cfg = _load_memcfg(mem_path)
            self.assertEqual(
                final_cfg[mapping.MAPPING_SECTION][sig_key(expected_bam)],
                second_mapping_sig,
            )
            self.assertNotEqual(
                final_cfg[sam_parsing.SAM_PARSING_SECTION][sig_key(dict_path)],
                second_dict_sig,
            )
            self.assertEqual(
                final_cfg[sam_parsing.SAM_PARSING_SECTION][sig_key(count_path)],
                final_cfg[sam_parsing.SAM_PARSING_SECTION][sig_key(dict_path)],
            )
            self.assertEqual(final_cfg["ADVANCED"]["maxhits"], "50")
            self.assertEqual(final_cfg["ADVANCED"]["mismat"], "1")


if __name__ == "__main__":
    unittest.main()
