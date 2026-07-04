from __future__ import annotations

import configparser
import io
import os
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from phasis import runtime as rt
from phasis.stages import cluster_scoring


def _serial_parallel_runner(func, iterable, **_kwargs):
    return [func(item) for item in list(iterable)]


def _write_pickle(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


def _make_minimal_scoring_inputs(tmpdir: str):
    akey = "libA-1"
    positions = [100 + (idx * 21) for idx in range(12)]
    lclust_path = os.path.join(tmpdir, f"{akey}.lclust")
    dict_path = os.path.join(tmpdir, "libA_21.dict")

    lclust = {"cluster1": positions}
    position_tags = {}
    for idx, pos in enumerate(positions):
        position_tags[pos] = [["1", "w", 1, f"TAG{idx}", f"tag{idx}", pos, 21, 1]]

    _write_pickle(lclust_path, lclust)
    _write_pickle(dict_path, {akey: [position_tags]})
    return akey, lclust_path, dict_path


class ClusterScoringBatchedNestdictTests(unittest.TestCase):
    def test_detects_path_only_sources_for_batched_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a = os.path.join(tmpdir, "a.dict")
            b = os.path.join(tmpdir, "b.dict")

            resolved = cluster_scoring._nestdict_pickle_sources([a, b, a])

        self.assertEqual(len(resolved), 2)
        self.assertTrue(resolved[0].endswith("a.dict"))
        self.assertTrue(resolved[1].endswith("b.dict"))

    def test_dict_sources_use_legacy_global_loader(self):
        self.assertIsNone(cluster_scoring._nestdict_pickle_sources([{"libA-1": []}]))

    def test_scoringprocess_uses_batched_loader_for_pickle_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            akey, lclust_path, dict_path = _make_minimal_scoring_inputs(tmpdir)
            scored_dir = os.path.join(tmpdir, "scored")
            mem_file = os.path.join(tmpdir, "phasis.mem")
            lib_path = os.path.join(tmpdir, "libA.fas")
            with open(lib_path, "w", encoding="utf-8") as handle:
                handle.write(">dummy\nTAG\n")

            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                runtime_patch = mock.patch.multiple(
                    rt,
                    phase=21,
                    mismat=0,
                    maxhits=25,
                    clustbuffer=300,
                    uniqueRatioCut=0.0,
                    memFile=mem_file,
                    ncores=1,
                    create=True,
                )
                with runtime_patch:
                    with mock.patch.object(
                        cluster_scoring,
                        "run_parallel_with_progress",
                        side_effect=_serial_parallel_runner,
                    ):
                        with mock.patch.object(
                            cluster_scoring,
                            "build_libchrs_nestdict",
                            side_effect=AssertionError("global nestdict loader should not run"),
                        ):
                            captured = io.StringIO()
                            with redirect_stdout(captured):
                                outputs = cluster_scoring.scoringprocess(
                                    [lib_path],
                                    [(akey, lclust_path)],
                                    [dict_path],
                                    tmpdir,
                                    force_rescore=True,
                                    verify_outputs=False,
                                    scored_dir=scored_dir,
                                    concat_mode=False,
                                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(outputs, ["libA.21-PHAS.candidate.clusters"])
            self.assertTrue(os.path.getsize(os.path.join(tmpdir, outputs[0])) > 0)
            log = captured.getvalue()
        self.assertIn("Using per-library batched parser loading", log)
        self.assertIn("Loading nestdict for library", log)


class ClusterScoringChunkBookkeepingTests(unittest.TestCase):
    def _write_chunks(self, tmpdir: str, phase: int = 24, count: int = 3):
        paths = []
        for idx in range(count):
            path = os.path.join(tmpdir, f"libA-{idx}.sRNA_{phase}.cluster")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(f"cluster {idx}\n")
            paths.append(os.path.realpath(path))
        with open(os.path.join(tmpdir, f"libA-ignore.sRNA_21.cluster"), "w", encoding="utf-8") as handle:
            handle.write("wrong phase\n")
        return paths

    def test_default_records_lightweight_manifest_not_per_chunk_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chunks(tmpdir, count=4)
            cfg = configparser.ConfigParser()
            cfg.add_section("SCORED_CHUNKS")
            cfg["SCORED_CHUNKS"]["stale"] = "old"
            bookkeeping_env = "Phasis_CLUSTER_SCORING_CHUNK_BOOKKEEPING"
            hash_mode_env = "Phasis_CLUSTER_SCORING_CHUNK_HASH_MODE"
            hash_limit_env = "Phasis_CLUSTER_SCORING_CHUNK_HASH_LIMIT"

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(bookkeeping_env, None)
                os.environ.pop(hash_mode_env, None)
                os.environ.pop(hash_limit_env, None)
                manifest = cluster_scoring._record_scored_chunk_bookkeeping(
                    cfg,
                    "SCORED_CHUNKS",
                    tmpdir,
                    phase_value=24,
                    max_worker_cap=2,
                )

        self.assertEqual(manifest["__bookkeeping_mode__"], "manifest")
        self.assertEqual(cfg["SCORED_CHUNKS"].get("__chunk_count__"), "4")
        self.assertIn("__manifest_signature__", cfg["SCORED_CHUNKS"])
        self.assertNotIn("stale", cfg["SCORED_CHUNKS"])
        per_chunk_keys = [
            key for key in cfg["SCORED_CHUNKS"].keys()
            if key.endswith(".srna_24.cluster")
        ]
        self.assertEqual(per_chunk_keys, [])

    def test_exact_mode_records_legacy_per_chunk_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_chunks(tmpdir, count=2)
            cfg = configparser.ConfigParser()
            cfg.add_section("SCORED_CHUNKS")

            with mock.patch.dict(
                os.environ,
                {"Phasis_CLUSTER_SCORING_CHUNK_BOOKKEEPING": "exact"},
                clear=False,
            ):
                manifest = cluster_scoring._record_scored_chunk_bookkeeping(
                    cfg,
                    "SCORED_CHUNKS",
                    tmpdir,
                    phase_value=24,
                    max_worker_cap=2,
                )

        self.assertEqual(manifest["__bookkeeping_mode__"], "exact")
        self.assertEqual(cfg["SCORED_CHUNKS"].get("__chunk_count__"), "2")
        for path in paths:
            self.assertTrue(cfg["SCORED_CHUNKS"].get(path))

    def test_skip_mode_keeps_manifest_and_omits_per_chunk_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_chunks(tmpdir, count=2)
            cfg = configparser.ConfigParser()
            cfg.add_section("SCORED_CHUNKS")

            with mock.patch.dict(
                os.environ,
                {"Phasis_CLUSTER_SCORING_CHUNK_BOOKKEEPING": "skip"},
                clear=False,
            ):
                manifest = cluster_scoring._record_scored_chunk_bookkeeping(
                    cfg,
                    "SCORED_CHUNKS",
                    tmpdir,
                    phase_value=24,
                    max_worker_cap=2,
                )

        self.assertEqual(manifest["__bookkeeping_mode__"], "skip")
        self.assertEqual(cfg["SCORED_CHUNKS"].get("__chunk_count__"), "2")
        for path in paths:
            self.assertIsNone(cfg["SCORED_CHUNKS"].get(path))

    def test_exact_mode_falls_back_to_manifest_when_hash_limit_is_exceeded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_chunks(tmpdir, count=3)
            cfg = configparser.ConfigParser()
            cfg.add_section("SCORED_CHUNKS")

            with mock.patch.dict(
                os.environ,
                {
                    "Phasis_CLUSTER_SCORING_CHUNK_BOOKKEEPING": "exact",
                    "Phasis_CLUSTER_SCORING_CHUNK_HASH_LIMIT": "2",
                },
                clear=False,
            ):
                manifest = cluster_scoring._record_scored_chunk_bookkeeping(
                    cfg,
                    "SCORED_CHUNKS",
                    tmpdir,
                    phase_value=24,
                    max_worker_cap=2,
                )

        self.assertEqual(manifest["__bookkeeping_mode__"], "manifest")
        self.assertEqual(cfg["SCORED_CHUNKS"].get("__chunk_count__"), "3")
        for path in paths:
            self.assertIsNone(cfg["SCORED_CHUNKS"].get(path))


if __name__ == "__main__":
    unittest.main()
