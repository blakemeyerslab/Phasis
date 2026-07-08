from __future__ import annotations

import gzip
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from phasis.cache import (
    MemCache,
    artifact_exists,
    compress_intermediates_enabled,
    finalize_text_artifact,
    resolve_artifact_path,
    stage_signature,
)
from phasis.env import legacy_env_name
from phasis import state as st
from phasis.stages.phas_clusters import _read_cached_phas_to_detect


def _write_gzip_text(path: str, text: str) -> str:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return path


class CompressedArtifactCacheTests(unittest.TestCase):
    def test_finalize_text_artifact_compresses_by_default_and_records_logical_key(self):
        compress_env = "Phasis_COMPRESS_INTERMEDIATES"
        legacy_compress_env = legacy_env_name(compress_env)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop(compress_env, None)
            os.environ.pop(legacy_compress_env, None)
            logical = os.path.join(tmpdir, "21_clusters_scored.tsv")
            mem_path = os.path.join(tmpdir, "phasis.mem")
            with open(logical, "w", encoding="utf-8") as handle:
                handle.write("cID\tphasis_score\nc1\t100\n")

            cache = MemCache.load(mem_path)
            fp = finalize_text_artifact(cache, "CLUSTERS_SCORED", logical, "input-signature")

            self.assertTrue(fp)
            self.assertFalse(os.path.exists(logical))
            self.assertTrue(os.path.isfile(f"{logical}.gz"))
            self.assertEqual(resolve_artifact_path(logical), f"{logical}.gz")
            self.assertTrue(cache.hit("CLUSTERS_SCORED", logical, "input-signature"))
            self.assertEqual(cache.get("CLUSTERS_SCORED", f"{logical}.artifact"), f"{logical}.gz")

    def test_compress_intermediates_accepts_legacy_env_alias(self):
        compress_env = "Phasis_COMPRESS_INTERMEDIATES"
        legacy_compress_env = legacy_env_name(compress_env)
        with mock.patch.dict(
            os.environ,
            {legacy_compress_env: "0"},
            clear=False,
        ):
            os.environ.pop(compress_env, None)
            self.assertFalse(compress_intermediates_enabled())

    def test_finalize_text_artifact_can_keep_plain_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logical = os.path.join(tmpdir, "21_clusters_scored.tsv")
            mem_path = os.path.join(tmpdir, "phasis.mem")
            with open(logical, "w", encoding="utf-8") as handle:
                handle.write("cID\tphasis_score\nc1\t100\n")

            cache = MemCache.load(mem_path)
            fp = finalize_text_artifact(
                cache,
                "CLUSTERS_SCORED",
                logical,
                "input-signature",
                compress=False,
            )

            self.assertTrue(fp)
            self.assertTrue(os.path.isfile(logical))
            self.assertFalse(os.path.exists(f"{logical}.gz"))
            self.assertTrue(cache.hit("CLUSTERS_SCORED", logical, "input-signature"))

    def test_cache_records_and_hits_gz_artifact_under_logical_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logical = os.path.join(tmpdir, "24_PHAS_to_detect.tab")
            gz_path = f"{logical}.gz"
            mem_path = os.path.join(tmpdir, "phasis.mem")
            _write_gzip_text(gz_path, "clusterID\tpos\nc1\t42\n")

            cache = MemCache.load(mem_path)
            fp = cache.record("PHAS_TO_DETECT", logical, "input-signature")

            self.assertTrue(fp)
            self.assertTrue(cache.hit("PHAS_TO_DETECT", logical, "input-signature"))
            self.assertEqual(resolve_artifact_path(logical), gz_path)
            self.assertEqual(cache.get("PHAS_TO_DETECT", f"{logical}.artifact"), gz_path)

    def test_stage_signature_resolves_gz_logical_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logical = os.path.join(tmpdir, "24_processed_clusters.tab")
            _write_gzip_text(f"{logical}.gz", "clusterID\tpos\nc1\t42\n")

            self.assertTrue(artifact_exists(logical))
            present_sig = stage_signature(files=[logical], params={"phase": 24})
            missing_sig = stage_signature(files=[f"{logical}.missing"], params={"phase": 24})

            self.assertNotEqual(present_sig, missing_sig)

    def test_cached_phas_to_detect_reader_loads_gz_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logical = os.path.join(tmpdir, "24_PHAS_to_detect.tab")
            mem_path = os.path.join(tmpdir, "phasis.mem")
            _write_gzip_text(
                f"{logical}.gz",
                "\t".join(
                    [
                        "alib",
                        "clusterID",
                        "chromosome",
                        "strand",
                        "pos",
                        "len",
                        "hits",
                        "abun",
                        "pval_h_f",
                        "N_f",
                        "X_f",
                        "pval_r_f",
                        "pval_corr_f",
                        "pval_h_r",
                        "N_r",
                        "X_r",
                        "pval_r_r",
                        "pval_corr_r",
                        "tag_id",
                        "tag_seq",
                        "identifier",
                    ]
                )
                + "\n"
                + "\t".join(
                    [
                        "ALL_LIBS",
                        "c1",
                        "1",
                        "w",
                        "42",
                        "24",
                        "1",
                        "10",
                        "0.01",
                        "1",
                        "1",
                        "0.02",
                        "0.02",
                        "0.03",
                        "1",
                        "1",
                        "0.04",
                        "0.04",
                        "tag1",
                        "ATGC",
                        "1:42..66",
                    ]
                )
                + "\n",
            )

            cache = MemCache.load(mem_path)
            cache.record("PHAS_TO_DETECT", logical, "input-signature")

            frame = _read_cached_phas_to_detect(logical, cache, "input-signature")

            self.assertIsInstance(frame, pd.DataFrame)
            self.assertEqual(frame.shape[0], 1)
            self.assertEqual(frame.loc[0, "clusterID"], "c1")
            self.assertEqual(float(frame.loc[0, "pos"]), 42.0)

    def test_win_score_lookup_loader_resolves_gz_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logical = os.path.join(tmpdir, "21_clusters_scored.tsv")
            _write_gzip_text(
                f"{logical}.gz",
                "cID\tphasis_score\tcombined_fishers\nc1\t123.5\t9.25\n",
            )

            st.clear_win_score_lookup()
            lookup = st.load_win_score_lookup_from_tsv(logical)

            self.assertEqual(lookup["c1"], (123.5, 9.25))


if __name__ == "__main__":
    unittest.main()
