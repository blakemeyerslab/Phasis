from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from phasis import ids
from phasis import runtime as rt
from phasis.stages import candidates_merge


def _run_serial(func, data, **_kwargs):
    return [func(item) for item in data]


class CandidateMergeReferenceIdTests(unittest.TestCase):
    def test_worker_preserves_chromosome_ids_as_text(self):
        for chromosome in ("chr1", "Mt", "1"):
            with self.subTest(chromosome=chromosome):
                clusters = pd.DataFrame(
                    {
                        "clusterID": ["A1", "A1"],
                        "chromosome": [chromosome, chromosome],
                        "pos": [100, 121],
                    }
                )

                rows = candidates_merge.chromosome_clusters_to_candidate_loci(
                    clusters,
                    minClusterLength=0,
                )

                self.assertEqual(
                    {tuple(row) for row in rows},
                    {("A1", 0, chromosome, 100, 121)},
                )
                self.assertTrue(all(isinstance(row[2], str) for row in rows))

    def test_non_pooled_locus_merge_retains_preserved_ids(self):
        all_clusters = pd.DataFrame(
            {
                "alib": ["libA", "libA", "libB", "libB", "libA", "libA"],
                "clusterID": ["A1", "A1", "B1", "B1", "A10", "A10"],
                "chromosome": ["chr1", "chr1", "chr1", "chr1", "chr10", "chr10"],
                "pos": [100, 121, 105, 126, 500, 521],
            }
        )

        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                mem_file = os.path.join(tmpdir, "phasis.mem")
                processed_path = os.path.join(tmpdir, "21_processed_clusters.tab")
                loci_path = os.path.join(tmpdir, "21_candidate.loci_table.tab")
                all_clusters.to_csv(processed_path, sep="\t", index=False)

                with (
                    mock.patch.multiple(
                        rt,
                        create=True,
                        phase=21,
                        concat_libs=False,
                        clustbuffer=0,
                        minClusterLength=0,
                        compress_intermediates=False,
                        run_dir=tmpdir,
                    ),
                    mock.patch.object(
                        candidates_merge,
                        "run_parallel_with_progress",
                        side_effect=_run_serial,
                    ),
                    mock.patch.object(
                        ids,
                        "run_parallel_with_progress",
                        side_effect=_run_serial,
                    ),
                ):
                    loci = candidates_merge.loci_table_from_clusters(
                        all_clusters,
                        memFile=mem_file,
                        minClusterLength=0,
                        outfname=loci_path,
                    )
                    cached_loci = candidates_merge.loci_table_from_clusters(
                        all_clusters,
                        memFile=mem_file,
                        minClusterLength=0,
                        outfname=loci_path,
                    )

                    merged = ids.ensure_mergedClusterDict_always(
                        concat_libs=False,
                        phase="21",
                        merged_out_path=os.path.join(tmpdir, "unused.tsv"),
                        loci_table_df=loci,
                        allClusters_df=all_clusters,
                        memFile=mem_file,
                    )

                    self.assertEqual(set(loci["chr"]), {"chr1", "chr10"})
                    self.assertEqual(set(cached_loci["chr"]), {"chr1", "chr10"})
                    self.assertEqual(
                        {key: set(values) for key, values in merged.items()},
                        {
                            "chr1:100..126": {"A1", "B1"},
                            "chr10:500..521": {"A10"},
                        },
                    )
                    self.assertEqual(rt.mergedClusterReverse["A1"], "chr1:100..126")

                persisted = pd.read_csv(
                    os.path.join(tmpdir, "21_mergedClusterDict.tab"),
                    sep="\t",
                    header=None,
                )
                self.assertEqual(set(persisted[0]), {"chr1:100..126", "chr10:500..521"})
        finally:
            os.chdir(original_cwd)

    def test_locus_builder_propagates_worker_errors(self):
        clusters = pd.DataFrame(
            {
                "clusterID": ["A1"],
                "chromosome": ["chr1"],
                "pos": [100],
            }
        )
        expected = RuntimeError("primary worker failure")

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(rt, "phase", 21, create=True),
                mock.patch.object(rt, "concat_libs", False, create=True),
                mock.patch.object(
                    candidates_merge,
                    "run_parallel_with_progress",
                    return_value=[expected],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "primary worker failure"):
                    candidates_merge.loci_table_from_clusters(
                        clusters,
                        memFile=os.path.join(tmpdir, "phasis.mem"),
                        minClusterLength=0,
                        outfname=os.path.join(tmpdir, "21_candidate.loci_table.tab"),
                    )


if __name__ == "__main__":
    unittest.main()
