from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from phasis.config import Phase2Config
from phasis.stages import phase2_pipeline


class Phase2MemorySafePhasHandoffTests(unittest.TestCase):
    """Exercise the write-only PHAS handoff used to avoid a raw+final peak."""

    def _cfg(self, tmpdir: str) -> Phase2Config:
        return Phase2Config(
            phase="24",
            outdir=os.path.join(tmpdir, "results"),
            concat_libs=False,
            steps="both",
            class_cluster_file=None,
            libs=[],
            memFile=os.path.join(tmpdir, "phasis.mem"),
        )

    @staticmethod
    def _raw_clusters() -> pd.DataFrame:
        # The PHAS builder is mocked here; these are only the fields the
        # orchestration layer needs before handing the table off to that stage.
        return pd.DataFrame(
            {
                "alib": ["libA"],
                "clusterID": ["cluster-1"],
                "chromosome": ["chr1"],
            }
        )

    def _run_until_empty_phas_result(
        self,
        *,
        tmpdir: str,
        phas_output: str | None,
    ):
        raw_clusters = self._raw_clusters()
        logical_names = lambda name: os.path.join(tmpdir, f"24_{name}")

        with (
            mock.patch.object(
                phase2_pipeline.st_cluster_aggregation,
                "aggregate_and_write_processed_clusters",
                return_value=raw_clusters,
            ) as aggregate,
            mock.patch.object(
                phase2_pipeline.st_cmerge,
                "loci_table_from_clusters",
                return_value=pd.DataFrame({"clusterID": ["cluster-1"]}),
            ),
            mock.patch.object(
                phase2_pipeline.ids,
                "ensure_mergedClusterDict_always",
                return_value={"cluster-1": "chr1:1..24"},
            ),
            mock.patch.object(
                phase2_pipeline.st_phas_clusters,
                "build_and_save_phas_clusters",
                return_value=phas_output,
            ) as build_phas,
            mock.patch.object(
                phase2_pipeline.st_phas_clusters,
                "load_phas_to_detect_output",
                return_value=pd.DataFrame(),
            ) as load_phas,
            mock.patch.object(phase2_pipeline, "phase2_basename", side_effect=logical_names),
        ):
            phase2_pipeline.run_phase2_pipeline(
                ["input.24-PHAS.candidate.clusters"],
                cfg=self._cfg(tmpdir),
            )

        return raw_clusters, aggregate, build_phas, load_phas

    def test_write_only_handoff_releases_raw_table_before_loading_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logical_output = os.path.join(tmpdir, "24_PHAS_to_detect.tab")
            with open(logical_output, "w", encoding="utf-8") as handle:
                handle.write("identifier\n")

            raw_clusters, aggregate, build_phas, load_phas = (
                self._run_until_empty_phas_result(
                    tmpdir=tmpdir,
                    phas_output=logical_output,
                )
            )

        aggregate.assert_called_once_with(
            ["input.24-PHAS.candidate.clusters"],
            memFile=mock.ANY,
        )
        build_phas.assert_called_once_with(
            raw_clusters,
            phase=24,
            memFile=mock.ANY,
            concat_libs=False,
            return_dataframe=False,
        )
        load_phas.assert_called_once_with(
            logical_output,
            columns=phase2_pipeline.PHASE2_RUNTIME_CLUSTER_COLUMNS,
        )

    def test_empty_write_only_result_exits_without_loading_phas_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, build_phas, load_phas = self._run_until_empty_phas_result(
                tmpdir=tmpdir,
                phas_output=None,
            )

        self.assertEqual(build_phas.call_args.kwargs["return_dataframe"], False)
        load_phas.assert_not_called()

    def test_missing_returned_logical_path_exits_without_loading_phas_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_output = os.path.join(tmpdir, "24_PHAS_to_detect.tab")
            _, _, _, load_phas = self._run_until_empty_phas_result(
                tmpdir=tmpdir,
                phas_output=missing_output,
            )

        load_phas.assert_not_called()

    def test_compressed_phas_artifact_is_accepted_via_logical_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logical_output = os.path.join(tmpdir, "24_PHAS_to_detect.tab")
            # There is deliberately no plain logical file.  artifact_exists()
            # must accept its compressed physical sibling before the loader is
            # invoked with the original logical name.
            with open(f"{logical_output}.gz", "wb") as handle:
                handle.write(b"compressed fixture")

            _, _, _, load_phas = self._run_until_empty_phas_result(
                tmpdir=tmpdir,
                phas_output=logical_output,
            )

        load_phas.assert_called_once_with(
            logical_output,
            columns=phase2_pipeline.PHASE2_RUNTIME_CLUSTER_COLUMNS,
        )

    def test_compact_phase2_frame_keeps_required_columns_and_stable_categories(self):
        frame = pd.DataFrame(
            {
                "alib": ["libB", "libA", "libB"],
                "clusterID": ["002", "001", "002"],
                "chromosome": [2, 1, 2],
                "strand": ["w", "c", "w"],
                "pos": [100, 200, 121],
                "len": [24, 24, 24],
                "hits": [1, 2, 1],
                "abun": [10, 5, 4],
                "pval_corr_f": [0.1, 0.2, 0.1],
                "pval_corr_r": [0.3, 0.4, 0.3],
                "tag_seq": ["NA", "ACGT", "NA"],
                "identifier": ["chr2:100..121", "chr1:200..200", "chr2:100..121"],
            }
        )

        compact = phase2_pipeline._compact_phase2_cluster_frame(frame.copy())

        for column in phase2_pipeline.PHASE2_RUNTIME_CATEGORICAL_COLUMNS:
            self.assertTrue(isinstance(compact[column].dtype, pd.CategoricalDtype))
        self.assertEqual(compact["clusterID"].astype(str).tolist(), ["002", "001", "002"])
        self.assertEqual(compact["tag_seq"].tolist(), ["NA", "ACGT", "NA"])
        self.assertEqual(
            compact["clusterID"].cat.categories.tolist(),
            ["002", "001"],
        )

    def test_reloads_compact_phas_fields_only_when_locus_plots_need_them(self):
        runtime_clusters = pd.DataFrame(
            {
                "alib": ["libA", "libA"],
                "clusterID": ["cluster-1", "cluster-1"],
                "chromosome": ["chr1", "chr1"],
                "strand": ["w", "c"],
                "pos": [100, 124],
                "len": [24, 24],
                "hits": [1, 1],
                "abun": [10, 8],
                "pval_corr_f": [0.1, 0.1],
                "pval_corr_r": [0.2, 0.2],
                "tag_seq": ["ACGT", "TGCA"],
                "identifier": ["chr1:100..124", "chr1:100..124"],
            }
        )
        labeled = pd.DataFrame(
            {
                "identifier": ["chr1:100..124"],
                "alib": ["libA"],
                "cID": ["cluster-1"],
                "final_class": ["PHAS"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            logical_output = os.path.join(tmpdir, "24_PHAS_to_detect.tab")
            with open(logical_output, "w", encoding="utf-8") as handle:
                handle.write("identifier\n")
            logical_names = lambda name: os.path.join(tmpdir, f"24_{name}")

            with (
                mock.patch.object(
                    phase2_pipeline.st_cluster_aggregation,
                    "aggregate_and_write_processed_clusters",
                    return_value=self._raw_clusters(),
                ),
                mock.patch.object(
                    phase2_pipeline.st_cmerge,
                    "loci_table_from_clusters",
                    return_value=pd.DataFrame({"clusterID": ["cluster-1"]}),
                ),
                mock.patch.object(
                    phase2_pipeline.ids,
                    "ensure_mergedClusterDict_always",
                    return_value={"cluster-1": "chr1:100..124"},
                ),
                mock.patch.object(
                    phase2_pipeline.st_phas_clusters,
                    "build_and_save_phas_clusters",
                    return_value=logical_output,
                ),
                mock.patch.object(
                    phase2_pipeline.st_phas_clusters,
                    "load_phas_to_detect_output",
                    side_effect=[runtime_clusters.copy(), runtime_clusters.copy()],
                ) as load_phas,
                mock.patch.object(
                    phase2_pipeline.st_winsel,
                    "select_scoring_windows",
                    return_value=pd.DataFrame({"cluster_id": ["cluster-1"]}),
                ),
                mock.patch.object(
                    phase2_pipeline.st_winscore,
                    "compute_and_save_phasis_scores",
                    return_value=pd.DataFrame({"cID": ["cluster-1"]}),
                ),
                mock.patch.object(
                    phase2_pipeline.st_feat,
                    "features_to_detection",
                    return_value=pd.DataFrame({"cID": ["cluster-1"]}),
                ),
                mock.patch.object(
                    phase2_pipeline.st_classify,
                    "gmm_classify",
                    return_value=labeled,
                ),
                mock.patch.object(
                    phase2_pipeline.st_classify,
                    "apply_evidence_classification",
                    return_value=labeled,
                ),
                mock.patch.object(
                    phase2_pipeline.st_locus_plots,
                    "write_individual_phas_locus_plots",
                ) as write_plots,
                mock.patch.object(
                    phase2_pipeline.st_output,
                    "finalize_and_write_results",
                ),
                mock.patch.object(phase2_pipeline, "phase2_basename", side_effect=logical_names),
            ):
                phase2_pipeline.run_phase2_pipeline(
                    ["input.24-PHAS.candidate.clusters"],
                    cfg=self._cfg(tmpdir),
                )

        expected_load = mock.call(
            logical_output,
            columns=phase2_pipeline.PHASE2_RUNTIME_CLUSTER_COLUMNS,
        )
        self.assertEqual(load_phas.call_args_list, [expected_load, expected_load])
        plotted_clusters = write_plots.call_args.args[2]
        self.assertTrue(
            isinstance(plotted_clusters["clusterID"].dtype, pd.CategoricalDtype)
        )


if __name__ == "__main__":
    unittest.main()
