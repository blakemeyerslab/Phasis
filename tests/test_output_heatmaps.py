from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from phasis.stages import output


class _RecordingPool:
    def __init__(self):
        self.jobs = []

    def map(self, _func, jobs):
        self.jobs = list(jobs)
        return []

    def close(self):
        return None

    def join(self):
        return None

    def terminate(self):
        return None


def _heatmap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "identifier": "1:100..200",
                "alib": "libA",
                "label": "PHAS",
                "log_clust_len_norm_counts": 2.0,
                "total_abund": 100.0,
                "Peak_Howell_score": 20.0,
                "Peak_Howell_score_strict": 18.0,
            },
            {
                "identifier": "1:100..200",
                "alib": "libB",
                "label": "non-PHAS",
                "log_clust_len_norm_counts": 1.0,
                "total_abund": 40.0,
                "Peak_Howell_score": 8.0,
                "Peak_Howell_score_strict": 7.0,
            },
            {
                "identifier": "2:300..400",
                "alib": "libA",
                "label": "non-PHAS",
                "log_clust_len_norm_counts": 0.5,
                "total_abund": 25.0,
                "Peak_Howell_score": 6.0,
                "Peak_Howell_score_strict": 5.0,
            },
            {
                "identifier": "2:300..400",
                "alib": "libB",
                "label": "non-PHAS",
                "log_clust_len_norm_counts": 0.25,
                "total_abund": 10.0,
                "Peak_Howell_score": 4.0,
                "Peak_Howell_score_strict": 3.0,
            },
        ]
    )


def _minimal_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "identifier": "1:100..200",
                "alib": "libA",
                "label": "PHAS",
                "phasis_score": 300.0,
                "complexity": 0.1,
                "strand_bias": 0.8,
                "log_clust_len_norm_counts": 2.0,
                "ratio_abund_len_phase": 5.0,
                "total_abund": 100.0,
                "Peak_Howell_score": 20.0,
                "Peak_Howell_score_strict": 18.0,
            }
        ]
    )


class HeatmapRowSelectionTests(unittest.TestCase):
    def test_filter_keeps_complete_locus_rows_only_when_called_at_least_once(self):
        filtered = output._filter_plot_df(_heatmap_frame())

        self.assertEqual(filtered["identifier"].unique().tolist(), ["1:100..200"])
        self.assertEqual(filtered["alib"].tolist(), ["libA", "libB"])
        self.assertEqual(filtered["label"].tolist(), ["PHAS", "non-PHAS"])

    def test_all_heatmaps_exclude_non_phas_only_loci_in_21_phase(self):
        frame = _heatmap_frame()
        plot_functions = (
            output.plot_report_heat_map,
            output.plot_phasAbundance_heat_map,
            output.plot_totalAbundance_heat_map,
            output.plot_howell_score_heat_maps,
        )

        with tempfile.TemporaryDirectory() as outdir:
            for plot_function in plot_functions:
                captured = []
                real_heatmap = output.sns.heatmap

                def record_heatmap(data, *args, **kwargs):
                    captured.append(data.copy())
                    return real_heatmap(data, *args, **kwargs)

                with self.subTest(plot_function=plot_function.__name__):
                    with (
                        mock.patch.object(output, "phase", 21),
                        mock.patch.object(output, "outdir", outdir),
                        mock.patch.object(output, "_savefig"),
                        mock.patch.object(output.sns, "heatmap", side_effect=record_heatmap),
                    ):
                        plot_function(frame, "GMM")

                    self.assertTrue(captured)
                    for matrix in captured:
                        self.assertEqual(matrix.index.tolist(), ["1:100..200"])
                        self.assertEqual(matrix.columns.tolist(), ["libA", "libB"])


class PooledHeatmapSuppressionTests(unittest.TestCase):
    def _finalize(self, outdir, *, pooled, pool=None):
        pool = pool or _RecordingPool()
        with (
            mock.patch.object(output.st_ids, "ensure_mergedClusterDict"),
            mock.patch.object(output.rt, "mergedClusterReverse", {}, create=True),
            mock.patch.object(output, "make_pool", return_value=pool) as make_pool,
        ):
            output.finalize_and_write_results(
                "GMM",
                _minimal_feature_frame(),
                job_outdir=outdir,
                job_phase=21,
                job_concat_libs=pooled,
            )
        return make_pool, pool

    def test_pooled_mode_skips_all_cross_library_heatmaps(self):
        with tempfile.TemporaryDirectory() as outdir:
            make_pool, pool = self._finalize(outdir, pooled=True)

            make_pool.assert_not_called()
            self.assertEqual(pool.jobs, [])
            self.assertTrue(os.path.isfile(os.path.join(outdir, "21_calls.tsv")))
            self.assertTrue(os.path.isfile(os.path.join(outdir, "21_PHAS.gff")))
            for filename in (
                "21_PHAS.pdf",
                "21_Abundance_PHAS.pdf",
                "21_Abundance_PHAS_and_nonPHAS.pdf",
                "21_Howell_scores.pdf",
            ):
                self.assertFalse(os.path.exists(os.path.join(outdir, filename)))

    def test_individual_mode_still_schedules_four_heatmaps(self):
        with tempfile.TemporaryDirectory() as outdir:
            pool = _RecordingPool()
            make_pool, pool = self._finalize(outdir, pooled=False, pool=pool)

            make_pool.assert_called_once_with(mock.ANY, kind="plot")
            self.assertEqual(len(pool.jobs), 4)
            self.assertEqual(
                {job[0] for job in pool.jobs},
                {
                    output.plot_report_heat_map,
                    output.plot_phasAbundance_heat_map,
                    output.plot_totalAbundance_heat_map,
                    output.plot_howell_score_heat_maps,
                },
            )


if __name__ == "__main__":
    unittest.main()
