from __future__ import annotations

import os
import random
import unittest
from unittest import mock

import numpy as np

from phasis.stages import feature_assembly as fa


def _loops():
    """Run the scans with the Python loops (array form switched off)."""
    return mock.patch.dict(os.environ, {"Phasis_HOWELL_ARRAY_SCAN": "0"})


def _random_pos_abun(rng, phase, span, phased_fraction, value_kind, start=500):
    pos_abun = {}
    for _ in range(rng.randint(1, 120)):
        if rng.random() < phased_fraction:
            pos = start + rng.randint(0, span // phase) * phase + rng.choice([-1, -1, 0, 0, 0, 1, 1])
        else:
            pos = start + rng.randint(0, span)
        if value_kind == "whole":
            value = float(rng.choice([1, 1, 1, 2, 2, 3, 7, 50, 900]))
        else:
            value = rng.choice([1, 2, 3, 7, 50]) / rng.choice([3, 7])
        pos_abun[pos] = pos_abun.get(pos, 0.0) + value
    return pos_abun


def _cases(seed, value_kind="whole", n=40):
    rng = random.Random(seed)
    for phase in (21, 24):
        for _ in range(n):
            span = rng.choice([40, 150, 400, 900, 2500])
            pos_abun = _random_pos_abun(rng, phase, span, rng.choice([0.0, 0.5, 0.9, 1.0]), value_kind)
            positions = sorted(pos_abun)
            bounds = [
                (positions[0], positions[-1]),
                (None, None),
                (positions[0] - rng.randint(1, 60), positions[-1] + rng.randint(1, 60)),
                (positions[0] + 3, positions[-1] - 3),
            ]
            for seq_start, seq_end in bounds:
                for forward in (True, False):
                    yield phase, fa.WINDOW_MULTIPLIER * phase, pos_abun, seq_start, seq_end, forward


class HowellArrayScanTests(unittest.TestCase):
    """The array form must return exactly what the Python loops return."""

    def test_whole_number_abundances_use_the_array_form(self):
        pos_abun = {100 + 24 * k: float(k + 1) for k in range(12)}
        self.assertIsNotNone(fa._HowellRegisterArrays.build(pos_abun, 24, 240, 100, 200))
        with _loops():
            self.assertIsNone(fa._HowellRegisterArrays.build(pos_abun, 24, 240, 100, 200))

    def test_inputs_outside_the_array_form_fall_back_to_the_loops(self):
        whole = {100 + 24 * k: float(k + 1) for k in range(12)}
        for pos_abun in (
            {**whole, 130: 2.5},            # fractional abundance
            {**whole, 130: -1.0},           # negative abundance
            {**whole, 130: float("nan")},
            {**whole, 130: 2.0 ** 60},      # sums would not be exact
            {},
        ):
            self.assertIsNone(fa._HowellRegisterArrays.build(pos_abun, 24, 240, 100, 200))
        # Window that is not a whole number of cycles, too few cycles, tiny phase.
        self.assertIsNone(fa._HowellRegisterArrays.build(whole, 24, 250, 100, 200))
        self.assertIsNone(fa._HowellRegisterArrays.build(whole, 24, 72, 100, 200))
        self.assertIsNone(fa._HowellRegisterArrays.build(whole, 2, 20, 100, 200))

    def test_candidate_windows_and_peak_detail_match_the_loops(self):
        checked = 0
        for phase, win_size, pos_abun, seq_start, seq_end, forward in _cases(31):
            got = fa._enumerate_relaxed_candidate_windows(
                pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=forward
            )
            with _loops():
                expected = fa._enumerate_relaxed_candidate_windows(
                    pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=forward
                )
            self.assertEqual(repr(expected), repr(got))
            checked += len(got[0])
        self.assertGreater(checked, 50000)

    def test_generic_scan_matches_the_loops_with_and_without_detail(self):
        for phase, win_size, pos_abun, seq_start, seq_end, forward in _cases(32, n=25):
            for return_detail in (False, True):
                got = fa._best_sliding_window_score_generic(
                    pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end,
                    forward=forward, return_detail=return_detail,
                )
                with _loops():
                    expected = fa._best_sliding_window_score_generic(
                        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end,
                        forward=forward, return_detail=return_detail,
                    )
                self.assertEqual(repr(expected), repr(got))

    def test_strict_scan_matches_the_loops(self):
        scored = 0
        for phase, win_size, pos_abun, seq_start, seq_end, forward in _cases(33):
            got = fa._best_sliding_window_score_generic_strict(
                pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=forward
            )
            with _loops():
                expected = fa._best_sliding_window_score_generic_strict(
                    pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=forward
                )
            self.assertEqual(repr(expected), repr(got))
            scored += 1 if got[0] > 0 else 0
        self.assertGreater(scored, 50)

    def test_relaxed_trace_matches_the_loops(self):
        rows = 0
        for phase, win_size, pos_abun, _seq_start, _seq_end, forward in _cases(34, n=25):
            got = fa._enumerate_relaxed_trace_for_strand(pos_abun, phase, win_size, forward=forward)
            with _loops():
                expected = fa._enumerate_relaxed_trace_for_strand(pos_abun, phase, win_size, forward=forward)
            self.assertEqual(repr(expected), repr(got))
            rows += len(got)
        self.assertGreater(rows, 5000)

    def test_dense_neighbouring_positions_match_the_loops(self):
        # Runs of adjacent coordinates exercise the +-1 wobble, its tie rule and the window edges.
        rng = random.Random(35)
        for phase in (21, 24):
            win_size = fa.WINDOW_MULTIPLIER * phase
            for _ in range(30):
                pos_abun = {}
                for pos in range(1000, 1000 + rng.choice([260, 500, 700])):
                    if rng.random() < rng.choice([0.3, 0.6, 0.95]):
                        pos_abun[pos] = float(rng.choice([1, 1, 2, 2, 5]))
                if not pos_abun:
                    continue
                positions = sorted(pos_abun)
                for forward in (True, False):
                    got = fa._enumerate_relaxed_candidate_windows(
                        pos_abun, phase, win_size, seq_start=positions[0], seq_end=positions[-1], forward=forward
                    )
                    got_trace = fa._enumerate_relaxed_trace_for_strand(pos_abun, phase, win_size, forward=forward)
                    with _loops():
                        expected = fa._enumerate_relaxed_candidate_windows(
                            pos_abun, phase, win_size, seq_start=positions[0], seq_end=positions[-1], forward=forward
                        )
                        expected_trace = fa._enumerate_relaxed_trace_for_strand(
                            pos_abun, phase, win_size, forward=forward
                        )
                    self.assertEqual(repr(expected), repr(got))
                    self.assertEqual(repr(expected_trace), repr(got_trace))

    def test_fractional_abundances_give_the_loop_results(self):
        for phase, win_size, pos_abun, seq_start, seq_end, forward in _cases(36, value_kind="fraction", n=6):
            got = fa._enumerate_relaxed_candidate_windows(
                pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=forward
            )
            with _loops():
                expected = fa._enumerate_relaxed_candidate_windows(
                    pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=forward
                )
            self.assertEqual(repr(expected), repr(got))

    def test_chunked_scan_equals_single_pass(self):
        rng = random.Random(37)
        pos_abun = _random_pos_abun(rng, 24, 2500, 0.9, "whole")
        starts = np.arange(min(pos_abun) - 10, max(pos_abun) + 10)
        whole = fa._howell_window_scan_arrays(pos_abun, 24, 240, starts, strict=True, register_scores=True)
        with mock.patch.object(fa, "_HOWELL_ARRAY_CHUNK_WINDOWS", 97):
            chunked = fa._howell_window_scan_arrays(pos_abun, 24, 240, starts, strict=True, register_scores=True)
        self.assertEqual(sorted(whole), sorted(chunked))
        for name in whole:
            self.assertTrue(np.array_equal(whole[name], chunked[name]), name)

    def test_scalar_logs_equal_one_call_per_value(self):
        values = np.array([1.0, 2.5, 2.5, 11.0, 1e6, 1.0000000001])
        self.assertEqual(
            fa._howell_scalar_logs(values).tolist(), [float(np.log(v)) for v in values.tolist()]
        )


if __name__ == "__main__":
    unittest.main()
