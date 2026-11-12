from __future__ import annotations

import random
import unittest

import numpy as np

from phasis.stages import feature_assembly as fa


def _reference_score_window_registers(window_positions, pos_abun, win_start, win_end, phase, *, forward, exact_only):
    """The former implementation: one evaluator call per register."""
    ph = int(phase)
    empty = {
        "score": 0.0, "best_register": None, "best_in_phase_sum": 0.0,
        "best_out_of_phase": 0.0, "best_filled": 0, "register_scores": [0.0] * ph,
    }
    if not window_positions:
        return empty
    num_cycles = max(0, (win_end - win_start + 1) // ph)
    if num_cycles < 4:
        return empty

    best_score = -float("inf")
    best_reg_sum = 0.0
    best_reg_out = 0.0
    best_reg_filled = 0
    best_reg = None
    register_scores = []
    evaluator = fa._evaluate_register_strict_exact if exact_only else fa._evaluate_register
    for reg in range(ph):
        in_sum, eff_total, n_filled = evaluator(window_positions, pos_abun, win_start, win_end, ph, reg, forward=forward)
        reg_score, out_of_phase = fa._compute_howell_score_from_register(in_sum, eff_total, n_filled, num_cycles)
        register_scores.append(float(reg_score))
        if reg_score > best_score:
            best_score = reg_score
            best_reg_sum = float(in_sum)
            best_reg_out = float(out_of_phase)
            best_reg_filled = int(n_filled)
            best_reg = reg
    return {
        "score": float(0.0 if best_score == -float("inf") else best_score),
        "best_register": best_reg,
        "best_in_phase_sum": float(best_reg_sum),
        "best_out_of_phase": float(best_reg_out),
        "best_filled": int(best_reg_filled),
        "register_scores": register_scores,
    }


def _reference_best_strict(pos_abun, phase, win_size, seq_start, seq_end, forward):
    """The former strict scan: list-comprehension window and one evaluator call per register."""
    positions = sorted(pos_abun.keys())
    if not positions:
        return 0.0, None, None
    lower_bound = seq_start if seq_start is not None else positions[0]
    upper_bound = (seq_end - win_size + 1) if seq_end is not None else positions[-1] - win_size + 1
    if upper_bound < lower_bound:
        lower_bound = positions[0]
        upper_bound = lower_bound
    best_score = -float("inf")
    best_window = (None, None)
    for win_start in range(lower_bound, upper_bound + 1):
        win_end = win_start + win_size - 1
        window_positions = [p for p in positions if win_start <= p <= win_end]
        if not window_positions:
            score = 0.0
        else:
            num_cycles = max(0, (win_end - win_start + 1) // int(phase))
            if num_cycles < 4:
                score = 0.0
            else:
                best_reg_sum = 0.0
                best_reg_total = 0.0
                best_reg_filled = 0
                for reg in range(int(phase)):
                    in_sum, total, n_filled = fa._evaluate_register_strict_exact(
                        window_positions, pos_abun, win_start, win_end, int(phase), reg, forward=forward
                    )
                    if in_sum > best_reg_sum:
                        best_reg_sum = in_sum
                        best_reg_total = total
                        best_reg_filled = n_filled
                out_of_phase = max(0.0, best_reg_total - best_reg_sum)
                numerator = best_reg_sum
                denominator = 1.0 + out_of_phase
                if numerator <= 0.0 or not (best_reg_filled > 3):
                    score = 0.0
                else:
                    log_arg = 1.0 + 10.0 * (numerator / denominator)
                    if log_arg <= 0.0 or log_arg != log_arg:
                        score = 0.0
                    else:
                        scale = max(min(best_reg_filled, num_cycles) - 2, 0)
                        score = scale * (0.0 if log_arg <= 0 else np.log(log_arg))
        if score > best_score:
            best_score = score
            best_window = (win_start, win_end)
    return best_score if best_score != -float("inf") else 0.0, best_window[0], best_window[1]


def _random_pos_abun(rng, phase, span, phased_fraction, value_kind):
    start = 500
    pos_abun = {}
    for _ in range(rng.randint(5, 90)):
        if rng.random() < phased_fraction:
            pos = start + rng.randint(0, span // phase) * phase + rng.choice([-1, 0, 0, 0, 1])
        else:
            pos = start + rng.randint(0, span)
        if value_kind == "int":
            value = rng.choice([1, 1, 2, 3, 7, 50, 900])
        elif value_kind == "float":
            value = float(rng.choice([1, 2, 3, 7, 50, 900]))
        else:
            value = rng.choice([1, 2, 3, 7, 50]) / rng.choice([1, 3, 7])  # non-integer floats
        pos_abun[pos] = pos_abun.get(pos, 0) + value
    return pos_abun


class HowellRegisterSharingTests(unittest.TestCase):
    def test_window_register_scores_match_per_register_evaluators(self):
        rng = random.Random(21)
        checked = 0
        for phase in (21, 24):
            for value_kind in ("int", "float", "fraction"):
                for _ in range(25):
                    span = rng.choice([150, 400, 900])
                    pos_abun = _random_pos_abun(rng, phase, span, rng.choice([0.0, 0.5, 0.9]), value_kind)
                    positions = sorted(pos_abun)
                    win_size = fa.WINDOW_MULTIPLIER * phase
                    for win_start in range(positions[0] - 5, positions[-1] + 5, 7):
                        win_end = win_start + win_size - 1
                        window_positions = [p for p in positions if win_start <= p <= win_end]
                        for forward in (True, False):
                            for exact_only in (True, False):
                                expected = _reference_score_window_registers(
                                    window_positions, pos_abun, win_start, win_end, phase,
                                    forward=forward, exact_only=exact_only,
                                )
                                got = fa._score_window_registers(
                                    window_positions, pos_abun, win_start, win_end, phase,
                                    forward=forward, exact_only=exact_only,
                                )
                                self.assertEqual(repr(expected), repr(got))
                                checked += 1
        self.assertGreater(checked, 5000)

    def test_short_and_empty_windows_keep_their_defaults(self):
        for exact_only in (True, False):
            empty = fa._score_window_registers([], {}, 1, 240, 24, forward=True, exact_only=exact_only)
            self.assertEqual(empty["score"], 0.0)
            self.assertIsNone(empty["best_register"])
            short = fa._score_window_registers([5], {5: 3}, 1, 60, 24, forward=True, exact_only=exact_only)
            self.assertEqual(short, empty)

    def test_strict_scan_matches_former_implementation(self):
        rng = random.Random(22)
        for phase in (21, 24):
            for value_kind in ("int", "fraction"):
                for _ in range(20):
                    span = rng.choice([150, 400, 900])
                    pos_abun = _random_pos_abun(rng, phase, span, rng.choice([0.0, 0.5, 0.9]), value_kind)
                    positions = sorted(pos_abun)
                    win_size = fa.WINDOW_MULTIPLIER * phase
                    for bounds in ((positions[0], positions[-1]), (None, None), (positions[0] - 30, positions[-1] + 30)):
                        for forward in (True, False):
                            expected = _reference_best_strict(pos_abun, phase, win_size, bounds[0], bounds[1], forward)
                            got = fa._best_sliding_window_score_generic_strict(
                                pos_abun, phase, win_size, seq_start=bounds[0], seq_end=bounds[1], forward=forward
                            )
                            self.assertEqual(repr(expected), repr(got))


if __name__ == "__main__":
    unittest.main()
