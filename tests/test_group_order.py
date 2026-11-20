import unittest

import numpy as np
import pandas as pd

from phasis.group_order import legacy_group_indices


class LegacyGroupOrderTests(unittest.TestCase):
    """The order is written out here, so the test does not depend on pandas."""

    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "chromosome": ["chr2", "chr1", "chr1", "chr3", "chr2", "chr1"],
                "clusterID": ["cA", "cB", "cA", "cC", "cB", "cC"],
            }
        )

    def test_groups_follow_first_key_then_second_key_first_appearance(self):
        groups = legacy_group_indices(self.frame, ["chromosome", "clusterID"])
        self.assertEqual(
            list(groups),
            [
                ("chr2", "cA"),
                ("chr2", "cB"),
                ("chr1", "cA"),
                ("chr1", "cB"),
                ("chr1", "cC"),
                ("chr3", "cC"),
            ],
        )
        np.testing.assert_array_equal(groups[("chr1", "cA")], [2])
        np.testing.assert_array_equal(groups[("chr2", "cB")], [4])

    def test_row_positions_are_ascending_within_a_group(self):
        frame = pd.DataFrame({"a": ["x", "y", "x", "x"], "b": ["p", "p", "p", "q"]})
        groups = legacy_group_indices(frame, ["a", "b"])
        self.assertEqual(list(groups), [("x", "p"), ("x", "q"), ("y", "p")])
        np.testing.assert_array_equal(groups[("x", "p")], [0, 2])

    def test_missing_keys_are_dropped_or_kept(self):
        frame = pd.DataFrame({"a": ["x", "x", "y"], "b": ["p", None, "p"]})
        self.assertEqual(list(legacy_group_indices(frame, ["a", "b"])), [("x", "p"), ("y", "p")])
        kept = list(legacy_group_indices(frame, ["a", "b"], dropna=False))
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[0], ("x", "p"))
        self.assertTrue(pd.isna(kept[1][1]))

    def test_empty_frame(self):
        self.assertEqual(legacy_group_indices(self.frame.iloc[0:0], ["chromosome", "clusterID"]), {})


if __name__ == "__main__":
    unittest.main()
