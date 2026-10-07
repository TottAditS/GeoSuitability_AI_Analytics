"""The shared experiment library: targets, the warehouse table and the spatial cross-validation folds."""
import unittest

import numpy as np

from _setup import DATA_READY, SKIP_REASON

import exp_lib as E


class TargetColumns(unittest.TestCase):
    def test_experiments_learn_the_1km_target(self):
        self.assertEqual(E.TARGET_RES, "1km")
        self.assertEqual(E.target_col("rice"), "gaez1k_index_rice")
        self.assertEqual(E.target_col("rice", "9km"), "gaez_index_rice")
        self.assertEqual(E.gaez_cols(), [f"gaez1k_index_{c}" for c in E.CROPS])

    def test_ten_crops_and_sentinel2_features(self):
        self.assertEqual(len(E.CROPS), 10)
        self.assertEqual(len(E.feature_columns()), 14)
        self.assertEqual(len(E.FEATURE_GROUPS["sentinel2"]), 23)


@unittest.skipUnless(DATA_READY, SKIP_REASON)
class WarehouseTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = E.load_table()
        cls.full = E.load_table(complete=False)

    def test_one_row_per_grid_cell(self):
        self.assertEqual(len(self.full), 9333)
        self.assertTrue(self.full["grid_id"].is_unique)
        self.assertFalse(self.full.duplicated(["row", "col"]).any())

    def test_complete_rows_have_every_target(self):
        self.assertGreater(len(self.df), 9000)
        self.assertFalse(self.df[E.gaez_cols()].isna().any().any())

    def test_targets_on_the_gaez_scale(self):
        values = self.df[E.gaez_cols()].to_numpy()
        self.assertTrue(((values >= 0) & (values <= 10000)).all())

    def test_feature_groups_exist(self):
        for group, cols in E.FEATURE_GROUPS.items():
            if group == "sentinel2" and not E.S2_CSV.exists():
                continue
            with self.subTest(group=group):
                self.assertEqual([c for c in cols if c not in self.df.columns], [])

    def test_pixel_folds_hold_out_whole_pixels(self):
        seen = np.zeros(len(self.df), dtype=int)
        for tr, te in E.make_splits(self.df, "pixel", 5, seed=0):
            self.assertEqual(set(self.df["pix"].iloc[tr]) & set(self.df["pix"].iloc[te]), set())
            seen[te] += 1
        self.assertTrue((seen == 1).all(), "every cell is tested exactly once")

    def test_block_folds_are_disjoint(self):
        for tr, te in E.make_splits(self.df, "block10", 5, seed=0):
            self.assertEqual(len(np.intersect1d(tr, te)), 0)

    def test_rule_labels(self):
        labels = E.crop_labels(self.df, "thresh")
        self.assertTrue(set(labels.unique()) <= set(E.CROPS) | {"Not recommended"})


if __name__ == "__main__":
    unittest.main()
