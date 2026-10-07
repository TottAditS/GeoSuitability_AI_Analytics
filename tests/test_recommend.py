"""The crop recommendation rule of the dashboard (common.recommend) on small synthetic tables; needs no data."""
import unittest

import numpy as np
import pandas as pd

import _setup  # noqa: F401  (import paths)

import common as C


def table(rows, res="1 km"):
    return pd.DataFrame(rows, columns=[C.gaez_col(c, "index", res) for c in C.CROPS], dtype=float)


def row(base=1000.0, **crops):
    values = [base] * len(C.CROPS)
    for crop, v in crops.items():
        values[C.CROPS.index(crop)] = v
    return values


class AbsoluteRule(unittest.TestCase):
    def recommend(self, rows, res="1 km"):
        return C.recommend(table(rows, res), C.CROPS, "Absolute", res=res)

    def test_highest_index_wins(self):
        out = self.recommend([row(rice=8000, maize=6000)])
        self.assertEqual(out["label"].iloc[0], "Rice")
        self.assertEqual(out["winner"].iloc[0], "rice")
        self.assertEqual(out["winner_index"].iloc[0], 8000)

    def test_runner_up_within_500_is_a_tie(self):
        self.assertEqual(self.recommend([row(rice=8000, maize=7600)])["label"].iloc[0], "Maize / Rice (tied)")

    def test_below_5000_is_not_recommended(self):
        out = self.recommend([row(base=4000, rice=4999)])
        self.assertEqual(out["label"].iloc[0], "Not recommended")
        self.assertTrue(pd.isna(out["winner"].iloc[0]))

    def test_crop_without_a_value_cannot_win(self):
        self.assertEqual(self.recommend([row(rice=np.nan, maize=6000)])["label"].iloc[0], "Maize")

    def test_cell_without_any_value_is_no_cropland(self):
        self.assertEqual(self.recommend([[np.nan] * len(C.CROPS)])["label"].iloc[0], "No cropland / water")

    def test_resolution_picks_the_columns(self):
        out = self.recommend([row(coconut=9000)], res="~9 km")
        self.assertEqual(out["label"].iloc[0], "Coconut")

    def test_margin_between_0_and_1(self):
        out = self.recommend([row(rice=8000, maize=6000), row(rice=9000, maize=8999), row(base=4000)])
        self.assertTrue(out["margin"].between(0, 1).all())


class RelativeRule(unittest.TestCase):
    def test_comparative_advantage(self):
        # rice is rarely this high in the reference, coconut often: equal indices, rice is relatively stronger
        reference = table([row(rice=5000, coconut=9000), row(rice=5100, coconut=9500), row(rice=9000, coconut=9000)])
        out = C.recommend(table([row(rice=9000, coconut=9000)]), C.CROPS, "Relative", reference=reference, res="1 km")
        self.assertEqual(out["label"].iloc[0], "Rice")

    def test_still_needs_5000(self):
        reference = table([row(rice=1000), row(rice=4000)])
        out = C.recommend(table([row(rice=4000)]), C.CROPS, "Relative", reference=reference, res="1 km")
        self.assertEqual(out["label"].iloc[0], "Not recommended")


class Columns(unittest.TestCase):
    def test_gaez_column_names(self):
        self.assertEqual(C.gaez_col("rice", "index", "1 km"), "gaez1k_index_rice")
        self.assertEqual(C.gaez_col("rice", "class", "~9 km"), "gaez_class_rice")
        self.assertEqual(C.gaez_classes("1 km")[9], "No cropland")
        self.assertEqual(C.gaez_classes("~9 km")[9], "Water")

    def test_catalogue_knows_every_family(self):
        cat = C.column_catalogue(["gaez1k_index_rice_rainfed", "gaez_class_coffee", "gcs2024_maize", "s2_NDVI_mean",
                                  "soil_ph", "bio1_temp_mean_c"]).set_index("column")
        self.assertEqual(cat.loc["gaez1k_index_rice_rainfed", "group"], "GAEZ index (rain-fed 1 km, audit)")
        self.assertEqual(cat.loc["s2_NDVI_mean", "group"], "Sentinel-2")
        self.assertNotIn("Other", set(cat["group"]))


if __name__ == "__main__":
    unittest.main()
