"""Headless smoke tests of the Streamlit dashboard: every page, view and section renders without an exception.

Run from the project root:  python -m unittest discover -s tests -v   (about two minutes; the AI Lab trains its models)
"""
import unittest

from _setup import DATA_READY, ROOT, SKIP_REASON

from streamlit.testing.v1 import AppTest

APP = str(ROOT / "scripts" / "dashboard" / "app.py")
PAGES = ["page_overview.py", "page_map.py", "page_gaez_gcs.py", "page_lake.py", "page_eda.py", "page_explorer.py",
         "page_experiments.py", "page_graph.py", "page_ai_lab.py"]


@unittest.skipUnless(DATA_READY, SKIP_REASON)
class DashboardSmoke(unittest.TestCase):
    def setUp(self):
        self.at = AppTest.from_file(APP, default_timeout=600)
        self.at.run()
        self.assertNoException("start")

    def assertNoException(self, where):
        self.assertEqual([str(e.value)[:500] for e in self.at.exception], [], where)

    def show(self, page, **state):
        for key, value in state.items():
            self.at.session_state[key] = value
        self.at.switch_page(page)
        self.at.run()
        self.assertNoException(f"{page} {state}")

    def test_every_page_at_both_resolutions(self):
        for res in ["1 km", "~9 km"]:
            for page in PAGES:
                with self.subTest(page=page, res=res):
                    self.show(page, gaez_res=res)

    def test_every_map_view(self):
        for view in ["Recommendation", "Every crop", "One crop on the map", "Uncertainty", "Raw data layers"]:
            with self.subTest(view=view):
                self.show("page_map.py", map_view=view)

    def test_every_recommendation_rule(self):
        for group in ["All 10 crops", "Food crops", "Tree & estate crops"]:
            for ranking in ["Relative", "Absolute"]:
                with self.subTest(group=group, ranking=ranking):
                    self.show("page_overview.py", rule_group=group, rule_ranking=ranking)

    def test_every_eda_section(self):
        for section in ["Column profiles", "Distributions by group", "Correlations", "Relationships", "Spatial patterns",
                        "Categories", "Irrigated vs rain-fed", "PCA"]:
            with self.subTest(section=section):
                self.show("page_eda.py", eda_section=section)

    def test_graph_views_and_extreme_thresholds(self):
        cases = [dict(graph_patch_view="Largest patch and bottlenecks"), dict(graph_matrix="Jaccard overlap"),
                 dict(graph_comm=1), dict(graph_crop="rubber"),
                 dict(graph_threshold=0), dict(graph_threshold=10000)]   # everything suitable / nothing suitable
        for state in cases:
            with self.subTest(**state):
                self.show("page_graph.py", **state)

    def test_every_experiment(self):
        self.show("page_experiments.py")
        options = next(s for s in self.at.selectbox if s.label == "Experiment").options
        for option in options:
            with self.subTest(experiment=option):
                next(s for s in self.at.selectbox if s.label == "Experiment").set_value(option)
                self.at.run()
                self.assertNoException(option)


if __name__ == "__main__":
    unittest.main()
