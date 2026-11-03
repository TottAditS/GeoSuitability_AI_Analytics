"""lake_path picks HDFS when the NameNode answers, and the local datasets/ folder when it does not."""
import os
import unittest
from pathlib import Path
from unittest import mock

import _setup  # noqa: F401  (import paths)
import exp_spark as S


def _probe(reachable: bool):
    if reachable:
        return mock.patch("socket.create_connection", return_value=mock.MagicMock())
    return mock.patch("socket.create_connection", side_effect=OSError("refused"))


class LakePathTests(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        os.environ.pop("LAKE_URI", None)

    def tearDown(self):
        self._env.stop()

    def test_hdfs_when_the_namenode_answers(self):
        with _probe(True):
            self.assertEqual(S.lake_path("processed", "features", "a.csv"),
                             "hdfs://localhost:9000/lake/processed/features/a.csv")

    def test_local_when_hdfs_is_down(self):
        with _probe(False):
            self.assertEqual(Path(S.lake_path("processed", "features", "a.csv")),
                             S.LOCAL_LAKE / "processed" / "features" / "a.csv")

    def test_local_override_wins_over_a_running_hdfs(self):
        os.environ["LAKE_URI"] = "local"
        with _probe(True):
            self.assertTrue(Path(S.lake_path("x")).is_relative_to(S.LOCAL_LAKE))

    def test_explicit_uri_is_used_without_probing(self):
        os.environ["LAKE_URI"] = "hdfs://other:9000/data/"
        with _probe(False):
            self.assertEqual(S.lake_path("/processed/", "a.csv"), "hdfs://other:9000/data/processed/a.csv")


if __name__ == "__main__":
    unittest.main()
