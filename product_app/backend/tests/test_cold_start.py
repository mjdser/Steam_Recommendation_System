"""Check cluster discovery against both a tiny fixture and the saved artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from product_app.backend.cold_start import ColdStartCatalog, ColdStartDataError


ROOT = Path(__file__).resolve().parents[3]


def _write_csv(root: Path, relative: str, fields: list[str], rows: list[dict]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path) -> None:
    _write_csv(root, "new_package/01_catalog/games_for_features.csv",
               ["appid", "name", "positive", "negative", "price"], [
                   {"appid": 10, "name": "Free Adventure", "positive": 100, "negative": 0, "price": 0},
                   {"appid": 20, "name": "Paid Adventure", "positive": 10, "negative": 10, "price": 30},
                   {"appid": 30, "name": "Other World", "positive": 0, "negative": 100, "price": 80},
               ])
    _write_csv(root, "new_package/04_clusters/game_clusters.csv",
               ["appid", "status", "cluster_id", "cluster_label"], [
                   {"appid": 10, "status": "ok", "cluster_id": 0, "cluster_label": "Adventures"},
                   {"appid": 20, "status": "ok", "cluster_id": 0, "cluster_label": "Adventures"},
                   {"appid": 30, "status": "ok", "cluster_id": 1, "cluster_label": "Other"},
               ])
    _write_csv(root, "new_package/04_clusters/cluster_summary.csv",
               ["cluster_id", "n_games", "label", "top_features", "examples"], [
                   {"cluster_id": 0, "n_games": 2, "label": "Adventures", "top_features": "Adventure 0.8", "examples": "Free Adventure; Paid Adventure"},
                   {"cluster_id": 1, "n_games": 1, "label": "Other", "top_features": "Casual 0.7", "examples": "Other World"},
               ])
    _write_csv(root, "cold_start/game_tiers.csv", ["appid", "hardware_tier"], [
        {"appid": 10, "hardware_tier": "low"},
        {"appid": 20, "hardware_tier": "medium"},
        {"appid": 30, "hardware_tier": ""},
    ])
    players = [{"appid": 10, "current_players": 100},
               {"appid": 20, "current_players": 1000},
               {"appid": 30, "current_players": 10}]
    (root / "cold_start/current_players.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in players), encoding="utf-8")


class ColdStartFixtureTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        _fixture(self.root)
        self.catalog = ColdStartCatalog(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def test_formula_and_current_cluster_membership(self):
        result = self.catalog.recommend(0)
        self.assertEqual(result["source"], "cluster_cold_start")
        self.assertEqual(result["cluster"]["label"], "Adventures")
        self.assertEqual(result["eligible_games"], 2)
        scores = {row["appid"]: row["score"] for row in result["recommendations"]}
        prior = 110 / 220
        expected = 100 * math.sqrt(((100 + 100 * prior) / 200) * (math.log1p(100) / math.log1p(1000)))
        self.assertAlmostEqual(scores[10], expected, places=4)
        self.assertEqual({row["appid"] for row in result["recommendations"]}, {10, 20})
        self.assertEqual(self.catalog.recommend(1)["recommendations"][0]["appid"], 30)
        self.assertEqual(result["recommendations"][0]["rank"], 1)
        self.assertEqual(result["recommendations"][0]["url"], "https://store.steampowered.com/app/10/")

    def test_budget_and_cumulative_hardware_filters(self):
        self.assertEqual([row["appid"] for row in self.catalog.recommend(0, budget="0")["recommendations"]], [10])
        self.assertEqual([row["appid"] for row in self.catalog.recommend(0, budget="20")["recommendations"]], [10])
        self.assertEqual([row["appid"] for row in self.catalog.recommend(0, hardware="low")["recommendations"]], [10])
        self.assertEqual({row["appid"] for row in self.catalog.recommend(0, hardware="medium")["recommendations"]}, {10, 20})
        self.assertEqual(self.catalog.recommend(1, hardware="high")["status"], "no_matches")
        self.assertEqual(self.catalog.recommend(1, hardware="high")["eligible_games"], 0)

    def test_invalid_selection_and_copy_isolation(self):
        for args in ((-1,), (True,), ("0",), (99,), (0, "25"), (0, "all", "ultra"),
                     (0, "all", "all", 0), (0, "all", "all", True), (0, "all", "all", 21)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.catalog.recommend(*args)
        clusters = self.catalog.list_clusters()
        clusters[0]["examples"].append("Changed")
        self.assertNotIn("Changed", self.catalog.list_clusters()[0]["examples"])
        result = self.catalog.recommend(0)
        result["recommendations"][0]["name"] = "Changed"
        self.assertNotEqual(self.catalog.recommend(0)["recommendations"][0]["name"], "Changed")

    def test_rejects_mixed_cluster_generations(self):
        assignment = self.root / "new_package/04_clusters/game_clusters.csv"
        assignment.write_text(assignment.read_text().replace("Adventures", "Old adventures", 1))
        with self.assertRaisesRegex(ColdStartDataError, "do not match"):
            ColdStartCatalog(self.root)


class SavedArtifactTest(unittest.TestCase):
    def test_all_current_clusters_and_filter_combinations(self):
        catalog = ColdStartCatalog(ROOT)
        clusters = catalog.list_clusters()
        self.assertEqual(len(clusters), 18)
        self.assertEqual(sum(cluster["n_games"] for cluster in clusters), 6750)
        by_id = {cluster["id"]: cluster for cluster in clusters}
        self.assertEqual(by_id[2]["label"], "FPS / Shooter / First-Person")
        self.assertEqual(catalog.recommend(2, limit=1)["recommendations"][0]["appid"], 730)
        for cluster in clusters:
            for budget in ("all", "0", "20", "80"):
                for hardware in ("all", "low", "medium", "high"):
                    with self.subTest(cluster=cluster["id"], budget=budget, hardware=hardware):
                        result = catalog.recommend(cluster["id"], budget, hardware)
                        self.assertEqual(result["status"], "ok")
                        self.assertGreaterEqual(len(result["recommendations"]), 1)
                        self.assertLessEqual(len(result["recommendations"]), 10)
                        self.assertEqual([row["rank"] for row in result["recommendations"]],
                                         list(range(1, len(result["recommendations"]) + 1)))
                        self.assertEqual([row["score"] for row in result["recommendations"]],
                                         sorted((row["score"] for row in result["recommendations"]), reverse=True))
                        if budget != "all":
                            self.assertTrue(all(row["price"] <= float(budget) for row in result["recommendations"]))
                        if hardware != "all":
                            levels = {"low": 0, "medium": 1, "high": 2}
                            self.assertTrue(all(row["hardware_tier"] in levels and
                                                levels[row["hardware_tier"]] <= levels[hardware]
                                                for row in result["recommendations"]))


if __name__ == "__main__":
    unittest.main()
