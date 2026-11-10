"""Contract checks for free-text extraction and catalog matching."""

from __future__ import annotations

import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "converting-description-to-vector"))
import gemini_query  # noqa: E402
import recommend_games  # noqa: E402
from recommend_games import CatalogError, TextGameRecommender  # noqa: E402


def mock_result(names, scores, excluded=()):
    vector = [scores.get(name, 0.0) for name in names]
    return {
        "names": names,
        "vector": vector,
        "active": [{"label": name, "score": score} for name, score in scores.items()],
        "excluded": list(excluded),
        "model": "test",
        "note": "",
    }


class TextRecommendationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ("01_catalog", "02_taxonomy", "03_features"):
            (self.root / "new_package" / folder).mkdir(parents=True)
        source = ROOT / "new_package/02_taxonomy/dimension_order.json"
        schema = json.loads(source.read_text(encoding="utf-8"))
        (self.root / "new_package/02_taxonomy/dimension_order.json").write_text(json.dumps(schema))
        self.names = schema["genre_dimension_order"] + schema["tag_dimension_order"]
        self._write_catalog()
        self._write_features()

    def _write_catalog(self):
        rows = [
            (1, "The brothers' journey", "Two brothers rescue their sister in a cooperative story."),
            (2, "Another adventure", "A multiplayer story through distant planets."),
            (3, "Shooter story", "Two brothers rescue their sister with shooting."),
            (4, "Unscored", "A game without a valid vector."),
        ]
        with (self.root / "new_package/01_catalog/games_for_features.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["appid", "name", "description", "short_description"])
            writer.writerows((appid, name, description, description) for appid, name, description in rows)

    def _write_features(self, names=None):
        names = names or self.names
        with (self.root / "new_package/03_features/game_features.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["appid", "name", "inclusion", "status", "note", "n_labels", *names])
            for appid, title, scores, status in [
                (1, "The brothers' journey", {"Multiplayer": 0.8, "Story Rich": 0.8}, "ok"),
                (2, "Another adventure", {"Multiplayer": 0.8, "Story Rich": 0.8}, "ok"),
                (3, "Shooter story", {"Multiplayer": 0.8, "Story Rich": 0.8, "Shooter": 0.9}, "ok"),
                (4, "Unscored", {}, "insufficient_description"),
            ]:
                writer.writerow([appid, title, "both", status, "", len(scores), *[scores.get(name, 0) for name in names]])

    def _edit_csv(self, relative_path, change):
        path = self.root / relative_path
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        change(rows)
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)

    def test_vector_ranking_exclusions_and_plot_tie_break(self):
        request = mock_result(self.names, {"Multiplayer": 0.9, "Story Rich": 0.9}, ["Shooter"])
        engine = TextGameRecommender(self.root, converter=lambda text: request)
        result = engine.recommend("Multiplayer story about two brothers rescuing their sister")
        self.assertEqual(result["status"], "ok")
        self.assertEqual([row["appid"] for row in result["recommendations"]], [1, 2])
        self.assertEqual(len(result["feature_vector"]), 93)
        self.assertEqual(result["feature_names"], self.names)
        self.assertIn("brother", result["recommendations"][0]["description_terms"])
        self.assertEqual(result["catalog_games"], 3)

    def test_schema_mismatch_fails_instead_of_misaligning_vectors(self):
        swapped = self.names.copy()
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self._write_features(swapped)
        with self.assertRaisesRegex(CatalogError, "taxonomy order"):
            TextGameRecommender(self.root, converter=lambda text: None)

    def test_unmatchable_request_does_not_offer_unrelated_games(self):
        request = mock_result(self.names, {"Puzzle": 0.9})
        engine = TextGameRecommender(self.root, converter=lambda text: request)
        result = engine.recommend("A puzzle game")
        self.assertEqual(result["status"], "no_matches")
        self.assertEqual(result["recommendations"], [])

    def test_invalid_input_is_rejected_before_conversion(self):
        calls = []
        engine = TextGameRecommender(self.root, converter=lambda text: calls.append(text))
        for value in (None, 1, "", " \t\n ", "x", "  ab  ", "x" * 2001,
                      "???", "\u200b" * 3, "😊" * 3):
            with self.subTest(text=repr(value)), self.assertRaisesRegex(ValueError, "3 to 2,000"):
                engine.recommend(value)
        for value in (False, True, None, "5", 0, -1, 21, 1.5):
            with self.subTest(limit=value), self.assertRaisesRegex(ValueError, "limit"):
                engine.recommend("a story game", limit=value)
        self.assertEqual(calls, [])

    def test_request_boundaries_and_limit(self):
        calls = []
        request = mock_result(self.names, {"Story Rich": 0.9})
        engine = TextGameRecommender(self.root, converter=lambda text: calls.append(text) or request)
        self.assertEqual(len(engine.recommend("RPG", limit=1)["recommendations"]), 1)
        self.assertEqual(len(engine.recommend("界" * 2000, limit=20)["recommendations"]), 3)
        self.assertEqual(calls, ["RPG", "界" * 2000])

    def test_converter_result_is_checked_before_ranking(self):
        valid = mock_result(self.names, {"Story Rich": 0.9})
        cases = [
            (None, ValueError),
            ({}, ValueError),
            ({**valid, "names": list(reversed(self.names))}, CatalogError),
            ({**valid, "vector": None}, ValueError),
            ({**valid, "vector": [0.0] * 93}, ValueError),
            ({**valid, "vector": [0.9]}, ValueError),
            ({**valid, "vector": recommend_games.np.array(0.9)}, ValueError),
            ({**valid, "vector": [10**1000] + [0.0] * 92}, ValueError),
            ({**valid, "vector": [float("nan")] + [0.0] * 92}, ValueError),
            ({**valid, "vector": [float("inf")] + [0.0] * 92}, ValueError),
            ({**valid, "vector": [-0.1] + [0.0] * 92}, ValueError),
            ({**valid, "vector": [1.1] + [0.0] * 92}, ValueError),
            ({**valid, "vector": ["0.9"] + [0.0] * 92}, ValueError),
            ({**valid, "vector": [True] + [0.0] * 92}, ValueError),
            ({**valid, "excluded": "Shooter"}, ValueError),
            ({**valid, "excluded": ["Shooter", "Shooter"]}, ValueError),
            ({**valid, "excluded": ["Unknown Tag"]}, ValueError),
            ({**valid, "excluded": ["Story Rich"]}, ValueError),
            ({**valid, "model": []}, ValueError),
            ({**valid, "note": []}, ValueError),
        ]
        for response, error in cases:
            with self.subTest(response=str(response)[:80]), self.assertRaises(error):
                TextGameRecommender(self.root, converter=lambda text: response).recommend("A story game")

    def test_query_features_are_derived_from_checked_vector(self):
        response = mock_result(self.names, {"Story Rich": 0.9})
        response["active"] = "malformed"
        result = TextGameRecommender(self.root, converter=lambda text: response).recommend("A story game")
        self.assertEqual(result["query_features"], [{"label": "Story Rich", "score": result["feature_vector"][self.names.index("Story Rich")]}])
        json.dumps(result)

    def test_exclusion_can_leave_no_matching_games(self):
        response = mock_result(self.names, {"Story Rich": 0.9}, ["Multiplayer"])
        result = TextGameRecommender(self.root, converter=lambda text: response).recommend("A story game without multiplayer")
        self.assertEqual(result["status"], "no_matches")
        self.assertEqual(result["recommendations"], [])

    def test_negated_words_do_not_reward_unwanted_description(self):
        self._edit_csv(
            "new_package/01_catalog/games_for_features.csv",
            lambda rows: (
                rows[1].__setitem__(2, "A peaceful multiplayer story"),
                rows[2].__setitem__(2, "A multiplayer story with shooting"),
            ),
        )
        response = mock_result(self.names, {"Multiplayer": 0.8, "Story Rich": 0.8}, ["Shooter"])
        result = TextGameRecommender(self.root, converter=lambda text: response).recommend(
            "A multiplayer story without shooting"
        )
        self.assertEqual([game["appid"] for game in result["recommendations"]], [1, 2])
        self.assertTrue(all("shooting" not in game["description_terms"] for game in result["recommendations"]))
        plot = TextGameRecommender(self.root, converter=lambda text: response).recommend(
            "A multiplayer story without shooting where two brothers rescue their sister"
        )
        self.assertIn("brother", plot["recommendations"][0]["description_terms"])

    def test_catalog_headers_and_ids_fail_cleanly(self):
        catalog = "new_package/01_catalog/games_for_features.csv"
        features = "new_package/03_features/game_features.csv"
        cases = [
            (catalog, lambda rows: rows[0].remove("appid"), "columns"),
            (catalog, lambda rows: rows.append(rows[1][:]), "Duplicate game ID"),
            (catalog, lambda rows: rows[1].__setitem__(0, "bad"), "game ID"),
            (catalog, lambda rows: rows[1].pop(), "Incomplete row"),
            (features, lambda rows: rows[0].__setitem__(3, "state"), "columns"),
            (features, lambda rows: rows.append(rows[1][:]), "Duplicate game ID"),
            (features, lambda rows: rows[1].__setitem__(3, "unknown"), "unknown feature status"),
            (features, lambda rows: rows[1].pop(), "Incomplete row"),
        ]
        for path, change, message in cases:
            with self.subTest(path=path, message=message):
                self._write_catalog()
                self._write_features()
                self._edit_csv(path, change)
                with self.assertRaisesRegex(CatalogError, message):
                    TextGameRecommender(self.root, converter=lambda text: None)

    def test_catalog_feature_values_fail_cleanly(self):
        path = "new_package/03_features/game_features.csv"
        for value, message in (("bad", "non-numeric"), ("nan", "invalid feature"),
                               ("inf", "invalid feature"), ("-0.1", "invalid feature"),
                               ("1.1", "invalid feature")):
            with self.subTest(value=value):
                self._write_features()
                self._edit_csv(path, lambda rows: rows[1].__setitem__(6, value))
                with self.assertRaisesRegex(CatalogError, message):
                    TextGameRecommender(self.root, converter=lambda text: None)

    def test_missing_metadata_or_usable_games_fails_cleanly(self):
        catalog = "new_package/01_catalog/games_for_features.csv"
        features = "new_package/03_features/game_features.csv"
        self._edit_csv(catalog, lambda rows: rows.pop(1))
        with self.assertRaisesRegex(CatalogError, "missing catalog metadata"):
            TextGameRecommender(self.root, converter=lambda text: None)
        self._write_catalog()
        self._edit_csv(features, lambda rows: [row.__setitem__(3, "non_game") for row in rows[1:]])
        with self.assertRaisesRegex(CatalogError, "no games"):
            TextGameRecommender(self.root, converter=lambda text: None)

    def test_unreadable_catalog_encoding_fails_cleanly(self):
        path = self.root / "new_package/03_features/game_features.csv"
        path.write_bytes(b"\xff\xfe")
        with self.assertRaisesRegex(CatalogError, "Could not parse game_features.csv"):
            TextGameRecommender(self.root, converter=lambda text: None)

    def test_cli_rejects_empty_input_without_loading_catalog_or_reading_stdin(self):
        class UnreadableStdin:
            def read(self):
                raise AssertionError("Explicit empty query must not read stdin")

        stderr = io.StringIO()
        with (patch.object(recommend_games, "TextGameRecommender", side_effect=AssertionError("catalog loaded")),
              patch.object(sys, "stdin", UnreadableStdin()), redirect_stderr(stderr)):
            self.assertEqual(recommend_games.main(["   "]), 2)
        self.assertIn("3 to 2,000", stderr.getvalue())

        stderr = io.StringIO()
        with (patch.object(recommend_games, "TextGameRecommender", side_effect=AssertionError("catalog loaded")),
              patch.object(sys, "stdin", io.StringIO("\n")), redirect_stderr(stderr)):
            self.assertEqual(recommend_games.main([]), 2)
        self.assertIn("3 to 2,000", stderr.getvalue())


class RealCatalogIntegrationTest(unittest.TestCase):
    def test_converter_and_catalog_taxonomy_definitions_match(self):
        converter_taxonomy = json.loads((ROOT / "converting-description-to-vector/taxonomy_v2.json").read_text(encoding="utf-8"))
        catalog_taxonomy = json.loads((ROOT / "new_package/02_taxonomy/taxonomy_v2.json").read_text(encoding="utf-8"))
        for kind in ("genres", "tags"):
            with self.subTest(kind=kind):
                converter_labels = {item["id"]: item["definition"] for item in converter_taxonomy[kind]["labels"]}
                catalog_labels = {item["id"]: item["definition"] for item in catalog_taxonomy[kind]["labels"]}
                self.assertEqual(converter_labels, catalog_labels)

    def test_mocked_gemini_request_matches_real_catalog(self):
        engine = TextGameRecommender()
        model_response = {
            "features": [
                {"label": "Multiplayer", "score": 0.9, "evidence": "Requests multiplayer."},
                {"label": "Story Rich", "score": 0.9, "evidence": "Requests story mode."},
            ],
            "excluded": ["Shooter"],
            "note": "A multiplayer story game.",
        }
        envelope = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(model_response)}]}}]}

        def fake_open(request, timeout):
            self.assertEqual(timeout, 50)
            self.assertIn("generateContent", request.full_url)
            return io.BytesIO(json.dumps(envelope).encode())

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch.object(gemini_query, "urlopen", fake_open):
            result = engine.recommend("A multiplayer game with a story", limit=3)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["catalog_games"], 6750)
        self.assertEqual(len(result["feature_vector"]), 93)
        self.assertEqual(len(result["recommendations"]), 3)
        self.assertEqual(result["feature_vector"][result["feature_names"].index("Shooter")], 0)
        for game in result["recommendations"]:
            self.assertGreater(len(game["matched_features"]), 0)
            self.assertTrue(game["url"].endswith(f"/{game['appid']}/"))

        output = io.StringIO()
        with (patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
              patch.object(gemini_query, "urlopen", fake_open),
              patch.object(recommend_games, "TextGameRecommender", return_value=engine),
              redirect_stdout(output)):
            exit_code = recommend_games.main(["A multiplayer story game", "--limit", "1", "--json"])
        self.assertEqual(exit_code, 0)
        cli_result = json.loads(output.getvalue())
        self.assertEqual(len(cli_result["recommendations"]), 1)
        self.assertEqual(len(cli_result["feature_vector"]), 93)

    def test_real_catalog_ranking_invariants_across_requests(self):
        engine = TextGameRecommender()
        rows_by_id = {game["appid"]: i for i, game in enumerate(engine.games)}
        rng = random.Random(20261009)
        for _ in range(20):
            chosen = rng.sample(engine.names, rng.randint(1, 5))
            excluded = rng.sample([name for name in engine.names if name not in chosen], rng.randint(0, 2))
            scores = {name: rng.choice((0.3, 0.6, 0.9)) for name in chosen}
            engine.converter = lambda text, scores=scores, excluded=excluded: mock_result(engine.names, scores, excluded)
            result = engine.recommend("A story adventure game without shooting", limit=12)
            games = result["recommendations"]
            self.assertLessEqual(len(games), 12)
            self.assertEqual(len({game["appid"] for game in games}), len(games))
            self.assertEqual([game["score"] for game in games], sorted((game["score"] for game in games), reverse=True))
            self.assertEqual(result["status"], "ok" if games else "no_matches")
            json.dumps(result)
            for game in games:
                self.assertTrue(game["matched_features"])
                self.assertTrue(0 <= game["score"] <= 1)
                row = rows_by_id[game["appid"]]
                self.assertTrue(all(engine.matrix[row, engine.index[label]] == 0 for label in excluded))


if __name__ == "__main__":
    unittest.main()
