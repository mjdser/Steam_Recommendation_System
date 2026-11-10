"""Steam Web API contract and real saved-model checks without a live key."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import unittest
from urllib.error import HTTPError, URLError

from product_app.backend.steam_recommend import (
    SteamHistoryRecommender,
    SteamRecommendationError,
    score_owned_games,
    validate_steam_request,
)


ROOT = Path(__file__).resolve().parents[3]
TEST_KEY = "a" * 32
TEST_STEAM_ID = "76561198000000000"


class SteamRequestTest(unittest.TestCase):
    def test_rejects_invalid_server_key_id_and_limit_before_steam_call(self):
        cases = [
            ("", TEST_STEAM_ID, 6, 503), ("short", TEST_STEAM_ID, 6, 503),
            (TEST_KEY + " ", TEST_STEAM_ID, 6, 503), (TEST_KEY, "123", 6, 400),
            (TEST_KEY, TEST_STEAM_ID, True, 400), (TEST_KEY, TEST_STEAM_ID, 0, 400),
            (TEST_KEY, TEST_STEAM_ID, 21, 400), (TEST_KEY, TEST_STEAM_ID, "6", 400),
        ]
        for key, steam_id, limit, status in cases:
            with self.subTest(key_length=len(key), steam_id=steam_id, limit=limit):
                with self.assertRaises(SteamRecommendationError) as caught:
                    validate_steam_request(key, steam_id, limit)
                self.assertEqual(caught.exception.status, status)
                self.assertNotIn(TEST_KEY, caught.exception.message)

    def test_library_interest_uses_playtime_and_skips_unplayed_non_games(self):
        games = [
            {"appid": 620, "name": "Portal 2", "playtime_forever": 600, "playtime_2weeks": 60},
            {"appid": 105600, "name": "Terraria", "playtime_forever": 1200, "playtime_2weeks": 0},
            {"appid": 620, "name": "Portal 2", "playtime_forever": 100, "playtime_2weeks": 0},
            {"appid": 431960, "name": "Wallpaper Engine", "playtime_forever": 9999},
            {"appid": 730, "name": "Counter-Strike 2", "playtime_forever": 0},
            {"appid": 999, "name": "Invalid", "playtime_forever": "nan"},
        ]
        scored = score_owned_games(games)
        self.assertEqual([game["appid"] for game in scored], [620, 105600])
        self.assertEqual(scored[0]["playtime_forever_hours"], 10.0)
        self.assertEqual(scored[0]["playtime_2weeks_hours"], 1.0)
        self.assertGreater(scored[0]["interest_score"], scored[1]["interest_score"])


class SteamTransportTest(unittest.TestCase):
    def engine_with_response(self, response):
        engine = object.__new__(SteamHistoryRecommender)
        engine.opener = lambda request, timeout: BytesIO(json.dumps(response).encode("utf-8"))
        return engine

    def test_empty_and_private_libraries_are_distinct(self):
        for response, code in (
            ({"response": {}}, "steam_library_private"),
            ({"response": {"game_count": 0}}, "steam_library_empty"),
            ({"response": {"game_count": 1, "games": []}}, "steam_library_private"),
        ):
            with self.subTest(code=code):
                with self.assertRaises(SteamRecommendationError) as caught:
                    self.engine_with_response(response)._fetch_library(TEST_KEY, TEST_STEAM_ID)
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn(TEST_KEY, str(caught.exception))

    def test_bad_key_and_network_errors_never_echo_credentials(self):
        for upstream_error, code in (
            (HTTPError("https://example.test/?key=" + TEST_KEY, 400, "Bad Request", {}, None), "steam_request_rejected"),
            (HTTPError("https://example.test/?key=" + TEST_KEY, 403, "Access Denied", {}, None), "steam_auth_failed"),
            (HTTPError("https://example.test/?key=" + TEST_KEY, 429, "Too Many Requests", {}, None), "steam_rate_limited"),
            (URLError("https://example.test/?key=" + TEST_KEY), "steam_unavailable"),
        ):
            with self.subTest(code=code):
                engine = object.__new__(SteamHistoryRecommender)

                def fail(_request, timeout):
                    raise upstream_error

                engine.opener = fail
                with self.assertRaises(SteamRecommendationError) as caught:
                    engine._fetch_library(TEST_KEY, TEST_STEAM_ID)
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn(TEST_KEY, str(caught.exception))

    def test_invalid_steam_json_and_schema_are_sanitized(self):
        for response in (b"not json", b"[]", b'{"response":{"games":[]}}'):
            with self.subTest(response=response[:15]):
                engine = object.__new__(SteamHistoryRecommender)
                engine.opener = lambda request, timeout: BytesIO(response)
                with self.assertRaises(SteamRecommendationError) as caught:
                    engine._fetch_library(TEST_KEY, TEST_STEAM_ID)
                self.assertEqual(caught.exception.code, "steam_unavailable")

    def test_owned_but_unplayed_and_unsupported_histories_get_distinct_errors(self):
        engine = object.__new__(SteamHistoryRecommender)
        engine._fetch_library = lambda _key, _id: ([
            {"appid": 620, "name": "Portal 2", "playtime_forever": 0},
        ], 1)
        with self.assertRaises(SteamRecommendationError) as caught:
            engine.recommend(TEST_KEY, TEST_STEAM_ID)
        self.assertEqual(caught.exception.code, "steam_no_played_games")

        engine._fetch_library = lambda _key, _id: ([
            {"appid": 99999999, "name": "Game outside the catalog", "playtime_forever": 120},
        ], 1)
        engine.appid_to_index = {620: 0}
        engine.n_slots = 6
        with self.assertRaises(SteamRecommendationError) as caught:
            engine.recommend(TEST_KEY, TEST_STEAM_ID)
        self.assertEqual(caught.exception.code, "steam_no_supported_games")


class SteamProfileTest(unittest.TestCase):
    def engine_with_response(self, response):
        engine = object.__new__(SteamHistoryRecommender)
        engine.opener = lambda request, timeout: BytesIO(response)
        return engine

    def test_username_comes_only_from_matching_steamid(self):
        payload = {"response": {"players": [
            {"steamid": "76561198000000001", "personaname": "Other account"},
            {"steamid": TEST_STEAM_ID, "personaname": "  Player One 🎮  "},
        ]}}
        engine = self.engine_with_response(json.dumps(payload).encode("utf-8"))
        self.assertEqual(engine._fetch_username(TEST_KEY, TEST_STEAM_ID), "Player One 🎮")

    def test_missing_or_invalid_profile_keeps_recommendations_available(self):
        for response in (
            b"not json", b"[]", b'{"response":{"players":[]}}',
            b'{"response":{"players":[{"steamid":"76561198000000001","personaname":"Other"}]}}',
            b'{"response":{"players":[{"steamid":"76561198000000000","personaname":"   "}]}}',
            b"x" * 128_001,
        ):
            with self.subTest(response=response[:30]):
                engine = self.engine_with_response(response)
                self.assertIsNone(engine._fetch_username(TEST_KEY, TEST_STEAM_ID))

        engine = object.__new__(SteamHistoryRecommender)
        def fail(request, timeout):
            raise HTTPError(request.full_url, 503, "Unavailable", {}, None)
        engine.opener = fail
        self.assertIsNone(engine._fetch_username(TEST_KEY, TEST_STEAM_ID))


class RealHybridModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import xgboost  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("XGBoost is not installed in this Python environment")
        cls.requests = []
        library = {
            "response": {
                "game_count": 4,
                "games": [
                    {"appid": 620, "name": "Portal 2", "playtime_forever": 900, "playtime_2weeks": 80},
                    {"appid": 105600, "name": "Terraria", "playtime_forever": 1800},
                    {"appid": 730, "name": "Counter-Strike 2", "playtime_forever": 0},
                    {"appid": 413150, "name": "Stardew Valley", "playtime_forever": 300},
                ],
            }
        }

        def fake_open(request, timeout):
            cls.requests.append((request.full_url, timeout))
            if "/ISteamUser/GetPlayerSummaries/" in request.full_url:
                return BytesIO(json.dumps({"response": {"players": [
                    {"steamid": TEST_STEAM_ID, "personaname": "Sample Player"}
                ]}}).encode("utf-8"))
            return BytesIO(json.dumps(library).encode("utf-8"))

        cls.engine = SteamHistoryRecommender(ROOT, opener=fake_open)

    def test_saved_hybrid_model_ranks_real_games_and_excludes_owned(self):
        self.requests.clear()
        result = self.engine.recommend(TEST_KEY, TEST_STEAM_ID, limit=4)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source"], "steam_history")
        self.assertEqual(result["model"], "clusters_v3")
        self.assertEqual(result["catalog_games"], 6750)
        self.assertEqual(result["owned_game_count"], 4)
        self.assertEqual(result["matched_history_count"], 3)
        self.assertEqual(len(result["recommendations"]), 4)
        self.assertEqual(len({game["appid"] for game in result["recommendations"]}), 4)
        self.assertFalse({game["appid"] for game in result["recommendations"]} & {620, 105600, 730, 413150})
        self.assertTrue(all(0 <= game["score"] <= 1 for game in result["recommendations"]))
        self.assertTrue(all(game["url"].startswith("https://store.steampowered.com/app/")
                            for game in result["recommendations"]))
        self.assertEqual(result["steam_id"], TEST_STEAM_ID)
        self.assertEqual(result["steam_username"], "Sample Player")
        self.assertNotIn(TEST_KEY, json.dumps(result))
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(self.requests[0][0].startswith("https://api.steampowered.com/"))
        self.assertIn("/ISteamUser/GetPlayerSummaries/", self.requests[1][0])
        self.assertIn("steamids=" + TEST_STEAM_ID, self.requests[1][0])

    def test_profile_lookup_failure_does_not_block_saved_model(self):
        original_opener = self.engine.opener
        def without_profile(request, timeout):
            if "/ISteamUser/GetPlayerSummaries/" in request.full_url:
                raise URLError("Profile temporarily unavailable")
            return original_opener(request, timeout)
        self.engine.opener = without_profile
        try:
            result = self.engine.recommend(TEST_KEY, TEST_STEAM_ID, limit=2)
        finally:
            self.engine.opener = original_opener
        self.assertIsNone(result["steam_username"])
        self.assertEqual(len(result["recommendations"]), 2)


if __name__ == "__main__":
    unittest.main()
