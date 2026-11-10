"""HTTP contracts for the Steam-history and current-cluster pathways."""

from __future__ import annotations

from http.client import HTTPConnection
import json
import os
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from product_app.backend.server import create_server, steam_api_key_settings
from product_app.backend.steam_recommend import SteamRecommendationError


class FakeSteamRecommender:
    def __init__(self):
        self.calls = []
        self.error = None

    def recommend(self, *, api_key, steam_id, limit):
        self.calls.append((api_key, steam_id, limit))
        if self.error is not None:
            raise self.error
        return {
            "status": "ok",
            "source": "steam_history",
            "model": "rank_hybrid",
            "steam_id": steam_id,
            "steam_username": "Mock Player",
            "history_count": 12,
            "matched_history_count": 6,
            "catalog_games": 6750,
            "recommendations": [{
                "rank": 1, "appid": 123, "name": "Example game",
                "url": "https://store.steampowered.com/app/123/",
                "score": 0.7, "reason": "Related to your library.",
                "matched_features": ["Adventure"],
            }][:limit],
        }


class SteamServerConfigurationTest(unittest.TestCase):
    def test_env_file_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            config = root / "product_app" / ".env"
            self.assertEqual(steam_api_key_settings(root), "")
            config.parent.mkdir()
            config.write_text('STEAM_API_KEY="' + "A" * 32 + '"\n', encoding="utf-8")
            self.assertEqual(steam_api_key_settings(root), "A" * 32)
            with patch.dict(os.environ, {"STEAM_API_KEY": "B" * 32}):
                self.assertEqual(steam_api_key_settings(root), "B" * 32)
            with patch.dict(os.environ, {"STEAM_API_KEY": ""}):
                self.assertEqual(steam_api_key_settings(root), "")


class NewRoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.steam = FakeSteamRecommender()
        cls.steam_key = "A" * 32
        cls.server = create_server(
            port=0,
            converter=lambda _: {},
            config_provider=lambda: (None, "test-model"),
            steam_config_provider=lambda: cls.steam_key,
            steam_recommender=cls.steam,
        )
        cls.worker = Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=2)

    def request(self, method, path, body=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=15)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        try:
            connection.request(method, path, json.dumps(body).encode() if body is not None else None, headers)
            response = connection.getresponse()
            value = json.loads(response.read())
            return response.status, dict(response.getheaders()), value
        finally:
            connection.close()

    def test_cluster_choices_use_current_18_cluster_artifact(self):
        status, _, body = self.request("GET", "/api/clusters")
        self.assertEqual(status, 200)
        self.assertEqual(body["source"], "cluster_cold_start")
        self.assertEqual({cluster["id"] for cluster in body["clusters"]}, set(range(18)))
        self.assertTrue(all(cluster["n_games"] > 0 for cluster in body["clusters"]))
        self.assertIn("Management", next(cluster["label"] for cluster in body["clusters"] if cluster["id"] == 0))

    def test_cluster_results_use_saved_signals_and_filters(self):
        status, _, body = self.request(
            "GET", "/api/cluster-recommendations?cluster_id=2&budget=20&hardware=medium&limit=3"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["source"], "cluster_cold_start")
        self.assertEqual(body["cluster"]["id"], 2)
        self.assertEqual(body["filters"], {"budget": "20", "hardware": "medium"})
        self.assertEqual(len(body["recommendations"]), 3)
        for game in body["recommendations"]:
            self.assertLessEqual(game["price"], 20)
            self.assertIn(game["hardware_tier"], ("low", "medium"))
            self.assertTrue(game["url"].startswith("https://store.steampowered.com/app/"))

    def test_cluster_query_errors_are_specific(self):
        for path in (
            "/api/cluster-recommendations",
            "/api/cluster-recommendations?cluster_id=18",
            "/api/cluster-recommendations?cluster_id=0&cluster_id=1",
            "/api/cluster-recommendations?cluster_id=0&budget=100",
            "/api/cluster-recommendations?cluster_id=0&hardware=ultra",
            "/api/cluster-recommendations?cluster_id=0&limit=0",
            "/api/cluster-recommendations?cluster_id=0&extra=yes",
            "/api/clusters?x=1",
        ):
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 400)
                self.assertIn("error", body)

    def test_steam_post_uses_server_key_without_returning_it(self):
        payload = {"steamid": "76561198000000000", "limit": 3}
        status, headers, body = self.request("POST", "/api/steam/recommend", payload)
        self.assertEqual(status, 200)
        self.assertEqual(self.steam.calls[-1], (self.steam_key, payload["steamid"], 3))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn(self.steam_key, json.dumps(body))
        self.assertEqual(body["source"], "steam_history")
        self.assertEqual(body["steam_username"], "Mock Player")
        status, _, health = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["steam_configured"])
        self.assertNotIn(self.steam_key, json.dumps(health))

    def test_steam_server_key_is_required_and_never_reflected(self):
        for key, code in (("", "missing_steam_api_key"), ("too-short", "steam_configuration_error")):
            with self.subTest(code=code):
                type(self).steam_key = key
                try:
                    status, _, body = self.request("POST", "/api/steam/recommend", {"steamid": "76561198000000000"})
                    self.assertEqual(status, 503)
                    self.assertEqual(body["error"]["code"], code)
                    if key:
                        self.assertNotIn(key, json.dumps(body))
                    status, _, health = self.request("GET", "/api/health")
                    self.assertEqual(status, 200)
                    self.assertFalse(health["steam_configured"])
                finally:
                    type(self).steam_key = "A" * 32

    def test_steam_validation_and_upstream_error_mapping(self):
        for payload in (
            {}, {"steamid": "123"},
            {"steamid": "76561198000000000", "limit": 0},
            {"steamid": "76561198000000000", "api_key": "x"},
            {"steamid": "76561198000000000", "extra": 1},
        ):
            with self.subTest(payload=payload):
                status, _, body = self.request("POST", "/api/steam/recommend", payload)
                self.assertEqual(status, 400)
                self.assertEqual(body["error"]["code"], "invalid_request")

        self.steam.error = SteamRecommendationError(403, "steam_library_private", "This library is private.")
        try:
            status, _, body = self.request("POST", "/api/steam/recommend", {
                "steamid": "76561198000000000",
            })
            self.assertEqual(status, 403)
            self.assertEqual(body["error"]["code"], "steam_library_private")
            self.assertNotIn("A" * 32, json.dumps(body))
        finally:
            self.steam.error = None

    def test_methods_are_restricted(self):
        for method, path in (("GET", "/api/steam/recommend"), ("POST", "/api/clusters"),
                             ("POST", "/api/cluster-recommendations")):
            with self.subTest(path=path):
                status, headers, body = self.request(method, path, {} if method == "POST" else None)
                self.assertEqual(status, 405)
                self.assertIn("method_not_allowed", json.dumps(body))
                self.assertIn(headers["Allow"], ("GET", "POST"))


if __name__ == "__main__":
    unittest.main()
