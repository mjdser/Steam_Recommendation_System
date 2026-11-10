"""HTTP contract tests using the real catalog and a local converter stand-in.

No test calls Gemini or requires an API key. Recommendation ranking is always
performed by the production TextGameRecommender against the real catalog.
"""

from __future__ import annotations

from http.client import HTTPConnection
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch

from product_app.backend.server import MAX_REQUEST_BYTES, create_server
import gemini_query


ROOT = Path(__file__).resolve().parents[3]


def _serve(server):
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return worker


def _request(server, method, path, body=None, content_type="application/json"):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    headers = {"Content-Type": content_type} if body is not None else {}
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode("utf-8")
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        value = json.loads(raw) if response.getheader("Content-Type", "").startswith("application/json") else raw
        return response.status, response.headers, value
    finally:
        connection.close()


class ServerHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frontend = TemporaryDirectory()
        Path(cls.frontend.name, "index.html").write_text("<!doctype html><title>Game finder</title>")
        names_data = json.loads((ROOT / "new_package/02_taxonomy/dimension_order.json").read_text())
        names = names_data["genre_dimension_order"] + names_data["tag_dimension_order"]
        cls.converted_inputs = []

        def converter(text):
            cls.converted_inputs.append(text)
            scores = {"Adventure": 0.8, "Story Rich": 0.9}
            return {
                "names": names,
                "vector": [scores.get(name, 0.0) for name in names],
                "excluded": [],
                "model": "test-converter",
                "note": "Test interpretation",
            }

        cls.converter = converter
        options = {"frontend_dir": Path(cls.frontend.name), "converter": converter,
                   "config_provider": lambda: (None, "test-model")}
        cls.server = create_server(port=0, **options)
        cls.worker = _serve(cls.server)
        cls.keyless_server = create_server(port=0, require_api_key=True, **options)
        cls.keyless_worker = _serve(cls.keyless_server)

    @classmethod
    def tearDownClass(cls):
        for server, worker in ((cls.server, cls.worker), (cls.keyless_server, cls.keyless_worker)):
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
        cls.frontend.cleanup()

    def test_static_page_and_health_are_available_without_key(self):
        status, headers, page = _request(self.server, "GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Game finder", page)
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
        status, _, health = _request(self.keyless_server, "GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "needs_api_key")
        self.assertEqual(health["catalog_games"], 6750)
        self.assertFalse(health["gemini_configured"])
        self.assertNotIn("key", str(health).lower().replace("api_key", ""))

    def test_real_frontend_assets_are_served(self):
        server = create_server(port=0)
        worker = _serve(server)
        try:
            for path, media_type, marker in (
                ("/", "text/html", b'id="search-form"'),
                ("/styles.css", "text/css", b".featured-card"),
                ("/app.js", "javascript", b"/api/recommend"),
                ("/favicon.svg", "image/svg+xml", b"<svg"),
            ):
                with self.subTest(path=path):
                    status, headers, body = _request(server, "GET", path)
                    self.assertEqual(status, 200)
                    self.assertIn(media_type, headers.get("Content-Type", ""))
                    self.assertIn(marker, body)
                    self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
            status, _, body = _request(server, "GET", "/../README.md")
            self.assertEqual(status, 404)
            self.assertNotIn(b"Steam Game Recommender", body)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_curated_discovery_uses_real_catalog_with_no_key(self):
        before = len(self.converted_inputs)
        for preset in ("story", "cozy", "coop", "strategy", "racing"):
            with self.subTest(preset=preset):
                status, _, payload = _request(self.keyless_server, "GET", f"/api/discover?preset={preset}&limit=3")
                self.assertEqual(status, 200)
                self.assertEqual(payload["source"], "curated_preset")
                self.assertEqual(payload["preset"]["id"], preset)
                self.assertEqual(len(payload["presets"]), 5)
                self.assertEqual(payload["result"]["catalog_games"], 6750)
                self.assertEqual(len(payload["result"]["recommendations"]), 3)
                self.assertTrue(all(item["matched_features"] for item in payload["result"]["recommendations"]))
        self.assertEqual(len(self.converted_inputs), before)

    def test_discovery_validates_query_parameters(self):
        cases = (
            "/api/discover?preset=unknown",
            "/api/discover?preset=story&preset=cozy",
            "/api/discover?limit=0",
            "/api/discover?limit=21",
            "/api/discover?limit=1.5",
            "/api/discover?limit=2&limit=3",
            "/api/discover?extra=1",
        )
        for path in cases:
            with self.subTest(path=path):
                status, _, payload = _request(self.server, "GET", path)
                self.assertEqual(status, 400)
                self.assertIn("code", payload["error"])

    def test_free_text_post_uses_existing_recommender(self):
        status, _, result = _request(self.server, "POST", "/api/recommend", {"query": "A story adventure", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["model"], "test-converter")
        self.assertEqual(result["query"], "A story adventure")
        self.assertEqual(len(result["feature_names"]), 93)
        self.assertEqual(len(result["feature_vector"]), 93)
        self.assertEqual(len(result["recommendations"]), 2)
        self.assertTrue(all(row["url"].startswith("https://store.steampowered.com/app/")
                            for row in result["recommendations"]))
        self.assertEqual(self.converted_inputs[-1], "A story adventure")

    def test_http_to_gemini_parser_to_real_ranker_with_mocked_transport(self):
        model_response = {
            "features": [
                {"label": "Adventure", "score": 0.8, "evidence": "Requests an adventure."},
                {"label": "Story Rich", "score": 0.9, "evidence": "Requests a story."},
            ],
            "excluded": ["Shooter"],
            "note": "A story adventure without shooting.",
        }
        envelope = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(model_response)}]}}]}
        calls = []

        def fake_open(request, timeout):
            calls.append(request.full_url)
            self.assertEqual(timeout, 50)
            return io.BytesIO(json.dumps(envelope).encode("utf-8"))

        server = create_server(port=0, config_provider=lambda: ("test-key", "test-model"))
        worker = _serve(server)
        try:
            with (patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"}),
                  patch.object(gemini_query, "urlopen", fake_open)):
                status, _, result = _request(server, "POST", "/api/recommend", {
                    "query": "A story adventure without shooting", "limit": 2,
                })
            self.assertEqual(status, 200)
            self.assertEqual(result["model"], "test-model")
            self.assertEqual(result["excluded"], ["Shooter"])
            self.assertEqual(len(result["recommendations"]), 2)
            self.assertEqual(len(calls), 1)
            self.assertIn("generateContent", calls[0])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_invalid_text_and_limits_never_reach_converter(self):
        before = len(self.converted_inputs)
        cases = (
            {}, {"query": None}, {"query": ""}, {"query": "  \t  "},
            {"query": "???!"}, {"query": "ab"}, {"query": "x" * 2001},
            {"query": "A story game", "limit": 0},
            {"query": "A story game", "limit": 21},
            {"query": "A story game", "limit": True},
            {"query": "A story game", "limit": "5"},
            {"query": "A story game", "unknown": 1},
            ["A story game"],
        )
        for body in cases:
            with self.subTest(body=str(body)[:60]):
                status, _, payload = _request(self.server, "POST", "/api/recommend", body)
                self.assertEqual(status, 400)
                self.assertIn("message", payload["error"])
        self.assertEqual(len(self.converted_inputs), before)

    def test_keyless_free_text_has_clear_error_but_invalid_text_is_400(self):
        before = len(self.converted_inputs)
        status, _, payload = _request(self.keyless_server, "POST", "/api/recommend", {"query": "A story adventure"})
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "missing_api_key")
        self.assertIn("GEMINI_API_KEY", payload["error"]["message"])
        status, _, payload = _request(self.keyless_server, "POST", "/api/recommend", {"query": " "})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertEqual(len(self.converted_inputs), before)

    def test_invalid_server_configuration_is_reported_without_api_call(self):
        app = self.keyless_server.app
        previous = app.config_provider
        before = len(self.converted_inputs)
        try:
            def invalid_config():
                raise RuntimeError("Invalid Gemini model name")

            app.config_provider = invalid_config
            status, _, health = _request(self.keyless_server, "GET", "/api/health")
            self.assertEqual(status, 200)
            self.assertEqual(health["status"], "configuration_error")
            status, _, payload = _request(self.keyless_server, "POST", "/api/recommend", {"query": "A story game"})
            self.assertEqual(status, 503)
            self.assertEqual(payload["error"]["code"], "configuration_error")
            self.assertEqual(len(self.converted_inputs), before)
        finally:
            app.config_provider = previous

    def test_protocol_errors_are_json_and_never_show_tracebacks(self):
        cases = (
            ("POST", "/api/recommend", b"{", "application/json", 400),
            ("POST", "/api/recommend", b"\xff", "application/json", 400),
            ("POST", "/api/recommend", b'{"query":"hello","limit":' + b"9" * 5000 + b"}", "application/json", 400),
            ("POST", "/api/recommend", b"{}", "text/plain", 415),
            ("POST", "/api/recommend", b"x" * (MAX_REQUEST_BYTES + 1), "application/json", 413),
            ("GET", "/api/recommend", None, "application/json", 405),
            ("POST", "/api/health", b"{}", "application/json", 405),
            ("GET", "/api/unknown", None, "application/json", 404),
        )
        for method, path, body, content_type, expected in cases:
            with self.subTest(method=method, path=path, expected=expected):
                status, headers, payload = _request(self.server, method, path, body, content_type)
                self.assertEqual(status, expected)
                self.assertEqual(headers.get("Cache-Control"), "no-store")
                self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
                self.assertNotIn("Traceback", json.dumps(payload))
                self.assertIn("error", payload)

    def test_unexpected_converter_error_is_generic(self):
        engine = self.server.app.engine
        original = engine.converter
        try:
            def broken(_text):
                raise RuntimeError("secret internal detail")
            engine.converter = broken
            status, _, payload = _request(self.server, "POST", "/api/recommend", {"query": "A story game"})
            self.assertEqual(status, 502)
            self.assertEqual(payload["error"]["code"], "converter_error")
            self.assertNotIn("secret", json.dumps(payload))
        finally:
            engine.converter = original

    def test_known_gemini_configuration_and_quota_errors(self):
        engine = self.server.app.engine
        original = engine.converter
        try:
            for message, code in (
                ("Gemini API authentication failed.", "invalid_api_key"),
                ("Gemini API access was denied.", "gemini_access_denied"),
                ("The configured Gemini model is unavailable.", "invalid_model"),
                ("Gemini quota or rate limit reached. Try later or check billing.", "gemini_unavailable"),
            ):
                with self.subTest(code=code):
                    def fail(_text, error=message):
                        raise RuntimeError(error)

                    engine.converter = fail
                    status, _, payload = _request(self.server, "POST", "/api/recommend", {"query": "A story game"})
                    self.assertEqual(status, 503)
                    self.assertEqual(payload["error"]["code"], code)
        finally:
            engine.converter = original


if __name__ == "__main__":
    unittest.main()
