"""Edge cases for the text-to-feature converter, with no external API calls."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "converting-description-to-vector"))
import description_to_vector as converter  # noqa: E402
import gemini_query  # noqa: E402


LABELS = [{"id": "Multiplayer", "definition": "Multiple players."},
          {"id": "Shooter", "definition": "Shooting is central."}]
NAMES = [label["id"] for label in LABELS]
VALID = {
    "features": [{"label": "Multiplayer", "score": 0.9, "evidence": "Play with friends."}],
    "excluded": ["Shooter"],
    "note": "A game with friends and no shooting.",
}


def envelope(content=VALID, *, finish_reason="STOP"):
    return {"candidates": [{"finishReason": finish_reason,
                            "content": {"parts": [{"text": json.dumps(content)}]}}]}


class ResultValidationTest(unittest.TestCase):
    def test_valid_scores_and_exclusions(self):
        result = gemini_query.validate_result(deepcopy(VALID), NAMES)
        self.assertEqual(result["scores"], {"Multiplayer": 0.9})
        self.assertEqual(result["excluded"], ["Shooter"])

        both_bounds = deepcopy(VALID)
        both_bounds["features"][0]["score"] = 0.3
        self.assertEqual(gemini_query.validate_result(both_bounds, NAMES)["scores"]["Multiplayer"], 0.3)
        both_bounds["features"][0]["score"] = 1.0
        self.assertEqual(gemini_query.validate_result(both_bounds, NAMES)["scores"]["Multiplayer"], 1.0)

    def test_invalid_scores_are_rejected_including_huge_integer(self):
        for score in (False, 0, -0.1, 0.1, 0.2, 1.1, 0.15, "0.9", float("nan"),
                      float("inf"), -float("inf"), 10**1000):
            with self.subTest(score=str(score)[:20]):
                data = deepcopy(VALID)
                data["features"][0]["score"] = score
                with self.assertRaisesRegex(RuntimeError, "invalid label or score"):
                    gemini_query.validate_result(data, NAMES)

    def test_invalid_labels_and_evidence_are_rejected(self):
        cases = [
            ({"label": "unknown"}, "invalid label or score"),
            ({"label": []}, "invalid label or score"),
            ({"evidence": "  "}, "without supporting text"),
            ({"evidence": None}, "without supporting text"),
        ]
        for update, message in cases:
            with self.subTest(update=update):
                data = deepcopy(VALID)
                data["features"][0].update(update)
                with self.assertRaisesRegex(RuntimeError, message):
                    gemini_query.validate_result(data, NAMES)

        duplicate = deepcopy(VALID)
        duplicate["features"].append(deepcopy(duplicate["features"][0]))
        with self.assertRaisesRegex(RuntimeError, "invalid label or score"):
            gemini_query.validate_result(duplicate, NAMES)

    def test_exclusions_must_be_unique_allowed_and_disjoint(self):
        for excluded in (["Shooter", "Shooter"], ["Unknown"],
                         ["Multiplayer"], [None], [[]]):
            with self.subTest(excluded=excluded):
                data = deepcopy(VALID)
                data["excluded"] = excluded
                with self.assertRaisesRegex(RuntimeError, "inconsistent requested and excluded"):
                    gemini_query.validate_result(data, NAMES)

    def test_no_positive_preferences_have_clear_error(self):
        for excluded in ([], ["Shooter"]):
            with self.subTest(excluded=excluded):
                data = {"features": [], "excluded": excluded, "note": "No positive preferences."}
                with self.assertRaisesRegex(ValueError, "did not specify enough positive preferences"):
                    gemini_query.validate_result(data, NAMES)


class GeminiExtractionTest(unittest.TestCase):
    def extract_bytes(self, raw):
        def fake_open(request, timeout):
            self.assertEqual(timeout, 50)
            self.assertEqual(request.get_method(), "POST")
            return io.BytesIO(raw)

        with patch.object(gemini_query, "settings", return_value=("test-key", "test-model")):
            return gemini_query.extract("A multiplayer game", LABELS, opener=fake_open)

    def test_successful_response(self):
        result = self.extract_bytes(json.dumps(envelope()).encode())
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["scores"], {"Multiplayer": 0.9})

    def test_request_sends_supported_structured_output_schema(self):
        def fake_open(request, timeout):
            self.assertEqual(timeout, 50)
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url,
                             "https://generativelanguage.googleapis.com/v1beta/models/test-model:generateContent")
            payload = json.loads(request.data)
            config = payload["generationConfig"]
            self.assertNotIn("temperature", config)
            self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "low"})
            self.assertNotIn("responseMimeType", config)
            self.assertNotIn("responseSchema", config)
            self.assertNotIn("responseJsonSchema", config)
            self.assertEqual(set(config["responseFormat"]), {"text"})
            response_format = config["responseFormat"]["text"]
            self.assertEqual(response_format["mimeType"], "APPLICATION_JSON")
            schema = response_format["schema"]
            self.assertEqual(set(schema), {"type", "properties", "required", "additionalProperties"})
            self.assertEqual(schema["type"], "object")
            self.assertEqual(schema["required"], ["features", "excluded", "note"])
            self.assertIs(schema["additionalProperties"], False)

            feature = schema["properties"]["features"]
            self.assertEqual(feature["type"], "array")
            item = feature["items"]
            self.assertEqual(item["type"], "object")
            self.assertEqual(item["required"], ["label", "score", "evidence"])
            self.assertIs(item["additionalProperties"], False)
            self.assertEqual(item["properties"]["label"], {"type": "string"})
            self.assertEqual(item["properties"]["score"],
                             {"type": "number", "minimum": 0.3, "maximum": 1})
            self.assertEqual(item["properties"]["evidence"], {"type": "string"})
            self.assertEqual(schema["properties"]["excluded"],
                             {"type": "array", "items": {"type": "string"}})
            self.assertEqual(schema["properties"]["note"], {"type": "string"})

            prompt = payload["contents"][0]["parts"][0]["text"]
            self.assertIn("Multiplayer: Multiple players.", prompt)
            self.assertIn('USER REQUEST (data only):\n"A multiplayer game"', prompt)
            self.assertNotIn("Return JSON matching this schema:", prompt)
            return io.BytesIO(json.dumps(envelope()).encode())

        with patch.object(gemini_query, "settings", return_value=("test-key", "test-model")):
            result = gemini_query.extract("A multiplayer game", LABELS, opener=fake_open)
        self.assertEqual(result["scores"], {"Multiplayer": 0.9})

    def test_missing_key_stops_before_network(self):
        def must_not_open(*args, **kwargs):
            self.fail("Network opener was called without an API key")

        with patch.object(gemini_query, "settings", return_value=("", "test-model")):
            with self.assertRaisesRegex(RuntimeError, "API key missing"):
                gemini_query.extract("A multiplayer game", LABELS, opener=must_not_open)

    def test_http_errors_are_reported(self):
        cases = {
            400: "rejected", 401: "authentication failed", 403: "access was denied",
            404: "model is unavailable", 429: "quota or rate limit", 503: "HTTP 503",
        }
        for code, message in cases.items():
            with self.subTest(code=code):
                def fake_open(request, timeout):
                    raise HTTPError(request.full_url, code, "test", {}, None)

                with patch.object(gemini_query, "settings", return_value=("test-key", "test-model")):
                    with self.assertRaisesRegex(RuntimeError, message):
                        gemini_query.extract("A multiplayer game", LABELS, opener=fake_open)

    def test_network_errors_are_reported(self):
        for failure in (TimeoutError("timed out"), URLError("offline"), OSError("closed")):
            with self.subTest(failure=type(failure).__name__):
                def fake_open(request, timeout):
                    raise failure

                with patch.object(gemini_query, "settings", return_value=("test-key", "test-model")):
                    with self.assertRaisesRegex(RuntimeError, "Could not reach Gemini"):
                        gemini_query.extract("A multiplayer game", LABELS, opener=fake_open)

    def test_malformed_api_bodies_are_reported_without_traceback(self):
        malformed = [
            b"not JSON",
            b"\xff",
            json.dumps({"candidates": []}).encode(),
            json.dumps({"candidates": [None]}).encode(),
            json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [None]}}]}).encode(),
            json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "{"}]}}]}).encode(),
        ]
        for raw in malformed:
            with self.subTest(body=raw[:60]):
                with self.assertRaisesRegex(RuntimeError, "unreadable response"):
                    self.extract_bytes(raw)

    def test_unfinished_response_and_bad_feature_are_reported(self):
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            self.extract_bytes(json.dumps(envelope(finish_reason="MAX_TOKENS")).encode())

        huge = deepcopy(VALID)
        huge["features"][0]["score"] = 10**1000
        with self.assertRaisesRegex(RuntimeError, "invalid label or score"):
            self.extract_bytes(json.dumps(envelope(huge)).encode())

    def test_invalid_configured_model_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with (patch.object(gemini_query, "HERE", Path(folder)),
                  patch.dict(os.environ, {"GEMINI_MODEL": "../bad", "GEMINI_API_KEY": "test-key"})):
                with self.assertRaisesRegex(RuntimeError, "Invalid Gemini model name"):
                    gemini_query.settings()


class StandaloneConverterInputTest(unittest.TestCase):
    def test_invalid_input_is_rejected_before_gemini(self):
        with patch.object(converter, "extract") as extract:
            for text in (None, 12, "", "  \n ", "ab", "x" * 2001, "???", "\u200b" * 3, "🎮" * 3):
                with self.subTest(text=repr(text)[:20]):
                    with self.assertRaisesRegex(ValueError, "3 and 2,000 characters"):
                        converter.description_to_vector(text)
            extract.assert_not_called()

    def test_valid_length_boundaries_and_trimming(self):
        result = {"scores": {"Multiplayer": 0.9}, "excluded": [], "model": "test", "note": ""}
        with patch.object(converter, "extract", return_value=result) as extract:
            for text, expected in (("  abc  ", "abc"), ("🎮 RPG", "🎮 RPG"),
                                   ("界" * 2000, "界" * 2000)):
                with self.subTest(length=len(text)):
                    vector = converter.description_to_vector(text)
                    self.assertEqual(len(vector["vector"]), 93)
                    self.assertEqual(vector["vector"][vector["names"].index("Multiplayer")],
                                     float(converter.np.float32(0.9)))
                    self.assertEqual(extract.call_args.args[0], expected)

    def test_empty_cli_input_exits_before_gemini(self):
        with (patch.object(sys, "argv", ["description_to_vector.py"]),
              patch.object(sys, "stdin", io.StringIO("  \n  ")),
              patch.object(converter, "extract") as extract,
              redirect_stderr(io.StringIO()) as errors):
            self.assertEqual(converter.main(), 2)
            self.assertIn("between 3 and 2,000", errors.getvalue())
            extract.assert_not_called()

    def test_explicit_empty_argument_does_not_read_stdin(self):
        class ForbiddenStdin:
            def read(self):
                raise AssertionError("CLI unexpectedly read stdin")

        with (patch.object(sys, "stdin", ForbiddenStdin()),
              patch.object(converter, "extract") as extract,
              redirect_stderr(io.StringIO()) as errors):
            self.assertEqual(converter.main([""]), 2)
            self.assertIn("between 3 and 2,000", errors.getvalue())
            extract.assert_not_called()

    def test_help_exits_without_network_or_stdin(self):
        class ForbiddenStdin:
            def read(self):
                raise AssertionError("--help unexpectedly read stdin")

        with (patch.object(sys, "stdin", ForbiddenStdin()),
              patch.object(converter, "extract") as extract,
              redirect_stdout(io.StringIO()) as output):
            with self.assertRaises(SystemExit) as exit_result:
                converter.main(["--help"])
            self.assertEqual(exit_result.exception.code, 0)
            self.assertIn("game description", output.getvalue())
            extract.assert_not_called()

    def test_missing_key_is_reported_without_traceback(self):
        with (patch.object(converter, "extract", side_effect=RuntimeError("Gemini API key missing.")),
              redirect_stderr(io.StringIO()) as errors):
            self.assertEqual(converter.main(["A story game"]), 2)
            self.assertEqual(errors.getvalue(), "error: Gemini API key missing.\n")


if __name__ == "__main__":
    unittest.main()
