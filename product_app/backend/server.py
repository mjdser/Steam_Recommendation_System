"""Serve the product UI and all three recommendation paths together.

Run from the repository root with ``python3 -m product_app.backend.server``.
The default interface is loopback only; Gemini and Steam keys stay on the server.
"""

from __future__ import annotations

import argparse
from copy import copy, deepcopy
import json
import logging
import os
from pathlib import Path
import sys
from threading import Lock
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .cold_start import ColdStartCatalog, ColdStartDataError
from .steam_recommend import API_KEY_RE, STEAM_ID_RE, SteamHistoryRecommender, SteamRecommendationError


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "product_app" / "frontend"
CONVERTER_DIR = ROOT / "converting-description-to-vector"
if str(CONVERTER_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERTER_DIR))

from gemini_query import settings as gemini_settings  # noqa: E402
from recommend_games import CatalogError, TextGameRecommender, _validate_request  # noqa: E402


MAX_REQUEST_BYTES = 32_768
DEFAULT_LIMIT = 6


def steam_api_key_settings(root: Path = ROOT) -> str:
    """Read the server-owned Steam key without putting it in API responses."""
    if "STEAM_API_KEY" in os.environ:
        return os.environ["STEAM_API_KEY"].strip()
    env_path = Path(root) / "product_app" / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        if name.strip() == "STEAM_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""

# These profiles are authored in the same scored taxonomy used by the catalog.
# They let visitors explore real catalog matches without pretending Gemini ran.
PRESETS = {
    "story": {
        "title": "Story adventures",
        "description": "Character driven journeys with choices that matter.",
        "query": "A story-rich adventure with meaningful choices",
        "scores": {"Adventure": 0.8, "Story Rich": 0.9, "Choices Matter": 0.6},
    },
    "cozy": {
        "title": "Cozy escapes",
        "description": "Relaxing, easygoing worlds to spend time in.",
        "query": "A relaxing casual simulation game",
        "scores": {"Relaxing": 0.9, "Casual": 0.6, "Simulation": 0.5},
    },
    "coop": {
        "title": "Co-op journeys",
        "description": "Adventures designed to share with friends.",
        "query": "A cooperative multiplayer adventure with friends",
        "scores": {"Co-op": 0.9, "Multiplayer": 0.8, "Adventure": 0.6},
    },
    "strategy": {
        "title": "Strategy worlds",
        "description": "Thoughtful decisions, planning, and long games.",
        "query": "A strategy game with management and turn-based tactics",
        "scores": {"Strategy": 0.9, "Management": 0.6, "Turn-Based Strategy": 0.6},
    },
    "racing": {
        "title": "Racing rush",
        "description": "Fast car racing and driving competition.",
        "query": "A competitive multiplayer car racing and driving game",
        "scores": {"Racing": 0.9, "Multiplayer": 0.5, "Simulation": 0.4},
    },
}


class ApiError(Exception):
    """An error safe to show as an API response."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _preset_info(preset_id: str) -> dict:
    preset = PRESETS[preset_id]
    return {"id": preset_id, "title": preset["title"], "description": preset["description"]}


def _single_parameter(parameters: dict[str, list[str]], name: str, default: str) -> str:
    values = parameters.get(name, [default])
    if len(values) != 1:
        raise ApiError(400, "invalid_request", f"Specify {name} only once.")
    return values[0]


def _limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
        raise ApiError(400, "invalid_request", "limit must be an integer from 1 to 20.")
    return value


class Application:
    """Own one loaded catalog and handle the recommendation API routes.

    Supplying ``converter`` is intended for in-process HTTP tests. Production
    uses the existing Gemini converter and requires a server-side API key.
    """

    def __init__(self, *, root: Path = ROOT, converter=None, require_api_key: bool | None = None,
                 config_provider=None, steam_config_provider=None, steam_recommender=None,
                 cold_start_catalog=None):
        self.root = Path(root)
        self.require_api_key = converter is None if require_api_key is None else require_api_key
        self.config_provider = config_provider or gemini_settings
        self.steam_config_provider = steam_config_provider or (lambda: steam_api_key_settings(self.root))
        self.engine = None
        self._discovery_cache: dict[str, dict] = {}
        self._cache_lock = Lock()
        self._steam_lock = Lock()
        self._steam_recommender = steam_recommender
        try:
            self.cold_start = cold_start_catalog or ColdStartCatalog(self.root)
        except (ColdStartDataError, OSError, ValueError):
            logging.exception("Could not load current cold-start cluster data")
            self.cold_start = None
        try:
            if converter is None:
                self.engine = TextGameRecommender(self.root)
            else:
                self.engine = TextGameRecommender(self.root, converter=converter)
        except Exception:
            logging.exception("Could not load the game recommendation catalog")

    def _configuration(self) -> tuple[bool, str | None, bool]:
        """Return (key present, model name, config valid) without exposing key."""
        try:
            key, model = self.config_provider()
            return bool(key), model, True
        except (OSError, RuntimeError, UnicodeError):
            logging.exception("Gemini configuration is invalid")
            return False, None, False

    def health(self) -> dict:
        configured, model, valid = self._configuration()
        try:
            steam_key = self.steam_config_provider()
            steam_configured = isinstance(steam_key, str) and bool(API_KEY_RE.fullmatch(steam_key))
        except (OSError, UnicodeError, ValueError):
            steam_configured = False
        if self.engine is None:
            status = "catalog_error"
        elif not valid:
            status = "configuration_error"
        elif not configured:
            status = "needs_api_key"
        else:
            status = "ok"
        return {
            "status": status,
            "catalog_games": len(self.engine.games) if self.engine is not None else 0,
            "gemini_configured": configured,
            "model": model,
            "clusters_available": self.cold_start is not None,
            "cluster_count": len(self.cold_start.list_clusters()) if self.cold_start is not None else 0,
            "steam_configured": steam_configured,
        }

    def _require_catalog(self) -> TextGameRecommender:
        if self.engine is None:
            raise ApiError(503, "catalog_unavailable", "The game catalog could not load. Check the server logs.")
        return self.engine

    def recommend(self, payload: object) -> dict:
        if not isinstance(payload, dict) or set(payload) - {"query", "limit"}:
            raise ApiError(400, "invalid_request", "Send a JSON object with query and optional limit.")
        query = payload.get("query")
        limit = payload.get("limit", DEFAULT_LIMIT)
        try:
            _validate_request(query, limit)
        except ValueError as exc:
            raise ApiError(400, "invalid_request", str(exc)) from None
        engine = self._require_catalog()
        if self.require_api_key:
            configured, _, valid = self._configuration()
            if not valid:
                raise ApiError(503, "configuration_error", "Gemini configuration is invalid. Check the server logs.")
            if not configured:
                raise ApiError(503, "missing_api_key", "Add GEMINI_API_KEY to converting-description-to-vector/.env, then try again.")
        try:
            return engine.recommend(query, limit=limit)
        except CatalogError:
            logging.exception("Catalog and converter schemas do not match")
            raise ApiError(503, "catalog_unavailable", "The game catalog and text converter do not match.") from None
        except ValueError as exc:
            # This includes valid prose from which Gemini found no positive features.
            raise ApiError(400, "unusable_request", str(exc)) from None
        except RuntimeError as exc:
            message = str(exc)
            if message.startswith("Gemini API key missing"):
                raise ApiError(503, "missing_api_key", "Add GEMINI_API_KEY to converting-description-to-vector/.env, then try again.") from None
            if message.startswith("Gemini API authentication failed"):
                raise ApiError(503, "invalid_api_key", message) from None
            if message.startswith("Gemini API access was denied"):
                raise ApiError(503, "gemini_access_denied", message) from None
            if message.startswith("The configured Gemini model is unavailable"):
                raise ApiError(503, "invalid_model", message) from None
            if message.startswith("Gemini quota or rate limit reached"):
                raise ApiError(503, "gemini_unavailable", message) from None
            if message.startswith("Could not reach Gemini within the time limit"):
                raise ApiError(503, "gemini_unavailable", message) from None
            if message.startswith("Gemini "):
                raise ApiError(502, "gemini_error", message) from None
            logging.exception("Unexpected converter error")
            raise ApiError(502, "converter_error", "The request could not be interpreted. Try again.") from None

    def discover(self, preset_id: str, limit: int) -> dict:
        engine = self._require_catalog()
        if preset_id not in PRESETS:
            raise ApiError(400, "unknown_preset", "Choose one of the listed discovery themes.")
        if not 1 <= limit <= 20:
            raise ApiError(400, "invalid_request", "limit must be an integer from 1 to 20.")
        with self._cache_lock:
            if preset_id not in self._discovery_cache:
                preset = PRESETS[preset_id]
                names = engine.names
                scores = preset["scores"]
                if any(name not in engine.index for name in scores):
                    raise ApiError(503, "catalog_unavailable", "A discovery theme does not match the game taxonomy.")
                vector = [scores.get(name, 0.0) for name in names]
                # A shallow copy shares immutable catalog data while giving this
                # one call its own converter; concurrent Gemini calls remain safe.
                preset_engine = copy(engine)
                preset_engine.converter = lambda _text: {
                    "names": names,
                    "vector": vector,
                    "excluded": [],
                    "model": "curated-preset",
                    "note": "A curated discovery theme scored with the catalog's 93-feature taxonomy.",
                }
                try:
                    self._discovery_cache[preset_id] = preset_engine.recommend(preset["query"], limit=20)
                except (CatalogError, ValueError):
                    logging.exception("Could not rank discovery theme %s", preset_id)
                    raise ApiError(503, "catalog_unavailable", "This discovery theme is unavailable.") from None
            result = self._discovery_cache[preset_id]
        # Return a separate value so one request cannot mutate a later result.
        response_result = deepcopy(result)
        response_result["recommendations"] = response_result["recommendations"][:limit]
        return {
            "source": "curated_preset",
            "preset": _preset_info(preset_id),
            "presets": [_preset_info(name) for name in PRESETS],
            "result": response_result,
        }

    def clusters(self) -> dict:
        if self.cold_start is None:
            raise ApiError(503, "clusters_unavailable", "Cluster discovery data could not load. Check the server logs.")
        return {"source": "cluster_cold_start", "clusters": self.cold_start.list_clusters()}

    def cluster_recommendations(self, cluster_id: int, budget: str, hardware: str, limit: int) -> dict:
        if self.cold_start is None:
            raise ApiError(503, "clusters_unavailable", "Cluster discovery data could not load. Check the server logs.")
        try:
            return self.cold_start.recommend(cluster_id, budget=budget, hardware=hardware, limit=limit)
        except ValueError as exc:
            raise ApiError(400, "invalid_request", str(exc)) from None
        except ColdStartDataError:
            logging.exception("Current cluster discovery data is invalid")
            raise ApiError(503, "clusters_unavailable", "Cluster discovery data could not load. Check the server logs.") from None

    def steam_recommend(self, payload: object) -> dict:
        if not isinstance(payload, dict) or set(payload) - {"steamid", "limit"}:
            raise ApiError(400, "invalid_request", "Send steamid and optional limit in a JSON object.")
        steam_id = payload.get("steamid")
        limit = _limit(payload.get("limit", DEFAULT_LIMIT))
        if not isinstance(steam_id, str) or not STEAM_ID_RE.fullmatch(steam_id):
            raise ApiError(400, "invalid_request", "Enter a 17-digit SteamID64.")
        try:
            api_key = self.steam_config_provider()
        except (OSError, UnicodeError, ValueError):
            raise ApiError(503, "steam_configuration_error", "The server's Steam API configuration could not be read.") from None
        if not api_key:
            raise ApiError(503, "missing_steam_api_key", "Steam recommendations need a server-side Steam Web API key. Ask the site owner to configure it.")
        if not isinstance(api_key, str) or not API_KEY_RE.fullmatch(api_key):
            raise ApiError(503, "steam_configuration_error", "The server's Steam Web API key is invalid. Ask the site owner to check it.")
        # Loading the XGBoost model is deferred so text and cluster discovery
        # stay available if the optional model dependency is not installed.
        with self._steam_lock:
            if self._steam_recommender is None:
                try:
                    self._steam_recommender = SteamHistoryRecommender(self.root)
                except SteamRecommendationError as exc:
                    raise ApiError(exc.status, exc.code, exc.message) from None
                except (ImportError, OSError, ValueError, RuntimeError):
                    logging.exception("Could not load the personalized Steam model")
                    raise ApiError(503, "model_unavailable", "The personalized model could not load. Check server dependencies and logs.") from None
            recommender = self._steam_recommender
        try:
            return recommender.recommend(api_key=api_key, steam_id=steam_id, limit=limit)
        except SteamRecommendationError as exc:
            raise ApiError(exc.status, exc.code, exc.message) from None


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    root: Path = ROOT,
    frontend_dir: Path = FRONTEND,
    converter=None,
    require_api_key: bool | None = None,
    config_provider=None,
    steam_config_provider=None,
    steam_recommender=None,
    cold_start_catalog=None,
) -> ThreadingHTTPServer:
    """Build a local server; tests may pass port=0 and a mocked converter."""
    app = Application(root=root, converter=converter, require_api_key=require_api_key,
                      config_provider=config_provider, steam_config_provider=steam_config_provider,
                      steam_recommender=steam_recommender,
                      cold_start_catalog=cold_start_catalog)
    frontend_dir = Path(frontend_dir)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(frontend_dir), **kwargs)

        def end_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
            super().end_headers()

        def list_directory(self, path):
            self.send_error(404, "Not found")
            return None

        def _json(self, status: int, value: dict):
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            if status == 405:
                self.send_header("Allow", "POST" if urlsplit(self.path).path in
                                 ("/api/recommend", "/api/steam/recommend") else "GET")
            self.end_headers()
            self.wfile.write(raw)

        def _error(self, error: ApiError):
            self._json(error.status, {"error": {"code": error.code, "message": error.message}})

        def do_GET(self):
            path = urlsplit(self.path)
            if path.path == "/api/health":
                self._json(200, app.health())
                return
            if path.path == "/api/clusters":
                try:
                    if path.query:
                        raise ApiError(400, "invalid_request", "This route takes no query parameters.")
                    self._json(200, app.clusters())
                except ApiError as exc:
                    self._error(exc)
                return
            if path.path == "/api/cluster-recommendations":
                try:
                    params = parse_qs(path.query, keep_blank_values=True)
                    if set(params) - {"cluster_id", "budget", "hardware", "limit"}:
                        raise ApiError(400, "invalid_request", "Unknown cluster filter.")
                    raw_id = _single_parameter(params, "cluster_id", "")
                    raw_limit = _single_parameter(params, "limit", str(DEFAULT_LIMIT))
                    if not raw_id.isdecimal() or len(raw_id) > 3:
                        raise ApiError(400, "invalid_request", "Choose a valid cluster ID.")
                    if not raw_limit.isdecimal() or len(raw_limit) > 2:
                        raise ApiError(400, "invalid_request", "limit must be an integer from 1 to 20.")
                    budget = _single_parameter(params, "budget", "all")
                    hardware = _single_parameter(params, "hardware", "all")
                    self._json(200, app.cluster_recommendations(int(raw_id), budget, hardware, _limit(int(raw_limit))))
                except ApiError as exc:
                    self._error(exc)
                except Exception:
                    logging.exception("Unexpected cluster recommendation API failure")
                    self._error(ApiError(500, "internal_error", "The server could not complete this request."))
                return
            if path.path == "/api/discover":
                try:
                    params = parse_qs(path.query, keep_blank_values=True)
                    if set(params) - {"preset", "limit"}:
                        raise ApiError(400, "invalid_request", "Only preset and limit are accepted.")
                    preset_id = _single_parameter(params, "preset", "story")
                    raw_limit = _single_parameter(params, "limit", str(DEFAULT_LIMIT))
                    if not raw_limit.isdecimal():
                        raise ApiError(400, "invalid_request", "limit must be an integer from 1 to 20.")
                    self._json(200, app.discover(preset_id, int(raw_limit)))
                except ApiError as exc:
                    self._error(exc)
                except Exception:
                    logging.exception("Unexpected discovery API failure")
                    self._error(ApiError(500, "internal_error", "The server could not complete this request."))
                return
            if path.path in ("/api/recommend", "/api/steam/recommend"):
                self._error(ApiError(405, "method_not_allowed", "This route accepts POST."))
                return
            if path.path.startswith("/api/"):
                self._error(ApiError(404, "not_found", "Unknown API route."))
                return
            SimpleHTTPRequestHandler.do_GET(self)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/recommend", "/api/steam/recommend"):
                if path in ("/api/health", "/api/discover", "/api/clusters", "/api/cluster-recommendations"):
                    self._error(ApiError(405, "method_not_allowed", "This route accepts GET."))
                elif path.startswith("/api/"):
                    self._error(ApiError(404, "not_found", "Unknown API route."))
                else:
                    self._error(ApiError(405, "method_not_allowed", "This route accepts GET."))
                return
            try:
                media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if media_type != "application/json":
                    raise ApiError(415, "unsupported_media_type", "Send application/json.")
                raw_length = self.headers.get("Content-Length")
                if raw_length is None or not raw_length.isdecimal():
                    raise ApiError(400, "invalid_request", "A valid Content-Length is required.")
                length = int(raw_length)
                if length > MAX_REQUEST_BYTES:
                    self.close_connection = True
                    raise ApiError(413, "request_too_large", "The request is too large.")
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, ValueError, RecursionError):
                    raise ApiError(400, "invalid_json", "Send valid UTF-8 JSON.") from None
                result = app.recommend(payload) if path == "/api/recommend" else app.steam_recommend(payload)
                self._json(200, result)
            except ApiError as exc:
                self._error(exc)
            except Exception as exc:
                if path == "/api/steam/recommend":
                    # urllib exceptions can contain the upstream URL and a
                    # user's API key. Keep unexpected failures out of logs.
                    logging.error("Unexpected Steam API failure (%s)", type(exc).__name__)
                else:
                    logging.exception("Unexpected recommendation API failure")
                self._error(ApiError(500, "internal_error", "The server could not complete this request."))

    class LocalServer(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = LocalServer((host, port), Handler)
    server.app = app
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local game recommendation app.")
    parser.add_argument("--host", default=os.environ.get("PRODUCT_APP_HOST") or "127.0.0.1")
    parser.add_argument("--port", type=int, default=os.environ.get("PRODUCT_APP_PORT") or "8000")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        server = create_server(args.host, args.port)
    except OSError as exc:
        parser.error(f"Could not start the server: {exc}")
    address, port = server.server_address[:2]
    print(f"Game recommendation app: http://{address}:{port}/", flush=True)
    print(f"Backend status: {server.app.health()['status']}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
