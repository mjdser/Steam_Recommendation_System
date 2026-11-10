"""Steam library recommendations using the repository's saved personalized model.

The server-owned Steam Web API key is passed in for each request. It is sent
only to Steam's fixed HTTPS endpoint and is never logged or included in the
response. XGBoost receives columns in the exact order saved in its bundle.
"""

from __future__ import annotations

import csv
from http.client import HTTPException
from io import BytesIO
import json
import math
from pathlib import Path
import re
import socket
import sys
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STEAM_OWNED_GAMES_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
STEAM_PLAYER_SUMMARIES_URL = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
STEAM_TIMEOUT_SECONDS = 15
STEAM_PROFILE_TIMEOUT_SECONDS = 5
MAX_STEAM_RESPONSE_BYTES = 4_000_000
MAX_STEAM_PROFILE_BYTES = 128_000
MAX_LIBRARY_GAMES = 20_000
API_KEY_RE = re.compile(r"[A-Za-z0-9]{16,128}\Z")
STEAM_ID_RE = re.compile(r"[0-9]{17}\Z")
NONGAME_NAME = re.compile(
    r"(wallpaper engine|soundpad|blender|crosshair|overlay|sharex|precision x|"
    r"borderless|aseprite|ovr toolkit|ovr advanced|hudsight|steamvr|source sdk|"
    r"driver booster|cpucores|esea|\bdlc\b|soundtrack|artbook|art book|"
    r"bonus content|season pass|dedicated server|server files|\bsdk\b|"
    r"benchmark|configurator|playtest\b|tmodloader|mod loader|"
    r"deleted scenes| - multiplayer$|gameplay recording)",
    re.IGNORECASE,
)


class SteamRecommendationError(Exception):
    """A sanitized failure that can be sent directly as an API error."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def validate_steam_request(api_key: str, steam_id: str, limit: int) -> None:
    if not isinstance(api_key, str) or not API_KEY_RE.fullmatch(api_key):
        raise SteamRecommendationError(503, "steam_configuration_error", "The server's Steam Web API key is invalid.")
    if not isinstance(steam_id, str) or not STEAM_ID_RE.fullmatch(steam_id):
        raise SteamRecommendationError(400, "invalid_request", "Enter your 17-digit SteamID64.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise SteamRecommendationError(400, "invalid_request", "limit must be an integer from 1 to 20.")


def _playtime(value: object) -> float:
    """Steam reports playtime in minutes; malformed entries do not become slots."""
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) and number >= 0 else 0.0


def score_owned_games(games: list[dict]) -> list[dict]:
    """Faithful request-time port of score_interest_top10.py::score_library.

    Steam's playtime fields are minutes. The score uses the project's 40% recent,
    35% lifetime, 15% momentum, 10% investment formula with a 200-hour cap.
    The full sorted list is returned so supported games beyond the first ten
    can still fill the model's six history slots.
    """
    by_app: dict[int, dict] = {}
    for game in games:
        if not isinstance(game, dict):
            continue
        try:
            raw_id = game.get("appid")
            if isinstance(raw_id, bool):
                continue
            appid = int(raw_id)
        except (TypeError, ValueError, OverflowError):
            continue
        if appid <= 0:
            continue
        name = game.get("name")
        name = name if isinstance(name, str) else ""
        if NONGAME_NAME.search(name):
            continue
        forever = _playtime(game.get("playtime_forever"))
        recent = _playtime(game.get("playtime_2weeks"))
        if forever <= 0 and recent <= 0:
            continue
        previous = by_app.get(appid)
        if previous is None or forever > previous["forever_min"]:
            by_app[appid] = {"appid": appid, "name": name, "forever_min": forever,
                             "recent_min": recent}

    rows = list(by_app.values())
    if not rows:
        return []
    max_forever = max(math.log1p(row["forever_min"] / 60) for row in rows)
    max_recent = max(math.log1p(row["recent_min"] / 60) for row in rows)
    investment_cap = math.log1p(200.0)
    scored = []
    for row in rows:
        lifetime_hours = row["forever_min"] / 60
        recent_hours = row["recent_min"] / 60
        log_forever = math.log1p(lifetime_hours)
        log_recent = math.log1p(recent_hours)
        s_play = log_forever / max_forever if max_forever > 0 else 0.0
        s_recent = log_recent / max_recent if max_recent > 0 else 0.0
        s_momentum = min(1.0, log_recent / log_forever) if lifetime_hours > 0 else 0.0
        s_invest = min(1.0, log_forever / investment_cap) if lifetime_hours > 0 else 0.0
        score = 0.40 * s_recent + 0.35 * s_play + 0.15 * s_momentum + 0.10 * s_invest
        scored.append({
            "appid": row["appid"], "name": row["name"],
            "playtime_forever_hours": round(lifetime_hours, 2),
            "playtime_2weeks_hours": round(recent_hours, 2),
            "interest_score": round(score, 4),
        })
    scored.sort(key=lambda row: (row["interest_score"], row["playtime_forever_hours"]), reverse=True)
    return scored


def _owned_appids(games: list[dict]) -> set[int]:
    owned = set()
    for game in games:
        if not isinstance(game, dict) or isinstance(game.get("appid"), bool):
            continue
        try:
            appid = int(game["appid"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if appid > 0:
            owned.add(appid)
    return owned


class SteamHistoryRecommender:
    """Load v3 plus the game clusters once for Steam requests."""

    def __init__(self, root: Path = ROOT, *, opener=urlopen):
        self.root = Path(root)
        self.opener = opener
        self._predict_lock = Lock()
        try:
            import pandas as pd
            import xgboost as xgb

            data_dir = str(self.root / "data")
            v3_dir = str(self.root / "src" / "v3_user_interaction_added")
            for path in (data_dir, v3_dir):
                if path not in sys.path:
                    sys.path.insert(0, path)

            from cluster_pipeline import ClusterRecommender, edition_blocklist
            from try_new_users import load_context

            self.xgb = xgb
            self.pd = pd
            self._edition_blocklist = edition_blocklist
            bundle_path = self.root / "models/model_rank_v3_interact_bundle.json"
            self.bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.feature_names = tuple(self.bundle["game_feats"])
            self.feature_cols = tuple(self.bundle["feature_cols"])
            self.n_slots = self.bundle["n_slots"]
            self.best_iteration = self.bundle["best_iteration"]
            if (self.n_slots != 6 or len(self.feature_names) != 94
                    or self.feature_names[0] != "n_labels"
                    or not isinstance(self.best_iteration, int)
                    or self.best_iteration < 0):
                raise ValueError("The v3 model bundle has an unexpected feature layout.")

            self.model = xgb.Booster()
            self.model.load_model(str(self.root / "models/model_rank_v3_interact.json"))
            if self.model.num_features() != len(self.feature_cols):
                raise ValueError("The model and bundle feature counts differ.")

            self._load_catalog()
            feats_df = pd.DataFrame(self.features, index=self.appids, columns=list(self.feature_names))
            clusters = pd.read_csv(self.root / "new_package/04_clusters/game_clusters.csv")
            clusters = clusters[clusters["cluster_id"].notna() & (clusters["cluster_id"] >= 0)]
            neighbors = pd.read_csv(self.root / "new_package/08_item_neighbors/game_top10_neighbors.csv")
            names = {int(appid): name for appid, name in zip(self.appids, self.names)}
            self._cluster = ClusterRecommender(feats_df, clusters, neighbors, names)
            self._feats_df = feats_df
            self._ctx, _ = load_context(feats_df)
            self._cluster_labels = (
                clusters.drop_duplicates("cluster_id")
                .set_index("cluster_id")["cluster_label"]
                .to_dict()
            )
        except (ImportError, OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
            raise SteamRecommendationError(
                503, "model_unavailable", "Steam recommendations are unavailable because the saved model or catalog could not load."
            ) from exc

    def _load_catalog(self) -> None:
        path = self.root / "new_package/03_features/game_features.csv"
        names: list[str] = []
        appids: list[int] = []
        vectors: list[list[float]] = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                    or not {"appid", "name", "status", *self.feature_names}.issubset(reader.fieldnames)):
                raise ValueError("The game catalog does not match the model bundle.")
            for row in reader:
                if row["status"] != "ok":
                    continue
                appid = int(row["appid"])
                name = row["name"]
                vector = [float(row[feature]) for feature in self.feature_names]
                if (appid <= 0 or not name or not all(math.isfinite(x) for x in vector)
                        or vector[0] < 1 or vector[0] > 93
                        or any(not 0 <= x <= 1 for x in vector[1:])):
                    raise ValueError("The game catalog contains invalid model features.")
                appids.append(appid)
                names.append(name)
                vectors.append(vector)
        if not vectors or len(appids) != len(set(appids)):
            raise ValueError("The game catalog is empty or contains duplicate IDs.")
        self.appids = np.asarray(appids, dtype=np.int64)
        self.names = tuple(names)
        self.features = np.asarray(vectors, dtype=np.float32)
        self.appid_to_index = {appid: i for i, appid in enumerate(appids)}

    def _fetch_library(self, api_key: str, steam_id: str) -> tuple[list[dict], int]:
        params = urlencode({
            "key": api_key,
            "steamid": steam_id,
            "include_appinfo": "1",
            "include_played_free_games": "1",
        })
        # Steam requires its key in the upstream query string. The URL is never
        # logged or reflected in exception text or responses.
        request = Request(f"{STEAM_OWNED_GAMES_URL}?{params}",
                          headers={"Accept": "application/json", "User-Agent": "NextQuest/1.0"})
        try:
            with self.opener(request, timeout=STEAM_TIMEOUT_SECONDS) as response:
                content = response.read(MAX_STEAM_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            if exc.code == 400:
                raise SteamRecommendationError(400, "steam_request_rejected", "Steam rejected the Web API key or SteamID64. Check both and try again.") from None
            if exc.code in (401, 403):
                raise SteamRecommendationError(401, "steam_auth_failed", "Steam rejected that Web API key. Check it and try again.") from None
            if exc.code == 429:
                raise SteamRecommendationError(503, "steam_rate_limited", "Steam is rate limiting requests. Try again shortly.") from None
            raise SteamRecommendationError(502, "steam_unavailable", "Steam could not provide the game library right now.") from None
        except (URLError, TimeoutError, socket.timeout, OSError):
            raise SteamRecommendationError(502, "steam_unavailable", "Steam could not provide the game library right now.") from None
        if len(content) > MAX_STEAM_RESPONSE_BYTES:
            raise SteamRecommendationError(502, "steam_unavailable", "Steam returned a library response that is too large.")
        try:
            payload = json.loads(content.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            raise SteamRecommendationError(502, "steam_unavailable", "Steam returned an unreadable game library.") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
            raise SteamRecommendationError(502, "steam_unavailable", "Steam returned an unexpected game library response.")
        library = payload["response"]
        if not library:
            raise SteamRecommendationError(403, "steam_library_private", "This Steam account's game details are private or unavailable. Make game details public in Steam privacy settings.")
        games = library.get("games", [])
        count = library.get("game_count")
        if (isinstance(count, bool) or not isinstance(count, int) or count < 0
                or not isinstance(games, list) or len(games) > MAX_LIBRARY_GAMES):
            raise SteamRecommendationError(502, "steam_unavailable", "Steam returned an unexpected game library response.")
        if count == 0:
            raise SteamRecommendationError(404, "steam_library_empty", "This Steam account has no games yet. Choose a discovery cluster instead.")
        if not games:
            raise SteamRecommendationError(403, "steam_library_private", "This Steam account's game details are private or unavailable. Make game details public in Steam privacy settings.")
        return games, count

    def _fetch_username(self, api_key: str, steam_id: str) -> str | None:
        """Fetch optional display metadata; a profile failure must not hide ranked games."""
        params = urlencode({"key": api_key, "steamids": steam_id})
        request = Request(
            f"{STEAM_PLAYER_SUMMARIES_URL}?{params}",
            headers={"Accept": "application/json", "User-Agent": "NextQuest/1.0"},
        )
        try:
            with self.opener(request, timeout=STEAM_PROFILE_TIMEOUT_SECONDS) as response:
                content = response.read(MAX_STEAM_PROFILE_BYTES + 1)
            if len(content) > MAX_STEAM_PROFILE_BYTES:
                return None
            payload = json.loads(content.decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, socket.timeout, OSError,
                HTTPException, ValueError, UnicodeError, RecursionError):
            return None
        if not isinstance(payload, dict):
            return None
        response = payload.get("response")
        players = response.get("players") if isinstance(response, dict) else None
        if not isinstance(players, list):
            return None
        for player in players:
            if not isinstance(player, dict) or player.get("steamid") != steam_id:
                continue
            name = player.get("personaname")
            if isinstance(name, str):
                name = name.strip()[:100]
                return name or None
        return None

    def _rank_v3(self, hist, cands):
        from rank_pipeline_v3_interact import N_SLOTS, _build_slot_cand
        from heuristic_features_v2 import add_heuristic_features

        if len(cands) == 0:
            return []
        top = hist.sort_values("interest", ascending=False).head(N_SLOTS)
        rows = _build_slot_cand(top, cands, self._feats_df, self.bundle["game_feats"])
        add_heuristic_features(rows, None, cands, top, self._ctx, self.bundle["game_feats"], N_SLOTS)
        matrix = self.pd.DataFrame(rows)[self.bundle["feature_cols"]]
        with self._predict_lock:
            scores = self.model.predict(
                self.xgb.DMatrix(matrix),
                iteration_range=(0, self.best_iteration + 1),
            )
        scores = np.asarray(scores)
        if scores.shape != (len(cands),) or not np.isfinite(scores).all():
            raise ValueError("v3 returned invalid scores")
        order = np.argsort(-scores)
        return [(int(cands[i]), float(scores[i])) for i in order]

    def _model_matrix(self, candidate_features: np.ndarray, history: list[dict]) -> np.ndarray:
        n_candidates = len(candidate_features)
        width = len(self.feature_names)
        n_slots = self.n_slots
        matrix = np.empty((n_candidates, len(self.feature_cols)), dtype=np.float32)
        for index in range(n_slots):
            start = index * width
            if index < len(history):
                game_index = self.appid_to_index[history[index]["appid"]]
                matrix[:, start:start + width] = self.features[game_index]
            else:
                matrix[:, start:start + width] = np.nan
            matrix[:, n_slots * width + index] = history[index]["interest_score"] if index < len(history) else np.nan
            matrix[:, n_slots * width + n_slots + index] = 1.0 if index < len(history) else 0.0
        matrix[:, n_slots * width + 2 * n_slots:] = candidate_features
        return matrix

    def recommend(self, api_key: str, steam_id: str, limit: int = 6) -> dict:
        validate_steam_request(api_key, steam_id, limit)
        games, owned_count = self._fetch_library(api_key, steam_id)
        owned_ids = _owned_appids(games)
        scored = score_owned_games(games)
        if not scored:
            raise SteamRecommendationError(422, "steam_no_played_games", "No played games were found yet. Choose a discovery cluster to get started.")
        in_catalog = [game for game in scored if game["appid"] in self.appid_to_index]
        history = in_catalog[:self.n_slots]
        if not history:
            raise SteamRecommendationError(422, "steam_no_supported_games", "Your played games are outside this catalog. Choose a discovery cluster instead.")

        hist = self.pd.DataFrame(
            {"appid": game["appid"], "interest": game["interest_score"]}
            for game in in_catalog
        )
        blocked = self._edition_blocklist(owned_ids, self._cluster.names, [int(a) for a in self.appids])
        blocked |= {int(a) for a in self.appids if int(a) not in self._ctx["appid_to_idx"]}
        try:
            cands, _, _ = self._cluster.candidates(hist, blocked)
            chosen, _ = self._cluster.top_clusters(hist, self._cluster.profile(hist))
            ranked = self._rank_v3(hist, cands)[:limit]
        except (ValueError, TypeError, RuntimeError, KeyError) as exc:
            raise SteamRecommendationError(503, "model_unavailable", "The saved Steam recommendation model could not score this library.") from exc
        if not len(cands) or not ranked:
            raise SteamRecommendationError(422, "steam_no_candidates", "All catalog games are already in your library.")
        cluster_names = [str(self._cluster_labels.get(c, c)) for c in chosen]

        # Explanations show feature overlap. The saved model also considers the
        # history slots and their interest weights, so overlap is contextual
        # evidence rather than a complete account of the model's decision.
        weights = np.asarray([game["interest_score"] for game in history], dtype=np.float32)
        history_features = self.features[[self.appid_to_index[game["appid"]] for game in history], 1:]
        taste = np.average(history_features, axis=0, weights=weights)
        recommendations = []
        for rank, (appid, score) in enumerate(ranked, 1):
            catalog_index = self.appid_to_index[appid]
            vector = self.features[catalog_index, 1:]
            overlaps = taste * vector
            feature_order = sorted(range(len(overlaps)), key=lambda i: (-float(overlaps[i]), self.feature_names[i + 1]))
            matched = [
                {"label": self.feature_names[i + 1], "requested": round(float(taste[i]), 3),
                 "game": round(float(vector[i]), 3)}
                for i in feature_order if overlaps[i] > 0
            ][:6]
            recommendations.append({
                "rank": rank,
                "appid": appid,
                "name": self.names[catalog_index],
                "url": f"https://store.steampowered.com/app/{appid}/",
                "score": round(score, 4),
                "matched_features": matched,
                "description_terms": [],
                "reason": ("Shares " + ", ".join(item["label"] for item in matched[:3])
                           + " with games in your library.") if matched else "Ranked from your played game history.",
            })
        top_features = np.argsort(-taste)[:8]
        steam_username = self._fetch_username(api_key, steam_id)
        return {
            "source": "steam_history",
            "status": "ok",
            "model": "clusters_v3",
            "note": "Picked 300 games from your top clusters (" + ", ".join(cluster_names)
                    + "), then ranked them with v3. Scores are relative ranking signals.",
            "steam_id": steam_id,
            "steam_username": steam_username,
            "owned_game_count": owned_count,
            "played_game_count": len(scored),
            "matched_history_count": len(history),
            "history_games": [
                {"appid": game["appid"], "name": game["name"] or self.names[self.appid_to_index[game["appid"]]],
                 "interest_score": game["interest_score"]}
                for game in history
            ],
            "query_features": [
                {"label": self.feature_names[i + 1], "score": round(float(taste[i]), 3)}
                for i in top_features if taste[i] > 0
            ],
            "excluded": [],
            "catalog_games": len(self.appids),
            "recommendations": recommendations,
        }
