"""Cluster discovery from the current game assignments and saved Steam signals.

The old ``cold_start_tables.xlsx`` was built with different cluster IDs. Reading
the current CSV assignments directly keeps the choices shown to a new player
aligned with the project's latest clustering output.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUDGETS = ("all", "0", "20", "80")
HARDWARE = ("all", "low", "medium", "high")
_HARDWARE_LEVEL = {"low": 0, "medium": 1, "high": 2}


class ColdStartDataError(ValueError):
    """A required saved catalog or cluster artifact is missing or inconsistent."""


def _csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    csv.field_size_limit(max(csv.field_size_limit(), 10_000_000))
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                    or not required.issubset(reader.fieldnames)):
                raise ColdStartDataError(f"Invalid columns in {path.name}.")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ColdStartDataError(f"Could not read {path.name}.") from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ColdStartDataError(f"Incomplete row in {path.name}.")
    return rows


def _id(value: str, path: Path) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ColdStartDataError(f"Invalid ID in {path.name}.") from exc
    if result <= 0:
        raise ColdStartDataError(f"Invalid ID in {path.name}.")
    return result


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _require_unique(rows: list[dict], key: str, path: Path) -> dict[int, dict]:
    result = {}
    for row in rows:
        appid = _id(row[key], path)
        if appid in result:
            raise ColdStartDataError(f"Duplicate ID in {path.name}.")
        result[appid] = row
    return result


class ColdStartCatalog:
    """Rank saved cluster members by review quality and cached player counts.

    The score and filter meanings match ``cold_start/build_cold_start_tables.py``.
    Loading is local and network free; the cached player count is a snapshot.
    """

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        catalog_path = self.root / "new_package/01_catalog/games_for_features.csv"
        assignment_path = self.root / "new_package/04_clusters/game_clusters.csv"
        summary_path = self.root / "new_package/04_clusters/cluster_summary.csv"
        tier_path = self.root / "cold_start/game_tiers.csv"
        players_path = self.root / "cold_start/current_players.jsonl"

        catalog = _require_unique(
            _csv_rows(catalog_path, {"appid", "name", "positive", "negative", "price"}),
            "appid", catalog_path,
        )
        assignments = _require_unique(
            _csv_rows(assignment_path, {"appid", "status", "cluster_id", "cluster_label"}),
            "appid", assignment_path,
        )
        tiers = _require_unique(
            _csv_rows(tier_path, {"appid", "hardware_tier"}), "appid", tier_path,
        )
        try:
            with players_path.open(encoding="utf-8") as stream:
                players = {}
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict) or "appid" not in row:
                        raise ColdStartDataError("Invalid current_players.jsonl row.")
                    appid = _id(row["appid"], players_path)
                    if appid in players:
                        raise ColdStartDataError("Duplicate ID in current_players.jsonl.")
                    players[appid] = _number(row.get("current_players"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ColdStartDataError("Could not read current_players.jsonl.") from exc

        summaries = _csv_rows(summary_path, {"cluster_id", "n_games", "label", "top_features", "examples"})
        self._clusters = []
        seen_clusters = set()
        for row in summaries:
            try:
                cluster_id = int(row["cluster_id"])
                n_games = int(row["n_games"])
            except (TypeError, ValueError) as exc:
                raise ColdStartDataError("Invalid cluster summary.") from exc
            if cluster_id < 0 or n_games <= 0 or cluster_id in seen_clusters or not row["label"].strip():
                raise ColdStartDataError("Invalid cluster summary.")
            seen_clusters.add(cluster_id)
            self._clusters.append({
                "id": cluster_id,
                "label": row["label"],
                "n_games": n_games,
                "top_features": row["top_features"],
                "examples": [part.strip() for part in row["examples"].split(";") if part.strip()],
            })
        if not self._clusters:
            raise ColdStartDataError("No clusters are available.")
        self._by_cluster = {item["id"]: item for item in self._clusters}

        # Ensure IDs and labels are from the same clustering run. This catches
        # accidental mixtures of old workbook IDs and newer CSV assignments.
        counts = {cluster_id: 0 for cluster_id in seen_clusters}
        for appid, row in assignments.items():
            if appid not in catalog:
                raise ColdStartDataError("A clustered game is missing from the catalog.")
            try:
                cluster_id = int(row["cluster_id"])
            except (TypeError, ValueError) as exc:
                raise ColdStartDataError("Invalid cluster assignment.") from exc
            if cluster_id == -1:
                continue
            if (cluster_id not in counts or row["status"] != "ok"
                    or row["cluster_label"] != self._by_cluster[cluster_id]["label"]):
                raise ColdStartDataError("Cluster assignments and summaries do not match.")
            counts[cluster_id] += 1
        if any(counts[item["id"]] != item["n_games"] for item in self._clusters):
            raise ColdStartDataError("Cluster assignments and summaries have different counts.")

        reviews = {}
        for appid, row in catalog.items():
            positive, negative = _number(row["positive"]), _number(row["negative"])
            if positive is not None and negative is not None and positive + negative > 0:
                reviews[appid] = (positive, negative)
        if not reviews:
            raise ColdStartDataError("The catalog has no usable review counts.")
        prior = sum(pair[0] for pair in reviews.values()) / sum(sum(pair) for pair in reviews.values())
        maximum = max((count for count in players.values() if count is not None and count > 0), default=0)
        if maximum <= 0:
            raise ColdStartDataError("The saved player counts have no positive values.")

        self._ranked = {cluster_id: [] for cluster_id in seen_clusters}
        for appid, assignment in assignments.items():
            cluster_id = int(assignment["cluster_id"])
            if cluster_id < 0 or appid not in reviews or players.get(appid) is None:
                continue
            positive, negative = reviews[appid]
            count = players[appid]
            quality = (positive + 100 * prior) / (positive + negative + 100)
            popularity = math.log1p(count) / math.log1p(maximum)
            score = 100 * math.sqrt(quality * popularity)
            price_raw = _number(catalog[appid]["price"])
            price = price_raw if price_raw is not None else 0.0
            tier = tiers.get(appid, {}).get("hardware_tier", "")
            if tier not in ("", "low", "medium", "high"):
                raise ColdStartDataError("Invalid hardware tier in game_tiers.csv.")
            self._ranked[cluster_id].append({
                "appid": appid,
                "name": catalog[appid]["name"],
                "url": f"https://store.steampowered.com/app/{appid}/",
                "score": score,
                "review_quality": round(quality, 4),
                "popularity": round(popularity, 4),
                "price": price,
                "price_missing": price_raw is None,
                "hardware_tier": tier or None,
                "current_players": int(count),
                "positive": int(positive),
                "negative": int(negative),
            })
        for records in self._ranked.values():
            records.sort(key=lambda game: (-game["score"], game["appid"]))

    def list_clusters(self) -> list[dict]:
        """Return display metadata for the current 18 (or configured) clusters."""
        return [{**cluster, "examples": cluster["examples"][:]} for cluster in self._clusters]

    def recommend(self, cluster_id: int, budget: str = "all", hardware: str = "all",
                  limit: int = 10) -> dict:
        if isinstance(cluster_id, bool) or not isinstance(cluster_id, int) or cluster_id not in self._by_cluster:
            raise ValueError("Choose one of the listed game clusters.")
        if budget not in BUDGETS:
            raise ValueError("Choose a budget of all, 0, 20, or 80.")
        if hardware not in HARDWARE:
            raise ValueError("Choose a hardware level of all, low, medium, or high.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("limit must be an integer from 1 to 20.")

        ceiling = None if budget == "all" else float(budget)
        maximum_tier = _HARDWARE_LEVEL.get(hardware)
        eligible = [
            game for game in self._ranked[cluster_id]
            if (ceiling is None or game["price"] <= ceiling)
            and (maximum_tier is None or (game["hardware_tier"] in _HARDWARE_LEVEL
                                          and _HARDWARE_LEVEL[game["hardware_tier"]] <= maximum_tier))
        ]
        recommendations = [
            {**game, "score": round(game["score"], 4), "rank": rank}
            for rank, game in enumerate(eligible[:limit], 1)
        ]
        return {
            "status": "ok" if recommendations else "no_matches",
            "source": "cluster_cold_start",
            "cluster": {**self._by_cluster[cluster_id], "examples": self._by_cluster[cluster_id]["examples"][:]},
            "filters": {"budget": budget, "hardware": hardware},
            "eligible_games": len(eligible),
            "recommendations": recommendations,
        }
