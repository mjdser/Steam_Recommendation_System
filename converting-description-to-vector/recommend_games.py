"""Recommend catalog games from a free-text description of the desired game.

The request and each catalog game share the ordered 93-label taxonomy. Gemini
extracts the request vector; local code retrieves and ranks existing games.
The saved XGBoost models are history-based and do not accept a request vector.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from numbers import Real
from pathlib import Path
import re
import sys

import numpy as np

from description_to_vector import description_to_vector


ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
NEGATED_RE = re.compile(
    r"\b(?:no|not|without|avoid|exclude|excluding)\b\s+(.+?)(?=[,.;!?]|\b(?:but|however|instead|while|with|where|which|when|who|whose|that)\b|$)",
    re.IGNORECASE,
)
STOPWORDS = {
    "about", "after", "also", "and", "are", "avoid", "but", "can", "could", "does",
    "exclude", "excluding",
    "for", "from", "game", "games", "have", "into", "like", "mode", "more",
    "not", "play", "playing", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "through", "try", "want", "who", "with", "without", "would", "you",
}


class CatalogError(ValueError):
    """The saved feature catalog cannot be matched to the taxonomy."""


def _validate_request(text: str, limit: int) -> None:
    if (not isinstance(text, str) or not 3 <= len(text.strip()) <= 2000
            or not any(char.isalnum() for char in text)):
        raise ValueError("Describe the game you want in 3 to 2,000 characters, using words or numbers.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("limit must be an integer from 1 to 20.")


def _terms(text: str) -> set[str]:
    """Small lexical supplement for names and plot details outside the taxonomy."""
    words = set()
    for word in WORD_RE.findall(text.casefold()):
        if len(word) < 3 or word in STOPWORDS:
            continue
        if len(word) > 5 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.add(word)
    return words


def _read_csv(path: Path):
    csv.field_size_limit(max(csv.field_size_limit(), 10_000_000))
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            yield reader.fieldnames, None
            yield from ((None, row) for row in reader)
    except (UnicodeError, csv.Error) as exc:
        raise CatalogError(f"Could not parse {path.name}.") from exc


def _check_header(header, required, path: Path):
    if (not header or len(header) != len(set(header))
            or any(name not in header for name in required)):
        raise CatalogError(f"Invalid columns in {path.name}.")


def _check_row(row, path: Path):
    if None in row or any(value is None for value in row.values()):
        raise CatalogError(f"Incomplete row in {path.name}.")


def _appid(row, path: Path) -> int:
    try:
        appid = int(row["appid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CatalogError(f"Invalid game ID in {path.name}.") from exc
    if appid <= 0:
        raise CatalogError(f"Invalid game ID in {path.name}: {appid}")
    return appid


class TextGameRecommender:
    """Load the scored catalog once and reuse it for free-text requests."""

    def __init__(self, root: Path = ROOT, *, converter=description_to_vector):
        self.root = Path(root)
        self.converter = converter
        schema = json.loads((self.root / "new_package/02_taxonomy/dimension_order.json").read_text(encoding="utf-8"))
        if (not isinstance(schema, dict)
                or not isinstance(schema.get("genre_dimension_order"), list)
                or not isinstance(schema.get("tag_dimension_order"), list)):
            raise CatalogError("The game taxonomy has an invalid feature order.")
        self.names = tuple(schema["genre_dimension_order"] + schema["tag_dimension_order"])
        if (len(self.names) != 93 or any(not isinstance(name, str) for name in self.names)
                or len(set(self.names)) != 93):
            raise CatalogError("The game taxonomy must contain 93 unique labels.")
        converter_schema = json.loads((Path(__file__).parent / "feature_schema.json").read_text(encoding="utf-8"))
        if (not isinstance(converter_schema, dict)
                or not isinstance(converter_schema.get("dimension_order"), list)
                or tuple(converter_schema["dimension_order"]) != self.names):
            raise CatalogError("The text converter and game catalog use different feature orders.")
        self.index = {name: i for i, name in enumerate(self.names)}

        catalog_path = self.root / "new_package/01_catalog/games_for_features.csv"
        catalog_rows = _read_csv(catalog_path)
        catalog_header, _ = next(catalog_rows)
        _check_header(catalog_header, ("appid", "name"), catalog_path)
        metadata = {}
        for _, row in catalog_rows:
            _check_row(row, catalog_path)
            appid = _appid(row, catalog_path)
            if appid in metadata:
                raise CatalogError(f"Duplicate game ID in catalog metadata: {appid}")
            metadata[appid] = row

        features_path = self.root / "new_package/03_features/game_features.csv"
        feature_rows = _read_csv(features_path)
        header, _ = next(feature_rows)
        _check_header(header, ("appid", "name", "status"), features_path)
        if tuple(header[:6]) != ("appid", "name", "inclusion", "status", "note", "n_labels") or tuple(header[6:]) != self.names:
            raise CatalogError("Game feature columns do not match the 93-label taxonomy order.")

        self.games, vectors = [], []
        seen = set()
        for _, row in feature_rows:
            _check_row(row, features_path)
            appid = _appid(row, features_path)
            if appid in seen:
                raise CatalogError(f"Duplicate game ID in feature catalog: {appid}")
            seen.add(appid)
            if row["status"] not in ("ok", "insufficient_description", "non_game"):
                raise CatalogError(f"Game {appid} has an unknown feature status.")
            if row["status"] != "ok":
                continue
            if appid not in metadata:
                raise CatalogError(f"Game {appid} is missing catalog metadata.")
            try:
                vector = np.asarray([float(row[name]) for name in self.names], dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise CatalogError(f"Game {appid} has a non-numeric feature.") from exc
            if not np.isfinite(vector).all() or np.any((vector < 0) | (vector > 1)) or not np.any(vector):
                raise CatalogError(f"Game {appid} has an invalid feature vector.")
            meta = metadata[appid]
            name = meta.get("name") or row["name"]
            if not name:
                raise CatalogError(f"Game {appid} has no name.")
            self.games.append({
                "appid": appid,
                "name": name,
                "description": meta.get("description") or meta.get("short_description") or "",
            })
            vectors.append(vector)
        if not vectors:
            raise CatalogError("The catalog has no games with usable feature vectors.")
        self.matrix = np.stack(vectors)
        self.norms = np.linalg.norm(self.matrix, axis=1)

        self.title_terms = [_terms(game["name"]) for game in self.games]
        self.description_terms = [_terms(game["description"]) for game in self.games]
        self.term_counts = Counter()
        for title, description in zip(self.title_terms, self.description_terms):
            self.term_counts.update(title | description)

    def _description_scores(self, query: str, excluded: list[str]) -> tuple[np.ndarray, set[str]]:
        query_terms = _terms(query)
        for label in excluded:
            query_terms.difference_update(_terms(label))
        for match in NEGATED_RE.finditer(query):
            query_terms.difference_update(_terms(match.group(1)))
        scores = np.zeros(len(self.games), dtype=np.float32)
        if not query_terms:
            return scores, query_terms
        weights = {
            term: math.log((len(self.games) + 1) / (self.term_counts[term] + 1)) + 1
            for term in query_terms
        }
        total = sum(weights.values())
        for i, (title, description) in enumerate(zip(self.title_terms, self.description_terms)):
            matched = sum(weights[term] * (1.5 if term in title else 1.0)
                          for term in query_terms if term in title or term in description)
            scores[i] = min(1.0, matched / total)
        return scores, query_terms

    def recommend(self, text: str, *, limit: int = 5) -> dict:
        """Return the best matching games and the interpreted request vector.

        Feature similarity is the main ranking signal. Literal title/description
        overlap has only a small influence for story details not in the taxonomy.
        Scores are relative match measures, not satisfaction probabilities.
        """
        _validate_request(text, limit)
        text = text.strip()
        interpreted = self.converter(text)
        if not isinstance(interpreted, dict) or not isinstance(interpreted.get("names"), (list, tuple)):
            raise ValueError("The text converter returned an invalid result.")
        if tuple(interpreted["names"]) != self.names:
            raise CatalogError("The text vector order does not match the saved game catalog.")
        raw_vector = interpreted.get("vector")
        if (not isinstance(raw_vector, (list, tuple, np.ndarray))
                or (isinstance(raw_vector, np.ndarray) and raw_vector.ndim != 1)
                or len(raw_vector) != 93
                or any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_vector)):
            raise ValueError("The text converter returned an invalid vector.")
        try:
            query = np.asarray(raw_vector, dtype=np.float32)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("The text converter returned an invalid vector.") from exc
        if (query.shape != (93,) or not np.isfinite(query).all()
                or np.any((query < 0) | (query > 1)) or not np.any(query)):
            raise ValueError("The text converter returned no usable game features.")
        excluded = interpreted.get("excluded", [])
        if (not isinstance(excluded, list) or any(not isinstance(name, str) for name in excluded)
                or len(excluded) != len(set(excluded)) or any(name not in self.index for name in excluded)
                or any(query[self.index[name]] > 0 for name in excluded)):
            raise ValueError("The text converter returned invalid excluded features.")
        if interpreted.get("model") is not None and not isinstance(interpreted["model"], str):
            raise ValueError("The text converter returned an invalid model name.")
        if not isinstance(interpreted.get("note", ""), str):
            raise ValueError("The text converter returned an invalid note.")

        dot = self.matrix @ query
        coverage = dot / float(np.sum(query))
        cosine = dot / (self.norms * float(np.linalg.norm(query)))
        feature_score = 0.7 * coverage + 0.3 * cosine
        lexical_score, query_terms = self._description_scores(text, excluded)
        score = 0.9 * feature_score + 0.1 * lexical_score

        # An excluded catalog label with any positive score is a hard rejection.
        eligible = dot > 0
        for name in excluded:
            eligible &= self.matrix[:, self.index[name]] == 0
        eligible_ids = np.flatnonzero(eligible)
        order = sorted(eligible_ids, key=lambda i: (-score[i], -feature_score[i], self.games[i]["appid"]))
        recommendations = []
        for i in order[:limit]:
            game = self.games[i]
            matches = [
                {"label": name, "requested": round(float(query[j]), 3), "game": round(float(self.matrix[i, j]), 3)}
                for j, name in enumerate(self.names) if query[j] > 0 and self.matrix[i, j] > 0
            ]
            matches.sort(key=lambda item: (-item["requested"] * item["game"], item["label"]))
            words = sorted(query_terms & (self.title_terms[i] | self.description_terms[i]))[:8]
            recommendations.append({
                "appid": game["appid"],
                "name": game["name"],
                "url": f"https://store.steampowered.com/app/{game['appid']}/",
                "score": round(float(score[i]), 4),
                "feature_score": round(float(feature_score[i]), 4),
                "description_score": round(float(lexical_score[i]), 4),
                "matched_features": matches,
                "description_terms": words,
            })
        return {
            "status": "ok" if recommendations else "no_matches",
            "query": text,
            "model": interpreted.get("model"),
            "note": interpreted.get("note", ""),
            "query_features": [
                {"label": name, "score": round(float(query[i]), 3)}
                for i, name in enumerate(self.names) if query[i] > 0
            ],
            "excluded": excluded,
            "feature_names": list(self.names),
            "feature_vector": [round(float(value), 3) for value in query],
            "catalog_games": len(self.games),
            "recommendations": recommendations,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recommend games from a free-text game description.")
    parser.add_argument("query", nargs="*", help="description of the game you want (or provide stdin)")
    parser.add_argument("--limit", type=int, default=5, help="number of games, 1 to 20")
    parser.add_argument("--json", action="store_true", help="print the complete result as JSON")
    args = parser.parse_args(argv)
    text = " ".join(args.query).strip() if args.query else sys.stdin.read().strip()
    try:
        _validate_request(text, args.limit)
        result = TextGameRecommender().recommend(text, limit=args.limit)
    except (CatalogError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if result["status"] == "no_matches":
        print("No catalog games matched the requested features and exclusions.")
        return 0
    print("Interpreted features: " + ", ".join(
        f"{item['label']}={item['score']:.1f}" for item in result["query_features"]
    ))
    if result["excluded"]:
        print("Excluded: " + ", ".join(result["excluded"]))
    for rank, game in enumerate(result["recommendations"], 1):
        features = ", ".join(item["label"] for item in game["matched_features"][:4])
        print(f"{rank}. {game['name']} ({game['appid']}) — match {game['score']:.3f}; features: {features}")
        print(f"   {game['url']}")
    print("Match scores are ranking signals, not satisfaction probabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
