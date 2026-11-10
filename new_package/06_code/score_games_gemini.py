"""Score catalog games with Gemini, one contract, under a shekel budget.

Eight games per call. The description is the score. Community genres and tags
do not raise a score the text already earned. Metadata-only genre <= 0.4,
metadata-only tag <= 0.3. The run stops before the spend reaches 80 ILS.

  python 06_code/score_games_gemini.py --limit 8
  python 06_code/score_games_gemini.py
  python 06_code/score_games_gemini.py --assemble
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "gemini_api_key.txt"
CATALOG = ROOT / "01_catalog" / "games_for_features.csv"
SCHEMA = ROOT / "02_taxonomy" / "dimension_order.json"
TAXONOMY = ROOT / "02_taxonomy" / "taxonomy_v2.json"
OUT_DIR = ROOT / "03_features"
CACHE = OUT_DIR / "gemini_scores.jsonl"
OUT_CSV = OUT_DIR / "gemini_game_features.csv"
SUMMARY = OUT_DIR / "gemini_score_summary.json"

MODEL = "gemini-3.8-flash"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PROMPT_ID = "v2"
BATCH = 8
DESC_CHARS = 1200
MIN_DESC = 80
# Intro rate through Dec 2026. Shekels use ~0.328 USD each.
USD_IN = 0.75 / 1_000_000
USD_OUT = 3.75 / 1_000_000
USD_PER_ILS = 0.328
# Google's spend meter was ₪24.11 when this script counted ~₪8.4, about 3×.
# Stop using the meter, so a 10-minute billing lag does not cross ₪80.
GOOGLE_METER_SCALE = 3.0
BUDGET_REAL_ILS = 75.0

RULES = """You assign interpretable feature scores to Steam games for a recommender.
Later these vectors are averaged across the games a person plays and clustered.
Do not rank quality, popularity, price, or reviews.

For each game, use its store description as the main evidence. Return a sparse set of applicable labels from ALLOWED_LABELS.

EVIDENCE RULES
1. Read the description for meaning. Do not score labels merely because related words appear.
2. Use the game name only to resolve identity or ambiguity. Do not add features from memory.
3. Ignore references to other games, soundtrack credits, marketing comparisons, and explicitly negated features.
4. Community genres and tags are secondary evidence:
   - When they agree with the description, use them as confirmation. Do not automatically increase the score.
   - A community genre supported only by metadata may receive 0.4.
   - A community tag supported only by metadata may receive 0.3.
   - Include metadata-only labels sparingly, only when relevant and plausible.
   - If the description contradicts metadata, trust the description and omit the contradicted label.
5. If the description clearly describes a different edition or product than the name, score the description and set note to "text_may_be_other_edition".
6. If the description is missing or too vague to support meaningful scoring, return an empty features list and set note to "insufficient_description".
7. If the description clearly identifies software, a soundtrack, a book, or another non-game product, return an empty features list and set note to "non_game".

LABEL RULES
- Use only exact label names from ALLOWED_LABELS.
- Follow the supplied label definitions.
- Most labels should be omitted. Usually return 4 to 12 labels, but this is a guideline, not a quota.
- Return fewer labels when evidence is limited and more only when clearly justified.
- Omit unsupported labels instead of returning zero scores.
- Ignore Indie, Free to Play, Early Access, Violent, Gore, Nudity, Sexual Content, and NSFW.
- Distinguish central gameplay from optional, minor, or incidental features.
- Do not assume related labels imply each other. Crafting a strategy does not establish a Crafting mechanic.

SCORING
Scores describe how strongly a feature characterizes the game:
- 0.9-1.0: defining or central gameplay feature.
- 0.7-0.8: substantial, recurring feature.
- 0.5-0.6: clearly supported secondary feature.
- 0.3-0.4: minor feature or limited supporting evidence.
Use one decimal place. Scores do not need to sum to 1.
Metadata-only genres must not exceed 0.4. Metadata-only tags must not exceed 0.3.
evidence is one short sentence, 12 words or fewer.

OUTPUT
Return only JSON: {"games":[{"appid":123,"features":[{"label":"Exact allowed label","score":0.8,"evidence_source":"description","evidence":"Brief gameplay evidence."}],"note":null}]}
Preserve each appid exactly. One record per input game.
Sort features by descending score, then alphabetically.
note is null unless one of the specified conditions applies.
Allowed evidence_source values: "description", "community_genre_only", "community_tag_only".
"""


def load_dims() -> tuple[list[str], dict[str, str]]:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    order = list(schema["genre_dimension_order"]) + list(schema["tag_dimension_order"])
    defs: dict[str, str] = {}
    tax = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    for bucket in (tax.get("genres") or {}, tax.get("tags") or {}):
        for lab in bucket.get("labels") or []:
            defs[str(lab["id"])] = clip(str(lab.get("definition") or ""), 140)
    return order, defs


def clip(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= n:
        return text
    return text[: n - 3].rstrip() + "..."


def api_key() -> str:
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key and KEY_FILE.exists():
        for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "paste" not in line.lower():
                key = line
                break
    if not key:
        raise SystemExit(f"Paste the Gemini key as the only line in {KEY_FILE}")
    return key


def load_cache() -> dict[int, dict]:
    done: dict[int, dict] = {}
    if not CACHE.exists():
        return done
    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("prompt") != PROMPT_ID:
                continue
            done[int(rec["appid"])] = rec
    return done


def append_cache(rec: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    rec["prompt"] = PROMPT_ID
    with CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def spend_usd(cache: dict[int, dict]) -> float:
    prompt = out = 0
    for rec in cache.values():
        usage = rec.get("usage") or {}
        prompt += int(usage.get("prompt") or 0)
        out += int(usage.get("output") or 0) + int(usage.get("thoughts") or 0)
    return prompt * USD_IN + out * USD_OUT


def ils(usd: float) -> float:
    return usd / USD_PER_ILS


def real_ils(usd: float) -> float:
    return ils(usd) * GOOGLE_METER_SCALE


def call_gemini(prompt: str, model: str, key: str) -> tuple[dict, dict]:
    url = URL.format(model=model)
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }
    last = "request failed"
    for attempt in range(8):
        if attempt and last != "429":
            time.sleep(min(60, 5 * attempt))
        r = requests.post(url, params={"key": key}, json=body, timeout=120)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 90))
            wait = max(wait, 90)
            print(f"rate limited, sleep {wait}s", flush=True)
            time.sleep(wait)
            last = "429"
            continue
        if r.status_code >= 500:
            last = f"{r.status_code}"
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata") or {}
        meta = {
            "prompt": usage.get("promptTokenCount"),
            "output": usage.get("candidatesTokenCount"),
            "thoughts": usage.get("thoughtsTokenCount"),
        }
        return json.loads(text), meta
    raise RuntimeError(last)


def label_block(order: list[str], defs: dict[str, str]) -> str:
    lines = []
    for name in order:
        definition = defs.get(name, "")
        lines.append(f"- {name}: {definition}" if definition else f"- {name}")
    return "\n".join(lines)


def game_block(row) -> str:
    desc = clip(getattr(row, "description", "") or getattr(row, "short_description", "") or "", DESC_CHARS)
    return "\n".join(
        [
            "{",
            f'  "appid": {int(row.appid)},',
            f'  "name": {json.dumps(str(getattr(row, "name", "") or ""))},',
            f'  "community_genres": {json.dumps(clip(getattr(row, "genres", "") or "", 300))},',
            f'  "community_tags": {json.dumps(clip(getattr(row, "tags", "") or "", 400))},',
            f'  "description": {json.dumps(desc)}',
            "}",
        ]
    )


def build_prompt(rows: list, order: list[str], defs: dict[str, str]) -> str:
    games = ",\n".join(game_block(row) for row in rows)
    return (
        RULES
        + "\nALLOWED_LABELS:\n"
        + label_block(order, defs)
        + "\n\nGAME_RECORDS:\n[\n"
        + games
        + "\n]\n"
    )


def clean_game(raw: dict, allowed: set[str], genres: set[str]) -> tuple[dict[str, float], str, list]:
    features = []
    scores: dict[str, float] = {}
    for item in raw.get("features") or []:
        label = str(item.get("label") or "").strip()
        if label not in allowed:
            continue
        try:
            value = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        source = str(item.get("evidence_source") or "description")
        if source not in {"description", "community_genre_only", "community_tag_only"}:
            source = "description"
        if source == "community_genre_only":
            value = min(value, 0.4)
        elif source == "community_tag_only":
            value = min(value, 0.3)
        value = round(max(0.0, min(1.0, value)), 1)
        if value < 0.3:
            continue
        if label in scores and scores[label] >= value:
            continue
        scores[label] = value
        features.append(
            {
                "label": label,
                "score": value,
                "evidence_source": source,
                "evidence": clip(str(item.get("evidence") or ""), 120),
            }
        )
    features.sort(key=lambda item: (-item["score"], item["label"]))
    note = raw.get("note")
    if note in ("", "null"):
        note = None
    if isinstance(note, str):
        note = note.strip() or None
    return scores, note if isinstance(note, str) or note is None else None, features


def records_from_response(raw: dict) -> list[dict]:
    if isinstance(raw, list):
        return raw
    for key in ("games", "records", "results"):
        if isinstance(raw.get(key), list):
            return raw[key]
    return []


def local_skip(row) -> dict | None:
    desc = " ".join(str(getattr(row, "description", "") or "").split())
    if len(desc) < MIN_DESC:
        return {
            "appid": int(row.appid),
            "status": "insufficient_description",
            "scores": {},
            "features": [],
            "note": "insufficient_description",
            "usage": {},
        }
    return None


def score_batch(rows: list, order: list[str], defs: dict[str, str], key: str, model: str) -> tuple[list[dict], dict]:
    allowed = set(order)
    genres = set(order[:9])
    prompt = build_prompt(rows, order, defs)
    raw, usage = call_gemini(prompt, model, key)
    wanted = {int(row.appid): row for row in rows}
    found: dict[int, dict] = {}
    for item in records_from_response(raw):
        try:
            appid = int(item.get("appid"))
        except (TypeError, ValueError):
            continue
        if appid not in wanted:
            continue
        scores, note, features = clean_game(item, allowed, genres)
        status = "ok"
        if note in {"insufficient_description", "non_game"} and not scores:
            status = note
        elif note == "text_may_be_other_edition":
            status = "ok"
        found[appid] = {
            "appid": appid,
            "status": status,
            "scores": scores,
            "features": features,
            "note": note,
            "usage": {},
        }
    missing = [row for row in rows if int(row.appid) not in found]
    share = max(1, len(wanted))
    per = {
        "prompt": int((usage.get("prompt") or 0) / share),
        "output": int((usage.get("output") or 0) / share),
        "thoughts": int((usage.get("thoughts") or 0) / share),
    }
    for rec in found.values():
        rec["usage"] = per
    if missing and len(rows) > 1:
        rest = []
        extra_prompt = extra_out = extra_thoughts = 0
        for row in missing:
            piece, extra = score_batch([row], order, defs, key, model)
            rest.extend(piece)
            extra_prompt += int((extra or {}).get("prompt") or 0)
            extra_out += int((extra or {}).get("output") or 0)
            extra_thoughts += int((extra or {}).get("thoughts") or 0)
        usage = {
            "prompt": int(usage.get("prompt") or 0) + extra_prompt,
            "output": int(usage.get("output") or 0) + extra_out,
            "thoughts": int(usage.get("thoughts") or 0) + extra_thoughts,
        }
        return list(found.values()) + rest, usage
    for appid in wanted:
        if appid not in found:
            found[appid] = {
                "appid": appid,
                "status": "error",
                "scores": {},
                "features": [],
                "note": "missing_from_batch",
                "usage": per,
            }
    return list(found.values()), usage


def assemble(catalog: pd.DataFrame, dims: list[str], cache: dict[int, dict]) -> None:
    rows = []
    for rec in catalog.itertuples(index=False):
        appid = int(rec.appid)
        hit = cache.get(appid)
        out = {
            "appid": appid,
            "name": rec.name,
            "inclusion": rec.inclusion,
            "status": "pending",
            "note": "",
            "n_labels": 0,
        }
        for dim in dims:
            out[dim] = None
        if hit:
            out["status"] = hit.get("status") or "ok"
            out["note"] = hit.get("note") or ""
            scores = hit.get("scores") or {}
            out["n_labels"] = len(scores)
            if out["status"] == "ok":
                for dim in dims:
                    out[dim] = float(scores.get(dim, 0.0))
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    usd = spend_usd(cache)
    summary = {
        "model": MODEL,
        "prompt": PROMPT_ID,
        "catalog_rows": int(len(df)),
        "scored": int((df["status"] == "ok").sum()),
        "insufficient_description": int((df["status"] == "insufficient_description").sum()),
        "non_game": int((df["status"] == "non_game").sum()),
        "errors": int((df["status"] == "error").sum()),
        "pending": int((df["status"] == "pending").sum()),
        "spent_usd": round(usd, 4),
        "spent_ils_script": round(ils(usd), 2),
        "spent_ils_google_estimate": round(real_ils(usd), 2),
        "budget_ils_google": BUDGET_REAL_ILS,
        "dimensions": len(dims),
        "files": {"cache": str(CACHE), "csv": str(OUT_CSV)},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="score at most this many new games")
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()

    order, defs = load_dims()
    catalog = pd.read_csv(CATALOG)
    cache = load_cache()
    print(
        f"catalog {len(catalog)} cached {len(cache)} dims {len(order)} google~{real_ils(spend_usd(cache)):.2f} ILS",
        flush=True,
    )
    if args.assemble:
        assemble(catalog, order, cache)
        return

    key = api_key()
    pending = []
    for row in catalog.itertuples(index=False):
        appid = int(row.appid)
        if appid in cache and cache[appid].get("status") in {"ok", "insufficient_description", "non_game"}:
            continue
        skip = local_skip(row)
        if skip:
            append_cache(skip)
            cache[appid] = skip
            continue
        pending.append(row)
        if args.limit and len(pending) >= args.limit:
            break

    scored_new = 0
    for start in range(0, len(pending), BATCH):
        batch = pending[start : start + BATCH]
        if ils(spend_usd(cache)) >= 80:
            print(f"budget stop at script {ils(spend_usd(cache)):.2f} ILS", flush=True)
            break
        time.sleep(args.delay)
        try:
            records, _usage = score_batch(batch, order, defs, key, args.model)
        except Exception as exc:
            print(f"batch error: {exc}", flush=True)
            for row in batch:
                rec = {
                    "appid": int(row.appid),
                    "status": "error",
                    "scores": {},
                    "features": [],
                    "note": str(exc)[:240],
                    "usage": {},
                }
                append_cache(rec)
                cache[int(row.appid)] = rec
            continue
        for rec in records:
            append_cache(rec)
            cache[int(rec["appid"])] = rec
            scored_new += 1
        sample = records[0]
        top = ", ".join(f"{k}={v}" for k, v in list(sample.get("scores", {}).items())[:5])
        print(
            f"{scored_new} appid {sample['appid']} {sample['status']} {top}  google~{real_ils(spend_usd(cache)):.2f} ILS",
            flush=True,
        )

    assemble(catalog, order, cache)


if __name__ == "__main__":
    main()
