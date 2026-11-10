"""LibrariesToTop10: one interest top-10 file per user, then Steam overlap.

Reads the owned-library export and writes 07_users/libraries_top10/<steamid>.csv.
Playtime in that export is minutes. This script converts to hours and keeps
hours for every term, including H_cap.

    interest = 0.40*s_recent + 0.35*s_play + 0.15*s_momentum + 0.10*s_invest

    s_recent   = ln(1 + hours_2weeks) / max ln(1 + hours_2weeks) for this user
    s_play     = ln(1 + hours_forever) / max ln(1 + hours_forever) for this user
    s_momentum = min(1, ln(1 + hours_2weeks) / ln(1 + hours_forever))
    s_invest   = min(1, ln(1 + hours_forever) / ln(1 + H_cap))

H_cap is 200 hours. In this library, played games sit at about 3.7h median,
50h at the 90th percentile, and 118h at the 95th. 200 hours is just past that
95th percentile, so a main game reaches 1 and multi-thousand-hour outliers
do not stretch the scale. Same unit as the hours above.

After the per-user files, the script ranks the 2026 Steam dataset by popularity
and writes how many of the top 2500 and top 5000 show up in those top 10s.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
USER_CSV = Path(os.environ.get("USER_LIBRARY_CSV", "steam_user_api_data_26_9.csv"))
STEAM_CSV = Path(
    os.environ.get(
        "STEAM_2026_CSV",
        str(
            Path.home()
            / ".cache/kagglehub/datasets/stefanotriscali/steam-database-2026-fixed/versions/5/steam_cleaned_2026.csv"
        ),
    )
)
OUT_DIR = ROOT / "07_users" / "libraries_top10"
WIDE_CSV = ROOT / "07_users" / "libraries_top10.csv"
COVERAGE_JSON = ROOT / "07_users" / "libraries_top10_coverage.json"

H_CAP_HOURS = 200.0
W_RECENT = 0.40
W_PLAY = 0.35
W_MOMENTUM = 0.15
W_INVEST = 0.10
TOP_N = 10

NONGAME_NAME = re.compile(
    r"(wallpaper engine|soundpad|blender|crosshair|overlay|sharex|precision x|"
    r"borderless|aseprite|ovr toolkit|ovr advanced|hudsight|steamvr|source sdk|"
    r"driver booster|cpucores|esea|\bdlc\b|soundtrack|artbook|art book|"
    r"bonus content|season pass|dedicated server|server files|\bsdk\b|"
    r"benchmark|configurator|playtest\b|tmodloader|mod loader|"
    r"deleted scenes| - multiplayer$|gameplay recording)",
    re.I,
)
NON_GAME_GENRES = {
    "Education",
    "Game Development",
    "Design & Illustration",
    "Software Training",
    "Utilities",
    "Animation & Modeling",
    "Accounting",
    "Web Publishing",
    "Video Production",
    "Photo Editing",
    "Movie",
    "Audio Production",
}
NON_GAME_APPIDS = {
    431960,  # Wallpaper Engine
    629520,  # Soundpad
    365670,  # Blender
    1366800,  # Crosshair X
    400040,  # ShareX
    1009850,
    268850,
    388080,
    431730,
    1172000,
    1281930,  # tModLoader
    282660,
    1068820,
    2250040,
    1477830,
    920490,  # Driver Booster
    384300,  # CPUCores
    228180,  # Action! recording
    479130,  # ESEA
    1812620,  # DSX
}

USER_FIELDS = [
    "steamid",
    "rank",
    "appid",
    "name",
    "playtime_forever_hours",
    "playtime_2weeks_hours",
    "s_recent",
    "s_play",
    "s_momentum",
    "s_invest",
    "interest_score",
]


def is_nongame_name(name: str) -> bool:
    return bool(name and NONGAME_NAME.search(str(name)))


def split_labels(val) -> list[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("[") or s.startswith("{"):
        try:
            import ast

            v = ast.literal_eval(s)
            if isinstance(v, dict):
                return [str(k) for k in v]
            if isinstance(v, list):
                return [str(x) for x in v]
        except (SyntaxError, ValueError):
            pass
    return [p.strip() for p in s.split(",") if p.strip()]


def is_nongame_catalog(appid: int, name: str, genres_val) -> bool:
    if int(appid) in NON_GAME_APPIDS:
        return True
    if is_nongame_name(name):
        return True
    genres = split_labels(genres_val)
    if genres and set(genres).issubset(NON_GAME_GENRES | {"Free To Play", "Indie", "Early Access"}):
        if any(g in NON_GAME_GENRES for g in genres):
            return True
    return False


def owners_mid(s) -> float:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return 0.0
    txt = str(s).replace(",", "")
    parts = [p for p in txt.replace("-", " ").split() if p.isdigit()]
    if not parts:
        return 0.0
    return sum(float(p) for p in parts) / len(parts)


def score_library(games: list[dict]) -> list[dict]:
    """Return up to 10 played games, best interest first."""
    by_app: dict[int, dict] = {}
    for g in games:
        name = "" if g.get("name") is None else str(g.get("name"))
        if is_nongame_name(name):
            continue
        try:
            appid = int(g.get("appid"))
        except (TypeError, ValueError):
            continue
        forever_min = float(g.get("playtime_forever") or 0)
        weeks_min = float(g.get("playtime_2weeks") or 0)
        if forever_min <= 0 and weeks_min <= 0:
            continue
        prev = by_app.get(appid)
        if prev is None or forever_min > prev["forever_min"]:
            by_app[appid] = {
                "appid": appid,
                "name": name,
                "forever_min": forever_min,
                "weeks_min": weeks_min,
            }

    rows = list(by_app.values())
    if not rows:
        return []

    for row in rows:
        row["forever_h"] = row["forever_min"] / 60.0
        row["weeks_h"] = row["weeks_min"] / 60.0
        row["log_forever"] = math.log1p(row["forever_h"])
        row["log_weeks"] = math.log1p(row["weeks_h"])

    max_forever = max(r["log_forever"] for r in rows)
    max_weeks = max(r["log_weeks"] for r in rows)
    invest_den = math.log1p(H_CAP_HOURS)

    scored = []
    for row in rows:
        s_play = (row["log_forever"] / max_forever) if max_forever > 0 else 0.0
        s_recent = (row["log_weeks"] / max_weeks) if max_weeks > 0 else 0.0
        if row["forever_h"] <= 0:
            s_momentum = 0.0
        else:
            s_momentum = min(1.0, row["log_weeks"] / row["log_forever"])
        s_invest = min(1.0, row["log_forever"] / invest_den) if row["forever_h"] > 0 else 0.0
        interest = (
            W_RECENT * s_recent
            + W_PLAY * s_play
            + W_MOMENTUM * s_momentum
            + W_INVEST * s_invest
        )
        scored.append(
            {
                "appid": row["appid"],
                "name": row["name"],
                "playtime_forever_hours": round(row["forever_h"], 2),
                "playtime_2weeks_hours": round(row["weeks_h"], 2),
                "s_recent": round(s_recent, 4),
                "s_play": round(s_play, 4),
                "s_momentum": round(s_momentum, 4),
                "s_invest": round(s_invest, 4),
                "interest_score": round(interest, 4),
            }
        )

    scored.sort(key=lambda r: (r["interest_score"], r["playtime_forever_hours"]), reverse=True)
    return scored[:TOP_N]


def wide_columns() -> list[str]:
    cols = ["steamid", "profile_name"]
    for i in range(1, TOP_N + 1):
        cols.extend(
            [
                f"top{i}_appid",
                f"top{i}_game",
                f"top{i}_hours",
                f"top{i}_hours_2weeks",
                f"top{i}_interest",
            ]
        )
    return cols


def load_profile_names() -> dict[str, str]:
    names: dict[str, str] = {}
    if not USER_CSV.exists():
        return names
    csv.field_size_limit(2**31 - 1)
    with USER_CSV.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("steamid") or "").strip()
            names[sid] = "" if row.get("profile_name") is None else str(row.get("profile_name"))
    return names


def write_wide_excel() -> pd.DataFrame:
    """One row per user: top1_game, top1_hours, ... top10_interest."""
    names = load_profile_names()
    rows = []
    for path in sorted(OUT_DIR.glob("*.csv")):
        steamid = path.stem
        rec = {c: None for c in wide_columns()}
        rec["steamid"] = steamid
        rec["profile_name"] = names.get(steamid, "")
        with path.open("r", encoding="utf-8", newline="") as f:
            for item in csv.DictReader(f):
                i = int(item["rank"])
                if i < 1 or i > TOP_N:
                    continue
                rec[f"top{i}_appid"] = int(item["appid"]) if item.get("appid") else None
                rec[f"top{i}_game"] = item.get("name") or ""
                rec[f"top{i}_hours"] = float(item["playtime_forever_hours"]) if item.get("playtime_forever_hours") else 0.0
                rec[f"top{i}_hours_2weeks"] = float(item["playtime_2weeks_hours"]) if item.get("playtime_2weeks_hours") else 0.0
                rec[f"top{i}_interest"] = float(item["interest_score"]) if item.get("interest_score") else 0.0
        rows.append(rec)

    df = pd.DataFrame(rows, columns=wide_columns())
    for i in range(1, TOP_N + 1):
        df[f"top{i}_appid"] = pd.to_numeric(df[f"top{i}_appid"], errors="coerce").astype("Int64")
    df["steamid"] = df["steamid"].astype(str)
    df.to_csv(WIDE_CSV, index=False)
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"wrote {WIDE_CSV} ({len(df)} users, one row each)")
    return df


def write_user_file(steamid: str, top: list[dict]) -> None:
    path = OUT_DIR / f"{steamid}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=USER_FIELDS)
        w.writeheader()
        for rank, row in enumerate(top, start=1):
            w.writerow({"steamid": steamid, "rank": rank, **row})


def build_user_top10s() -> dict:
    if not USER_CSV.exists():
        raise FileNotFoundError(USER_CSV)
    csv.field_size_limit(2**31 - 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    users = 0
    users_with_top = 0
    top_rows = 0
    appid_users: dict[int, int] = {}
    bad_json = 0

    with USER_CSV.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users += 1
            steamid = str(row.get("steamid") or "").strip()
            raw = row.get("game_library") or ""
            try:
                library = json.loads(raw) if raw.strip() else []
            except json.JSONDecodeError:
                bad_json += 1
                library = []
            if not isinstance(library, list):
                bad_json += 1
                library = []
            top = score_library(library)
            write_user_file(steamid, top)
            if top:
                users_with_top += 1
                top_rows += len(top)
                for item in top:
                    appid_users[item["appid"]] = appid_users.get(item["appid"], 0) + 1
            if users % 2000 == 0:
                print(f"scored {users} users", flush=True)

    print(f"wrote {users} user files to {OUT_DIR}", flush=True)
    return {
        "users": users,
        "users_with_a_top10_game": users_with_top,
        "top_rows": top_rows,
        "unique_top10_games": len(appid_users),
        "bad_json": bad_json,
        "appid_users": appid_users,
    }


def rank_steam_games() -> pd.DataFrame:
    if not STEAM_CSV.exists():
        raise FileNotFoundError(STEAM_CSV)
    usecols = [
        "AppID",
        "Name",
        "Estimated owners",
        "Peak CCU",
        "Positive",
        "Recommendations",
        "Genres",
    ]
    raw = pd.read_csv(STEAM_CSV, usecols=usecols)
    raw["appid"] = raw["AppID"].astype(int)
    raw["name"] = raw["Name"].fillna("").astype(str)
    raw["_nongame"] = [
        is_nongame_catalog(a, n, g)
        for a, n, g in zip(raw["appid"], raw["name"], raw["Genres"])
    ]
    games = raw.loc[~raw["_nongame"]].copy()
    games["pop"] = (
        games["Peak CCU"].fillna(0).astype(float) * 1000
        + games["Positive"].fillna(0).astype(float)
        + games["Recommendations"].fillna(0).astype(float)
        + games["Estimated owners"].map(owners_mid) / 1000
    )
    games = games.sort_values(["pop", "appid"], ascending=[False, True])
    games = games.drop_duplicates(subset=["appid"], keep="first")
    games["popularity_rank"] = range(1, len(games) + 1)
    return games[["appid", "name", "pop", "popularity_rank"]]


def coverage_block(ranked: pd.DataFrame, user_ids: set[int], n: int) -> dict:
    top = ranked.loc[ranked["popularity_rank"] <= n, "appid"].astype(int)
    top_ids = set(top.tolist())
    appeared = top_ids & user_ids
    force_ids = user_ids - top_ids
    in_dataset = set(ranked["appid"].astype(int))
    force_in_dataset = force_ids & in_dataset
    force_missing = force_ids - in_dataset
    return {
        "steam_top_n": n,
        "steam_top_count": len(top_ids),
        "appear_in_any_user_top10": len(appeared),
        "appear_percent_of_steam_top": round(100.0 * len(appeared) / len(top_ids), 2) if top_ids else 0.0,
        "user_top10_games_inside": len(user_ids & top_ids),
        "user_top10_games_inside_percent": round(100.0 * len(user_ids & top_ids) / len(user_ids), 2) if user_ids else 0.0,
        "force_add_in_steam_dataset": len(force_in_dataset),
        "force_add_missing_from_steam_dataset": len(force_missing),
        "catalog_size_top_plus_force": len(top_ids | force_in_dataset),
    }


def write_coverage(stats: dict) -> dict:
    ranked = rank_steam_games()
    user_ids = set(stats["appid_users"])
    in_dataset = set(ranked["appid"].astype(int))
    report = {
        "h_cap_hours": H_CAP_HOURS,
        "weights": {
            "s_recent": W_RECENT,
            "s_play": W_PLAY,
            "s_momentum": W_MOMENTUM,
            "s_invest": W_INVEST,
        },
        "equation": (
            "0.40*s_recent + 0.35*s_play + 0.15*s_momentum + 0.10*s_invest; "
            "hours throughout; H_cap = 200"
        ),
        "user_csv": USER_CSV.name,
        "steam_csv": STEAM_CSV.name,
        "user_files": "07_users/libraries_top10.csv",
        "users": stats["users"],
        "users_with_a_top10_game": stats["users_with_a_top10_game"],
        "top_rows": stats["top_rows"],
        "unique_user_top10_games": stats["unique_top10_games"],
        "user_top10_games_in_steam_dataset": len(user_ids & in_dataset),
        "user_top10_games_missing_from_steam_dataset": len(user_ids - in_dataset),
        "steam_games_ranked": int(len(ranked)),
        "popularity": "peak_ccu*1000 + positive + recommendations + owner_midpoint/1000, non-games removed",
        "top_2500": coverage_block(ranked, user_ids, 2500),
        "top_5000": coverage_block(ranked, user_ids, 5000),
        "bad_json": stats["bad_json"],
    }
    COVERAGE_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "equation"}, indent=2))
    print(f"wrote {COVERAGE_JSON}")
    return report


def main() -> None:
    stats = build_user_top10s()
    write_wide_excel()
    write_coverage(stats)


if __name__ == "__main__":
    main()
