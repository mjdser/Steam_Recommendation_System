"""Build one game catalog for LLM / feature tagging.

Catalog = Steam 2026 top 2500 (non-games dropped) UNION every unique user top-10 game.
That is the pool for later clusters + XGBoost: popular candidates plus what users
actually play.

Descriptions:
  1. 2026 dump (stefanotriscali)
  2. March 2025 dump (artermiloff) if the 2026 row is missing or empty
  3. Steam store appdetails for leftover force-adds (owned top-10 ids with no dump row)

Community tags, genres, positives/negatives come from the dumps when present.
Steam appdetails does not return community tags; those stay dump-only.

  python 06_code/build_game_catalog.py
  python 06_code/build_game_catalog.py --no-fetch
  python 06_code/build_game_catalog.py --fetch-all-force
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "06_code"))
from score_interest_top10 import (  # noqa: E402
    STEAM_CSV,
    WIDE_CSV,
    is_nongame_catalog,
    is_nongame_name,
    rank_steam_games,
)

DUMP_2026 = STEAM_CSV
DUMP_2025 = (
    Path.home()
    / ".cache/kagglehub/datasets/artermiloff/steam-games-dataset/versions/2/games_march2025_cleaned.csv"
)
OUT_DIR = ROOT / "01_catalog"
OUT_CSV = OUT_DIR / "games_for_features.csv"
OUT_SUM = OUT_DIR / "games_for_features_summary.json"
CACHE = OUT_DIR / "steam_appdetails_cache.jsonl"
TOP_N = 2500
MIN_DESC = 80
# Library app ids whose store page lives under another dump id.
DESC_ALIAS = {
    620: 659,  # Portal 2
    400: 52003,  # Portal
    12120: 1547000,  # GTA SA -> Definitive
    12110: 1546990,  # Vice City -> Definitive
    12100: 1546970,  # GTA III -> Definitive
    211420: 570940,  # Dark Souls PTDE -> Remastered
    22490: 22380,  # New Vegas PCR
    1811260: 1962391,  # FIFA 23 other id
    3405690: 3484530,  # FC 26 other id
    2669320: 2898350,  # FC 25 other id
    4080220: 3484530,  # FC 27 -> FC 26
    1293830: 1551360,  # Forza Horizon 4 -> 5
    2483190: 1551360,  # Forza Horizon 6 -> 5
    863550: 1659040,  # HITMAN 2 -> World of Assassination
    236870: 382940,  # HITMAN 2016 other id
    34330: 201270,  # Shogun 2 other id
    212910: 202990,  # BO2 Zombies -> BO2
    10180: 10190,  # MW2 2009 other id
    42680: 115300,  # MW3 2011 other id
    3595230: 1962660,  # MWII other id
    3595270: 2519060,  # MWIII other id
    1190460: 1850570,  # Death Stranding -> Director's Cut
    63380: 728740,  # Sniper Elite V2 -> Remastered
    91310: 383150,  # Dead Island -> Definitive
    44350: 1307710,  # GRID 2 -> GRID Legends
    295110: 2394590,  # Just Survive other id
    996470: 1665460,  # PES 2020 -> eFootball
    1259970: 1665460,  # PES 2021 -> eFootball
    1919590: 2878980,  # NBA 2K23 -> 2K25
    1644960: 2878980,  # NBA 2K22 -> 2K25
}

COLS = [
    "appid",
    "name",
    "inclusion",
    "popularity_rank",
    "top10_users",
    "source_description",
    "source_tags",
    "short_description",
    "description",
    "genres",
    "tags",
    "categories",
    "positive",
    "negative",
    "recommendations",
    "pct_positive",
    "peak_ccu",
    "estimated_owners",
    "price",
    "release_date",
    "developers",
    "publishers",
    "has_description",
]

HTML_TAG = re.compile(r"<[^>]+>", re.I)
WS = re.compile(r"\s+")
EDITION_BITS = re.compile(
    r"\b("
    r"definitive edition|the definitive edition|definitive|"
    r"remastered|remaster|complete edition|complete|"
    r"game of the year edition|game of the year|goty|"
    r"gold edition|enhanced edition|enhanced|"
    r"directors cut|director s cut|"
    r"prepare to die edition|pcr|legacy|redux|"
    r"master collection version|master collection|season update|"
    r"friend s pass|zombies|warzone|multiplayer"
    r")\b"
)
YEARLY_PATTERNS = [
    (re.compile(r"^(?:ea sports )?fifa (\d{2})$"), "fifa"),
    (re.compile(r"^(?:ea sports )?fc (\d{2})$"), "ea sports fc"),
    (re.compile(r"^f1 (20\d{2})$"), "f1"),
    (re.compile(r"^nba 2k(\d{2})$"), "nba 2k"),
    (re.compile(r"^(?:efootball(?: pes)?|pes)(?: (\d{4}))?$"), "efootball"),
    (re.compile(r"^forza horizon (\d+)$"), "forza horizon"),
    (re.compile(r"^madden nfl (\d{2})$"), "madden"),
    (re.compile(r"^nhl (\d{2})$"), "nhl"),
    (re.compile(r"^dirt (\d+)$"), "dirt"),
    (re.compile(r"^call of duty modern warfare(?: (\d+|[ivxl]+))?$"), "cod mw"),
    (re.compile(r"^call of duty black ops(?: (\d+|iii|ii|i))?$"), "cod bo"),
]


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def stem_name(s: str) -> str:
    return re.sub(r"\s+", " ", EDITION_BITS.sub(" ", norm_name(s))).strip()


def yearly_family(stem: str) -> tuple[str, int | None] | None:
    for pat, family in YEARLY_PATTERNS:
        m = pat.match(stem)
        if not m:
            continue
        raw = m.group(1) if m.lastindex else None
        year = None
        if raw:
            roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}
            year = roman.get(raw) if raw in roman else int(raw)
        return family, year
    return None


def apply_donor(row: dict, donor: dict, donor_id: int, why: str) -> None:
    row["description"] = donor["description"]
    row["short_description"] = donor.get("short_description") or donor["description"][:400]
    row["source_description"] = f"sibling:{donor_id}:{why}"
    row["has_description"] = True
    for k in ("genres", "tags", "categories", "developers", "publishers"):
        if not as_text(row.get(k)) and as_text(donor.get(k)):
            row[k] = donor[k]
            if k == "tags":
                row["source_tags"] = f"sibling:{donor_id}"


def fill_siblings(rows: list[dict], dump26: dict, dump25: dict, steam: dict) -> int:
    """Copy store text from same title / yearly sequel onto empty rows."""
    donors: dict[int, dict] = {}
    for src in (dump25, dump26, steam):
        for aid, rec in src.items():
            if rec.get("ok") is False:
                continue
            if len(as_text(rec.get("description"))) < MIN_DESC:
                continue
            donors[int(aid)] = rec
    for row in rows:
        if row.get("has_description") and len(as_text(row.get("description"))) >= MIN_DESC:
            donors[int(row["appid"])] = row

    by_exact: dict[str, list[int]] = {}
    by_stem: dict[str, list[int]] = {}
    by_family: dict[str, list[tuple[int, int | None]]] = {}
    for aid, rec in donors.items():
        n = norm_name(rec.get("name") or "")
        st = stem_name(rec.get("name") or "")
        if n:
            by_exact.setdefault(n, []).append(aid)
        if st and (len(st) >= 10 or " " in st):
            by_stem.setdefault(st, []).append(aid)
        fam = yearly_family(st)
        if fam:
            by_family.setdefault(fam[0], []).append((aid, fam[1]))

    filled = 0
    for row in rows:
        if row.get("has_description"):
            continue
        aid = int(row["appid"])
        name = row.get("name") or ""
        n = norm_name(name)
        st = stem_name(name)

        donor_id = None
        why = ""
        hits = [x for x in by_exact.get(n, []) if x != aid]
        if hits and (" " in n or len(n) >= 12):
            donor_id, why = hits[0], "same_name"
        if donor_id is None:
            hits = [x for x in by_stem.get(st, []) if x != aid]
            if len(hits) == 1 or (hits and st in {"grand theft auto san andreas", "grand theft auto vice city", "grand theft auto iii", "dark souls", "fallout new vegas", "dead island", "sniper elite v2", "death stranding"}):
                donor_id, why = hits[0], "edition"
        if donor_id is None:
            fam = yearly_family(st)
            if fam:
                cands = [(d, y) for d, y in by_family.get(fam[0], []) if d != aid]
                if cands:
                    my_year = fam[1]

                    def score(item: tuple[int, int | None]) -> tuple:
                        d, y = item
                        dist = abs((y or 0) - (my_year or 0)) if my_year and y else 99
                        return (dist, -(y or 0), -len(as_text(donors[d].get("description"))))

                    donor_id, why = min(cands, key=score)[0], f"series:{fam[0]}"

        if donor_id is None or donor_id not in donors:
            continue
        apply_donor(row, donors[donor_id], donor_id, why)
        filled += 1
    print(f"filled {filled} empty descriptions from siblings/series")
    return filled


def strip_html(text: str) -> str:
    if not text:
        return ""
    s = html.unescape(str(text))
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = HTML_TAG.sub(" ", s)
    return WS.sub(" ", s).strip()


def as_text(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def load_top10() -> tuple[dict[int, int], dict[int, str]]:
    if not WIDE_CSV.exists():
        raise FileNotFoundError(WIDE_CSV)
    df = pd.read_csv(WIDE_CSV)
    users: dict[int, int] = {}
    names: dict[int, str] = {}
    for i in range(1, 11):
        for aid, name in zip(df[f"top{i}_appid"], df[f"top{i}_game"]):
            if pd.isna(aid):
                continue
            appid = int(aid)
            users[appid] = users.get(appid, 0) + 1
            if appid not in names and pd.notna(name):
                names[appid] = str(name)
    return users, names


def _records(df: pd.DataFrame, id_col: str, name_col: str, about_col: str, extra: dict, source: str) -> dict[int, dict]:
    recs = {}
    for data in df.to_dict("records"):
        aid = int(data[id_col])
        about = strip_html(as_text(data.get(about_col)))
        short_src = extra.get("short")
        short = strip_html(as_text(data.get(short_src))) if short_src else about[:400]
        detailed_src = extra.get("detailed")
        detailed = strip_html(as_text(data.get(detailed_src))) if detailed_src else ""
        desc = detailed or about or short
        recs[aid] = {
            "name": as_text(data.get(name_col)),
            "description": desc,
            "short_description": short or desc[:400],
            "genres": as_text(data.get(extra["genres"])),
            "tags": as_text(data.get(extra["tags"])),
            "categories": as_text(data.get(extra["categories"])),
            "positive": data.get(extra["positive"]),
            "negative": data.get(extra["negative"]),
            "recommendations": data.get(extra["recommendations"]),
            "peak_ccu": data.get(extra["peak_ccu"]),
            "estimated_owners": as_text(data.get(extra["owners"])),
            "price": data.get(extra["price"]),
            "release_date": as_text(data.get(extra["release"])),
            "developers": as_text(data.get(extra["developers"])),
            "publishers": as_text(data.get(extra["publishers"])),
            "source": source,
        }
    return recs


def load_2026() -> dict[int, dict]:
    print("Loading Steam 2026 dump...", flush=True)
    usecols = [
        "AppID",
        "Name",
        "About the game",
        "Genres",
        "Tags",
        "Categories",
        "Positive",
        "Negative",
        "Recommendations",
        "Peak CCU",
        "Estimated owners",
        "Price",
        "Release date",
        "Developers",
        "Publishers",
    ]
    raw = pd.read_csv(DUMP_2026, usecols=usecols)
    return _records(
        raw,
        "AppID",
        "Name",
        "About the game",
        {
            "genres": "Genres",
            "tags": "Tags",
            "categories": "Categories",
            "positive": "Positive",
            "negative": "Negative",
            "recommendations": "Recommendations",
            "peak_ccu": "Peak CCU",
            "owners": "Estimated owners",
            "price": "Price",
            "release": "Release date",
            "developers": "Developers",
            "publishers": "Publishers",
        },
        "dump_2026",
    )


def load_2025() -> dict[int, dict]:
    print("Loading March 2025 dump...", flush=True)
    usecols = [
        "appid",
        "name",
        "detailed_description",
        "about_the_game",
        "short_description",
        "genres",
        "tags",
        "categories",
        "positive",
        "negative",
        "recommendations",
        "peak_ccu",
        "estimated_owners",
        "price",
        "release_date",
        "developers",
        "publishers",
    ]
    raw = pd.read_csv(DUMP_2025, usecols=usecols)
    return _records(
        raw,
        "appid",
        "name",
        "about_the_game",
        {
            "detailed": "detailed_description",
            "short": "short_description",
            "genres": "genres",
            "tags": "tags",
            "categories": "categories",
            "positive": "positive",
            "negative": "negative",
            "recommendations": "recommendations",
            "peak_ccu": "peak_ccu",
            "owners": "estimated_owners",
            "price": "price",
            "release": "release_date",
            "developers": "developers",
            "publishers": "publishers",
        },
        "dump_2025",
    )


def load_cache() -> dict[int, dict]:
    cached = {}
    if not CACHE.exists():
        return cached
    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cached[int(rec["appid"])] = rec
    return cached


def append_cache(rec: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def steam_to_row(appid: int, payload: dict) -> dict:
    data = (payload or {}).get("data") or {}
    about = strip_html(as_text(data.get("about_the_game")))
    detailed = strip_html(as_text(data.get("detailed_description")))
    short = strip_html(as_text(data.get("short_description")))
    genres = ", ".join(g.get("description", "") for g in (data.get("genres") or []) if g)
    cats = ", ".join(c.get("description", "") for c in (data.get("categories") or []) if c)
    recs = (data.get("recommendations") or {}).get("total")
    return {
        "appid": appid,
        "ok": bool((payload or {}).get("success")),
        "name": as_text(data.get("name")),
        "description": detailed or about or short,
        "short_description": short,
        "genres": genres,
        "tags": "",
        "categories": cats,
        "positive": None,
        "negative": None,
        "recommendations": recs,
        "peak_ccu": None,
        "estimated_owners": "",
        "price": (data.get("price_overview") or {}).get("final", 0) / 100.0 if data.get("price_overview") else None,
        "release_date": as_text((data.get("release_date") or {}).get("date")),
        "developers": ", ".join(data.get("developers") or []),
        "publishers": ", ".join(data.get("publishers") or []),
        "source": "steam_appdetails",
        "type": as_text(data.get("type")),
    }


def fetch_appdetails(appid: int, delay: float, session: requests.Session) -> dict:
    last = "failed"
    url = "https://store.steampowered.com/api/appdetails"
    for attempt in range(6):
        time.sleep(delay if attempt == 0 else delay + attempt * 15)
        try:
            r = session.get(url, params={"appids": appid, "cc": "us", "l": "english"}, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"rate limited, sleep {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            body = r.json() or {}
            payload = body.get(str(appid)) or body.get(appid) or {}
            rec = steam_to_row(appid, payload)
            rec["http"] = r.status_code
            return rec
        except Exception as e:
            last = str(e)
            continue
    return {"appid": appid, "ok": False, "error": last, "source": "steam_appdetails", "description": ""}


def pick_meta(aid: int, dump26: dict, dump25: dict, steam: dict, lib_name: str) -> dict:
    order = []
    if aid in steam and steam[aid].get("ok") and len(as_text(steam[aid].get("description"))) >= MIN_DESC:
        order.append(steam[aid])
    if aid in dump26:
        order.append(dump26[aid])
    alias = DESC_ALIAS.get(aid)
    if alias and alias in dump26:
        order.append(dump26[alias])
    if aid in dump25:
        order.append(dump25[aid])
    if alias and alias in dump25:
        order.append(dump25[alias])
    if aid in steam and steam[aid].get("ok"):
        if steam[aid] not in order:
            order.append(steam[aid])
    base = {
        "name": lib_name,
        "description": "",
        "short_description": "",
        "genres": "",
        "tags": "",
        "categories": "",
        "positive": None,
        "negative": None,
        "recommendations": None,
        "peak_ccu": None,
        "estimated_owners": "",
        "price": None,
        "release_date": "",
        "developers": "",
        "publishers": "",
        "source": "library_name_only",
    }
    desc_src = "none"
    tag_src = "none"
    for src in order:
        if not as_text(base["name"]) and as_text(src.get("name")):
            base["name"] = src["name"]
        if len(as_text(base["description"])) < MIN_DESC and len(as_text(src.get("description"))) >= MIN_DESC:
            base["description"] = src["description"]
            base["short_description"] = src.get("short_description") or src["description"][:400]
            desc_src = src.get("source", "")
        for k in (
            "genres",
            "tags",
            "categories",
            "positive",
            "negative",
            "recommendations",
            "peak_ccu",
            "estimated_owners",
            "price",
            "release_date",
            "developers",
            "publishers",
        ):
            cur = base.get(k)
            empty = cur is None or (isinstance(cur, float) and pd.isna(cur)) or as_text(cur) == ""
            if empty and src.get(k) not in (None, ""):
                base[k] = src[k]
                if k == "tags":
                    tag_src = src.get("source", "")
    if not as_text(base["name"]):
        base["name"] = lib_name or f"appid_{aid}"
    base["source_description"] = desc_src
    base["source_tags"] = tag_src
    base["has_description"] = len(as_text(base["description"])) >= MIN_DESC
    return base


def write_outputs(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df = df[COLS]
    df = df.sort_values(["inclusion", "top10_users", "popularity_rank"], ascending=[True, False, True])
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} games)")


def summarize(df: pd.DataFrame) -> dict:
    def n_src(col, val):
        return int((df[col] == val).sum())

    summary = {
        "rows": int(len(df)),
        "top2500_only": n_src("inclusion", "top2500"),
        "force_only": n_src("inclusion", "force"),
        "both": n_src("inclusion", "both"),
        "has_description": int(df["has_description"].sum()),
        "missing_description": int((~df["has_description"]).sum()),
        "desc_from_2026": n_src("source_description", "dump_2026"),
        "desc_from_2025": n_src("source_description", "dump_2025"),
        "desc_from_steam": n_src("source_description", "steam_appdetails"),
        "desc_from_sibling": int(df["source_description"].astype(str).str.startswith("sibling:").sum()),
        "desc_none": n_src("source_description", "none"),
        "files": {
            "csv": "01_catalog/games_for_features.csv",
            "cache": "01_catalog/steam_appdetails_cache.jsonl",
        },
    }
    OUT_SUM.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def build_rows(dump26, dump25, steam, top_users, lib_names, ranked) -> list[dict]:
    top_ids = set(ranked.loc[ranked["popularity_rank"] <= TOP_N, "appid"].astype(int))
    force_ids = set(top_users) - top_ids
    catalog_ids = top_ids | set(top_users)
    rank_map = dict(zip(ranked["appid"].astype(int), ranked["popularity_rank"].astype(int)))
    name_rank = dict(zip(ranked["appid"].astype(int), ranked["name"].astype(str)))
    rows = []
    for aid in sorted(catalog_ids):
        in_top = aid in top_ids
        in_user = aid in top_users
        if in_top and in_user:
            inclusion = "both"
        elif in_top:
            inclusion = "top2500"
        else:
            inclusion = "force"
        lib_name = lib_names.get(aid) or name_rank.get(aid, "")
        if inclusion == "top2500" and is_nongame_name(lib_name):
            continue
        meta = pick_meta(aid, dump26, dump25, steam, lib_name)
        rec = {
            "appid": aid,
            "inclusion": inclusion,
            "popularity_rank": rank_map.get(aid),
            "top10_users": int(top_users.get(aid, 0)),
            **meta,
        }
        rows.append(rec)
    return rows


def fetch_needed(ids: list[int], cached: dict, delay: float) -> dict:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research catalog build)"
    todo = [a for a in ids if a not in cached]
    print(f"steam fetch {len(todo)} / {len(ids)} (cached {len(ids) - len(todo)})")
    for i, aid in enumerate(todo, 1):
        rec = fetch_appdetails(aid, delay, session)
        append_cache(rec)
        cached[aid] = rec
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} last={aid} ok={rec.get('ok')} name={as_text(rec.get('name'))[:40]}")
    return cached


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true", help="skip Steam store calls")
    p.add_argument("--fetch-all-force", action="store_true", help="fetch every force game, even if a dump has text")
    p.add_argument("--delay", type=float, default=2.0)
    args = p.parse_args()

    print("Loading user top 10s...", flush=True)
    top_users, lib_names = load_top10()
    dump26 = load_2026()
    dump25 = load_2025()
    print("Ranking 2026 games...", flush=True)
    ranked = rank_steam_games()
    steam = load_cache()

    top_ids = set(ranked.loc[ranked["popularity_rank"] <= TOP_N, "appid"].astype(int))
    force_ids = sorted(set(top_users) - top_ids)
    if args.fetch_all_force:
        need = force_ids
    else:
        need = []
        for aid in force_ids:
            d26 = as_text((dump26.get(aid) or {}).get("description"))
            d25 = as_text((dump25.get(aid) or {}).get("description"))
            if len(d26) < MIN_DESC and len(d25) < MIN_DESC:
                need.append(aid)

    if not args.no_fetch and need:
        steam = fetch_needed(need, steam, args.delay)
    else:
        print(f"skip fetch; would have requested {len(need)} apps")

    rows = build_rows(dump26, dump25, steam, top_users, lib_names, ranked)
    fill_siblings(rows, dump26, dump25, steam)
    write_outputs(rows)
    summarize(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
