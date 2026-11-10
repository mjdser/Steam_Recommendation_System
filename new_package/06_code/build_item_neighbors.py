"""Build top-10 co-Top10 neighbors for each game (item-item CF table).

Run from the package root:

    python 06_code/build_item_neighbors.py

Reads:
  07_users/libraries_top10.csv
  01_catalog/games_for_features.csv

Writes:
  08_item_neighbors/game_top10_neighbors.csv
  08_item_neighbors/game_top10_neighbors.json
  08_item_neighbors/game_top10_neighbors.xlsx
  08_item_neighbors/README.md
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "08_item_neighbors"
TOP10_CSV = PKG / "07_users" / "libraries_top10.csv"
CATALOG_CSV = PKG / "01_catalog" / "games_for_features.csv"
TOP_N = 10

FOCUS = [
    730,
    440,
    570,
    105600,
    252490,
    271590,
    359550,
    1245620,
    1086940,
    413150,
    252950,
    1172470,
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading...", flush=True)
    top = pd.read_csv(TOP10_CSV, dtype={"steamid": str})
    cat = pd.read_csv(CATALOG_CSV)
    names = dict(zip(cat.appid.astype(int), cat.name.astype(str)))

    item_users: dict[int, set[str]] = defaultdict(set)
    pair: Counter = Counter()
    n_users = 0
    for _, r in top.iterrows():
        apps = []
        for i in range(1, 11):
            a = r.get(f"top{i}_appid")
            if pd.isna(a):
                continue
            apps.append(int(a))
        if not apps:
            continue
        n_users += 1
        apps = list(dict.fromkeys(apps))
        for a in apps:
            item_users[a].add(str(r["steamid"]))
        for i in range(len(apps)):
            for j in range(i + 1, len(apps)):
                x, y = (apps[i], apps[j]) if apps[i] < apps[j] else (apps[j], apps[i])
                pair[(x, y)] += 1

    neighbors: dict[int, list] = defaultdict(list)
    for (x, y), c in pair.items():
        nx, ny = len(item_users[x]), len(item_users[y])
        sim = c / math.sqrt(nx * ny) if nx and ny else 0.0
        neighbors[x].append((y, c, sim, ny))
        neighbors[y].append((x, c, sim, nx))

    for a in neighbors:
        neighbors[a].sort(key=lambda t: (-t[2], -t[1], t[0]))
        neighbors[a] = neighbors[a][:TOP_N]

    all_out = {}
    rows_csv = []
    wide_rows = []
    for a, neigh in sorted(neighbors.items(), key=lambda kv: -len(item_users[kv[0]])):
        entry = {
            "appid": a,
            "name": names.get(a, str(a)),
            "users_with_in_top10": len(item_users[a]),
            "top_neighbors": [
                {
                    "appid": b,
                    "name": names.get(b, str(b)),
                    "co_count": int(c),
                    "similarity": round(float(sim), 6),
                    "neighbor_users_top10": int(nu),
                }
                for b, c, sim, nu in neigh
            ],
        }
        all_out[str(a)] = entry
        wide = {
            "game_appid": a,
            "game_name": names.get(a, str(a)),
            "users_with_in_top10": len(item_users[a]),
        }
        for rank, (b, c, sim, nu) in enumerate(neigh, 1):
            rows_csv.append(
                {
                    "game_appid": a,
                    "game_name": names.get(a, str(a)),
                    "game_users_top10": len(item_users[a]),
                    "rank": rank,
                    "neighbor_appid": b,
                    "neighbor_name": names.get(b, str(b)),
                    "co_count": c,
                    "similarity": round(float(sim), 6),
                    "neighbor_users_top10": nu,
                }
            )
            wide[f"neighbor{rank}_name"] = names.get(b, str(b))
            wide[f"neighbor{rank}_appid"] = b
            wide[f"neighbor{rank}_co_count"] = c
            wide[f"neighbor{rank}_similarity"] = round(float(sim), 6)
        for rank in range(len(neigh) + 1, TOP_N + 1):
            wide[f"neighbor{rank}_name"] = ""
            wide[f"neighbor{rank}_appid"] = ""
            wide[f"neighbor{rank}_co_count"] = ""
            wide[f"neighbor{rank}_similarity"] = ""
        wide_rows.append(wide)

    (OUT / "game_top10_neighbors.json").write_text(
        json.dumps(all_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    long_df = pd.DataFrame(rows_csv)
    wide_df = pd.DataFrame(wide_rows)
    long_df.to_csv(OUT / "game_top10_neighbors.csv", index=False)

    formula = pd.DataFrame(
        [
            {
                "item": "users(i)",
                "meaning": "How many users have game i in their Top-10",
            },
            {
                "item": "co_count(i, j)",
                "meaning": "How many users have BOTH games i and j in their Top-10",
            },
            {
                "item": "similarity",
                "meaning": "sim(i,j) = co_count(i,j) / sqrt(users(i) * users(j))",
            },
            {
                "item": "note",
                "meaning": "This is co-play closeness, NOT genre/feature similarity",
            },
        ]
    )
    xlsx_path = OUT / "game_top10_neighbors.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        wide_df.to_excel(writer, sheet_name="one_row_per_game", index=False)
        long_df.to_excel(writer, sheet_name="long_format", index=False)
        formula.to_excel(writer, sheet_name="how_calculated", index=False)

    lines = [
        "# 08_item_neighbors",
        "",
        "Item–item co-Top10 neighbor table for collaborative filtering.",
        "",
        f"Built from `07_users/libraries_top10.csv` ({n_users:,} users).",
        "",
        "## Formula",
        "",
        "| Name | Meaning |",
        "|---|---|",
        "| `users(i)` | How many users have game i in their Top-10 |",
        "| `co_count(i, j)` | How many users have **both** games in their Top-10 |",
        "| `similarity` | `co_count(i,j) / sqrt(users(i) * users(j))` |",
        "",
        "This is cosine similarity over user sets. It measures how often games appear together, not how similar they are as genres.",
        "",
        "Script: `06_code/build_item_neighbors.py`",
        "",
        "## Files",
        "",
        "| File | What it is |",
        "|---|---|",
        "| `game_top10_neighbors.xlsx` | Easy Excel view (one row per game + top 10 neighbors) |",
        "| `game_top10_neighbors.csv` | Long format (one row per neighbor) |",
        "| `game_top10_neighbors.json` | Same data as JSON |",
        "",
        f"Games with at least one neighbor: **{len(neighbors):,}**",
        "",
        "## Focus examples",
        "",
    ]
    for a in FOCUS:
        if a not in neighbors:
            continue
        lines.append(f"### {names.get(a, a)} (`{a}`)")
        lines.append(f"Users with this game in Top-10: **{len(item_users[a]):,}**")
        lines.append("")
        lines.append("| Rank | Neighbor | Co-count | Similarity |")
        lines.append("|---:|---|---:|---:|")
        for rank, (b, c, sim, _nu) in enumerate(neighbors[a], 1):
            lines.append(f"| {rank} | {names.get(b, b)} | {c} | {sim:.4f} |")
        lines.append("")

    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("games_with_neighbors", len(neighbors), flush=True)
    print("Wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
