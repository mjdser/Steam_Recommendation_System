"""Cold-start top-10 tables and a hardware tier per game.

See README.md. Catalog and clusters come from new_package. Hardware
specs come from HARDWARE_SPECS. This script writes cold_start_tables.xlsx
and game_tiers.csv next to itself.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CATALOG = ROOT / "new_package" / "01_catalog" / "games_for_features.csv"
CLUSTERS = ROOT / "new_package" / "04_clusters" / "game_clusters.csv"
SUMMARIES = ROOT / "new_package" / "04_clusters" / "cluster_summary.csv"
SPEC_FILE = Path(os.environ.get("HARDWARE_SPECS", ""))
OUT = HERE / "cold_start_tables.xlsx"
PLAYERS = HERE / "current_players.jsonl"

HTML = re.compile(r"<[^>]+>")
RAM = re.compile(r"(\d+(?:\.\d+)?)\s*GB", re.I)
GPU_NUM = re.compile(
    r"(rtx|gtx|gt|rx|radeon|arc)\s*[a-z]?\s*(\d{3,4})",
    re.I,
)

# Rough class vs an RTX 3060 (score 20 = mid). Higher means heavier.
GPU_CLASS = {
    (10, "gt"): 4,
    (1030, "gt"): 6,
    (1050, "gtx"): 8,
    (1060, "gtx"): 10,
    (1650, "gtx"): 12,
    (1660, "gtx"): 14,
    (1070, "gtx"): 14,
    (1080, "gtx"): 16,
    (2060, "rtx"): 16,
    (2070, "rtx"): 18,
    (2080, "rtx"): 20,
    (3050, "rtx"): 14,
    (3060, "rtx"): 20,
    (3070, "rtx"): 24,
    (3080, "rtx"): 28,
    (3090, "rtx"): 30,
    (4050, "rtx"): 18,
    (4060, "rtx"): 22,
    (4070, "rtx"): 28,
    (4080, "rtx"): 32,
    (4090, "rtx"): 36,
    (5050, "rtx"): 22,
    (5060, "rtx"): 26,
    (5070, "rtx"): 32,
    (5080, "rtx"): 36,
    (550, "rx"): 6,
    (560, "rx"): 7,
    (570, "rx"): 9,
    (580, "rx"): 10,
    (590, "rx"): 12,
    (5500, "rx"): 12,
    (5600, "rx"): 14,
    (5700, "rx"): 16,
    (6500, "rx"): 12,
    (6600, "rx"): 18,
    (6700, "rx"): 22,
    (6800, "rx"): 26,
    (6900, "rx"): 28,
    (7600, "rx"): 20,
    (7700, "rx"): 26,
    (7800, "rx"): 28,
    (7900, "rx"): 32,
}


def strip(html: str) -> str:
    return re.sub(r"\s+", " ", HTML.sub(" ", html or "")).strip()


def section(text: str, label: str) -> str:
    labels = r"OS|Processor|CPU|Memory|RAM|Graphics|Video Card|Storage|Hard Disk|DirectX|Network|Additional|Sound"
    m = re.search(rf"{label}\s*:\s*(.*?)(?=(?:{labels})\s*:|$)", text, re.I)
    return m.group(1).strip() if m else ""


def ram_gb(text: str) -> float | None:
    nums = [float(x) for x in RAM.findall(text)]
    return max(nums) if nums else None


def gpu_score(text: str) -> float | None:
    low = text.lower()
    if any(k in low for k in ("integrated", "uhd graphics", "intel hd", "vega 8", "vega 11")):
        return 5.0
    best = None
    for m in GPU_NUM.finditer(text):
        brand, num = m.group(1).lower(), int(m.group(2))
        family = brand if brand in ("rtx", "gtx", "gt", "rx") else ("rx" if "radeon" in brand else None)
        if family is None:
            continue
        # Match the closest known model in the same family by number.
        same = [(n, s) for (n, fam), s in GPU_CLASS.items() if fam == family]
        if not same:
            continue
        n, s = min(same, key=lambda pair: abs(pair[0] - num))
        if abs(n - num) > 200 and num < 1000:
            continue
        best = s if best is None else max(best, s)
    return best


def source_label(url: str) -> str:
    u = (url or "").lower()
    if "zenodo.org" in u:
        return "zenodo_2025"
    if "huggingface.co" in u:
        return "hf_steam"
    if "github.com" in u:
        return "github_2019"
    if "steampowered.com" in u:
        return "steam_live"
    return ""


def tier_from(ram: float | None, gpu: float | None) -> str | None:
    parts = []
    if ram is not None:
        parts.append("low" if ram <= 8 else "medium" if ram <= 16 else "high")
    if gpu is not None:
        parts.append("low" if gpu < 16 else "medium" if gpu <= 22 else "high")
    if not parts:
        return None
    order = {"low": 0, "medium": 1, "high": 2}
    return max(parts, key=lambda t: order[t])


def reparse(row: dict) -> dict:
    used = row.get("recommended_text") or ""
    memory = section(used, "Memory") or section(used, "RAM")
    graphics = section(used, "Graphics") or section(used, "Video Card")
    ram = ram_gb(memory) if memory else None
    gpu = gpu_score(graphics) if graphics else None
    row["memory"] = (memory or "")[:180]
    row["graphics"] = (graphics or "")[:180]
    row["ram_gb"] = ram
    row["gpu_score"] = gpu
    row["hardware_tier"] = tier_from(ram, gpu)
    return row


def load_cache() -> dict[int, dict]:
    out = {}
    if not CACHE.exists():
        return out
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("ok"):
            continue
        out[int(row["appid"])] = reparse(row)
    return out


def fetch_one(appid: int) -> dict:
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&filters=basic,pc_requirements"
    req = urllib.request.Request(url, headers={"User-Agent": "NextPlayColdStart/1.0"})
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = 30 if "403" in str(exc) else 2 * (attempt + 1)
            time.sleep(wait)
    else:
        return {"appid": appid, "ok": False, "error": str(last)}
    node = payload.get(str(appid), {})
    data = node.get("data") or {}
    reqs = data.get("pc_requirements") or {}
    rec = strip(reqs.get("recommended") or "")
    minimum = strip(reqs.get("minimum") or "")
    used = rec or minimum
    memory = section(used, "Memory")
    graphics = section(used, "Graphics")
    ram = ram_gb(memory or used)
    gpu = gpu_score(graphics or used)
    return {
        "appid": appid,
        "ok": True,
        "store_success": bool(node.get("success")),
        "name_store": data.get("name") or "",
        "used": "recommended" if rec else ("minimum" if minimum else ""),
        "recommended_text": (rec or minimum)[:500],
        "memory": memory[:180],
        "graphics": graphics[:180],
        "ram_gb": ram,
        "gpu_score": gpu,
        "hardware_tier": tier_from(ram, gpu),
    }


def load_players() -> dict[int, float | None]:
    out = {}
    if not PLAYERS.exists():
        return out
    for line in PLAYERS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[int(row["appid"])] = row.get("current_players")
    return out


def fetch_players(appid: int) -> dict:
    url = f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={appid}"
    req = urllib.request.Request(url, headers={"User-Agent": "NextPlayColdStart/1.0"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        return {"appid": appid, "current_players": None, "error": str(last)}
    body = payload.get("response") or {}
    count = body.get("player_count") if body.get("result") == 1 else None
    return {"appid": appid, "current_players": count}


def main() -> None:
    if not SPEC_FILE.is_file():
        raise FileNotFoundError("Set HARDWARE_SPECS to game_hardware_specs.csv")
    catalog = pd.read_csv(CATALOG)
    assignments = pd.read_csv(CLUSTERS)
    summaries = pd.read_csv(SUMMARIES)
    games = catalog.merge(assignments[["appid", "cluster_id"]], on="appid", how="inner")
    for col in ["positive", "negative", "peak_ccu", "price"]:
        games[col] = pd.to_numeric(games[col], errors="coerce")
        games.loc[games[col] < 0, col] = np.nan
    games["price_missing"] = games["price"].isna()
    games["price"] = games["price"].fillna(0)
    total = games.positive + games.negative
    known = games.positive.notna() & games.negative.notna() & total.gt(0)
    prior = float(games.loc[known, "positive"].sum() / total[known].sum())
    games["quality"] = (games.positive + 100 * prior) / (total + 100)
    players = load_players()
    need = [int(a) for a in games.loc[known, "appid"] if int(a) not in players]
    print(f"player fetch {len(need)} cached {len(players)}", flush=True)
    if need:
        with PLAYERS.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch_players, aid) for aid in need]
            done = 0
            for fut in as_completed(futures):
                row = fut.result()
                if "error" not in row:
                    fh.write(json.dumps(row) + "\n")
                    players[int(row["appid"])] = row["current_players"]
                done += 1
                if done % 400 == 0:
                    print(f"players {done}/{len(need)}", flush=True)
                    fh.flush()
    games["current_players"] = games.appid.map(lambda a: players.get(int(a), None))
    games["current_players"] = pd.to_numeric(games["current_players"], errors="coerce")
    live = games.loc[games.current_players.gt(0), "current_players"]
    player_max = float(live.max()) if len(live) else 1.0
    games["popularity"] = np.log1p(games.current_players) / np.log1p(player_max)
    games["rankable"] = known & games.current_players.notna()
    games["score"] = 100 * np.sqrt(games.quality * games.popularity)
    games.loc[~games.rankable, "score"] = np.nan
    labels = {
        int(r.cluster_id): str(r.label)
        for r in summaries.itertuples()
    }
    games["cluster"] = games.cluster_id.map(labels)

    hw = pd.read_csv(
        SPEC_FILE,
        usecols=[
            "appid", "recommended_ram_gb", "recommended_gpu", "recommended_source",
            "minimum_ram_gb", "minimum_gpu", "minimum_source",
        ],
    )
    rows = []
    for rec in hw.itertuples(index=False):
        used = "recommended"
        ram = pd.to_numeric(rec.recommended_ram_gb, errors="coerce")
        gpu_text = rec.recommended_gpu if isinstance(rec.recommended_gpu, str) else ""
        src = rec.recommended_source if isinstance(rec.recommended_source, str) else ""
        if pd.isna(ram) and not gpu_text.strip():
            used = "minimum"
            ram = pd.to_numeric(rec.minimum_ram_gb, errors="coerce")
            gpu_text = rec.minimum_gpu if isinstance(rec.minimum_gpu, str) else ""
            src = rec.minimum_source if isinstance(rec.minimum_source, str) else ""
        ram_v = None if pd.isna(ram) else float(ram)
        gpu_v = gpu_score(gpu_text) if gpu_text else None
        has_tier = ram_v is not None or gpu_v is not None
        rows.append({
            "appid": int(rec.appid),
            "ram_gb": ram_v,
            "gpu_score": gpu_v,
            "graphics": (gpu_text or "")[:180],
            "used": used if has_tier else "",
            "spec_source": source_label(src) if has_tier else "",
            "spec_url": src if has_tier else "",
            "hardware_tier": tier_from(ram_v, gpu_v),
        })
    spec = pd.DataFrame(rows)
    placed = catalog[["appid", "name"]].merge(spec, on="appid", how="left")
    placed_path = HERE / "game_tiers.csv"
    placed.to_csv(placed_path, index=False)
    print("tiers file", placed_path, placed.hardware_tier.value_counts(dropna=False).to_dict(), flush=True)
    games = games.merge(spec, on="appid", how="left")
    games["hardware_tier"] = games["hardware_tier"].where(games["hardware_tier"].isin(["low", "medium", "high"]))
    print(games.hardware_tier.value_counts(dropna=False).to_dict(), flush=True)

    # Do not write game_metadata.csv until every rankable game has a tier.
    # A partial file would make the live UI rank only the fetched games.

    def top_rows(frame: pd.DataFrame, scope: str, budget: str, hardware: str) -> pd.DataFrame:
        mask = frame.rankable.copy()
        if budget != "all":
            mask &= frame.price.le(float(budget))
        if hardware != "all":
            allowed = {"low": ["low"], "medium": ["low", "medium"], "high": ["low", "medium", "high"]}[hardware]
            mask &= frame.hardware_tier.isin(allowed)
        picked = frame.loc[mask].sort_values(["score", "appid"], ascending=[False, True]).head(10)
        out = picked[
            ["appid", "name", "cluster_id", "cluster", "price", "price_missing", "positive", "negative", "current_players", "peak_ccu", "quality", "popularity", "score", "hardware_tier", "used", "spec_source", "spec_url", "ram_gb", "graphics"]
        ].copy()
        out.insert(0, "rank", range(1, len(out) + 1))
        out.insert(0, "hardware", hardware)
        out.insert(0, "budget", budget)
        out.insert(0, "scope", scope)
        return out

    scopes = [("all", games)] + [(f"cluster_{cid}", games[games.cluster_id.eq(cid)]) for cid in sorted(labels)]
    budgets = ["all", "0", "20", "80"]
    hardwares = ["all", "low", "medium", "high"]
    tables = [
        top_rows(frame, scope, budget, hardware)
        for scope, frame in scopes
        for budget in budgets
        for hardware in hardwares
    ]
    tops = pd.concat(tables, ignore_index=True)
    formula = pd.DataFrame(
        [
            {"item": "score", "value": "100 * sqrt(Q * P)"},
            {"item": "Q", "value": "(positive + 100 * catalog positive fraction) / (positive + negative + 100)"},
            {"item": "prior", "value": round(prior, 6)},
            {"item": "P", "value": "log1p(current_players) / log1p(max current_players). Live Steam count, not peak_ccu."},
            {"item": "player_max", "value": round(player_max, 2)},
            {"item": "mid_pc", "value": "16 GB RAM and RTX 3060-class GPU. Above that is high, below is low."},
            {"item": "pc_filter", "value": "low PC plays low; medium plays low+medium; high plays all classified games"},
            {"item": "budget", "value": "price <= 0, 20, or 80 in dataset units. A missing price is treated as 0 (free)."},
            {"item": "hardware_source", "value": "Recommended RAM and GPU first. Minimum only when Recommended has neither. spec_source is zenodo_2025, hf_steam, github_2019, or steam_live."},
        ]
    )
    spec_out = games.loc[
        :,
        ["appid", "name", "cluster_id", "cluster", "price", "price_missing", "positive", "negative", "current_players", "peak_ccu", "score", "hardware_tier", "used", "spec_source", "spec_url", "ram_gb", "gpu_score", "graphics"],
    ]
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        formula.to_excel(writer, sheet_name="formula", index=False)
        tops.to_excel(writer, sheet_name="tops", index=False)
        spec_out.to_excel(writer, sheet_name="game_specs", index=False)
    print("wrote", OUT, "rows", len(tops), "specs", int(games.hardware_tier.notna().sum()), flush=True)


if __name__ == "__main__":
    main()
