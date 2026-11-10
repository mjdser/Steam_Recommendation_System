# 08_item_neighbors

Item–item co-Top10 neighbor table for collaborative filtering.

Built from `07_users/libraries_top10.csv` (20,064 users).

## Formula

| Name | Meaning |
|---|---|
| `users(i)` | How many users have game i in their Top-10 |
| `co_count(i, j)` | How many users have **both** games in their Top-10 |
| `similarity` | `co_count(i,j) / sqrt(users(i) * users(j))` |

This is cosine similarity over user sets. It measures how often games appear together, not how similar they are as genres.

Script: `06_code/build_item_neighbors.py`

## Files

| File | What it is |
|---|---|
| `game_top10_neighbors.xlsx` | Easy Excel view (one row per game + top 10 neighbors) |
| `game_top10_neighbors.csv` | Long format (one row per neighbor) |
| `game_top10_neighbors.json` | Same data as JSON |

Games with at least one neighbor: **6,902**

## Focus examples

### Counter-Strike 2 (`730`)
Users with this game in Top-10: **13,973**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Counter-Strike | 6140 | 0.6416 |
| 2 | Grand Theft Auto V Legacy | 3743 | 0.4631 |
| 3 | PUBG: BATTLEGROUNDS | 2955 | 0.4452 |
| 4 | Team Fortress 2 | 3852 | 0.3973 |
| 5 | Garry's Mod | 3781 | 0.3939 |
| 6 | Tom Clancy's Rainbow Six Siege | 2505 | 0.3868 |
| 7 | Counter-Strike: Source | 1989 | 0.3572 |
| 8 | Rust | 1680 | 0.3223 |
| 9 | Euro Truck Simulator 2 | 1575 | 0.3190 |
| 10 | Left 4 Dead 2 | 1977 | 0.3099 |

### Team Fortress 2 (`440`)
Users with this game in Top-10: **6,726**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Garry's Mod | 3756 | 0.5639 |
| 2 | Counter-Strike 2 | 3852 | 0.3973 |
| 3 | Terraria | 1676 | 0.3480 |
| 4 | Left 4 Dead 2 | 1329 | 0.3003 |
| 5 | PAYDAY 2 | 1017 | 0.2643 |
| 6 | Grand Theft Auto V Legacy | 1345 | 0.2398 |
| 7 | Fallout: New Vegas | 700 | 0.2351 |
| 8 | The Binding of Isaac: Rebirth | 597 | 0.2217 |
| 9 | Tom Clancy's Rainbow Six Siege | 950 | 0.2114 |
| 10 | Deadlock | 475 | 0.1958 |

### Dota 2 (`570`)
Users with this game in Top-10: **1,530**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Counter-Strike 2 | 1319 | 0.2853 |
| 2 | Counter-Strike | 789 | 0.2491 |
| 3 | PUBG: BATTLEGROUNDS | 416 | 0.1894 |
| 4 | Path of Exile | 102 | 0.1336 |
| 5 | Team Fortress 2 | 401 | 0.1250 |
| 6 | Dota Underlords | 53 | 0.1247 |
| 7 | Counter-Strike: Source | 225 | 0.1221 |
| 8 | Grand Theft Auto V Legacy | 317 | 0.1185 |
| 9 | Warframe | 141 | 0.1013 |
| 10 | Rust | 168 | 0.0974 |

### Terraria (`105600`)
Users with this game in Top-10: **3,448**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Team Fortress 2 | 1676 | 0.3480 |
| 2 | Garry's Mod | 1533 | 0.3215 |
| 3 | Counter-Strike 2 | 2011 | 0.2897 |
| 4 | The Binding of Isaac: Rebirth | 497 | 0.2578 |
| 5 | Stardew Valley | 342 | 0.1923 |
| 6 | Bloons TD 6 | 301 | 0.1879 |
| 7 | Geometry Dash | 301 | 0.1839 |
| 8 | Risk of Rain 2 | 256 | 0.1795 |
| 9 | ELDEN RING | 339 | 0.1664 |
| 10 | Tom Clancy's Rainbow Six Siege | 528 | 0.1641 |

### Rust (`252490`)
Users with this game in Top-10: **1,945**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Counter-Strike 2 | 1680 | 0.3223 |
| 2 | PUBG: BATTLEGROUNDS | 577 | 0.2330 |
| 3 | Grand Theft Auto V Legacy | 689 | 0.2285 |
| 4 | Tom Clancy's Rainbow Six Siege | 537 | 0.2222 |
| 5 | Garry's Mod | 660 | 0.1843 |
| 6 | Counter-Strike | 608 | 0.1703 |
| 7 | DayZ | 175 | 0.1615 |
| 8 | Team Fortress 2 | 548 | 0.1515 |
| 9 | Unturned | 245 | 0.1453 |
| 10 | Rocket League | 242 | 0.1359 |

### Grand Theft Auto V Legacy (`271590`)
Users with this game in Top-10: **4,676**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Counter-Strike 2 | 3743 | 0.4631 |
| 2 | PUBG: BATTLEGROUNDS | 1109 | 0.2888 |
| 3 | Garry's Mod | 1585 | 0.2854 |
| 4 | Tom Clancy's Rainbow Six Siege | 1018 | 0.2717 |
| 5 | Counter-Strike | 1478 | 0.2670 |
| 6 | Team Fortress 2 | 1345 | 0.2398 |
| 7 | Grand Theft Auto V Enhanced | 397 | 0.2392 |
| 8 | Rust | 689 | 0.2285 |
| 9 | Rocket League | 595 | 0.2155 |
| 10 | Euro Truck Simulator 2 | 605 | 0.2118 |

### Tom Clancy's Rainbow Six Siege (`359550`)
Users with this game in Top-10: **3,002**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Counter-Strike 2 | 2505 | 0.3868 |
| 2 | Grand Theft Auto V Legacy | 1018 | 0.2717 |
| 3 | PUBG: BATTLEGROUNDS | 748 | 0.2431 |
| 4 | Rust | 537 | 0.2222 |
| 5 | Team Fortress 2 | 950 | 0.2114 |
| 6 | Garry's Mod | 935 | 0.2101 |
| 7 | Apex Legends | 447 | 0.2068 |
| 8 | Dead by Daylight | 383 | 0.1856 |
| 9 | Rocket League | 378 | 0.1708 |
| 10 | Terraria | 528 | 0.1641 |

### ELDEN RING (`1245620`)
Users with this game in Top-10: **1,203**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | DARK SOULS™ III | 271 | 0.3350 |
| 2 | ELDEN RING NIGHTREIGN | 76 | 0.2043 |
| 3 | DARK SOULS™ II: Scholar of the First Sin | 100 | 0.1935 |
| 4 | DARK SOULS™: REMASTERED | 98 | 0.1816 |
| 5 | Terraria | 339 | 0.1664 |
| 6 | Counter-Strike 2 | 647 | 0.1578 |
| 7 | Team Fortress 2 | 434 | 0.1526 |
| 8 | Sekiro™: Shadows Die Twice | 49 | 0.1434 |
| 9 | The Binding of Isaac: Rebirth | 158 | 0.1387 |
| 10 | Baldur's Gate 3 | 107 | 0.1296 |

### Baldur's Gate 3 (`1086940`)
Users with this game in Top-10: **567**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Divinity: Original Sin 2 | 32 | 0.1386 |
| 2 | ELDEN RING | 107 | 0.1296 |
| 3 | Cyberpunk 2077 | 96 | 0.1279 |
| 4 | The Witcher 3: Wild Hunt - Complete Edition | 70 | 0.0937 |
| 5 | Team Fortress 2 | 181 | 0.0927 |
| 6 | Terraria | 127 | 0.0908 |
| 7 | The Elder Scrolls V: Skyrim | 64 | 0.0900 |
| 8 | Marvel Rivals | 57 | 0.0875 |
| 9 | Garry's Mod | 169 | 0.0874 |
| 10 | Fallout 4 | 69 | 0.0861 |

### Stardew Valley (`413150`)
Users with this game in Top-10: **917**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Terraria | 342 | 0.1923 |
| 2 | Counter-Strike 2 | 486 | 0.1358 |
| 3 | Team Fortress 2 | 327 | 0.1317 |
| 4 | Garry's Mod | 286 | 0.1163 |
| 5 | The Binding of Isaac: Rebirth | 95 | 0.0955 |
| 6 | Bloons TD 6 | 73 | 0.0884 |
| 7 | Hollow Knight | 41 | 0.0846 |
| 8 | The Elder Scrolls V: Skyrim Special Edition | 78 | 0.0842 |
| 9 | Fallout: New Vegas | 92 | 0.0837 |
| 10 | Grand Theft Auto V Legacy | 167 | 0.0806 |

### Rocket League (`252950`)
Users with this game in Top-10: **1,631**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Counter-Strike 2 | 1446 | 0.3029 |
| 2 | Grand Theft Auto V Legacy | 595 | 0.2155 |
| 3 | PUBG: BATTLEGROUNDS | 479 | 0.2112 |
| 4 | Counter-Strike | 598 | 0.1829 |
| 5 | Tom Clancy's Rainbow Six Siege | 378 | 0.1708 |
| 6 | Euro Truck Simulator 2 | 271 | 0.1606 |
| 7 | Rust | 242 | 0.1359 |
| 8 | Z1 Battle Royale | 114 | 0.1234 |
| 9 | Paladins | 152 | 0.1155 |
| 10 | Team Fortress 2 | 371 | 0.1120 |

### Apex Legends (`1172470`)
Users with this game in Top-10: **1,556**

| Rank | Neighbor | Co-count | Similarity |
|---:|---|---:|---:|
| 1 | Counter-Strike 2 | 1338 | 0.2870 |
| 2 | Tom Clancy's Rainbow Six Siege | 447 | 0.2068 |
| 3 | PUBG: BATTLEGROUNDS | 400 | 0.1806 |
| 4 | Counter-Strike | 564 | 0.1766 |
| 5 | Grand Theft Auto V Legacy | 387 | 0.1435 |
| 6 | Marvel Rivals | 143 | 0.1325 |
| 7 | Rust | 226 | 0.1299 |
| 8 | Overwatch® | 142 | 0.1278 |
| 9 | Team Fortress 2 | 395 | 0.1221 |
| 10 | Terraria | 282 | 0.1217 |

