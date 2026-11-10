# Text to game features and recommendations

Send a description of the game you want to Gemini. `description_to_vector.py` returns the catalog's 93-label feature vector. `recommend_games.py` uses that vector to find matching games in the saved catalog.

## Setup

```powershell
copy .env.example .env
```

Put your Gemini API key in `.env` as `GEMINI_API_KEY`, or set it in your environment. That file stays on your machine and is not pushed. Python and NumPy are required; the recommender uses no other third-party library.

## Get a feature vector

```powershell
py -3.12 description_to_vector.py "A relaxing co-op farming simulator. You grow crops, craft tools, and explore a colorful countryside with friends. No combat or shooting."
```

`vector` has 93 scores in `feature_schema.json` order. Labels Gemini leaves out are `0.0`. `active` lists only the nonzero scores.

## Recommend games

Run from this folder:

```bash
python3 recommend_games.py "A multiplayer game with a story about two brothers rescuing their sister" --limit 5
python3 recommend_games.py "A co-op farming game without shooting" --limit 3 --json
```

The command prints ranked game names, Steam links, and the features that matched. `--json` includes the full request vector, ordered feature names, explicit exclusions, and scores. You can also reuse the catalog in Python:

```python
from recommend_games import TextGameRecommender

recommender = TextGameRecommender()
result = recommender.recommend("A cooperative story game", limit=5)
print(result["recommendations"][0])
```

Only catalog rows with `status=ok` and usable 93-feature vectors are candidates. Explicitly excluded features are hard filters. Ranking is 90% feature match (positive coverage and cosine similarity) and 10% literal title/description overlap, which can help with story details. The scores are relative ranking signals, not satisfaction probabilities. The 93 labels cannot verify a particular plot, character relationship, platform, or game quality; read the linked game description before treating such details as confirmed. A request with no identifiable positive game features returns an error, and impossible constraints return no matches.

Requests must contain 3–2,000 characters and at least one letter or number. Invalid requests are rejected before calling Gemini. Words in explicitly unwanted phrases do not earn the description-overlap bonus.

The saved XGBoost models predict interest from candidate games and, for personalized models, a real Steam play history. None has an input for a text request vector, so this text-only feature ranks directly against the catalog vectors. It does not change the existing models, data, or training scripts.

## Test

From the repository root, run `python3 -m unittest discover -s tests -v`. The tests mock Gemini, so no API key or paid request is needed. A live end-to-end check still requires a configured `GEMINI_API_KEY`.
