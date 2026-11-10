"""Turn one description into the catalog's 93-label Gemini feature vector.

Usage:
    python description_to_vector.py "cozy farming game, no combat"

Reads GEMINI_API_KEY the same way as gemini_query.py. Prints JSON and stops.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gemini_query import extract

HERE = Path(__file__).resolve().parent
SCHEMA = HERE / "feature_schema.json"
TAXONOMY = HERE / "taxonomy_v2.json"


def load_labels():
    names = json.loads(SCHEMA.read_text(encoding="utf-8"))["dimension_order"]
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    by_name = {item["id"]: item for kind in ("genres", "tags") for item in taxonomy[kind]["labels"]}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(f"Taxonomy is missing labels: {missing}")
    return names, [by_name[name] for name in names]


def description_to_vector(text):
    if (not isinstance(text, str) or not 3 <= len(text.strip()) <= 2000
            or not any(char.isalnum() for char in text)):
        raise ValueError("Enter a description between 3 and 2,000 characters, using words or numbers.")
    text = text.strip()
    names, labels = load_labels()
    result = extract(text, labels)
    vector = np.array([result["scores"].get(name, 0.0) for name in names], dtype=np.float32)
    active = [
        {"label": name, "score": float(vector[i])}
        for i, name in enumerate(names)
        if vector[i] > 0
    ]
    return {
        "model": result["model"],
        "note": result["note"],
        "excluded": result["excluded"],
        "names": names,
        "vector": [float(x) for x in vector],
        "active": active,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a game description to 93 feature scores.")
    parser.add_argument("text", nargs="*", help="game description (or provide stdin)")
    args = parser.parse_args(argv)
    text = " ".join(args.text).strip() if args.text else sys.stdin.read().strip()
    try:
        result = description_to_vector(text)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
