"""Fetch the crop recommendation CSV from a public mirror into data/.

The authoritative source is Kaggle (see data/README.md). This script exists so the project
can be run without a Kaggle account; it verifies that whatever it downloaded has the schema
the rest of the code expects.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_recommendation.data import DEFAULT_DATA_PATH, load_dataset

MIRROR_URL = (
    "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/"
    "Data-processed/crop_recommendation.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=MIRROR_URL, help="source URL for the CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url}")
    with urllib.request.urlopen(args.url, timeout=60) as response:
        payload = response.read()
    args.output.write_bytes(payload)

    frame = load_dataset(args.output)
    print(f"wrote {args.output} ({len(frame)} rows, {frame['label'].nunique()} crops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
