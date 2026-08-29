"""Predict a crop from soil and weather readings using the saved model."""

import argparse
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from crop_recommendation.data import DEFAULT_MODEL_PATH, FEATURES


def load_model(path: Path | str = DEFAULT_MODEL_PATH) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train a model first: python train.py")
    return joblib.load(path)


def predict_frame(bundle: Dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Predicted crop, plus confidence, for every row of the input."""
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"input is missing the columns {missing}")

    model = bundle["model"]
    features = frame[FEATURES]
    result = pd.DataFrame({"crop": model.predict(features)}, index=frame.index)
    if hasattr(model, "predict_proba"):
        result["confidence"] = model.predict_proba(features).max(axis=1)
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--csv", type=Path, help="predict for every row of a CSV")
    parser.add_argument("--top", type=int, default=3, help="alternatives to list")
    options = (
        ("n", "nitrogen in soil"),
        ("p", "phosphorus in soil"),
        ("k", "potassium in soil"),
        ("temperature", "degrees Celsius"),
        ("humidity", "relative humidity, percent"),
        ("ph", "soil pH"),
        ("rainfall", "millimetres"),
    )
    for name, help_text in options:
        parser.add_argument(f"--{name}", type=float, help=help_text)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bundle = load_model(args.model_path)

    if args.csv:
        frame = pd.read_csv(args.csv)
        print(predict_frame(bundle, frame).to_string())
        return 0

    values = {
        "N": args.n,
        "P": args.p,
        "K": args.k,
        "temperature": args.temperature,
        "humidity": args.humidity,
        "ph": args.ph,
        "rainfall": args.rainfall,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise SystemExit(f"missing values for {missing}; pass every feature or use --csv")

    frame = pd.DataFrame([values])
    model = bundle["model"]
    print(f"recommended crop: {model.predict(frame)[0]}")

    if hasattr(model, "predict_proba") and args.top > 0:
        probabilities = pd.Series(
            model.predict_proba(frame)[0], index=model.classes_
        ).sort_values(ascending=False)
        print(f"top {args.top}:")
        for crop, probability in probabilities.head(args.top).items():
            print(f"  {crop:<14} {probability:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
