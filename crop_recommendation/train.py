"""Train the candidate models, keep the best one and write an evaluation report."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib

from crop_recommendation.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_DIR,
    FEATURES,
    RANDOM_SEED,
    TARGET,
    load_dataset,
    split_dataset,
)
from crop_recommendation.evaluate import (
    CV_FOLDS,
    cross_validate,
    evaluate_holdout,
    feature_importance,
)
from crop_recommendation.models import build_models


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip the confusion matrix PNG (the CSV is still written)",
    )
    return parser.parse_args(argv)


def save_confusion_plot(matrix, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = max(6.0, 0.45 * len(matrix.columns))
    figure, axes = plt.subplots(figsize=(size, size))
    axes.imshow(matrix.values, cmap="Blues")
    axes.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=7)
    axes.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    axes.set_xlabel("predicted")
    axes.set_ylabel("actual")
    axes.set_title("Confusion matrix (held-out test set)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.values[row, column]
            if value:
                axes.text(column, row, int(value), ha="center", va="center", fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None) -> int:
    args = parse_args(argv)

    frame = load_dataset(args.data)
    print(f"loaded {len(frame)} rows, {frame[TARGET].nunique()} crops from {args.data}")

    x_train, x_test, y_train, y_test = split_dataset(
        frame, test_size=args.test_size, seed=args.seed
    )
    print(f"train {len(x_train)} rows / test {len(x_test)} rows (stratified)")

    cv_results: Dict[str, Dict] = {}
    for name, model in build_models(seed=args.seed).items():
        scores = cross_validate(
            model, x_train, y_train, folds=args.folds, seed=args.seed
        )
        cv_results[name] = scores
        mean = scores["cv_mean_accuracy"]
        deviation = scores["cv_std_accuracy"]
        print(f"{name:<22} cv accuracy {mean:.4f} +/- {deviation:.4f}")

    best_name = max(cv_results, key=lambda name: cv_results[name]["cv_mean_accuracy"])
    best_model = build_models(seed=args.seed)[best_name]
    best_model.fit(x_train, y_train)
    print(f"selected {best_name}, refitted on the full training split")

    holdout = evaluate_holdout(best_model, x_test, y_test)
    importance = feature_importance(best_model, x_test, y_test, seed=args.seed)

    accuracy = holdout["test_accuracy"]
    print(f"held-out test accuracy {accuracy:.4f}")
    print(holdout["classification_report_text"])
    print(importance.to_string(index=False))

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "features": FEATURES,
            "labels": holdout["labels"],
            "seed": args.seed,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_accuracy": accuracy,
        },
        args.model_path,
    )
    print(f"model written to {args.model_path}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    holdout["confusion_matrix"].to_csv(args.report_dir / "confusion_matrix.csv")
    importance.to_csv(args.report_dir / "feature_importance.csv", index=False)
    (args.report_dir / "classification_report.txt").write_text(
        holdout["classification_report_text"], encoding="utf-8"
    )
    report = holdout["classification_report"]
    metrics = {
        "data_path": str(args.data),
        "rows": int(len(frame)),
        "n_classes": int(frame[TARGET].nunique()),
        "test_size": args.test_size,
        "seed": args.seed,
        "cross_validation": cv_results,
        "selected_model": best_name,
        "test_accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "feature_importance": importance.to_dict(orient="records"),
    }
    (args.report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    if not args.no_plot:
        save_confusion_plot(
            holdout["confusion_matrix"], args.report_dir / "confusion_matrix.png"
        )
    print(f"reports written to {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
