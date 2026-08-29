"""Cross-validation, held-out scoring and feature importance."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from crop_recommendation.data import FEATURES, RANDOM_SEED

CV_FOLDS = 5


def cross_validate(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """Stratified k-fold accuracy on the training data only."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        model, features, target, cv=splitter, scoring="accuracy", n_jobs=-1
    )
    return {
        "cv_folds": folds,
        "cv_mean_accuracy": float(scores.mean()),
        "cv_std_accuracy": float(scores.std()),
        "cv_fold_accuracies": [float(score) for score in scores],
    }


def evaluate_holdout(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
) -> Dict[str, object]:
    """Accuracy, per-class report and confusion matrix on the held-out test set."""
    predictions = model.predict(features)
    labels: List[str] = sorted(target.unique())
    matrix = confusion_matrix(target, predictions, labels=labels)
    return {
        "test_accuracy": float(accuracy_score(target, predictions)),
        "labels": labels,
        "classification_report_text": classification_report(
            target, predictions, labels=labels, digits=3, zero_division=0
        ),
        "classification_report": classification_report(
            target, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": pd.DataFrame(matrix, index=labels, columns=labels),
    }


def feature_importance(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Impurity-based importance where the model exposes it, permutation otherwise."""
    classifier = model[-1] if isinstance(model, Pipeline) else model

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
        method = "impurity"
    else:
        result = permutation_importance(
            model, features, target, n_repeats=10, random_state=seed, n_jobs=-1
        )
        values = np.asarray(result.importances_mean, dtype=float)
        method = "permutation"

    frame = pd.DataFrame({"feature": FEATURES, "importance": values, "method": method})
    return frame.sort_values("importance", ascending=False, ignore_index=True)
