"""The candidate classifiers.

These are scikit-learn ports of the models in notebooks/exploration.ipynb, which used
Spark MLlib. Every estimator that needs feature scaling gets it inside a Pipeline, so the
scaler is fitted on training folds only and never sees the held-out data.
"""

from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from crop_recommendation.data import RANDOM_SEED


def build_models(seed: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """Return the candidate models, keyed by name."""
    return {
        "decision_tree": Pipeline(
            [("classifier", DecisionTreeClassifier(random_state=seed))]
        ),
        "random_forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "gaussian_naive_bayes": Pipeline(
            [("scaler", StandardScaler()), ("classifier", GaussianNB())]
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=seed),
                ),
            ]
        ),
    }
