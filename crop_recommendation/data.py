"""Dataset loading, validation and splitting."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "crop_model.joblib"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

# Used for every split, every shuffle and every model that takes a seed.
RANDOM_SEED = 42


class DatasetError(RuntimeError):
    """Raised when the CSV is missing or does not have the expected shape."""


def load_dataset(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the crop CSV and check it looks like the dataset the models expect."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"{path} not found. See data/README.md for where to get the dataset, "
            "or run: python scripts/download_data.py"
        )

    frame = pd.read_csv(path)

    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise DatasetError(
            f"{path} is missing the columns {missing}. Expected {FEATURES + [TARGET]}."
        )

    frame = frame[FEATURES + [TARGET]]

    non_numeric = [
        column
        for column in FEATURES
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise DatasetError(f"{path} has non-numeric feature columns: {non_numeric}")

    if frame.isna().any().any():
        counts = frame.isna().sum()
        raise DatasetError(f"{path} contains missing values:\n{counts[counts > 0]}")

    if frame[TARGET].nunique() < 2:
        raise DatasetError(f"{path} needs at least two distinct crop labels")

    return frame


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the crop label."""
    features = frame[FEATURES]
    target = frame[TARGET]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )
