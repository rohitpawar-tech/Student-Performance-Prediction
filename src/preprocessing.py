"""
preprocessing.py
================
Handles data loading, cleaning, feature selection, encoding,
train/test splitting, and standardisation.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "student_data.csv")

FEATURE_COLS = [
    "hours_studied",
    "attendance_percentage",
    "previous_marks",
    "assignments_completed",
    "class_tests_score",
    "participation_score",
]

TARGET_COL = "final_status"


class DataPreprocessor:
    """
    End-to-end preprocessor that loads the raw CSV, cleans it,
    encodes the target, splits into train/test sets, and scales
    the feature matrix.
    """

    def __init__(
        self,
        data_path: str = DATA_PATH,
        feature_cols: list = None,
        target_col: str = TARGET_COL,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.feature_cols = feature_cols or FEATURE_COLS
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def run(self) -> tuple:
        """
        Execute the full preprocessing pipeline.

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test) — all as NumPy arrays.
        """
        self._load_data()
        self._inspect()
        self._handle_missing()
        self._remove_duplicates()
        self._encode_target()
        self._split_and_scale()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_feature_names(self) -> list:
        """Return the list of feature column names."""
        return self.feature_cols

    def _load_data(self) -> None:
        """Read CSV into a DataFrame."""
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Run `python data/generate_data.py` first."
            )
        self.df = pd.read_csv(self.data_path)
        print(f"[INFO] Loaded dataset: {self.df.shape[0]} rows, "
              f"{self.df.shape[1]} columns.")

    def _inspect(self) -> None:
        """Print basic dataset info and a statistical summary."""
        print("\n--- Dataset Info ---")
        print(self.df.info())
        print("\n--- Statistical Summary ---")
        print(self.df.describe().round(2))
        print("\n--- Class Distribution ---")
        print(self.df[self.target_col].value_counts())

    def _handle_missing(self) -> None:
        """Drop rows with any missing values."""
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            print(f"\n[WARN] Found {missing} missing values. Dropping rows.")
            self.df.dropna(inplace=True)
        else:
            print("\n[INFO] No missing values found.")

    def _remove_duplicates(self) -> None:
        """Remove exact duplicate rows based on student_id."""
        before = len(self.df)
        self.df.drop_duplicates(subset="student_id", inplace=True)
        removed = before - len(self.df)
        if removed > 0:
            print(f"[INFO] Removed {removed} duplicate rows.")
        else:
            print("[INFO] No duplicate student IDs found.")

    def _encode_target(self) -> None:
        """Encode target: Pass -> 1, Fail -> 0"""
        self.df["target_encoded"] = self.label_encoder.fit_transform(
            self.df[self.target_col]
        )
        # Split into two lines to prevent terminal wrapping issues
        classes = self.label_encoder.classes_
        transformed = self.label_encoder.transform(classes)
        mapping = dict(zip(classes, transformed))
        print(f"\n[INFO] Target encoding mapping: {mapping}")

    def _split_and_scale(self) -> None:
        """Split into train/test, then apply StandardScaler to features."""
        X = self.df[self.feature_cols].values
        y = self.df["target_encoded"].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"\n[INFO] Train set size: {self.X_train.shape[0]}")
        print(f"[INFO] Test set size : {self.X_test.shape[0]}")


def load_and_preprocess(**kwargs) -> tuple:
    """
    One-call helper to run the full preprocessing pipeline.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, preprocessor)
    """
    preprocessor = DataPreprocessor(**kwargs)
    X_train, X_test, y_train, y_test = preprocessor.run()
    return X_train, X_test, y_train, y_test, preprocessor