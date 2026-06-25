"""
train.py
========
Trains multiple classification models on the preprocessed student
performance data and returns the best-performing model based on
F1-score.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Model registry — each entry is a (name, constructor, kwargs) tuple
# ---------------------------------------------------------------------------
MODEL_REGISTRY = [
    (
        "Logistic Regression",
        LogisticRegression,
        {
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs",
        },
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier,
        {
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
        },
    ),
    (
        "Random Forest",
        RandomForestClassifier,
        {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 4,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
]


def train_models(X_train, y_train) -> dict:
    """
    Instantiate, fit, and return all registered models.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (already scaled).
    y_train : np.ndarray
        Training target vector.

    Returns
    -------
    dict
        Mapping of model name -> fitted sklearn estimator.
    """
    trained = {}

    for name, cls, params in MODEL_REGISTRY:
        print(f"\n[TRAIN] Training {name} ...")
        model = cls(**params)
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  -> {name} trained successfully.")

    return trained


def get_model_names() -> list:
    """Return a list of registered model names."""
    return [entry[0] for entry in MODEL_REGISTRY]