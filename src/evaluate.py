"""
evaluate.py
===========
Evaluates trained models using multiple classification metrics,
generates a comparison table, and selects the best model.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Seaborn style
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def evaluate_all(
    models: dict,
    X_test,
    y_test,
    save_confusion_matrix: bool = True,
) -> pd.DataFrame:
    """
    Evaluate every model in *models* and return a summary DataFrame
    sorted by F1-score (descending).

    Parameters
    ----------
    models : dict
        {model_name: fitted_estimator}
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test target vector.
    save_confusion_matrix : bool
        If True, saves a confusion-matrix heatmap for each model.

    Returns
    -------
    pd.DataFrame
        Summary table with Accuracy, Precision, Recall, F1 per model.
    """
    records = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
        })

        # Detailed classification report
        print(f"\n{'='*50}")
        print(f"  Classification Report — {name}")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix ({name}):\n{cm}\n")

        if save_confusion_matrix:
            _save_cm_plot(cm, name)

    summary_df = pd.DataFrame(records).sort_values(
        "F1 Score", ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60 + "\n")

    return summary_df


def select_best_model(
    models: dict,
    summary_df: pd.DataFrame,
) -> tuple:
    """
    Pick the model with the highest F1-score.

    Parameters
    ----------
    models : dict
        {model_name: fitted_estimator}
    summary_df : pd.DataFrame
        Output of evaluate_all().

    Returns
    -------
    tuple
        (best_model_name, best_model_object, best_f1_score)
    """
    best_row = summary_df.iloc[0]
    best_name = best_row["Model"]
    best_f1 = best_row["F1 Score"]
    best_model = models[best_name]

    print(f"\n[RESULT] Best model: {best_name} (F1 = {best_f1})\n")
    return best_name, best_model, best_f1


def _save_cm_plot(cm, model_name: str) -> None:
    """Save a confusion-matrix heatmap as a PNG file."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")

    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(REPORT_DIR, f"cm_{safe_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")