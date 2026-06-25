"""
main.py
=======
Entry point for the Student Performance Prediction System.

Usage
-----
1.  Train & evaluate (produces models/student_model.pkl):
        python main.py --train

2.  Predict for a single student:
        python main.py --predict

3.  Full pipeline (train + predict demo):
        python main.py --all
"""

import argparse
import os
import pickle
import sys

import numpy as np

# Ensure the project root is on sys.path so `src` is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import DataPreprocessor
from src.train import train_models
from src.evaluate import evaluate_all, select_best_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "student_model.pkl")


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def run_training() -> None:
    """Load data, preprocess, train models, evaluate, and save the best."""
    print("=" * 60)
    print("  STUDENT PERFORMANCE PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run()

    # 2. Train
    models = train_models(X_train, y_train)

    # 3. Evaluate
    summary_df = evaluate_all(models, X_test, y_test)

    # 4. Select best
    best_name, best_model, best_f1 = select_best_model(models, summary_df)

    # 5. Save best model + preprocessor (scaler + encoder)
    artefact = {
        "model": best_model,
        "scaler": preprocessor.scaler,
        "label_encoder": preprocessor.label_encoder,
        "feature_cols": preprocessor.get_feature_names(),
        "model_name": best_name,
        "f1_score": best_f1,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artefact, f)

    print(f"[SAVED] Best model artefact -> {MODEL_PATH}")
    print(f"  Model : {best_name}")
    print(f"  F1    : {best_f1}")


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def load_artefact(path: str = MODEL_PATH) -> dict:
    """Load the saved model artefact from disk."""
    if not os.path.isfile(path):
        print(f"[ERROR] Model file not found: {path}")
        print("        Run `python main.py --train` first.")
        sys.exit(1)

    with open(path, "rb") as f:
        artefact = pickle.load(f)
    return artefact


def predict_student(
    hours_studied: float,
    attendance_percentage: float,
    previous_marks: float,
    assignments_completed: int,
    class_tests_score: float,
    participation_score: int,
    model_path: str = MODEL_PATH,
) -> str:
    """
    Predict Pass/Fail for a single student.

    Parameters
    ----------
    hours_studied : float
        Hours studied per day.
    attendance_percentage : float
        Attendance percentage (0-100).
    previous_marks : float
        Previous exam marks (0-100).
    assignments_completed : int
        Number of assignments completed (0-10).
    class_tests_score : float
        Class test score (0-100).
    participation_score : int
        Participation score (0-10).
    model_path : str
        Path to the pickled model artefact.

    Returns
    -------
    str
        "Pass" or "Fail"
    """
    artefact = load_artefact(model_path)

    # Build feature vector in the correct column order
    features = np.array([[
        hours_studied,
        attendance_percentage,
        previous_marks,
        assignments_completed,
        class_tests_score,
        participation_score,
    ]])

    # Scale
    features_scaled = artefact["scaler"].transform(features)

    # Predict
    pred_encoded = artefact["model"].predict(features_scaled)
    label = artefact["label_encoder"].inverse_transform(pred_encoded)[0]

    return label


# ---------------------------------------------------------------------------
# Interactive prediction
# ---------------------------------------------------------------------------
def run_prediction() -> None:
    """Accept student data from the user and display the prediction."""
    artefact = load_artefact()
    print(
        f"\n[INFO] Loaded model: {artefact['model_name']} "
        f"(F1 = {artefact['f1_score']})"
    )

    print("\n" + "=" * 60)
    print("  STUDENT PERFORMANCE PREDICTION")
    print("=" * 60)
    print("Enter the student's details:\n")

    try:
        hours = float(input("  Hours studied per day       : "))
        attendance = float(input("  Attendance percentage (%)   : "))
        prev_marks = float(input("  Previous marks (0-100)      : "))
        assignments = int(input("  Assignments completed (0-10): "))
        test_score = float(input("  Class tests score (0-100)   : "))
        participation = int(input("  Participation score (0-10)  : "))
    except ValueError:
        print("\n[ERROR] Invalid input. Please enter numeric values.")
        sys.exit(1)

    result = predict_student(
        hours_studied=hours,
        attendance_percentage=attendance,
        previous_marks=prev_marks,
        assignments_completed=assignments,
        class_tests_score=test_score,
        participation_score=participation,
    )

    print("\n" + "-" * 60)
    print(f"  PREDICTION RESULT:  >>>  {result.upper()}  <<<")
    print("-" * 60)

    # Also show demo predictions
    print("\n--- Demo Predictions ---")
    demo_students = [
        {
            "name": "High Performer",
            "data": {
                "hours_studied": 7.5,
                "attendance_percentage": 92.0,
                "previous_marks": 88.0,
                "assignments_completed": 10,
                "class_tests_score": 85.0,
                "participation_score": 9,
            },
        },
        {
            "name": "Average Student",
            "data": {
                "hours_studied": 3.5,
                "attendance_percentage": 70.0,
                "previous_marks": 58.0,
                "assignments_completed": 5,
                "class_tests_score": 52.0,
                "participation_score": 5,
            },
        },
        {
            "name": "At-Risk Student",
            "data": {
                "hours_studied": 1.0,
                "attendance_percentage": 45.0,
                "previous_marks": 28.0,
                "assignments_completed": 1,
                "class_tests_score": 18.0,
                "participation_score": 1,
            },
        },
    ]

    for student in demo_students:
        pred = predict_student(**student["data"])
        print(f"  {student['name']:20s} -> {pred}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Student Performance Prediction System"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train",
        action="store_true",
        help="Run the training pipeline and save the best model.",
    )
    group.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and predict for a student.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Train first, then run prediction demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train:
        run_training()

    elif args.predict:
        run_prediction()

    elif args.all:
        run_training()
        print("\n\n")
        run_prediction()


if __name__ == "__main__":
    main()