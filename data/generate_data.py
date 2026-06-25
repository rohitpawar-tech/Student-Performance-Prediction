"""
Data Generator for Student Performance Prediction System.

Generates a synthetic dataset of 1000 student records with realistic
distributions and correlations between academic features. Uses a
weighted scoring model with Gaussian noise to determine Pass/Fail status.

Run this script to regenerate student_data.csv:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_STUDENTS = 1000
SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "student_data.csv")

# Feature weights used to compute the latent "performance score"
# that drives the Pass/Fail label.
WEIGHTS = {
    "hours_studied":         0.15,
    "attendance_percentage": 0.20,
    "previous_marks":        0.25,
    "assignments_completed": 0.15,
    "class_tests_score":     0.20,
    "participation_score":   0.05,
}

# Decision threshold — values above this map to "Pass"
THRESHOLD = 0.48
NOISE_STD = 0.08  # Gaussian noise added to the composite score


def generate_student_data(n: int = NUM_STUDENTS, seed: int = SEED) -> pd.DataFrame:
    """
    Generate a DataFrame containing *n* synthetic student records.

    Each feature is drawn from a distribution that mimics real-world
    academic data, then a weighted composite score determines the label.

    Parameters
    ----------
    n : int
        Number of student records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        student_id, hours_studied, attendance_percentage, previous_marks,
        assignments_completed, class_tests_score, participation_score,
        final_status
    """
    rng = np.random.default_rng(seed)

    # --- Generate raw features with realistic distributions ---

    # Hours studied per day: right-skewed, most students study 1-6 hours
    hours_studied = np.clip(rng.exponential(scale=2.8, size=n), 0.5, 12.0)

    # Attendance percentage: roughly normal, centred around 72 %
    attendance_percentage = np.clip(
        rng.normal(loc=72, scale=15, size=n), 35.0, 100.0
    )

    # Previous exam marks: normal, centred around 62
    previous_marks = np.clip(
        rng.normal(loc=62, scale=18, size=n), 15.0, 100.0
    )

    # Assignments completed out of 10: slight right skew
    assignments_completed = rng.integers(low=0, high=11, size=n).astype(float)
    # Add a small bias so more students complete 5-9 assignments
    boost_idx = rng.choice(n, size=int(n * 0.4), replace=False)
    assignments_completed[boost_idx] = rng.integers(5, 10, size=len(boost_idx))

    # Class test scores: normal, centred around 55
    class_tests_score = np.clip(
        rng.normal(loc=55, scale=20, size=n), 5.0, 100.0
    )

    # Participation score out of 10: discrete, slight right skew
    participation_score = rng.integers(0, 11, size=n).astype(float)
    part_boost_idx = rng.choice(n, size=int(n * 0.35), replace=False)
    participation_score[part_boost_idx] = rng.integers(
        4, 9, size=len(part_boost_idx)
    )

    # --- Compute composite performance score ---
    composite = (
        (hours_studied / 12.0) * WEIGHTS["hours_studied"]
        + (attendance_percentage / 100.0) * WEIGHTS["attendance_percentage"]
        + (previous_marks / 100.0) * WEIGHTS["previous_marks"]
        + (assignments_completed / 10.0) * WEIGHTS["assignments_completed"]
        + (class_tests_score / 100.0) * WEIGHTS["class_tests_score"]
        + (participation_score / 10.0) * WEIGHTS["participation_score"]
    )

    # Add Gaussian noise to create some misalignment (realistic overlap)
    composite += rng.normal(loc=0.0, scale=NOISE_STD, size=n)

    # --- Determine Pass / Fail ---
    final_status = np.where(composite >= THRESHOLD, "Pass", "Fail")

    # --- Assemble DataFrame ---
    df = pd.DataFrame(
        {
            "student_id": np.arange(1001, 1001 + n),
            "hours_studied": np.round(hours_studied, 1),
            "attendance_percentage": np.round(attendance_percentage, 1),
            "previous_marks": np.round(previous_marks, 1),
            "assignments_completed": assignments_completed,
            "class_tests_score": np.round(class_tests_score, 1),
            "participation_score": participation_score,
            "final_status": final_status,
        }
    )

    return df


def main() -> None:
    """Generate the dataset and save it to CSV."""
    df = generate_student_data()

    # Print summary
    print(f"Dataset shape : {df.shape}")
    print(f"Pass count    : {(df['final_status'] == 'Pass').sum()}")
    print(f"Fail count    : {(df['final_status'] == 'Fail').sum()}")
    print(f"Pass ratio    : {(df['final_status'] == 'Pass').mean():.2%}")
    print(f"\nSaving to {OUTPUT_PATH} ...")

    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()