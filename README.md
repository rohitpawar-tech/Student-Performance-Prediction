cat << 'EOF' > ~/Desktop/Student-Performance-Prediction/README.md
# Student Performance Prediction System

A complete Machine Learning project that predicts whether a student will **PASS** or **FAIL** based on academic performance and attendance records.

---

## Project Overview

This project implements a supervised binary classification pipeline using Python and scikit-learn. It covers the full ML workflow: data generation, exploratory data analysis, preprocessing, model training, evaluation, and deployment-ready prediction script.

### Key Features
- **Synthetic dataset** of 1,000 realistic student records
- **Three models** compared: Logistic Regression, Decision Tree, Random Forest
- **Automatic best-model selection** based on F1-score
- **Modular codebase** with clean separation of concerns
- **Interactive prediction** from the command line
- **Comprehensive EDA** with publication-quality visualisations

---

## Project Structure
cat << 'EOF' > ~/Desktop/Student-Performance-Prediction/README.md



---

## Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| student_id | int | Unique identifier (1001-2000) |
| hours_studied | float | Hours studied per day (0.5-12.0) |
| attendance_percentage | float | Class attendance % (35-100) |
| previous_marks | float | Previous exam marks (15-100) |
| assignments_completed | int | Assignments completed out of 10 (0-10) |
| class_tests_score | float | Class test average score (5-100) |
| participation_score | int | Participation rating (0-10) |
| final_status | str | Target: "Pass" or "Fail" |

**Generation method:** Features are drawn from realistic distributions (exponential, normal, discrete). A weighted composite score with Gaussian noise determines the Pass/Fail label, producing natural class overlap.

---

## Setup Instructions

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Navigate to the project
cd ~/Desktop/Student-Performance-Prediction

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset (already done, but can regenerate)
python data/generate_data.py