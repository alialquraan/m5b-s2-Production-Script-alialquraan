# m5b-s2-Production-Script-alialquraan


# Model Comparison CLI

A production-ready command-line tool for comparing multiple machine learning models using 5-fold stratified cross-validation, PR curves, and calibration analysis on a telecom churn dataset.

The tool evaluates 6 model configurations (Dummy, Logistic Regression, Decision Tree, and Random Forest variants), including balanced versions to handle class imbalance.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python compare_models_production_script.py --data-path data/telecom_churn.csv
```

---

## Arguments

* **--data-path (required)**: Path to input CSV dataset
* **--output-dir (optional)**: Directory to save results (default: `./output`)
* **--n-folds (optional)**: Number of cross-validation folds (default: 5)
* **--random-seed (optional)**: Random seed for reproducibility (default: 42)
* **--dry-run (flag)**: Validate data and print configuration without training

---

## Examples

### 🔹 Normal run

```bash
python compare_models_production_script.py --data-path data/telecom_churn.csv
```

### 🔹 Dry run (no training)

```bash
python compare_models_production_script.py --data-path data/telecom_churn.csv --dry-run
```

---

## Output

All results are saved in the output directory:

* `comparison_table.csv` → model performance metrics
* `pr_curves.png` → Precision-Recall curves (top 3 models)
* `calibration.png` → calibration plots
* `best_model.joblib` → best model by PR-AUC
* `experiment_log.csv` → experiment tracking with timestamp

---

## Notes

* The script automatically creates the output directory if it does not exist.
* Logging is used instead of print statements for better traceability.
* The `--dry-run` mode helps validate configuration before running full training.
* The dataset is imbalanced, and class_weight='balanced' is used in selected models to address this.
