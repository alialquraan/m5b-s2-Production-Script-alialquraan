import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (PrecisionRecallDisplay, average_precision_score,
                             precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]


# -----------------------------
# Load & Validate
# -----------------------------
def load_data(path):
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    logging.info(f"Loaded data {df.shape}")

    return df


def validate_data(df):
    required = set(NUMERIC_FEATURES + ["churned"])

    if not required.issubset(df.columns):
        logging.error("Missing required columns")
        sys.exit(1)

    logging.info(f"Class distribution:\n{df['churned'].value_counts()}")


# -----------------------------
# Split
# -----------------------------
def split_data(df, seed):
    X = df[NUMERIC_FEATURES]
    y = df["churned"]

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


# -----------------------------
# Models
# -----------------------------
def define_models(seed):
    return {
        'Dummy': Pipeline([
            ('scaler', 'passthrough'),
            ('model', DummyClassifier(strategy='most_frequent'))
        ]),
        'LR_default': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=seed))
        ]),
        'LR_balanced': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed))
        ]),
        'DT_depth5': Pipeline([
            ('scaler', 'passthrough'),
            ('model', DecisionTreeClassifier(max_depth=5, random_state=seed))
        ]),
        'RF_default': Pipeline([
            ('scaler', 'passthrough'),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed))
        ]),
        'RF_balanced': Pipeline([
            ('scaler', 'passthrough'),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=seed))
        ])
    }


# -----------------------------
# CV Evaluation
# -----------------------------
def train_and_evaluate(models, X, y, folds, seed):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = []

    for name, model in models.items():
        logging.info(f"Evaluating {name}")

        metrics = {'acc': [], 'pre': [], 'rec': [], 'f1': [], 'auc': []}

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            metrics['acc'].append(accuracy_score(y_val, y_pred))
            metrics['pre'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics['rec'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            metrics['auc'].append(average_precision_score(y_val, y_proba))

        results.append({
            'model': name,
            'accuracy_mean': np.mean(metrics['acc']),
            'precision_mean': np.mean(metrics['pre']),
            'recall_mean': np.mean(metrics['rec']),
            'f1_mean': np.mean(metrics['f1']),
            'pr_auc_mean': np.mean(metrics['auc'])
        })

    return pd.DataFrame(results)


# -----------------------------
# Save Outputs
# -----------------------------
def save_results(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "comparison_table.csv")
    df.to_csv(path, index=False)
    logging.info(f"Saved table to {path}")


def plot_pr_curves(models, X_test, y_test, output_dir):
    path = os.path.join(output_dir, "pr_curves.png")

    scores = {}
    for name, model in models.items():
        scores[name] = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])

    top3 = sorted(scores, key=scores.get, reverse=True)[:3]

    fig, ax = plt.subplots()
    for name in top3:
        PrecisionRecallDisplay.from_estimator(models[name], X_test, y_test, ax=ax, name=name)

    plt.savefig(path)
    plt.close()
    logging.info(f"Saved PR curves to {path}")


def plot_calibration(models, X_test, y_test, output_dir):
    path = os.path.join(output_dir, "calibration.png")

    scores = {}
    for name, model in models.items():
        scores[name] = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])

    top3 = sorted(scores, key=scores.get, reverse=True)[:3]

    fig, ax = plt.subplots()
    for name in top3:
        CalibrationDisplay.from_estimator(models[name], X_test, y_test, ax=ax, name=name)

    plt.savefig(path)
    plt.close()
    logging.info(f"Saved calibration plot to {path}")


def save_best_model(models, results_df, output_dir):
    best = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    path = os.path.join(output_dir, "best_model.joblib")

    dump(models[best], path)
    logging.info(f"Saved best model ({best})")


def log_experiment(results_df, output_dir):
    log_df = results_df.copy()
    log_df["timestamp"] = datetime.now().isoformat()

    path = os.path.join(output_dir, "experiment_log.csv")
    log_df.to_csv(path, index=False)
    logging.info("Saved experiment log")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Production ML Pipeline CLI")

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    logging.info("Starting pipeline")

    df = load_data(args.data_path)
    validate_data(df)

    if args.dry_run:
        logging.info("DRY RUN")
        logging.info(f"Folds: {args.n_folds}")
        logging.info(f"Output: {args.output_dir}")
        logging.info(f"Models: {list(define_models(args.random_seed).keys())}")
        return

    X_train, X_test, y_train, y_test = split_data(df, args.random_seed)

    models = define_models(args.random_seed)

    results_df = train_and_evaluate(models, X_train, y_train, args.n_folds, args.random_seed)

    save_results(results_df, args.output_dir)

    # Fit models
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

    plot_pr_curves(fitted, X_test, y_test, args.output_dir)
    plot_calibration(fitted, X_test, y_test, args.output_dir)

    save_best_model(fitted, results_df, args.output_dir)
    log_experiment(results_df, args.output_dir)

    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()