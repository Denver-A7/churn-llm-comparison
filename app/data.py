"""Data loading and metric helpers for the Streamlit demo.

The app reads only ``app/data/`` (exported by ``export_app_data.py``) and
``data/processed/``. It never touches ``data/raw`` and never calls an API.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
APP_DATA_DIR = PROJECT_ROOT / "app" / "data"

TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

PROFILE_COLUMNS = [
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "PaymentMethod",
    "PaperlessBilling",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PhoneService",
    "MultipleLines",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "gender",
]


# --------------------------------------------------------------------------- #
# Loading (app/data + data/processed only)
# --------------------------------------------------------------------------- #
def load_customer_table() -> pd.DataFrame:
    """194 compared customers with features, predictions, reasons, and derived flags."""
    df = pd.read_csv(APP_DATA_DIR / "comparison_with_features.csv")
    df["agree"] = df["agree"].astype(bool)
    df["mentions_no_internet"] = df["mentions_no_internet"].astype(bool)
    return df


def load_test_customers() -> pd.DataFrame:
    """All 1,409 held-out customers with features and baseline outputs."""
    return pd.read_csv(APP_DATA_DIR / "test_customers.csv")


def load_model_spec() -> dict:
    with open(APP_DATA_DIR / "baseline_model.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_metrics() -> dict:
    with open(APP_DATA_DIR / "metrics.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_segments() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "disagreement_by_segment.csv")


# --------------------------------------------------------------------------- #
# Derived columns (shared by the export script and the app)
# --------------------------------------------------------------------------- #
def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Add tenure bucket and monthly-charge quartile, computed within ``df``."""
    out = df.copy()
    out["tenure_bucket"] = pd.cut(
        out["tenure"], bins=[-1, 12, 24, 48, float("inf")], labels=TENURE_LABELS
    ).astype(str)
    out["cost_quartile"] = pd.qcut(
        out["MonthlyCharges"].rank(method="first"), q=4, labels=QUARTILE_LABELS
    ).astype(str)
    return out


def who_was_right(row: pd.Series) -> str:
    base_ok = row["baseline_predicted"] == row["actual_churn"]
    llm_ok = row["llm_predicted"] == row["actual_churn"]
    if base_ok and llm_ok:
        return "Both right"
    if base_ok:
        return "Baseline right"
    if llm_ok:
        return "Haiku right"
    return "Both wrong"


def flag_no_internet_reason(reasons: pd.Series) -> pd.Series:
    # Haiku phrases it as "no internet service" or "lack of internet service".
    return reasons.fillna("").str.contains(
        r"no internet|lack of internet|without internet", case=False, regex=True
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def confusion_counts(actual: pd.Series, predicted: pd.Series) -> dict[str, int]:
    tp = int(((actual == "Yes") & (predicted == "Yes")).sum())
    tn = int(((actual == "No") & (predicted == "No")).sum())
    fp = int(((actual == "No") & (predicted == "Yes")).sum())
    fn = int(((actual == "Yes") & (predicted == "No")).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    c = confusion_counts(actual, predicted)
    n = sum(c.values())
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (c["tp"] + c["tn"]) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **c,
    }


def cost_weighted_score(
    counts: dict[str, int], fn_cost: float = 5.0, fp_cost: float = 1.0
) -> float:
    """1 minus cost as a share of the worst case (every churner missed, every stayer flagged)."""
    positives = counts["tp"] + counts["fn"]
    negatives = counts["tn"] + counts["fp"]
    worst = positives * fn_cost + negatives * fp_cost
    if worst == 0:
        return 0.0
    cost = counts["fn"] * fn_cost + counts["fp"] * fp_cost
    return 1.0 - cost / worst


def sign_test_p_value(wins_a: int, wins_b: int) -> float:
    """Two-sided exact sign test on paired disagreements."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = max(wins_a, wins_b)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)
