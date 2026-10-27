"""Export everything the Streamlit demo needs into ``app/data/``.

This is the only script that reads ``data/raw`` or uses scikit-learn. Run it once
after notebooks 01 and 02 have produced ``data/processed/``:

    python app/export_app_data.py

Outputs (all in app/data/):
  comparison_with_features.csv  194 compared customers + raw features + derived flags
  test_customers.csv            1,409 held-out customers + features + baseline outputs
  baseline_model.json           fitted preprocessing + logistic-regression coefficients
  metrics.json                  headline numbers used by the app
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as d  # noqa: E402

RAW_PATH = d.PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def fit_baseline(raw: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    """Refit notebook 01's pipeline: same features, split, preprocessing, classifier."""
    X = raw[d.FEATURE_COLUMNS]
    y = raw["Churn"]
    # Mirror the notebook: object columns first, then low-cardinality numerics (SeniorCitizen).
    categorical = [c for c in d.FEATURE_COLUMNS if X[c].dtype == object]
    categorical += [c for c in d.FEATURE_COLUMNS if c not in categorical and X[c].nunique(dropna=False) <= 2]
    numeric = [c for c in d.FEATURE_COLUMNS if c not in categorical]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return model, numeric, categorical


def model_to_spec(model: Pipeline, numeric: list[str], categorical: list[str]) -> dict:
    """Flatten the fitted pipeline into plain numbers the pure-Python predictor can use."""
    pre: ColumnTransformer = model.named_steps["preprocessor"]
    clf: LogisticRegression = model.named_steps["classifier"]
    classes = [str(c) for c in clf.classes_]
    assert classes == ["No", "Yes"], classes
    coef = clf.coef_[0].tolist()

    num_pipe = pre.named_transformers_["numeric"]
    medians = num_pipe.named_steps["imputer"].statistics_
    means = num_pipe.named_steps["scaler"].mean_
    scales = num_pipe.named_steps["scaler"].scale_

    cat_pipe = pre.named_transformers_["categorical"]
    most_frequent = cat_pipe.named_steps["imputer"].statistics_
    categories = cat_pipe.named_steps["onehot"].categories_

    spec = {
        "model": "LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)",
        "split": "train_test_split(test_size=0.2, stratify=Churn, random_state=42)",
        "positive_class": "Yes",
        "threshold": 0.5,
        "intercept": float(clf.intercept_[0]),
        "numeric": {},
        "categorical": {},
    }
    i = 0
    for j, name in enumerate(numeric):
        spec["numeric"][name] = {
            "median": float(medians[j]),
            "mean": float(means[j]),
            "scale": float(scales[j]),
            "coef": coef[i],
        }
        i += 1
    for j, name in enumerate(categorical):
        cats = [str(c) for c in categories[j]]
        spec["categorical"][name] = {
            "most_frequent": str(most_frequent[j]),
            "coefs": {cat: coef[i + k] for k, cat in enumerate(cats)},
        }
        i += len(cats)
    assert i == len(coef), (i, len(coef))
    return spec


def main() -> None:
    if not RAW_PATH.exists():
        sys.exit(f"Raw CSV not found at {RAW_PATH}. Place it there and rerun.")
    out_dir = d.APP_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    comparison = pd.read_csv(d.PROCESSED_DIR / "model_comparison.csv")
    comparison["agree"] = comparison["agree"].astype(bool)
    baseline = pd.read_csv(d.PROCESSED_DIR / "baseline_predictions.csv")
    features = raw[["customerID", *d.FEATURE_COLUMNS]]

    # 1) 194 compared customers with features and derived flags
    table = comparison.merge(baseline[["customerID", "baseline_probability"]], on="customerID", how="left")
    table["who_was_right"] = table.apply(d.who_was_right, axis=1)
    table["mentions_no_internet"] = d.flag_no_internet_reason(table["llm_reason"])
    table = table.merge(features, on="customerID", how="left")
    table = d.add_segments(table)
    assert len(table) == len(comparison) and table["tenure"].notna().all()
    table.to_csv(out_dir / "comparison_with_features.csv", index=False)

    # 2) Full held-out test set with features
    test = baseline.merge(features, on="customerID", how="left")
    assert len(test) == len(baseline) and test["tenure"].notna().all()
    test.to_csv(out_dir / "test_customers.csv", index=False)

    # 3) Model spec; sanity-check that the refit reproduces the saved probabilities
    model, numeric, categorical = fit_baseline(raw)
    refit = model.predict_proba(test[d.FEATURE_COLUMNS])[:, list(model.classes_).index("Yes")]
    max_diff = float(np.abs(refit - test["baseline_probability"].to_numpy()).max())
    assert max_diff < 1e-9, f"Refit differs from saved baseline probabilities (max {max_diff:.2e})"
    spec = model_to_spec(model, numeric, categorical)
    with open(out_dir / "baseline_model.json", "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)

    # 4) Headline metrics
    m_base = d.classification_metrics(table["actual_churn"], table["baseline_predicted"])
    m_llm = d.classification_metrics(table["actual_churn"], table["llm_predicted"])
    m_full = d.classification_metrics(test["actual_churn"], test["baseline_predicted"])
    dis = table[~table["agree"]]
    haiku_wins = int((dis["who_was_right"] == "Haiku right").sum())
    base_wins = int((dis["who_was_right"] == "Baseline right").sum())
    ni = table[table["mentions_no_internet"]]
    llm_raw = pd.read_csv(d.PROCESSED_DIR / "llm_predictions.csv")
    metrics = {
        "n_compared": int(len(table)),
        "n_sampled": int(len(llm_raw)),
        "n_churners_compared": int((table["actual_churn"] == "Yes").sum()),
        "n_agree": int(table["agree"].sum()),
        "n_disagree": int(len(dis)),
        "haiku_right_on_disagreements": haiku_wins,
        "baseline_right_on_disagreements": base_wins,
        "sign_test_p_value": d.sign_test_p_value(haiku_wins, base_wins),
        "cost_weights": {"false_negative": 5, "false_positive": 1},
        "baseline_compared": {**m_base, "cost_weighted_score": d.cost_weighted_score(m_base)},
        "haiku_compared": {**m_llm, "cost_weighted_score": d.cost_weighted_score(m_llm)},
        "baseline_full_test": {"n": int(len(test)), **m_full},
        "no_internet_reasons": {
            "count": int(len(ni)),
            "stayed": int((ni["actual_churn"] == "No").sum()),
            "dataset_churn_rate_no_internet": float(
                raw.loc[raw["InternetService"] == "No", "Churn"].eq("Yes").mean()
            ),
        },
        "dataset": {"rows": int(len(raw)), "churn_rate": float(raw["Churn"].eq("Yes").mean())},
        "refit_max_abs_prob_diff": max_diff,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Wrote 4 files to {out_dir}")
    print(f"  comparison_with_features.csv: {len(table)} rows")
    print(f"  test_customers.csv:           {len(test)} rows")
    print(f"  baseline_model.json:          {len(spec['numeric'])} numeric + {len(spec['categorical'])} categorical columns")
    print(f"  metrics.json (refit max |Δp| = {max_diff:.2e})")


if __name__ == "__main__":
    main()
