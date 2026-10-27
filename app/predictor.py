"""Pure-Python scorer for the exported logistic-regression baseline.

The spec (``app/data/baseline_model.json``) captures the fitted preprocessing
(median imputation + standardization for numeric columns, most-frequent
imputation + one-hot encoding for categorical columns) and the classifier's
coefficients, so no scikit-learn is needed at prediction time.
"""

from __future__ import annotations

import math
from typing import Any


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def contributions(spec: dict, customer: dict) -> list[dict]:
    """Per-feature additive terms of the linear score for one customer.

    Each entry has ``feature``, ``label`` (feature = value as used by the model),
    and ``contribution`` (positive pushes toward churn). The intercept is excluded.
    """
    rows: list[dict] = []
    for name, col in spec["numeric"].items():
        value = customer.get(name)
        x = col["median"] if _is_missing(value) else float(value)
        rows.append(
            {
                "feature": name,
                "label": f"{name} = {x:g}",
                "contribution": col["coef"] * (x - col["mean"]) / col["scale"],
            }
        )
    for name, col in spec["categorical"].items():
        value = customer.get(name)
        key = col["most_frequent"] if _is_missing(value) else str(value)
        # Unknown categories contribute nothing (handle_unknown="ignore").
        rows.append(
            {"feature": name, "label": f"{name} = {key}", "contribution": col["coefs"].get(key, 0.0)}
        )
    return rows


def linear_score(spec: dict, customer: dict) -> float:
    z = spec["intercept"]
    for row in contributions(spec, customer):
        z += row["contribution"]
    return z


def predict_proba(spec: dict, customer: dict) -> float:
    """Probability of the positive class (``Churn == "Yes"``)."""
    z = linear_score(spec, customer)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)
