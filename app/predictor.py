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


def linear_score(spec: dict, customer: dict) -> float:
    z = spec["intercept"]
    for name, col in spec["numeric"].items():
        value = customer.get(name)
        x = col["median"] if _is_missing(value) else float(value)
        z += col["coef"] * (x - col["mean"]) / col["scale"]
    for name, col in spec["categorical"].items():
        value = customer.get(name)
        key = col["most_frequent"] if _is_missing(value) else str(value)
        # Unknown categories contribute nothing (handle_unknown="ignore").
        z += col["coefs"].get(key, 0.0)
    return z


def predict_proba(spec: dict, customer: dict) -> float:
    """Probability of the positive class (``Churn == "Yes"``)."""
    z = linear_score(spec, customer)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)
