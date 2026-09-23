"""Deterministic, target-free feature transformations used by both entry points."""
import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ["Previous qualification (grade)", "Admission grade", "Age at enrollment", *[f"Curricular units {s} sem ({m})" for s in ["1st", "2nd"] for m in ["credited", "enrolled", "evaluations", "approved", "grade", "without evaluations"]], "Unemployment rate", "Inflation rate", "GDP"]
DERIVED_CATEGORICAL = ["Age group", "Zero approved 1st sem", "Zero approved 2nd sem", "Zero enrolled 1st sem", "Zero enrolled 2nd sem"]

def engineer_features(frame):
    """Deterministic transformations; no statistics learned from the data."""
    if "Target" in frame.columns:
        raise ValueError("Target must not enter feature engineering.")
    result = frame.copy()
    all_cat_cols = [c for c in frame.columns if c not in NUMERIC_COLUMNS] + DERIVED_CATEGORICAL

    for semester in ["1st", "2nd"]:
        enrolled = result[f"Curricular units {semester} sem (enrolled)"]
        approved = result[f"Curricular units {semester} sem (approved)"]
        no_evaluation = result[
            f"Curricular units {semester} sem (without evaluations)"
        ]

        denominator = enrolled.where(enrolled > 0)

        result[f"Approval ratio {semester} sem"] = (
            approved / denominator
        )
        result[f"Without evaluation ratio {semester} sem"] = (
            no_evaluation / denominator
        )

        result[f"Zero approved {semester} sem"] = (
            approved.eq(0).astype(float).where(approved.notna())
        )
        result[f"Zero enrolled {semester} sem"] = (
            enrolled.eq(0).astype(float).where(enrolled.notna())
        )

    result["Grade change"] = (
        result["Curricular units 2nd sem (grade)"]
        - result["Curricular units 1st sem (grade)"]
    )

    result["Approved units change"] = (
        result["Curricular units 2nd sem (approved)"]
        - result["Curricular units 1st sem (approved)"]
    )

    result["Age group"] = pd.cut(
        result["Age at enrollment"],
        bins=[-np.inf, 20, 25, 35, np.inf],
        labels=["Under 20", "20–24", "25–34", "35+"],
        right=False,
    )

    # Normalize categorical values to strings while preserving missingness.
    for col in all_cat_cols:
        values = result[col].astype("string")
        result[col] = values.astype(object).where(
            values.notna(), np.nan
        )

    return result

