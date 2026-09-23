"""Aggregate per-group, one-vs-rest audit metrics using explicit definitions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(metrics):
    rows = []
    for (attribute, outcome), frame in metrics.groupby(["Attribute", "Outcome"], sort=False):
        def gap(column):
            v = frame[column]
            # A missing group rate must not silently become evidence of parity.
            return float(v.max() - v.min()) if len(v) >= 2 and v.notna().all() else np.nan
        selection = frame["Selection rate"]
        ratio = (selection.min() / selection.max()
                 if len(selection) >= 2 and selection.notna().all() and selection.max() > 0
                 else np.nan)
        tpr, fpr = gap("TPR"), gap("FPR")
        rows.append({
            "Attribute": attribute, "Outcome": outcome,
            "Groups observed": len(frame), "Minimum group count": frame["Count"].min(),
            "Demographic parity difference": gap("Selection rate"),
            "Selection min/max ratio": ratio, "TPR difference": tpr, "FPR difference": fpr,
            "Equalised odds difference": max(tpr, fpr) if np.isfinite([tpr, fpr]).all() else np.nan,
            "Any small group": frame["Small group"].any(),
            "Any sparse outcome support": frame["Sparse outcome support"].any(),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summarize(pd.read_csv(args.input)).to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
