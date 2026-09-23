from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
from student_dropout.features import engineer_features

ROOT = Path(__file__).resolve().parents[1]

def test_source_integrity():
    p = ROOT / "data/raw/data.csv"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == "3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae"
    df = pd.read_csv(p, sep=";")
    assert df.shape == (4424, 37)
    assert set(df.Target) == {"Dropout", "Enrolled", "Graduate"}

def test_undefined_ratios_and_no_mutation():
    df = pd.read_csv(ROOT / "data/raw/data.csv", sep=";").drop(columns="Target")
    original = df.copy(deep=True)
    out = engineer_features(df)
    pd.testing.assert_frame_equal(df, original)
    for s in ["1st", "2nd"]:
        zero = df[f"Curricular units {s} sem (enrolled)"].eq(0)
        assert out[f"Approval ratio {s} sem"].isna().equals(zero)
        assert not np.isinf(out[f"Approval ratio {s} sem"]).any()
    # Preserve the full-dataset anomalies (one falls in the training partition) for investigation instead of silently clipping.
    assert (out["Without evaluation ratio 2nd sem"] > 1).sum() == 3

def test_target_rejected():
    with pytest.raises(ValueError, match="Target"):
        engineer_features(pd.DataFrame({"Target": ["Dropout"]}))
