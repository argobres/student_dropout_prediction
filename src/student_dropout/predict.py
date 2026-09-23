"""Predict recorded outcomes from a trusted, locally generated model bundle."""
import argparse
from pathlib import Path
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    data = pd.read_csv(args.input, sep=';').drop(columns='Target', errors='ignore')
    data.columns = data.columns.str.strip()
    missing = set(bundle['columns']) - set(data.columns)
    if missing:
        raise ValueError(f'Missing predictors: {sorted(missing)}')
    labels = bundle['model'].predict(data[bundle['columns']])
    result = pd.DataFrame({'Predicted': bundle['target_encoder'].inverse_transform(labels)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
