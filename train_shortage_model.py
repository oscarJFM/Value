"""Train a model that predicts upcoming medicine shortages for each hospital.

The script ingests the historical weekly inventory CSVs (generated via
`generate_weekly_history.py`) and learns to classify whether a given
hospital/medicine pair will hit a shortage threshold within the next N weeks.
It outputs evaluation metrics and saves the trained model for later inference.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score


@dataclass
class SampleRecord:
    hospital: str
    medicine_id: str
    medicine_name: str
    week_start: pd.Timestamp
    features: Dict[str, float]
    target: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shortage prediction model")
    parser.add_argument(
        "--base-dir",
        default="medicine_inventory_dummy_data_v2",
        help="Root directory that contains hospital inventories and historical data.",
    )
    parser.add_argument(
        "--history-dir",
        default="historical_inventory",
        help="Subdirectory under base-dir containing weekly history CSVs.",
    )
    parser.add_argument(
        "--lag-weeks",
        type=int,
        default=6,
        help="Number of past weeks to include as explicit lag features.",
    )
    parser.add_argument(
        "--horizon-weeks",
        type=int,
        default=2,
        help="Forecast horizon in weeks for detecting upcoming shortages.",
    )
    parser.add_argument(
        "--shortage-threshold",
        type=float,
        default=15.0,
        help="Amount threshold below which a medicine is considered a shortage.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of the timeline to use for training (chronological split).",
    )
    parser.add_argument(
        "--model-path",
        default="models/shortage_model.joblib",
        help="Where to store the trained model artifact.",
    )
    parser.add_argument(
        "--report-path",
        default="models/shortage_model_report.json",
        help="Where to store evaluation metrics.",
    )
    return parser.parse_args()


def load_inventory_metadata(base_dir: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    metadata: Dict[Tuple[str, str], Dict[str, float]] = {}
    for inventory_file in sorted(base_dir.glob("Hospital_*_inventory.csv")):
        hospital_code = inventory_file.stem.replace("_inventory", "")
        df = pd.read_csv(inventory_file)
        for _, row in df.iterrows():
            med_id = str(row.get("ID", "")).strip()
            if not med_id:
                continue
            metadata[(hospital_code, med_id)] = {
                "urgency": float(row.get("Urgency", 1)),
                "amount": float(row.get("Amount", 0)),
            }
    return metadata


def build_sample_records(
    csv_path: Path,
    metadata: Dict[Tuple[str, str], Dict[str, float]],
    lag_weeks: int,
    horizon_weeks: int,
    shortage_threshold: float,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    df = pd.read_csv(csv_path, parse_dates=["Week_Start_Date"])
    if df.empty:
        return [], []

    df = df.sort_values("Week_Start_Date").reset_index(drop=True)
    df["Amount"] = df["Amount"].astype(float)
    hospital = csv_path.parent.name  # e.g. Hospital_B
    medicine_id = str(df["Medicine_ID"].iloc[0])
    medicine_name = df["Medicine"].iloc[0]
    urgency = metadata.get((hospital, medicine_id), {}).get("urgency", 1.0)

    records: List[SampleRecord] = []
    inference_candidates: List[SampleRecord] = []
    amounts = df["Amount"].to_numpy()
    weeks = df["Week_Start_Date"].to_list()

    for idx in range(len(df)):
        if idx < lag_weeks:
            continue  # not enough history yet

        feature_map: Dict[str, float] = {
            "current_amount": float(amounts[idx]),
            "urgency": urgency,
            "week_of_year": float(df["Week_Start_Date"].dt.isocalendar().week.iloc[idx]),
            "month": float(df["Week_Start_Date"].dt.month.iloc[idx]),
            "year": float(df["Week_Start_Date"].dt.year.iloc[idx]),
            "series_age_weeks": float(idx),
        }

        window = amounts[idx - lag_weeks : idx]
        feature_map["rolling_mean"] = float(np.mean(window))
        feature_map["rolling_min"] = float(np.min(window))
        feature_map["rolling_max"] = float(np.max(window))
        feature_map["rolling_std"] = float(np.std(window))
        feature_map["delta_1"] = float(amounts[idx] - amounts[idx - 1])
        feature_map["delta_3"] = float(amounts[idx] - amounts[idx - min(3, lag_weeks)])

        for lag in range(1, lag_weeks + 1):
            feature_map[f"lag_{lag}"] = float(amounts[idx - lag])

        record = SampleRecord(
            hospital=hospital,
            medicine_id=medicine_id,
            medicine_name=str(medicine_name),
            week_start=weeks[idx],
            features=feature_map,
            target=None,
        )

        future_end_index = idx + horizon_weeks
        if future_end_index < len(df):
            future_window = amounts[idx + 1 : future_end_index + 1]
            record.target = int(np.min(future_window) < shortage_threshold)
            records.append(record)
        else:
            inference_candidates.append(record)

    return records, inference_candidates


def records_to_dataframe(records: Sequence[SampleRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {
            "Hospital": rec.hospital,
            "Medicine_ID": rec.medicine_id,
            "Medicine": rec.medicine_name,
            "Week_Start_Date": rec.week_start,
        }
        row.update(rec.features)
        if rec.target is not None:
            row["Target"] = rec.target
        rows.append(row)
    return pd.DataFrame(rows)


def encode_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    categorical_cols: List[str],
    existing_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    X = df[feature_columns + categorical_cols].copy()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    if existing_columns is not None:
        for col in existing_columns:
            if col not in X:
                X[col] = 0.0
        X = X[existing_columns]
        return X, existing_columns
    else:
        ordered_cols = X.columns.tolist()
        return X, ordered_cols


def chronological_split(df: pd.DataFrame, fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_weeks = sorted(df["Week_Start_Date"].unique())
    if not unique_weeks:
        return df, df
    split_index = int(len(unique_weeks) * fraction)
    split_index = max(1, min(len(unique_weeks) - 1, split_index))
    split_date = unique_weeks[split_index]
    train_df = df[df["Week_Start_Date"] <= split_date].copy()
    test_df = df[df["Week_Start_Date"] > split_date].copy()
    return train_df, test_df


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    history_dir = base_dir / args.history_dir
    history_files = sorted(history_dir.glob("Hospital_*/*.csv"))
    if not history_files:
        raise SystemExit(f"No historical CSVs found under {history_dir}")

    metadata = load_inventory_metadata(base_dir)

    training_records: List[SampleRecord] = []
    inference_candidates: List[SampleRecord] = []

    for csv_path in history_files:
        train_rec, inference_rec = build_sample_records(
            csv_path,
            metadata,
            lag_weeks=args.lag_weeks,
            horizon_weeks=args.horizon_weeks,
            shortage_threshold=args.shortage_threshold,
        )
        training_records.extend(train_rec)
        inference_candidates.extend(inference_rec)

    train_df = records_to_dataframe(training_records)
    if train_df.empty:
        raise SystemExit("No training samples could be created. Check historical data.")

    train_df = train_df.sort_values("Week_Start_Date").reset_index(drop=True)
    train_split, test_split = chronological_split(train_df, args.train_fraction)

    feature_columns = [
        "current_amount",
        "urgency",
        "week_of_year",
        "month",
        "year",
        "series_age_weeks",
        "rolling_mean",
        "rolling_min",
        "rolling_max",
        "rolling_std",
        "delta_1",
        "delta_3",
    ] + [f"lag_{lag}" for lag in range(1, args.lag_weeks + 1)]
    categorical_cols = ["Hospital", "Medicine_ID"]

    X_train, feature_vector_columns = encode_features(train_split, feature_columns, categorical_cols)
    y_train = train_split["Target"].astype(int)

    X_test, _ = encode_features(test_split, feature_columns, categorical_cols, feature_vector_columns)
    y_test = test_split["Target"].astype(int)

    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=800, random_state=42)
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    report = classification_report(y_test, test_preds, output_dict=True)
    report["roc_auc"] = float(roc_auc_score(y_test, test_probs))

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_vector_columns,
            "feature_names": feature_columns,
            "categorical_cols": categorical_cols,
            "lag_weeks": args.lag_weeks,
            "horizon_weeks": args.horizon_weeks,
            "shortage_threshold": args.shortage_threshold,
        },
        args.model_path,
    )

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=== Evaluation ===")
    print(json.dumps(report, indent=2))
    print(f"Model saved to {args.model_path}")

    if inference_candidates:
        inference_df = records_to_dataframe(inference_candidates)
        inference_df = inference_df.dropna(subset=[col for col in feature_columns if col.startswith("lag_")])
        if not inference_df.empty:
            X_inf, _ = encode_features(inference_df, feature_columns, categorical_cols, feature_vector_columns)
            inf_probs = model.predict_proba(X_inf)[:, 1]
            inference_df["shortage_probability"] = inf_probs
            latest_warnings = (
                inference_df.sort_values("shortage_probability", ascending=False)
                .head(15)[["Week_Start_Date", "Hospital", "Medicine_ID", "Medicine", "shortage_probability"]]
            )
            print("=== Highest Risk Upcoming Weeks ===")
            print(latest_warnings.to_string(index=False))


if __name__ == "__main__":
    main()
