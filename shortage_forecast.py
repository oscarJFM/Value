"""Utilities for predicting upcoming medicine shortages using the trained model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


@dataclass
class SampleRecord:
    hospital: str
    medicine_id: str
    medicine_name: str
    week_start: pd.Timestamp
    features: Dict[str, float]


def load_inventory_metadata(base_dir: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    metadata: Dict[Tuple[str, str], Dict[str, float]] = {}
    for inventory_file in sorted(base_dir.glob("Hospital_*_inventory.csv")):
        hospital_code = inventory_file.stem.replace("_inventory", "")
        hospital_label = hospital_code.replace("_", " ")
        df = pd.read_csv(inventory_file)
        for _, row in df.iterrows():
            med_id = str(row.get("ID", "")).strip()
            if not med_id:
                continue
            metadata[(hospital_label, med_id)] = {
                "urgency": float(row.get("Urgency", 1)),
                "amount": float(row.get("Amount", 0)),
            }
    return metadata


def build_inference_record(
    csv_path: Path,
    metadata: Dict[Tuple[str, str], Dict[str, float]],
    lag_weeks: int,
) -> SampleRecord | None:
    df = pd.read_csv(csv_path, parse_dates=["Week_Start_Date"])
    if df.empty:
        return None

    df = df.sort_values("Week_Start_Date").reset_index(drop=True)
    if len(df) <= lag_weeks:
        return None

    df["Amount"] = df["Amount"].astype(float)
    hospital = csv_path.parent.name.replace("_", " ")
    medicine_id = str(df["Medicine_ID"].iloc[0])
    medicine_name = str(df["Medicine"].iloc[0])
    urgency = metadata.get((hospital, medicine_id), {}).get("urgency", 1.0)

    amounts = df["Amount"].to_numpy()
    weeks = df["Week_Start_Date"].to_list()
    idx = len(df) - 1

    features: Dict[str, float] = {
        "current_amount": float(amounts[idx]),
        "urgency": float(urgency),
        "week_of_year": float(df["Week_Start_Date"].dt.isocalendar().week.iloc[idx]),
        "month": float(df["Week_Start_Date"].dt.month.iloc[idx]),
        "year": float(df["Week_Start_Date"].dt.year.iloc[idx]),
        "series_age_weeks": float(idx),
    }

    window = amounts[idx - lag_weeks : idx]
    features["rolling_mean"] = float(np.mean(window))
    features["rolling_min"] = float(np.min(window))
    features["rolling_max"] = float(np.max(window))
    features["rolling_std"] = float(np.std(window))
    features["delta_1"] = float(amounts[idx] - amounts[idx - 1])
    back_idx = max(0, idx - min(3, lag_weeks))
    features["delta_3"] = float(amounts[idx] - amounts[back_idx])

    for lag in range(1, lag_weeks + 1):
        features[f"lag_{lag}"] = float(amounts[idx - lag])

    return SampleRecord(
        hospital=hospital,
        medicine_id=medicine_id,
        medicine_name=medicine_name,
        week_start=weeks[idx],
        features=features,
    )


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
        rows.append(row)
    return pd.DataFrame(rows)


def encode_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    categorical_cols: List[str],
    existing_columns: List[str],
) -> pd.DataFrame:
    X = df[feature_columns + categorical_cols].copy()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    for col in existing_columns:
        if col not in X:
            X[col] = 0.0
    X = X[existing_columns]
    return X


def predict_upcoming_shortages(
    base_dir: Path,
    model_path: Path,
    probability_threshold: float = 0.7,
) -> List[Dict[str, object]]:
    if not model_path.exists():
        return []

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    base_feature_names = artifact["feature_names"]
    categorical_cols = artifact["categorical_cols"]
    lag_weeks = artifact["lag_weeks"]
    horizon_weeks = artifact["horizon_weeks"]

    metadata = load_inventory_metadata(base_dir)
    history_dir = base_dir / "historical_inventory"
    history_files = sorted(history_dir.glob("Hospital_*/*.csv"))
    inference_records: List[SampleRecord] = []

    for csv_path in history_files:
        record = build_inference_record(csv_path, metadata, lag_weeks=lag_weeks)
        if record:
            inference_records.append(record)

    if not inference_records:
        return []

    inference_df = records_to_dataframe(inference_records)
    X_inf = encode_features(inference_df, base_feature_names, categorical_cols, feature_columns)
    probabilities = model.predict_proba(X_inf)[:, 1]

    results: List[Dict[str, object]] = []
    for rec, prob in zip(inference_records, probabilities):
        if prob < probability_threshold:
            continue
        results.append(
            {
                "Hospital_Source": rec.hospital,
                "Medicine_ID": rec.medicine_id,
                "Medicine": rec.medicine_name,
                "Week_Start_Date": rec.week_start.isoformat(),
                "current_amount": rec.features["current_amount"],
                "urgency": rec.features["urgency"],
                "probability": float(prob),
                "forecast_window_weeks": horizon_weeks,
            }
        )

    return results
