"""Handle hospital-to-hospital (H2H) medicine transfers.

Given a lender hospital, receiver hospital, medicine, and quantity, this module
updates both hospitals' inventory CSVs and the medicine-specific expiry logs.
Batches with the furthest expiry dates leave the lender first (i.e., highest
Expiry_Date values are prioritized for transfer).

It can be executed as a standalone CLI script or imported as a module.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

DATE_FORMAT = "%Y-%m-%d"


@dataclass
class TransferResult:
    medicine_id: str
    medicine_name: str
    lender: str
    receiver: str
    quantity: int
    transferred_batches: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute an H2H transfer between hospitals.")
    parser.add_argument("--base-dir", default="medicine_inventory_dummy_data_v2", help="Root folder containing hospital CSVs.")
    parser.add_argument("--medicine-id", required=True, help="Medicine identifier (e.g., M004)")
    parser.add_argument("--lender", required=True, help="Hospital lending stock (e.g., Hospital_A or 'Hospital A')")
    parser.add_argument("--receiver", required=True, help="Hospital receiving stock")
    parser.add_argument("--amount", type=int, required=True, help="Quantity to transfer")
    return parser.parse_args()


def canonical_hospital(value: str) -> str:
    value = (value or "").strip().replace(" ", "_")
    if not value:
        raise ValueError("Hospital name is required")
    if not value.lower().startswith("hospital_"):
        value = f"Hospital_{value.split('_')[-1]}" if "_" in value else f"Hospital_{value}"  # fallback
    return value


def read_csv(path: Path) -> tuple[Sequence[str], List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = [row for row in reader if any((value or "").strip() for value in row.values())]
    return headers, rows


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_inventory_row(rows: List[Dict[str, str]], medicine_id: str) -> Dict[str, str]:
    for row in rows:
        if row.get("ID") == medicine_id:
            return row
    raise ValueError(f"Medicine {medicine_id} not found in inventory")


def parse_expiry(value: str) -> datetime:
    return datetime.strptime(value, DATE_FORMAT)


def ensure_batch_capacity(
    log_rows: List[Dict[str, str]],
    medicine_id: str,
    medicine_name: str,
    required: int,
) -> None:
    valid_rows = [row for row in log_rows if (row.get("Batch_ID") and row.get("Expiry_Date"))]
    deficit = required - len(valid_rows)
    if deficit <= 0:
        return

    max_expiry = max((parse_expiry(row["Expiry_Date"]) for row in valid_rows), default=datetime.utcnow())
    placeholder_expiry = (max_expiry + timedelta(days=365)).strftime(DATE_FORMAT)
    start_suffix = next_batch_suffix(log_rows)

    for offset in range(deficit):
        log_rows.append(
            {
                "Medicine_ID": medicine_id,
                "Medicine": medicine_name,
                "Batch_ID": f"{medicine_id}_B{start_suffix + offset:03d}",
                "Expiry_Date": placeholder_expiry,
            }
        )


def select_batches(
    log_rows: List[Dict[str, str]],
    medicine_id: str,
    medicine_name: str,
    amount: int,
) -> List[Dict[str, str]]:
    if amount <= 0:
        raise ValueError("Transfer amount must be positive")

    ensure_batch_capacity(log_rows, medicine_id, medicine_name, amount)
    sortable = [row for row in log_rows if (row.get("Batch_ID") and row.get("Expiry_Date"))]

    sortable.sort(key=lambda row: parse_expiry(row["Expiry_Date"]), reverse=True)
    selected = sortable[:amount]
    selected_ids = {row["Batch_ID"] for row in selected}

    remaining = [row for row in log_rows if row.get("Batch_ID") not in selected_ids]
    log_rows.clear()
    log_rows.extend(remaining)
    return selected


def next_batch_suffix(rows: List[Dict[str, str]]) -> int:
    suffix = 0
    for row in rows:
        batch_id = row.get("Batch_ID", "")
        if "_B" in batch_id:
            try:
                suffix = max(suffix, int(batch_id.split("_B")[-1]))
            except ValueError:
                continue
    return suffix + 1


def attach_batches(rows: List[Dict[str, str]], medicine_id: str, medicine_name: str, batches: List[Dict[str, str]]) -> List[str]:
    transferred_ids: List[str] = []
    start_suffix = next_batch_suffix(rows)
    for idx, batch in enumerate(batches, start=start_suffix):
        new_row = {
            "Medicine_ID": medicine_id,
            "Medicine": medicine_name,
            "Batch_ID": f"{medicine_id}_B{idx:03d}",
            "Expiry_Date": batch.get("Expiry_Date", ""),
        }
        rows.append(new_row)
        transferred_ids.append(new_row["Batch_ID"])
    return transferred_ids


def find_log_file(base_dir: Path, hospital: str, medicine_id: str) -> Path:
    logs_dir = base_dir / f"{hospital}_medicine_logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"Log directory missing: {logs_dir}")
    matches = sorted(logs_dir.glob(f"{medicine_id}_*_expiry_log.csv"))
    if not matches:
        raise FileNotFoundError(f"No log file found for {hospital} {medicine_id}")
    return matches[0]


def execute_transfer(
    base_dir: Path,
    medicine_id: str,
    lender: str,
    receiver: str,
    amount: int,
) -> TransferResult:
    lender = canonical_hospital(lender)
    receiver = canonical_hospital(receiver)
    if lender == receiver:
        raise ValueError("Lender and receiver hospitals must differ")

    inventory_lender_path = base_dir / f"{lender}_inventory.csv"
    inventory_receiver_path = base_dir / f"{receiver}_inventory.csv"

    lender_headers, lender_rows = read_csv(inventory_lender_path)
    receiver_headers, receiver_rows = read_csv(inventory_receiver_path)

    lender_row = find_inventory_row(lender_rows, medicine_id)
    receiver_row = find_inventory_row(receiver_rows, medicine_id)
    medicine_name = lender_row.get("Medicine", receiver_row.get("Medicine", "Unknown"))

    lender_amount = int(lender_row.get("Amount", 0))
    if lender_amount < amount:
        raise ValueError(f"{lender} only has {lender_amount} units available")

    lender_row["Amount"] = str(lender_amount - amount)
    receiver_row["Amount"] = str(int(receiver_row.get("Amount", 0)) + amount)

    lender_log_path = find_log_file(base_dir, lender, medicine_id)
    receiver_log_path = find_log_file(base_dir, receiver, medicine_id)

    lender_log_headers, lender_log_rows = read_csv(lender_log_path)
    receiver_log_headers, receiver_log_rows = read_csv(receiver_log_path)

    selected_batches = select_batches(lender_log_rows, medicine_id, medicine_name, amount)
    transferred_ids = attach_batches(receiver_log_rows, medicine_id, medicine_name, selected_batches)

    write_csv(inventory_lender_path, lender_headers, lender_rows)
    write_csv(inventory_receiver_path, receiver_headers, receiver_rows)
    write_csv(lender_log_path, lender_log_headers, lender_log_rows)
    write_csv(receiver_log_path, receiver_log_headers, receiver_log_rows)

    return TransferResult(
        medicine_id=medicine_id,
        medicine_name=medicine_name,
        lender=lender,
        receiver=receiver,
        quantity=amount,
        transferred_batches=transferred_ids,
    )


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    result = execute_transfer(base_dir, args.medicine_id, args.lender, args.receiver, args.amount)
    print(
        f"Transferred {result.quantity} units of {result.medicine_name} ({result.medicine_id}) "
        f"from {result.lender} to {result.receiver}. Batches: {', '.join(result.transferred_batches)}"
    )


if __name__ == "__main__":
    main()
