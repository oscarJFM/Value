"""Automate inventory updates based on medicine expiry logs.

This script scans every hospital inventory CSV alongside the per-medicine
expiry logs located inside the `medicine_inventory_dummy_data_v2` folder.
For each hospital it:

1. Removes any batch rows whose expiry date is on/before the provided
   cutoff date.
2. Reduces the aggregate `Amount` column for the corresponding medicine
   in the hospital inventory by the number of removed batches.

Run with `--dry-run` to preview the changes without touching the files.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

DATE_FORMAT = "%Y-%m-%d"


@dataclass
class MedicineChange:
    medicine_id: str
    medicine_name: str
    expired_batches: int
    inventory_before: int
    inventory_after: int
    log_path: Path


@dataclass
class ExpiredBatch:
    hospital: str
    medicine_id: str
    medicine_name: str
    batch_id: str
    expiry_date: str
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove expired medicine batches and update inventory counts."
    )
    parser.add_argument(
        "--base-dir",
        default="medicine_inventory_dummy_data_v2",
        help="Path to the folder that contains hospital inventories and logs.",
    )
    parser.add_argument(
        "--date",
        dest="cutoff",
        default=date.today().strftime(DATE_FORMAT),
        help=(
            "Expiry cutoff in YYYY-MM-DD. Batches with dates on/before this value "
            "are considered expired. Defaults to today."
        ),
    )
    parser.add_argument(
        "--hospital",
        action="append",
        dest="hospitals",
        help=(
            "Restrict processing to one or more hospitals (e.g. --hospital Hospital_A). "
            "Provide multiple flags to include several hospitals."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned modifications without rewriting any files.",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, DATE_FORMAT).date()


def read_csv(path: Path) -> tuple[Sequence[str], List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

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


def partition_expired(
    rows: Iterable[Dict[str, str]],
    cutoff: date,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    kept: List[Dict[str, str]] = []
    expired: List[Dict[str, str]] = []

    for row in rows:
        expiry_str = (row.get("Expiry_Date") or "").strip()
        if not expiry_str:
            kept.append(row)
            continue

        try:
            expiry_date = parse_date(expiry_str)
        except ValueError:
            print(f"⚠️  Skipping row with invalid date '{expiry_str}': {row}")
            kept.append(row)
            continue

        if expiry_date <= cutoff:
            expired.append(row)
        else:
            kept.append(row)

    return kept, expired


def update_inventory_amounts(
    inventory_rows: List[Dict[str, str]],
    expired_counts: Dict[str, int],
) -> List[MedicineChange]:
    changes: List[MedicineChange] = []

    for row in inventory_rows:
        medicine_id = row.get("ID")
        if not medicine_id or medicine_id not in expired_counts:
            continue

        expired_batches = expired_counts[medicine_id]
        if expired_batches <= 0:
            continue

        try:
            current_amount = int(row["Amount"])
        except (ValueError, KeyError):
            print(f"⚠️  Unable to parse Amount for {medicine_id}: {row}")
            continue

        updated_amount = max(0, current_amount - expired_batches)
        row["Amount"] = str(updated_amount)
        changes.append(
            MedicineChange(
                medicine_id=medicine_id,
                medicine_name=row.get("Medicine", "Unknown"),
                expired_batches=expired_batches,
                inventory_before=current_amount,
                inventory_after=updated_amount,
                log_path=Path(),  # placeholder, filled later
            )
        )

    return changes


def process_hospital(
    hospital: str,
    inventory_path: Path,
    logs_dir: Path,
    cutoff: date,
    dry_run: bool,
) -> tuple[List[MedicineChange], List[ExpiredBatch]]:
    headers, inventory_rows = read_csv(inventory_path)
    expired_counts: Dict[str, int] = defaultdict(int)
    change_records: Dict[str, MedicineChange] = {}
    removed_batches: List[ExpiredBatch] = []

    if not logs_dir.exists():
        print(f"⚠️  No log directory found for {hospital}: {logs_dir}")
        return [], []

    log_files = sorted(logs_dir.glob("*.csv"))
    if not log_files:
        print(f"⚠️  No log files present in {logs_dir}")

    for log_file in log_files:
        log_headers, log_rows = read_csv(log_file)
        if not log_rows:
            continue

        kept_rows, expired_rows = partition_expired(log_rows, cutoff)
        expired_count = len(expired_rows)
        if expired_count == 0:
            continue

        medicine_id = expired_rows[0].get("Medicine_ID") or kept_rows[0].get("Medicine_ID")
        medicine_name = expired_rows[0].get("Medicine") or kept_rows[0].get("Medicine") or "Unknown"
        if not medicine_id:
            print(f"⚠️  Unable to determine medicine ID for log {log_file}")
            continue

        expired_counts[medicine_id] += expired_count

        for row in expired_rows:
            removed_batches.append(
                ExpiredBatch(
                    hospital=hospital,
                    medicine_id=medicine_id,
                    medicine_name=medicine_name,
                    batch_id=row.get("Batch_ID", "Unknown"),
                    expiry_date=row.get("Expiry_Date", "Unknown"),
                    log_path=log_file,
                )
            )

        if not dry_run:
            write_csv(log_file, log_headers, kept_rows)

        change_records.setdefault(
            medicine_id,
            MedicineChange(
                medicine_id=medicine_id,
                medicine_name=medicine_name,
                expired_batches=0,
                inventory_before=0,
                inventory_after=0,
                log_path=log_file,
            ),
        ).expired_batches += expired_count

    if not expired_counts:
        return [], removed_batches

    inventory_changes = update_inventory_amounts(inventory_rows, expired_counts)
    if not dry_run and inventory_changes:
        write_csv(inventory_path, headers, inventory_rows)

    for change in inventory_changes:
        if change.medicine_id in change_records:
            change.log_path = change_records[change.medicine_id].log_path
            change.expired_batches = change_records[change.medicine_id].expired_batches

    return inventory_changes, removed_batches


def summarize(
    hospital: str,
    changes: Sequence[MedicineChange],
    removed_batches: Sequence[ExpiredBatch],
    dry_run: bool,
    cutoff: date,
) -> None:
    if not changes and not removed_batches:
        print(f"{hospital}: no batches expired on/before {cutoff}.")
        return

    print(
        f"{hospital}: {len(changes)} medicine(s) updated, {len(removed_batches)} "
        f"batch(es) removed (cutoff {cutoff})."
    )
    for change in changes:
        delta = change.inventory_before - change.inventory_after
        log_info = f"log: {change.log_path.name}" if change.log_path else "log: unknown"
        dry_run_note = " [dry-run]" if dry_run else ""
        print(
            f"  - {change.medicine_name} ({change.medicine_id}): removed "
            f"{change.expired_batches} batches ({log_info}); inventory {change.inventory_before} -> "
            f"{change.inventory_after} (Δ {delta}).{dry_run_note}"
        )

    if removed_batches:
        print("    Deleted batch entries:")
        for batch in removed_batches:
            dry_run_note = " [dry-run]" if dry_run else ""
            print(
                f"      • {batch.medicine_name} ({batch.medicine_id}) | Batch {batch.batch_id} | "
                f"Expiry {batch.expiry_date} | {batch.log_path.name}{dry_run_note}"
            )


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    cutoff = parse_date(args.cutoff)

    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    requested = set(args.hospitals or [])
    inventory_files = sorted(base_dir.glob("Hospital_*_inventory.csv"))
    if not inventory_files:
        raise SystemExit(f"No hospital inventory CSV files found in {base_dir}")

    for inventory_path in inventory_files:
        hospital = inventory_path.stem.replace("_inventory", "")
        if requested and hospital not in requested:
            continue

        logs_dir = base_dir / f"{hospital}_medicine_logs"
        changes, removed_batches = process_hospital(
            hospital, inventory_path, logs_dir, cutoff, args.dry_run
        )
        summarize(hospital, changes, removed_batches, args.dry_run, cutoff)

    if args.dry_run:
        print("Dry-run complete; no files were modified.")


if __name__ == "__main__":
    main()
