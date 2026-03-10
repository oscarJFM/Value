"""Generate weekly inventory history CSVs per hospital and medicine.

Each output CSV contains one row per week starting from January 6, 2025 (first
Monday of 2025) through the Monday of the current week. The final row always
matches the current inventory amount stored in Hospital_X_inventory.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

START_DATE = date(2025, 1, 6)  # First Monday of 2025
HEADER = ["Week_Start_Date", "Hospital", "Medicine_ID", "Medicine", "Amount"]


@dataclass
class HistorySummary:
    hospital: str
    medicine_id: str
    medicine_name: str
    weeks: int
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weekly history CSVs")
    parser.add_argument(
        "--base-dir",
        default="medicine_inventory_dummy_data_v2",
        help="Directory containing Hospital_X_inventory.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="historical_inventory",
        help="Subdirectory (inside base-dir) where history CSVs will be written.",
    )
    parser.add_argument(
        "--start",
        default=START_DATE.isoformat(),
        help="First week start date (YYYY-MM-DD). Defaults to first Monday of 2025.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned files without writing them.",
    )
    return parser.parse_args()


def read_inventory(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing inventory CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if any((value or "").strip() for value in row.values())]


def week_starts(start: date, end: date) -> List[date]:
    weeks: List[date] = []
    current = start
    while current <= end:
        weeks.append(current)
        current += timedelta(days=7)
    return weeks


def align_to_monday(target: date) -> date:
    return target - timedelta(days=target.weekday())


def generate_global_events(num_weeks: int) -> List[Dict[str, float]]:
    rng = random.Random("global-events")
    events: List[Dict[str, float]] = []
    for i in range(num_weeks):
        base = 1.0 + math.sin(i / 8.0) * 0.1 + rng.uniform(-0.05, 0.05)
        events.append(
            {
                "demand": base,
                "supply_block": False,
                "emergency_draw": False,
                "surge_supply": False,
            }
        )

    i = 0
    while i < num_weeks:
        if rng.random() < 0.22:
            duration = rng.randint(2, 6)
            severity = 1.3 + rng.random() * 0.9
            supply_block = rng.random() < 0.7
            emergency = rng.random() < 0.5
            for j in range(i, min(num_weeks, i + duration)):
                events[j]["demand"] *= severity
                if supply_block:
                    events[j]["supply_block"] = True
                if emergency:
                    events[j]["emergency_draw"] = True
            i += duration
        else:
            i += 1

    i = 0
    while i < num_weeks:
        if rng.random() < 0.12:
            duration = rng.randint(2, 4)
            relief = 0.6 + rng.random() * 0.3
            for j in range(i, min(num_weeks, i + duration)):
                events[j]["demand"] *= relief
                events[j]["surge_supply"] = True
            i += duration
        else:
            i += 1

    return events


def build_series(
    weeks: List[date],
    final_amount: int,
    urgency: int,
    global_events: List[Dict[str, float]],
    seed: str,
) -> List[int]:
    rng = random.Random(seed)
    if not weeks:
        return []

    # Determine baseline behavior
    urgency_factor = max(1, urgency)
    avg_stock = max(15, final_amount * rng.uniform(0.6, 1.4) + rng.randint(0, 40))
    amount = max(0, avg_stock + rng.randint(-40, 40))
    series: List[int] = []

    base_demand = max(3, final_amount * 0.07 + urgency_factor * 2)
    restock_size = max(20, final_amount * rng.uniform(0.2, 0.5))

    for index, _ in enumerate(weeks):
        event = global_events[index] if index < len(global_events) else {
            "demand": 1.0,
            "supply_block": False,
            "emergency_draw": False,
            "surge_supply": False,
        }
        demand = max(0, rng.gauss(base_demand, base_demand * 0.5)) * event["demand"]
        seasonal = math.sin(index / rng.uniform(3.0, 5.5)) * rng.uniform(5, 25)
        amount = max(0, amount - demand + seasonal)

        needs_restock = amount < final_amount * rng.uniform(0.1, 0.35)
        scheduled_restock = rng.random() < 0.22
        if needs_restock or scheduled_restock:
            restock = restock_size * rng.uniform(0.6, 1.6)
            if event["supply_block"]:
                restock *= rng.uniform(0.1, 0.4)
            if event["surge_supply"]:
                restock *= rng.uniform(1.3, 1.9)
            amount += restock

        if rng.random() < 0.12:
            # occasional shock (e.g., urgent transfer out)
            shock = rng.randint(int(amount * 0.2) if amount else 5, int(amount * 0.6) + 10)
            amount = max(0, amount - shock)

        if event["emergency_draw"]:
            amount = max(0, amount - rng.uniform(0.2, 0.6) * max(amount, final_amount * 0.5))

        amount = max(0, round(amount + rng.uniform(-8, 8)))
        series.append(amount)

    if not series:
        return []

    delta = final_amount - series[-1]
    adjustment_span = min(10, len(series))
    for i in range(adjustment_span):
        idx = -adjustment_span + i
        series[idx] = max(0, series[idx] + round(delta * (i + 1) / adjustment_span))
    series[-1] = final_amount
    return series


def build_rows(
    hospital: str,
    medicine_id: str,
    medicine_name: str,
    weeks: List[date],
    amounts: List[int],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for week, amount in zip(weeks, amounts):
        rows.append(
            {
                "Week_Start_Date": week.isoformat(),
                "Hospital": hospital.replace("_", " "),
                "Medicine_ID": medicine_id,
                "Medicine": medicine_name,
                "Amount": str(amount),
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_hospital(
    inventory_path: Path,
    output_dir: Path,
    weeks: List[date],
    global_events: List[Dict[str, float]],
    dry_run: bool,
) -> List[HistorySummary]:
    hospital = inventory_path.stem.replace("_inventory", "")
    inventory_rows = read_inventory(inventory_path)
    summaries: List[HistorySummary] = []

    for row in inventory_rows:
        medicine_id = row.get("ID")
        medicine_name = row.get("Medicine", "Unknown")
        amount_str = row.get("Amount", "0")
        if not medicine_id:
            continue
        try:
            final_amount = int(amount_str)
        except ValueError:
            final_amount = 0

        urgency_val = row.get("Urgency")
        try:
            urgency_level = int(urgency_val) if urgency_val is not None else 1
        except ValueError:
            urgency_level = 1

        amounts = build_series(
            weeks,
            final_amount,
            urgency_level,
            global_events,
            seed=f"{hospital}-{medicine_id}"
        )
        data_rows = build_rows(hospital, medicine_id, medicine_name, weeks, amounts)

        safe_name = medicine_name.replace(" ", "")
        output_path = output_dir / hospital / f"{medicine_id}_{safe_name}_weekly_history.csv"

        if not dry_run:
            write_csv(output_path, data_rows)

        summaries.append(
            HistorySummary(
                hospital=hospital,
                medicine_id=medicine_id,
                medicine_name=medicine_name,
                weeks=len(weeks),
                output_path=output_path,
            )
        )

    return summaries


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    start_date = date.fromisoformat(args.start)
    week_end = align_to_monday(date.today())
    output_dir = base_dir / args.output_dir
    weeks = week_starts(start_date, week_end)
    global_events = generate_global_events(len(weeks))

    inventory_files = sorted(base_dir.glob("Hospital_*_inventory.csv"))
    if not inventory_files:
        raise SystemExit(f"No Hospital_*_inventory.csv files found in {base_dir}")

    all_summaries: List[HistorySummary] = []
    for inventory_path in inventory_files:
        all_summaries.extend(
            process_hospital(inventory_path, output_dir, weeks, global_events, args.dry_run)
        )

    for summary in all_summaries:
        print(
            f"{summary.hospital} {summary.medicine_name} ({summary.medicine_id}): "
            f"{summary.weeks} weeks -> {summary.output_path.relative_to(base_dir)}"
        )

    if args.dry_run:
        print("Dry-run complete; no files created.")


if __name__ == "__main__":
    main()
