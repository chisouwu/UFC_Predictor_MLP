#!/usr/bin/env python3
"""Append BestFightOdds values to UFC fight rows and save as a new CSV.

This script joins odds for each fighter in a fight onto UFC rows twice:
- once for the red corner fighter (r_id)
- once for the blue corner fighter (b_id)

Output defaults to data/ufc_with_odds.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append BestFightOdds odds columns to UFC.csv and write a new file."
    )
    parser.add_argument(
        "--ufc",
        default="data/UFC.csv",
        help="Path to UFC.csv (default: data/UFC.csv)",
    )
    parser.add_argument(
        "--odds",
        default="data/BestFightOdds_odds.csv",
        help="Path to BestFightOdds_odds.csv (default: data/BestFightOdds_odds.csv)",
    )
    parser.add_argument(
        "--out",
        default="data/ufc_with_odds.csv",
        help="Output CSV path (default: data/ufc_with_odds.csv)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    ufc_path = Path(args.ufc)
    odds_path = Path(args.odds)
    out_path = Path(args.out)

    if not ufc_path.exists():
        raise FileNotFoundError(f"UFC file not found: {ufc_path}")
    if not odds_path.exists():
        raise FileNotFoundError(f"Odds file not found: {odds_path}")

    ufc_df = pd.read_csv(ufc_path)
    odds_df = pd.read_csv(odds_path)

    required_ufc_cols = {"fight_id", "r_id", "b_id"}
    required_odds_cols = {
        "fight_id",
        "fighter_id",
        "opening",
        "closing_range_min",
        "closing_range_max",
    }

    missing_ufc = required_ufc_cols - set(ufc_df.columns)
    missing_odds = required_odds_cols - set(odds_df.columns)

    if missing_ufc:
        raise ValueError(f"UFC file missing required columns: {sorted(missing_ufc)}")
    if missing_odds:
        raise ValueError(f"Odds file missing required columns: {sorted(missing_odds)}")

    # If the odds source has duplicate (fight_id, fighter_id) rows, keep last.
    odds_df = odds_df.drop_duplicates(subset=["fight_id", "fighter_id"], keep="last")

    red_odds = odds_df.rename(
        columns={
            "fighter_id": "r_id",
            "opening": "r_opening",
            "closing_range_min": "r_closing_range_min",
            "closing_range_max": "r_closing_range_max",
        }
    )

    blue_odds = odds_df.rename(
        columns={
            "fighter_id": "b_id",
            "opening": "b_opening",
            "closing_range_min": "b_closing_range_min",
            "closing_range_max": "b_closing_range_max",
        }
    )

    merged = ufc_df.merge(
        red_odds[
            [
                "fight_id",
                "r_id",
                "r_opening",
                "r_closing_range_min",
                "r_closing_range_max",
            ]
        ],
        on=["fight_id", "r_id"],
        how="left",
    )

    merged = merged.merge(
        blue_odds[
            [
                "fight_id",
                "b_id",
                "b_opening",
                "b_closing_range_min",
                "b_closing_range_max",
            ]
        ],
        on=["fight_id", "b_id"],
        how="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"Wrote {len(merged):,} rows to {out_path}")
    print(
        "Added columns: "
        "r_opening, r_closing_range_min, r_closing_range_max, "
        "b_opening, b_closing_range_min, b_closing_range_max"
    )


if __name__ == "__main__":
    main()
