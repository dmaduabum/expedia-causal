"""
src/analysis/descriptive.py
----------------------------
Produces:
  - results/tables/summary_stats.csv   (means/SDs of key variables)
  - results/tables/rates_by_condition.csv  (booking/click rate by random_bool)

Usage:
    python src/analysis/descriptive.py
"""

import os
import pandas as pd
import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN    = os.path.join(ROOT, "data", "processed", "train_clean.csv")
OUT_DIR  = os.path.join(ROOT, "results", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

# Variables to summarize — split by hotel vs. search characteristics
HOTEL_VARS  = ["price_usd", "prop_starrating", "prop_review_score",
               "prop_brand_bool", "prop_location_score1",
               "prop_location_score2", "promotion_flag"]

SEARCH_VARS = ["srch_length_of_stay", "srch_booking_window",
               "srch_adults_count", "srch_children_count",
               "srch_saturday_night_bool", "visitor_hist_adr_usd",
               "no_purchase_history", "orig_destination_distance"]

def summarise(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return mean, SD, median, % missing for each column."""
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        rows.append({
            "variable": c,
            "mean":     round(s.mean(), 4),
            "sd":       round(s.std(),  4),
            "median":   round(s.median(), 4),
            "pct_miss": round(s.isna().mean() * 100, 2),
            "n":        s.notna().sum(),
        })
    return pd.DataFrame(rows)

def main():
    print("Loading cleaned data ...")
    df = pd.read_csv(CLEAN, low_memory=False)
    print(f"  {len(df):,} rows | {df['srch_id'].nunique():,} searches")

    # ── overall summary stats ─────────────────────────────────────────────────
    all_vars = HOTEL_VARS + SEARCH_VARS
    stats = summarise(df, all_vars)
    out = os.path.join(OUT_DIR, "summary_stats.csv")
    stats.to_csv(out, index=False)
    print(f"Summary stats saved → {out}")
    print(stats.to_string(index=False))

    # ── booking & click rates by condition ───────────────────────────────────
    rates = (
        df.groupby("random_bool")
          .agg(
              n_rows      = ("srch_id", "count"),
              n_searches  = ("srch_id", "nunique"),
              booking_rate= ("booking_bool", "mean"),
              click_rate  = ("click_bool",   "mean"),
          )
          .reset_index()
    )
    rates["condition"] = rates["random_bool"].map({0: "Algorithmic", 1: "Random"})
    out2 = os.path.join(OUT_DIR, "rates_by_condition.csv")
    rates.to_csv(out2, index=False)
    print(f"\nRates by condition saved → {out2}")
    print(rates.to_string(index=False))

    # ── quick check: mean position by condition ───────────────────────────────
    pos = df.groupby("random_bool")["position"].mean().round(2)
    print("\nMean position by condition:")
    print(pos)

if __name__ == "__main__":
    main()
