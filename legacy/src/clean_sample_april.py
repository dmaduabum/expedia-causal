"""
src/pipeline/clean_sample.py
-----------------------------
Loads data/raw/train.csv, applies cleaning rules, samples 50k searches
at the search level, and writes data/processed/train_clean.csv.

Cleaning rules (document every drop so the paper can cite them):
  1.  Keep only the columns needed for analysis.
  2.  Drop rows where random_bool is missing (cannot assign to condition).
  3.  Drop rows where booking_bool or click_bool is missing (outcome missing).
  4.  Drop rows where position is missing or <= 0.
  5.  Drop rows where price_usd is missing, zero, or negative.
  6.  Drop rows where prop_starrating is 0
          (no stars / unknown — ambiguous covariate).
  7.  Drop rows where prop_review_score is missing
          (key covariate; too many would bias balance).
  8.  Drop rows where prop_location_score1 is missing.
  9.  Impute visitor_hist_adr_usd and visitor_hist_starrating missing → 0
          and add indicator flags (no_purchase_history).
  10. Impute srch_query_affinity_score missing → column median.
  11. Impute orig_destination_distance missing → column median
          and add indicator flag.
  12. Winsorise price_usd at the 99th percentile to remove outliers.
  13. Drop searches where ALL hotels have the same booking_bool
          (no within-search variation — uninformative for IV).
  14. Subsample 50k unique srch_ids at random (seed=571).

All rules are logged to results/tables/cleaning_log.txt.

Usage:
    python src/pipeline/clean_sample.py
"""

import os
import io
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CSV   = os.path.join(ROOT, "data", "raw",       "train.csv")
OUT_CSV   = os.path.join(ROOT, "data", "processed", "train_clean.csv")
LOG_PATH  = os.path.join(ROOT, "results", "tables",  "cleaning_log.txt")

os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "results", "tables"),  exist_ok=True)

# ── columns to keep ──────────────────────────────────────────────────────────
KEEP_COLS = [
    # identifiers
    "srch_id", "prop_id", "date_time",
    # treatment / instrument
    "random_bool",
    # outcomes
    "booking_bool", "click_bool", "gross_bookings_usd",
    # ranking
    "position",
    # hotel characteristics
    "prop_starrating", "prop_review_score", "prop_brand_bool",
    "prop_location_score1", "prop_location_score2",
    "prop_log_historical_price", "price_usd", "promotion_flag",
    "prop_country_id",
    # visitor characteristics
    "visitor_hist_starrating", "visitor_hist_adr_usd",
    "visitor_location_country_id",
    # search characteristics
    "srch_destination_id", "srch_length_of_stay", "srch_booking_window",
    "srch_adults_count", "srch_children_count", "srch_room_count",
    "srch_saturday_night_bool", "srch_query_affinity_score",
    "orig_destination_distance",
    # platform
    "site_id",
]

N_SAMPLE  = 50_000
SEED      = 571

def log(lines: list, path: str):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    for l in lines:
        print(l)

def winsorise(s: pd.Series, upper_pct: float = 99) -> pd.Series:
    cap = np.percentile(s.dropna(), upper_pct)
    return s.clip(upper=cap)

def clean():
    log_lines = ["=" * 60, "CLEANING LOG — expedia-causal", "=" * 60]

    # ── load ─────────────────────────────────────────────────────────────────
    print("Reading train.csv (this takes ~30s) ...")
    df = pd.read_csv(RAW_CSV, usecols=lambda c: c in KEEP_COLS, low_memory=False)
    log_lines.append(f"\nRaw rows        : {len(df):>10,}")
    log_lines.append(f"Raw searches    : {df['srch_id'].nunique():>10,}")

    # ── Rule 2: drop if random_bool missing ──────────────────────────────────
    before = len(df)
    df = df[df["random_bool"].notna()]
    log_lines.append(f"\nRule 2 (drop missing random_bool)  : -{before - len(df):>8,} rows")

    # ── Rule 3: drop if booking_bool or click_bool missing ───────────────────
    before = len(df)
    df = df[df["booking_bool"].notna() & df["click_bool"].notna()]
    log_lines.append(f"Rule 3 (drop missing outcomes)     : -{before - len(df):>8,} rows")

    # ── Rule 4: drop if position missing or <= 0 ─────────────────────────────
    before = len(df)
    df = df[df["position"].notna() & (df["position"] > 0)]
    log_lines.append(f"Rule 4 (drop bad position)         : -{before - len(df):>8,} rows")

    # ── Rule 5: drop if price_usd missing, zero, or negative ─────────────────
    before = len(df)
    df = df[df["price_usd"].notna() & (df["price_usd"] > 0)]
    log_lines.append(f"Rule 5 (drop bad price_usd)        : -{before - len(df):>8,} rows")

    # ── Rule 6: drop if prop_starrating == 0 (unknown) ───────────────────────
    before = len(df)
    df = df[df["prop_starrating"] > 0]
    log_lines.append(f"Rule 6 (drop unknown star rating)  : -{before - len(df):>8,} rows")

    # ── Rule 7: drop if prop_review_score missing ─────────────────────────────
    before = len(df)
    df = df[df["prop_review_score"].notna()]
    log_lines.append(f"Rule 7 (drop missing review score) : -{before - len(df):>8,} rows")

    # ── Rule 8: drop if prop_location_score1 missing ─────────────────────────
    before = len(df)
    df = df[df["prop_location_score1"].notna()]
    log_lines.append(f"Rule 8 (drop missing location sc1) : -{before - len(df):>8,} rows")

    # ── Rule 9: impute visitor purchase history ───────────────────────────────
    df["no_purchase_history"] = df["visitor_hist_adr_usd"].isna().astype(int)
    df["visitor_hist_adr_usd"]    = df["visitor_hist_adr_usd"].fillna(0)
    df["visitor_hist_starrating"]  = df["visitor_hist_starrating"].fillna(0)
    log_lines.append(
        f"Rule 9 (impute visitor history)    : "
        f"{df['no_purchase_history'].sum():>8,} rows flagged as no_purchase_history"
    )

    # ── Rule 10: impute srch_query_affinity_score → median ───────────────────
    med_affinity = df["srch_query_affinity_score"].median()
    df["srch_query_affinity_score"] = df["srch_query_affinity_score"].fillna(med_affinity)
    log_lines.append(
        f"Rule 10 (impute affinity score)    : filled with median={med_affinity:.3f}"
    )

    # ── Rule 11: impute orig_destination_distance → median + flag ────────────
    df["dist_missing"] = df["orig_destination_distance"].isna().astype(int)
    med_dist = df["orig_destination_distance"].median()
    df["orig_destination_distance"] = df["orig_destination_distance"].fillna(med_dist)
    log_lines.append(
        f"Rule 11 (impute distance)          : "
        f"{df['dist_missing'].sum():>8,} rows flagged as dist_missing"
    )

    # ── Rule 12: winsorise price_usd at 99th percentile ──────────────────────
    p99 = np.percentile(df["price_usd"], 99)
    before_max = df["price_usd"].max()
    df["price_usd"] = winsorise(df["price_usd"])
    log_lines.append(
        f"Rule 12 (winsorise price_usd)      : capped at ${p99:.0f} (was ${before_max:.0f})"
    )

    # ── Rule 13: drop searches with no within-search outcome variation ────────
    before_srch = df["srch_id"].nunique()
    booking_var = df.groupby("srch_id")["booking_bool"].std()
    valid_srch  = booking_var[booking_var > 0].index
    df = df[df["srch_id"].isin(valid_srch)]
    log_lines.append(
        f"Rule 13 (drop no-variation srch)   : "
        f"-{before_srch - df['srch_id'].nunique():>6,} searches"
    )

    log_lines.append(f"\nAfter cleaning  : {len(df):>10,} rows")
    log_lines.append(f"Searches left   : {df['srch_id'].nunique():>10,}")

    # ── Rule 14: subsample 50k searches ──────────────────────────────────────
    rng = np.random.default_rng(SEED)
    all_srch = df["srch_id"].unique()

    if len(all_srch) < N_SAMPLE:
        print(f"Warning: only {len(all_srch):,} searches available; using all.")
        sampled = all_srch
    else:
        sampled = rng.choice(all_srch, size=N_SAMPLE, replace=False)

    df = df[df["srch_id"].isin(sampled)].copy()
    log_lines.append(f"\nSampled searches: {df['srch_id'].nunique():>10,}")
    log_lines.append(f"Final rows      : {len(df):>10,}")

    # ── dtype cleanup ─────────────────────────────────────────────────────────
    bool_cols = ["random_bool", "booking_bool", "click_bool",
                 "prop_brand_bool", "promotion_flag", "srch_saturday_night_bool",
                 "no_purchase_history", "dist_missing"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # ── quick sanity checks ───────────────────────────────────────────────────
    log_lines.append("\n── Booking rate by condition ──")
    rates = df.groupby("random_bool")[["booking_bool", "click_bool"]].mean().round(4)
    log_lines.append(rates.to_string())

    log_lines.append("\n── random_bool distribution (searches) ──")
    dist = df.groupby("random_bool")["srch_id"].nunique()
    log_lines.append(dist.to_string())

    # ── save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUT_CSV, index=False)
    log_lines.append(f"\nSaved to: {OUT_CSV}")
    log_lines.append("=" * 60)

    log(log_lines, LOG_PATH)
    print(f"\nCleaning log saved to: {LOG_PATH}")

if __name__ == "__main__":
    clean()
