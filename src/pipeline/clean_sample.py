"""
src/pipeline/clean_sample.py
-----------------------------
Builds the two analysis files from data/raw/train.csv.

Design notes (why this differs from the April version):
  * No rows are dropped on the basis of outcomes. The old Rule 13 kept only
    searches with a booking, which conditioned on the outcome.
  * No rows are dropped *within* a search. Dropping individual hotels
    (e.g. unrated ones) breaks the page structure that the position and
    competitor analyses rely on. Missing covariates are flagged instead.
  * The causal analysis uses only the random-ordering arm. Expedia released
    a sample in which searches ending in a booking were over-sampled, at a
    rate that differs from the random arm (Ursu 2018, Sec. 3.3), so the two
    arms are not comparable populations. Within the random arm, the order of
    hotels on each page was randomised, which is what we use.

Outputs
  data/processed/searches.csv      one row per search, ALL searches
                                   (used only to document the sampling issue)
  data/processed/random_arm.pkl    one row per (search, hotel), random arm only
  results/tables/cleaning_log.txt

Usage:  python src/pipeline/clean_sample.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import RAW_CSV, SEARCHES, RANDOM_ARM, TAB_DIR  # noqa: E402

CHUNK = 1_000_000
RESERVED_SLOTS = {5, 11, 17, 23}  # page slots that almost never hold a listing

SEARCH_COLS = [
    "srch_id", "date_time", "site_id", "visitor_location_country_id",
    "prop_country_id", "srch_destination_id", "srch_length_of_stay",
    "srch_booking_window", "srch_adults_count", "srch_children_count",
    "srch_room_count", "srch_saturday_night_bool", "visitor_hist_adr_usd",
    "random_bool",
]
PAIR_COLS = [
    "srch_id", "prop_id", "position", "click_bool", "booking_bool",
    "prop_starrating", "prop_review_score", "prop_brand_bool",
    "prop_location_score1", "prop_log_historical_price", "price_usd",
    "promotion_flag",
]
DTYPES = {
    "srch_id": "int32", "prop_id": "int32", "position": "int16",
    "click_bool": "int8", "booking_bool": "int8", "random_bool": "int8",
    "prop_starrating": "int8", "prop_brand_bool": "int8",
    "promotion_flag": "int8", "price_usd": "float64",
    "prop_review_score": "float32", "prop_location_score1": "float32",
    "prop_log_historical_price": "float32",
}


def main():
    log = ["=" * 64, "CLEANING LOG - expedia-causal (random-arm design)", "=" * 64]
    search_parts, pair_parts = [], []
    n_rows = 0

    cols = sorted(set(SEARCH_COLS) | set(PAIR_COLS))
    reader = pd.read_csv(RAW_CSV, usecols=cols, na_values="NULL",
                         dtype=DTYPES, chunksize=CHUNK)
    for i, ch in enumerate(reader):
        n_rows += len(ch)
        # search level: per-chunk aggregates (a search can straddle chunks;
        # combined again below)
        g = ch.groupby("srch_id", sort=False)
        first = g[[c for c in SEARCH_COLS if c != "srch_id"]].first()
        agg = g.agg(n_listings=("position", "size"),
                    n_clicks=("click_bool", "sum"),
                    booked=("booking_bool", "max"))
        search_parts.append(first.join(agg))
        # pair level: random arm only
        pair_parts.append(ch.loc[ch["random_bool"] == 1, PAIR_COLS].copy())
        print(f"  chunk {i + 1}: {n_rows:,} rows read", flush=True)

    # ── search-level file (all searches) ─────────────────────────────────────
    s = pd.concat(search_parts)
    s = s.groupby(level=0).agg({
        **{c: "first" for c in SEARCH_COLS if c != "srch_id"},
        "n_listings": "sum", "n_clicks": "sum", "booked": "max"})
    s["month"] = s["date_time"].str[:7]
    s["has_purchase_history"] = s["visitor_hist_adr_usd"].notna().astype(int)
    s["domestic"] = (s["visitor_location_country_id"] ==
                     s["prop_country_id"]).astype(int)
    s = s.drop(columns=["date_time", "visitor_hist_adr_usd"])
    s.reset_index().to_csv(SEARCHES, index=False)

    log.append(f"\nRaw rows                 : {n_rows:>12,}")
    log.append(f"Raw searches             : {len(s):>12,}")
    log.append(f"  algorithmic (random=0) : {(s.random_bool == 0).sum():>12,}")
    log.append(f"  random      (random=1) : {(s.random_bool == 1).sum():>12,}")
    log.append(f"Searches with >= 1 click : {(s.n_clicks > 0).mean():>12.3f}  "
               "(Expedia released only clicked searches)")

    # ── random-arm pair file ─────────────────────────────────────────────────
    d = pd.concat(pair_parts, ignore_index=True)
    d = d.sort_values(["srch_id", "position"]).reset_index(drop=True)
    g = d.groupby("srch_id")
    d["N"] = g["position"].transform("size").astype("int16")
    d["rank"] = (g.cumcount() + 1).astype("int16")      # order among listed hotels
    d["reserved_slot"] = d["position"].isin(RESERVED_SLOTS).astype("int8")

    # covariate flags (rows are kept; nothing is dropped within a search)
    d["star_unknown"] = (d["prop_starrating"] == 0).astype("int8")
    d["review_missing"] = d["prop_review_score"].isna().astype("int8")
    d["review_none"] = (d["prop_review_score"] == 0).astype("int8")
    d["log_price"] = np.log(d["price_usd"].clip(lower=1)).astype("float32")
    d["log_price_rel"] = (d["log_price"] -
                          g["log_price"].transform("median")).astype("float32")
    d["price_usd"] = d["price_usd"].astype("float32")

    log.append(f"\nRandom-arm rows          : {len(d):>12,}")
    log.append(f"Random-arm searches      : {d.srch_id.nunique():>12,}")
    log.append(f"  rows in reserved slots : {d.reserved_slot.sum():>12,} "
               f"(positions {sorted(RESERVED_SLOTS)}; kept, flagged)")
    log.append(f"  unknown star rating    : {d.star_unknown.sum():>12,} (kept, flagged)")
    log.append(f"  missing review score   : {d.review_missing.sum():>12,} (kept, flagged)")
    log.append(f"  click rate (pair)      : {d.click_bool.mean():>12.4f}")
    log.append(f"  booking rate (pair)    : {d.booking_bool.mean():>12.4f}")
    log.append(f"  searches with booking  : "
               f"{d.groupby('srch_id').booking_bool.max().mean():>12.4f}")
    d.to_pickle(RANDOM_ARM)

    log.append(f"\nSaved: {SEARCHES}\nSaved: {RANDOM_ARM}\n" + "=" * 64)
    with open(os.path.join(TAB_DIR, "cleaning_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
