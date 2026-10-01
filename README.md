# expedia-causal

Causal inference project for STATS 571/671 (Causal Inference, University of Michigan).

**Research question:** Does Expedia's algorithmic hotel ranking causally increase
booking and click-through rates relative to a random ordering? Which channel —
attention (clicks) or conversion (bookings conditional on click) — drives the effect?

**Identification:** The Expedia Personalized Sort dataset (Kaggle ICDM 2013) contains
`random_bool`, a randomised indicator that assigns ~40% of searches to receive
randomly-ordered hotel results rather than Expedia's proprietary algorithm. This
provides clean causal identification without requiring strong ignorability.

**Methods:** ITT via DiM + ANCOVA (Lec 2–5); LATE via IV/Wald estimator (Lec 21–22).

---

## Setup

```bash
# 1. Python dependencies
pip install -r requirements.txt

# 2. R dependencies
Rscript packages.R
```

## Data

Place the raw Kaggle zip at:
```
data/raw/expedia-personalized-sort.zip
```
Download from: https://www.kaggle.com/c/expedia-personalized-sort/data

The zip and all processed files are gitignored. Only source code is tracked.

## Run

```bash
make          # full pipeline: extract → clean → balance → ITT → IV → robustness
make extract  # unzip train.csv only
make clean_data  # extract + clean/sample
make balance  # balance table + Love plot
make itt      # ITT: DiM + ANCOVA
make iv       # IV: Wald + iv_robust
make robustness  # all robustness checks
make purge    # delete all generated files (keeps raw zip)
```

## Structure

```
expedia-causal/
├── data/
│   ├── raw/            ← expedia-personalized-sort.zip (gitignored)
│   └── processed/      ← train_clean.csv, generated (gitignored)
├── src/
│   ├── pipeline/
│   │   ├── extract.py          ← unzips train.csv
│   │   └── clean_sample.py     ← 14 cleaning rules → 50k search sample
│   ├── analysis/
│   │   ├── descriptive.py      ← summary stats, rates by condition
│   │   └── balance.R           ← SMD table + Love plot
│   └── causal/
│       ├── itt_estimator.R     ← DiM + ANCOVA for booking & click
│       ├── iv_estimator.R      ← Wald IV + iv_robust, first stage F
│       └── robustness.R        ← placebo, subsamples, same-position, monotonicity
├── results/
│   ├── figures/        ← PDFs (gitignored)
│   ├── tables/         ← CSVs (gitignored)
│   └── estimates/      ← RDS files (gitignored)
├── paper/
│   └── main.tex
├── Makefile
├── packages.R
└── requirements.txt
```

## Cleaning Rules (clean_sample.py)

| Rule | Description |
|---|---|
| 1 | Keep only analysis columns |
| 2 | Drop if `random_bool` missing |
| 3 | Drop if `booking_bool` or `click_bool` missing |
| 4 | Drop if `position` missing or ≤ 0 |
| 5 | Drop if `price_usd` missing, zero, or negative |
| 6 | Drop if `prop_starrating` == 0 (unknown) |
| 7 | Drop if `prop_review_score` missing |
| 8 | Drop if `prop_location_score1` missing |
| 9 | Impute visitor purchase history → 0, add `no_purchase_history` flag |
| 10 | Impute `srch_query_affinity_score` → column median |
| 11 | Impute `orig_destination_distance` → median, add `dist_missing` flag |
| 12 | Winsorise `price_usd` at 99th percentile |
| 13 | Drop searches with no within-search booking variation |
| 14 | Sample 50,000 searches at random (seed = 571) |
