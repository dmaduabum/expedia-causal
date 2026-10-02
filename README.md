# expedia-causal

How hotel placement — a hotel's own position and the competitors placed around
it — shapes clicks and bookings, using Expedia's randomized ordering experiment.

## Research questions

1. **Own position.** How much does a hotel lose by being shown lower on the page?
2. **Competitor interference.** Holding a hotel's own position fixed, does it
   matter which rivals were placed above it (higher-rated, better-reviewed,
   cheaper)?
3. **Heterogeneity.** Which hotels depend most on placement (chains vs.
   independents, star rating, reviews, price)?

## Design

The Expedia Personalized Sort data (Kaggle / ICDM 2013) records, for each
search, whether the hotels were shown in Expedia's algorithmic order or in a
**random order** (`random_bool`).

**Only the random arm is analysed.** Expedia released only searches with at
least one click and over-sampled searches ending in a booking (Ursu 2018,
*Marketing Science*, Sec. 3.3). The two arms are therefore differently selected
samples: search characteristics fixed before results are shown (length of stay,
booking window, purchase history) differ sharply between arms
(`results/tables/sampling_check.tex`). Algorithmic and random searches are
not compared.

Within the random arm, the order of hotels on each page was randomized, so
comparisons between hotels **within the same search** are causal:

- **Position effects:** outcome on rank, search fixed effects.
- **Interference:** for each hotel, count the stronger rivals placed above it
  (`A`). Under random ordering its expectation given the hotel's rank is known
  (`mu`, hypergeometric), so controlling for `mu` gives a design-based
  estimate (recentered exposure; Borusyak & Hull 2023; Aronow & Samii 2017).
- All standard errors are clustered by search.

A randomization check (`randomization_check.tex`) tests that hotel
characteristics are unrelated to rank within the random arm.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download `expedia-personalized-sort.zip` from
https://www.kaggle.com/c/expedia-personalized-sort/data and place it at
`data/raw/expedia-personalized-sort.zip`. Data files are not tracked.

## Run

```bash
make            # everything
make checks     # sampling + randomization checks
make position   # own-position effects
make interference
make hetero
```

## Structure

```
src/
  utils.py                     paths, OLS with HC2 / clustered SEs, LaTeX tables
  pipeline/extract.py          unzip train.csv
  pipeline/clean_sample.py     searches.csv (all searches), random_arm.pkl
  pipeline/build_exposures.py  competitor exposures (A, K, mu)
  analysis/sampling_check.py   why the arms are not compared
  analysis/randomization_check.py
  causal/position_effects.py
  causal/interference.py
  causal/heterogeneity.py
legacy/                        April 2026 class-project version (superseded)
results/tables/  results/figures/
```

## Cleaning

No observations are dropped on the basis of outcomes, and no hotels are dropped
from within a search. Missing covariates (unknown star rating, missing review
score) are flagged and kept. Positions 5, 11, 17 and 23 are reserved slots that
rarely hold a listing; the analysis uses each hotel's rank among the listed
hotels.
