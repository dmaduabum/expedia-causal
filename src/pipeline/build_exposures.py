"""
src/pipeline/build_exposures.py
--------------------------------
Competitor-exposure variables for the random arm.

For hotel i in search s, and a relation "j is a stronger competitor than i":
    K_i  = number of other hotels in the search that are stronger than i
    A_i  = number of those placed ABOVE i on the page
    B_i  = K_i - A_i  (placed below i)

Under random ordering, given i's rank r_i among the N listed hotels, the
r_i - 1 slots above i are a uniformly random subset of the other N - 1 hotels,
so A_i is hypergeometric with expectation
    mu_i = (r_i - 1) * K_i / (N - 1).
The recentered exposure A_i - mu_i is mean-zero by design. Controlling for
mu_i (Borusyak & Hull 2023) makes the coefficient on A_i a design-based
estimate of the effect of having one more stronger rival above you, holding
your own position fixed.

Relations (j vs i, both values observed):
    higher_star : j has strictly more stars
    higher_rev  : j has a strictly higher review score (score 0 = no reviews,
                  treated as unobserved)
    cheaper     : j is at least 5% cheaper
    dominant    : j has at least as many stars AND is at least 5% cheaper

Output: data/processed/random_arm_exposures.pkl
Usage:  python src/pipeline/build_exposures.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import RANDOM_ARM, EXPOSURES  # noqa: E402

CHEAPER_GAP = np.log(1 / 0.95)   # j at least 5% cheaper than i
CHUNK_SEARCHES = 10_000
RELATIONS = ["higher_star", "higher_rev", "cheaper", "dominant"]


def relation_matrices(star, rev, lp, valid):
    """
    Arrays are (C, L) with padding where valid == False.
    Returns dict name -> bool array R[c, j, i] = 'j stronger than i'.
    """
    sj, si = star[:, :, None], star[:, None, :]
    rj, ri = rev[:, :, None], rev[:, None, :]
    pj, pi = lp[:, :, None], lp[:, None, :]
    both = valid[:, :, None] & valid[:, None, :]
    L = star.shape[1]
    both &= ~np.eye(L, dtype=bool)[None]
    star_ok = (sj > 0) & (si > 0)
    rev_ok = np.isfinite(rj) & np.isfinite(ri)
    cheaper = both & (pj < pi - CHEAPER_GAP)
    return {
        "higher_star": both & star_ok & (sj > si),
        "higher_rev": both & rev_ok & (rj > ri),
        "cheaper": cheaper,
        "dominant": cheaper & star_ok & (sj >= si),
    }


def main():
    d = pd.read_pickle(RANDOM_ARM)
    d = d.sort_values(["srch_id", "rank"]).reset_index(drop=True)
    rev = d["prop_review_score"].astype(float).where(d["prop_review_score"] > 0)

    sid = d["srch_id"].values
    starts = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
    sizes = np.diff(np.r_[starts, len(d)])
    L = int(sizes.max())
    out = {f"{k}_{v}": np.zeros(len(d), dtype=np.float32)
           for k in RELATIONS for v in ("K", "A", "B")}

    star_all = d["prop_starrating"].values.astype(float)
    rev_all = rev.values
    lp_all = d["log_price"].values.astype(float)

    for c0 in range(0, len(starts), CHUNK_SEARCHES):
        st = starts[c0:c0 + CHUNK_SEARCHES]
        sz = sizes[c0:c0 + CHUNK_SEARCHES]
        C = len(st)
        col = np.arange(L)[None, :]
        valid = col < sz[:, None]
        rows = np.where(valid, st[:, None] + col, 0)
        star = np.where(valid, star_all[rows], 0)
        rv = np.where(valid, rev_all[rows], np.nan)
        lp = np.where(valid, lp_all[rows], 0)
        R = relation_matrices(star, rv, lp, valid)
        above = np.triu(np.ones((L, L), dtype=bool), k=1)[None]  # j < i
        flat = rows[valid]
        for name, M in R.items():
            K = M.sum(axis=1)                 # over j -> (C, L) indexed by i
            A = (M & above).sum(axis=1)
            out[f"{name}_K"][flat] = K[valid]
            out[f"{name}_A"][flat] = A[valid]
            out[f"{name}_B"][flat] = (K - A)[valid]
        print(f"  searches {c0 + C:,}/{len(starts):,}", flush=True)

    for k, v in out.items():
        d[k] = v
    denom = (d["N"] - 1).clip(lower=1).astype(float)
    for name in RELATIONS:
        d[f"{name}_mu"] = ((d["rank"] - 1) * d[f"{name}_K"] / denom).astype("float32")
        d[f"{name}_Ac"] = (d[f"{name}_A"] - d[f"{name}_mu"]).astype("float32")

    d.to_pickle(EXPOSURES)
    print("\nMean exposures (A = stronger rivals above, mu = expected):")
    print(d[[f"{r}_{v}" for r in RELATIONS for v in ("K", "A", "mu", "Ac")]]
          .mean().round(3).to_string())
    print(f"\nSaved: {EXPOSURES}")


if __name__ == "__main__":
    main()
