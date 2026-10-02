"""
src/causal/position_effects.py
-------------------------------
Effect of a hotel's position on clicks and bookings, random-order searches only.

Because hotels were randomly ordered, comparing hotels at different ranks
within the same search identifies the causal effect of position.

Produces every position result in the paper:

  Table 2 / Figure 1  Searches listing at least 25 hotels, so every rank is
                      computed from the same searches (list length otherwise
                      confounds comparisons across ranks: short lists give each
                      hotel a larger share of clicks).
                      - outcome rates by rank 1..25        position_means_balanced.csv
                      - within-search changes between      position_pairs.csv
                        ranks 1->5, 5->10, 1->10, 10->25
                      - figure                             figures/position_effects.pdf
  Table 3             All random searches, search fixed effects, SEs clustered
                      by search: click / booking on rank (per 10) and log rank,
                      with and without hotel controls, plus booking among
                      clicked hotels (descriptive: conditions on a click).
                                                           position_effects.csv / .tex

Usage: python src/causal/position_effects.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (RANDOM_ARM, TAB_DIR, FIG_DIR, ols,  # noqa: E402
                   write_tex_table, nfmt, own_controls)

MIN_LISTED = 25
PAIRS = [(1, 5), (5, 10), (1, 10), (10, 25)]


def fit(y, X, names, d):
    r = ols(y, X, names, se="cluster",
            clusters=d["srch_id"].values, absorb=d["srch_id"].values)
    e = r.iloc[0]
    return e.estimate, e.se, e.p, r.attrs["n"], r.attrs["G"]


def balanced(d):
    """Rates by rank and pairwise changes in searches listing >= MIN_LISTED hotels."""
    b = d.loc[(d.N >= MIN_LISTED) & (d["rank"] <= MIN_LISTED),
              ["srch_id", "rank", "click_bool", "booking_bool"]]
    m = b.groupby("rank").agg(n=("click_bool", "size"),
                              click=("click_bool", "mean"),
                              book=("booking_bool", "mean"))
    for v in ("click", "book"):
        se = np.sqrt(m[v] * (1 - m[v]) / m.n)
        m[f"{v}_lo"], m[f"{v}_hi"] = m[v] - 1.96 * se, m[v] + 1.96 * se
    m.reset_index().to_csv(os.path.join(TAB_DIR, "position_means_balanced.csv"), index=False)

    ranks = sorted({r for p in PAIRS for r in p})
    w = b[b["rank"].isin(ranks)].pivot(index="srch_id", columns="rank",
                                        values=["click_bool", "booking_bool"])
    rows = []
    for y, lab in (("click_bool", "Click"), ("booking_bool", "Booking")):
        for a, c in PAIRS:
            diff = 100 * (w[(y, c)] - w[(y, a)])
            start = 100 * w[(y, a)].mean()
            rows.append({"outcome": lab, "from_rank": a, "to_rank": c,
                         "change_pp": diff.mean(), "se": diff.std() / np.sqrt(len(diff)),
                         "start_rate_pct": start, "pct_of_start": 100 * diff.mean() / start,
                         "searches": len(diff)})
            print(f"{lab:8s} rank {a:>2} -> {c:>2}: {diff.mean():+.3f} pp "
                  f"({100 * diff.mean() / start:+.0f}% of {start:.2f}%)")
    pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "position_pairs.csv"), index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for a, v, t in ((ax[0], "click", "A. Click rate"), (ax[1], "book", "B. Booking rate")):
        a.fill_between(m.index, 100 * m[f"{v}_lo"], 100 * m[f"{v}_hi"], color="0.75", lw=0)
        a.plot(m.index, 100 * m[v], color="black", lw=1.2, marker="o", ms=2.2)
        a.set_title(t, loc="left", fontsize=10)
        a.set_xlabel("Rank on page (1 = top)")
        a.set_ylabel("Percent of listings")
        a.set_ylim(bottom=0)
        a.set_xlim(0.5, MIN_LISTED + 0.5)
        a.set_xticks([1, 5, 10, 15, 20, 25])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "position_effects.pdf"))


def regressions(d):
    clicked = d[d.click_bool == 1]
    Xc = own_controls(d)
    cn = [f"own{j}" for j in range(Xc.shape[1])]
    r10, lr = d["rank"].values / 10.0, np.log(d["rank"].values)
    specs = [
        ("Click", d, "click_bool", False), ("Click", d, "click_bool", True),
        ("Booking", d, "booking_bool", False), ("Booking", d, "booking_bool", True),
        ("Booking | click", clicked, "booking_bool", False),
    ]
    res, csv = {}, []
    for k, (lab, x, y, ctrl) in enumerate(specs):
        for form in ("rank10", "logrank"):
            if ctrl and form == "logrank":
                continue
            base = (x["rank"].values / 10.0) if form == "rank10" else np.log(x["rank"].values)
            if ctrl:
                X, names = np.column_stack([base, Xc]), [form] + cn
            else:
                X, names = base[:, None], [form]
            est, se, p, n, G = fit(x[y].values, X, names, x)
            res[(k, form)] = (100 * est, 100 * se, p)
            csv.append({"outcome": lab, "controls": ctrl, "regressor": form,
                        "estimate_pp": 100 * est, "se_pp": 100 * se, "p": p,
                        "mean_y_pct": 100 * x[y].mean(),
                        "pct_of_mean": 100 * est / x[y].mean(), "n": n, "searches": G})
            print(f"{lab:16s} controls={ctrl!s:5s} {form:8s} {100 * est:+.3f} pp "
                  f"(se {100 * se:.3f})")
    pd.DataFrame(csv).to_csv(os.path.join(TAB_DIR, "position_effects.csv"), index=False)

    write_tex_table(
        os.path.join(TAB_DIR, "position_effects.tex"),
        columns=[f"({k + 1}) {s[0]}" for k, s in enumerate(specs)],
        rows=[("Rank, per 10 positions", [res.get((k, "rank10")) for k in range(len(specs))]),
              ("Log rank", [res.get((k, "logrank")) for k in range(len(specs))])],
        caption="Effect of Position on Clicks and Bookings",
        label="tab:position",
        notes=("Coefficients in percentage points. All specifications include search fixed "
               "effects; standard errors clustered by search. Columns (2) and (4) add hotel "
               "characteristics. Column (5) is restricted to clicked hotels and is descriptive."),
        extra_rows=[("Hotel characteristics", ["Yes" if s[3] else "No" for s in specs]),
                    ("Listings", [nfmt(len(s[1])) for s in specs]),
                    ("Searches", [nfmt(s[1].srch_id.nunique()) for s in specs])],
        digits=3,
    )


def main():
    d = pd.read_pickle(RANDOM_ARM)
    balanced(d)
    regressions(d)
    print("Saved position tables and figure.")


if __name__ == "__main__":
    main()
