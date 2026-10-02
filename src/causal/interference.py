"""
src/causal/interference.py
---------------------------
Competitor interference within the random arm.

Question: holding a hotel's own position fixed, does its click / booking
probability depend on WHICH rivals were randomly placed above it?

For each relation (see build_exposures.py), A_i counts stronger rivals above
hotel i and K_i counts stronger rivals anywhere on the page. With K_i held
fixed, a one-unit increase in A_i means one stronger rival moved from below i
to above i. Under random ordering, E[A_i | rank, K, N] = mu_i, so we estimate

    y_i = beta * A_i + gamma * mu_i + delta * K_i + f(rank_i) + X_i'theta
          + search FE + e_i

where beta is identified by the randomization (Borusyak & Hull 2023,
"recentered" exposure). f(rank) = rank + log(rank). X_i are the hotel's own
characteristics (precision only). SEs clustered by search.

Outputs:
  results/tables/interference.csv / .tex
Usage: python src/causal/interference.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import EXPOSURES, TAB_DIR, ols, write_tex_table, nfmt  # noqa: E402

RELATIONS = [
    ("higher_star", "Higher-star rival above"),
    ("higher_rev", "Better-reviewed rival above"),
    ("cheaper", "Cheaper rival ($\\geq$5\\%) above"),
    ("dominant", "Dominant rival above (as many stars, cheaper)"),
]
OUTCOMES = [("click_bool", "Click"), ("booking_bool", "Booking")]


def own_controls(d):
    star = d["prop_starrating"].astype(float)
    rev = d["prop_review_score"].astype(float).fillna(0)
    loc = d["prop_location_score1"].astype(float).fillna(0)
    return np.column_stack([
        star, d["star_unknown"], rev, d["review_missing"], d["review_none"],
        d["log_price_rel"], d["prop_brand_bool"], loc, d["promotion_flag"],
    ])


def main():
    d = pd.read_pickle(EXPOSURES)
    Xown = own_controls(d)
    rank = d["rank"].values.astype(float)
    base = np.column_stack([rank, np.log(rank)])
    g = d["srch_id"].values

    res, csv = {}, []
    for rel, lab in RELATIONS:
        X = np.column_stack([d[f"{rel}_A"], d[f"{rel}_mu"], d[f"{rel}_K"], base, Xown])
        names = ["A", "mu", "K", "rank", "logrank"] + [f"own{j}" for j in range(Xown.shape[1])]
        for y, ylab in OUTCOMES:
            r = ols(d[y].values, X, names, se="cluster", clusters=g, absorb=g)
            e = r.set_index("term").loc["A"]
            res[(rel, ylab)] = (e.estimate, e.se, e.p)
            csv.append({"relation": rel, "outcome": ylab, "estimate": e.estimate,
                        "se": e.se, "p": e.p, "mean_y": d[y].mean(),
                        "rel_to_mean": e.estimate / d[y].mean(),
                        "mean_K": d[f"{rel}_K"].mean(), "n": r.attrs["n"],
                        "searches": r.attrs["G"]})
            print(f"{rel:12s} {ylab:8s} beta = {e.estimate:+.5f} (se {e.se:.5f}) "
                  f"= {100 * e.estimate / d[y].mean():+.1f}% of mean")
    # ── joint model: all rival types together (they overlap, e.g. cheaper
    #    rivals tend to have fewer stars) ──────────────────────────────────────
    Xj = np.column_stack(
        [d[f"{rel}_{v}"] for rel, _ in RELATIONS for v in ("A", "mu", "K")] + [base, Xown])
    names_j = ([f"{rel}_{v}" for rel, _ in RELATIONS for v in ("A", "mu", "K")]
               + ["rank", "logrank"] + [f"own{j}" for j in range(Xown.shape[1])])
    for y, ylab in OUTCOMES:
        r = ols(d[y].values, Xj, names_j, se="cluster", clusters=g, absorb=g).set_index("term")
        for rel, _ in RELATIONS:
            e = r.loc[f"{rel}_A"]
            res[(rel, ylab + " (joint)")] = (e.estimate, e.se, e.p)
            csv.append({"relation": rel, "outcome": ylab + " (joint)",
                        "estimate": e.estimate, "se": e.se, "p": e.p,
                        "mean_y": d[y].mean(), "rel_to_mean": e.estimate / d[y].mean(),
                        "mean_K": d[f"{rel}_K"].mean(), "n": len(d),
                        "searches": d.srch_id.nunique()})
            print(f"JOINT {rel:12s} {ylab:8s} beta = {e.estimate:+.5f} (se {e.se:.5f})")
    pd.DataFrame(csv).to_csv(os.path.join(TAB_DIR, "interference.csv"), index=False)

    colkeys = [ylab for _, ylab in OUTCOMES] + [ylab + " (joint)" for _, ylab in OUTCOMES]
    write_tex_table(
        os.path.join(TAB_DIR, "interference.tex"),
        columns=[f"({i + 1}) {k}" for i, k in enumerate(colkeys)],
        rows=[(lab, [res[(rel, k)] for k in colkeys]) for rel, lab in RELATIONS],
        caption="Competitor Interference: Effect of Stronger Rivals Placed Above a Hotel",
        label="tab:interference",
        notes=("Random arm. In columns (1)--(2) each cell is a separate regression for one "
               "rival type; columns (3)--(4) include all rival types jointly. The coefficient "
               "is the effect of one stronger rival being placed above (rather than below) the "
               "hotel, holding fixed the hotel's own rank and the number of such rivals on the "
               "page. All models control for the expected exposure under random ordering, own "
               "rank and log rank, own characteristics, and search fixed effects; standard "
               "errors clustered by search."),
        extra_rows=[("Mean outcome", [f"{d[y].mean():.4f}" for y, _ in OUTCOMES] * 2),
                    ("Observations", [nfmt(len(d))] * 4),
                    ("Searches", [nfmt(d.srch_id.nunique())] * 4)],
        digits=5,
    )
    print("Saved interference tables.")


if __name__ == "__main__":
    main()
