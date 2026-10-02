"""
src/causal/position_effects.py
-------------------------------
Effect of a hotel's own position on clicks and bookings, random arm only.

Because hotels were randomly ordered, comparing hotels at different ranks
within the same search identifies the causal effect of position. All models
include search fixed effects and cluster standard errors by search.

  (1) click   ~ rank
  (2) booking ~ rank
  (3) booking ~ rank, among clicked hotels   [descriptive: conditions on a
      post-treatment outcome; mirrors Ursu's search-vs-purchase distinction]
Each is estimated with rank in units of 10 positions, and with log(rank).

Outputs:
  results/tables/position_effects.csv / .tex
  results/tables/position_means.csv      outcome means by rank with 95% CIs
  results/figures/position_effects.pdf
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
                   write_tex_table, nfmt)


def fit(y, x, d, name):
    r = ols(y, x[:, None], [name], se="cluster",
            clusters=d["srch_id"].values, absorb=d["srch_id"].values)
    e = r.iloc[0]
    return e.estimate, e.se, e.p, r.attrs["n"], r.attrs["G"]


def main():
    d = pd.read_pickle(RANDOM_ARM)
    clicked = d[d.click_bool == 1]

    specs = [("Click", d, "click_bool"), ("Booking", d, "booking_bool"),
             ("Booking | click", clicked, "booking_bool")]
    res, csv = {}, []
    for lab, x, y in specs:
        for form, xv in (("rank10", x["rank"].values / 10.0),
                         ("logrank", np.log(x["rank"].values))):
            est, se, p, n, G = fit(x[y].values, xv, x, form)
            res[(lab, form)] = (est, se, p)
            csv.append({"outcome": lab, "regressor": form, "estimate": est,
                        "se": se, "p": p, "mean_y": x[y].mean(),
                        "rel_to_mean": est / x[y].mean(), "n": n, "searches": G})
            print(f"{lab:16s} {form:8s} {est:+.5f} (se {se:.5f})  mean y = {x[y].mean():.4f}")
    pd.DataFrame(csv).to_csv(os.path.join(TAB_DIR, "position_effects.csv"), index=False)

    cols = [lab for lab, _, _ in specs]
    write_tex_table(
        os.path.join(TAB_DIR, "position_effects.tex"),
        columns=[f"({i + 1}) {c}" for i, c in enumerate(cols)],
        rows=[("Rank (per 10 positions)", [res[(c, "rank10")] for c in cols]),
              ("Log rank", [res[(c, "logrank")] for c in cols])],
        caption="Effect of Own Position on Clicks and Bookings (Random Arm)",
        label="tab:position",
        notes=("Each cell is a separate regression with search fixed effects; standard "
               "errors clustered by search. Rank is the hotel's order among listed hotels "
               "(1 = top). Column (3) conditions on a click, an outcome of position, and "
               "is descriptive."),
        extra_rows=[("Mean outcome", [f"{x[y].mean():.4f}" for _, x, y in specs]),
                    ("Observations", [nfmt(len(x)) for _, x, _ in specs]),
                    ("Searches", [nfmt(x.srch_id.nunique()) for _, x, _ in specs])],
    )

    # ── means by rank (one obs per search per rank, so SEs are i.i.d.) ──────
    m = d.groupby("rank").agg(n=("click_bool", "size"),
                              click=("click_bool", "mean"),
                              book=("booking_bool", "mean"))
    m = m[m.n >= 500]
    for v in ("click", "book"):
        se = np.sqrt(m[v] * (1 - m[v]) / m.n)
        m[f"{v}_lo"], m[f"{v}_hi"] = m[v] - 1.96 * se, m[v] + 1.96 * se
    m.reset_index().to_csv(os.path.join(TAB_DIR, "position_means.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for a, v, lab in ((ax[0], "click", "Click probability"),
                      (ax[1], "book", "Booking probability")):
        a.fill_between(m.index, m[f"{v}_lo"], m[f"{v}_hi"], color="#2c7bb6", alpha=0.25, lw=0)
        a.plot(m.index, m[v], color="#2c7bb6", lw=1.6)
        a.set_title(lab, fontsize=11)
        a.set_xlabel("Rank on page (1 = top)")
        a.grid(alpha=0.3)
        a.set_ylim(bottom=0)
    fig.suptitle("Random arm: outcome by randomly assigned rank (95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "position_effects.pdf"))
    print("Saved position tables and figure.")


if __name__ == "__main__":
    main()
