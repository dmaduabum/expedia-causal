"""
src/causal/heterogeneity.py
----------------------------
Who depends most on placement? Random arm only.

For each split (chain vs. independent, star rating, review score, price
relative to the search), estimate the within-search slope of clicks and
bookings on rank separately by group, and the difference:

    y = b1 * rank10 + b2 * rank10 x G + c * G + search FE + e

Because baseline rates differ across groups, the table also reports each
slope relative to the group's mean outcome (percent change per 10 positions).

Outputs:
  results/tables/heterogeneity.csv / .tex
Usage: python src/causal/heterogeneity.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import RANDOM_ARM, TAB_DIR, ols, lincom, stars, nfmt  # noqa: E402

SPLITS = [
    ("Chain vs.\\ independent", lambda d: d.prop_brand_bool == 1, None,
     "Chain", "Independent"),
    ("4--5 stars vs.\\ 1--3 stars", lambda d: d.prop_starrating >= 4,
     lambda d: d.star_unknown == 0, "4--5 stars", "1--3 stars"),
    ("Review $\\geq$ 4.5 vs.\\ below", lambda d: d.prop_review_score >= 4.5,
     lambda d: (d.review_missing == 0) & (d.review_none == 0), "Review $\\geq$ 4.5", "Review $<$ 4.5"),
    ("Cheaper vs.\\ pricier than search median", lambda d: d.log_price_rel < 0, None,
     "Below median price", "Above median price"),
]
OUTCOMES = [("click_bool", "Click"), ("booking_bool", "Booking")]


def main():
    d = pd.read_pickle(RANDOM_ARM)
    csv, lines = [], []
    for title, grp, keep, g1, g0 in SPLITS:
        x = d if keep is None else d[keep(d)]
        G = grp(x).astype(float).values
        r10 = x["rank"].values / 10.0
        X = np.column_stack([r10, r10 * G, G])
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\")
        cells = {}
        for y, ylab in OUTCOMES:
            res = ols(x[y].values, X, ["rank10", "rank10xG", "G"], se="cluster",
                      clusters=x.srch_id.values, absorb=x.srch_id.values)
            s0 = lincom(res, {"rank10": 1})
            s1 = lincom(res, {"rank10": 1, "rank10xG": 1})
            dif = res.set_index("term").loc["rank10xG"]
            m1, m0 = x[y][G == 1].mean(), x[y][G == 0].mean()
            cells[ylab] = (s1, s0, dif, m1, m0)
            csv.append({"split": title, "outcome": ylab,
                        "slope_g1": s1[0], "se_g1": s1[1], "mean_g1": m1,
                        "slope_g0": s0[0], "se_g0": s0[1], "mean_g0": m0,
                        "diff": dif.estimate, "se_diff": dif.se, "p_diff": dif.p,
                        "rel_g1": s1[0] / m1, "rel_g0": s0[0] / m0,
                        "n": len(x)})
        for lab, k, mi in ((g1, 0, 3), (g0, 1, 4)):
            row = [lab]
            for _, ylab in OUTCOMES:
                (est, se), mean = cells[ylab][k], cells[ylab][mi]
                row += [f"{est:+.4f} ({se:.4f})", f"{100 * est / mean:+.1f}\\%"]
            lines.append(" & ".join(row) + " \\\\")
        row = ["Difference"]
        for _, ylab in OUTCOMES:
            dif = cells[ylab][2]
            row += [f"{dif.estimate:+.4f}{stars(dif.p)} ({dif.se:.4f})", ""]
        lines.append(" & ".join(row) + " \\\\[0.4em]")

    t = pd.DataFrame(csv)
    t.to_csv(os.path.join(TAB_DIR, "heterogeneity.csv"), index=False)
    print(t[["split", "outcome", "slope_g1", "slope_g0", "diff", "se_diff",
             "rel_g1", "rel_g0"]].round(4).to_string(index=False))

    tex = [
        "\\begin{table}[htbp]\\centering",
        "\\caption{Who Depends Most on Placement? Rank Slopes by Hotel Type (Random Arm)}",
        "\\label{tab:heterogeneity}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        " & \\multicolumn{2}{c}{Click} & \\multicolumn{2}{c}{Booking} \\\\",
        " & Slope (SE) & \\% of mean & Slope (SE) & \\% of mean \\\\", "\\midrule",
    ] + lines + [
        "\\bottomrule", "\\end{tabular}",
        "\\begin{minipage}{0.92\\textwidth}\\footnotesize\\vspace{0.3em}",
        "\\textit{Notes:} Slopes are the change in click or booking probability per 10 "
        "positions lower on the page, from regressions with search fixed effects and "
        "standard errors clustered by search. ``\\% of mean'' divides the slope by the "
        "group's mean outcome. Difference tests whether the slopes differ across groups. "
        "$^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.",
        "\\end{minipage}", "\\end{table}",
    ]
    with open(os.path.join(TAB_DIR, "heterogeneity.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
