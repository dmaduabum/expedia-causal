"""
src/analysis/randomization_check.py
------------------------------------
Within the random arm, is a hotel's rank on the page unrelated to its
characteristics? Under random ordering it should be. Because Expedia released
only searches with at least one click, a search whose random draw put an
attractive hotel near the top could be more likely to appear in the data; this
check shows whether that selection is visible.

For each characteristic X: regress X on rank with search fixed effects,
clustered by search. Slopes should be close to zero.

Outputs:
  results/tables/randomization_check.csv
  results/tables/randomization_check.tex
Usage: python src/analysis/randomization_check.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import RANDOM_ARM, TAB_DIR, ols, stars, nfmt  # noqa: E402

CHARS = [
    ("prop_starrating", "Star rating", lambda d: d.star_unknown == 0),
    ("prop_review_score", "Review score", lambda d: d.review_missing == 0),
    ("log_price_rel", "Log price (rel.\\ to search median)", None),
    ("prop_brand_bool", "Chain hotel", None),
    ("prop_location_score1", "Location score", lambda d: d.prop_location_score1.notna()),
    ("promotion_flag", "On promotion", None),
]


def main():
    d = pd.read_pickle(RANDOM_ARM)
    rows = []
    for v, lab, keep in CHARS:
        x = d if keep is None else d[keep(d)]
        # slope per 10 ranks, so magnitudes are readable
        r = ols(x[v].values, (x["rank"].values / 10.0)[:, None], ["rank10"],
                se="cluster", clusters=x["srch_id"].values,
                absorb=x["srch_id"].values)
        e = r.iloc[0]
        rows.append({"characteristic": lab, "mean": x[v].mean(), "sd": x[v].std(),
                     "slope_per_10_ranks": e.estimate, "se": e.se, "p": e.p,
                     "slope_in_sd": e.estimate / x[v].std(), "n": len(x)})
        print(f"{lab:40s} slope/10 ranks = {e.estimate:+.4f} (se {e.se:.4f})")
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_DIR, "randomization_check.csv"), index=False)

    lines = [
        "\\begin{table}[htbp]\\centering",
        "\\caption{Randomization Check: Hotel Characteristics and Page Rank (Random Arm)}",
        "\\label{tab:randcheck}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        " & Mean & Slope per 10 ranks & Slope (SD units) & N \\\\", "\\midrule",
    ]
    for _, r in t.iterrows():
        lines.append(f"{r.characteristic} & {r['mean']:.3f} & "
                     f"{r.slope_per_10_ranks:+.4f}{stars(r.p)} ({r.se:.4f}) & "
                     f"{r.slope_in_sd:+.3f} & {nfmt(r.n)} \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\begin{minipage}{0.92\\textwidth}\\footnotesize\\vspace{0.3em}",
        "\\textit{Notes:} Each row regresses the characteristic on the hotel's rank "
        "among listed hotels (divided by 10), with search fixed effects and standard "
        "errors clustered by search. Under random ordering the slope is zero. "
        "$^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.",
        "\\end{minipage}", "\\end{table}",
    ]
    with open(os.path.join(TAB_DIR, "randomization_check.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
