"""
src/analysis/sampling_check.py
-------------------------------
Documents why the algorithmic and random arms cannot be compared.

Search characteristics entered by the user (dates, party size, booking window)
and visitor purchase history are fixed before any ranking is shown. If the
released sample were a random sample of searches in each arm, these would be
balanced across arms. They are not, because Expedia over-sampled searches that
ended in a booking, and bookings are far more common under its ranking
(Ursu 2018, Sec. 3.3).

Outputs:
  results/tables/sampling_check.csv
  results/tables/sampling_check.tex
Usage: python src/analysis/sampling_check.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import SEARCHES, TAB_DIR, nfmt  # noqa: E402

VARS = [
    ("srch_length_of_stay", "Length of stay (nights)"),
    ("srch_booking_window", "Booking window (days)"),
    ("srch_adults_count", "Adults"),
    ("srch_children_count", "Children"),
    ("srch_room_count", "Rooms"),
    ("srch_saturday_night_bool", "Includes Saturday night"),
    ("has_purchase_history", "Has purchase history"),
    ("domestic", "Domestic search"),
]
OUTCOMES = [
    ("n_clicks", "Clicks per search"),
    ("booked", "Search ends in a booking"),
]


def main():
    s = pd.read_csv(SEARCHES)
    alg, rnd = s[s.random_bool == 0], s[s.random_bool == 1]
    rows = []
    for v, lab in VARS + OUTCOMES:
        m0, m1 = alg[v].mean(), rnd[v].mean()
        sd = np.sqrt((alg[v].var() + rnd[v].var()) / 2)
        rows.append({"variable": lab, "algorithmic": m0, "random": m1,
                     "smd": (m0 - m1) / sd,
                     "type": "pre-treatment" if (v, lab) in VARS else "outcome"})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_DIR, "sampling_check.csv"), index=False)

    lines = [
        "\\begin{table}[htbp]\\centering",
        "\\caption{Search Characteristics by Ranking Arm in the Released Data}",
        "\\label{tab:sampling}",
        "\\begin{tabular}{lccc}", "\\toprule",
        " & Algorithmic & Random & Std.\\ diff. \\\\", "\\midrule",
        "\\multicolumn{4}{l}{\\textit{Fixed before results are shown}} \\\\",
    ]
    for _, r in t[t.type == "pre-treatment"].iterrows():
        lines.append(f"{r.variable} & {r.algorithmic:.3f} & {r.random:.3f} & {r.smd:+.3f} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\textit{Outcomes}} \\\\")
    for _, r in t[t.type == "outcome"].iterrows():
        lines.append(f"{r.variable} & {r.algorithmic:.3f} & {r.random:.3f} & {r.smd:+.3f} \\\\")
    lines += [
        "\\midrule",
        f"Searches & {nfmt(len(alg))} & {nfmt(len(rnd))} & \\\\",
        "\\bottomrule", "\\end{tabular}",
        "\\begin{minipage}{0.88\\textwidth}\\footnotesize\\vspace{0.3em}",
        "\\textit{Notes:} One observation per search, all 399{,}344 searches in the "
        "released training data. Std.\\ diff.\\ is the difference in means divided by "
        "the pooled standard deviation. Every released search has at least one click. "
        "Expedia over-sampled searches that ended in a booking (Ursu 2018), so the "
        "two arms are differently selected samples and are not compared in the analysis.",
        "\\end{minipage}", "\\end{table}",
    ]
    with open(os.path.join(TAB_DIR, "sampling_check.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
