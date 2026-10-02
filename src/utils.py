"""
src/utils.py
------------
Shared paths and a small, dependency-free regression helper (numpy only).

ols(y, X, ...) fits OLS and returns coefficients with either
  * HC2 heteroskedasticity-robust SEs (se="HC2"), or
  * cluster-robust SEs (se="cluster", clusters=...), CR1 small-sample
    correction as in Stata / fixest:  G/(G-1) * (n-1)/(n-k).

Fixed effects can be absorbed with `absorb=` (one categorical variable).
The outcome and regressors are demeaned within groups (Frisch-Waugh-Lovell),
so the coefficients equal those from including group dummies. When the
absorbed groups are nested within clusters (e.g. search FE, clustered by
search) the FE are not counted in k, following the usual convention.
"""

import os
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, "data", "raw", "train.csv")
PROC_DIR = os.path.join(ROOT, "data", "processed")
SEARCHES = os.path.join(PROC_DIR, "searches.csv")
RANDOM_ARM = os.path.join(PROC_DIR, "random_arm.pkl")
EXPOSURES = os.path.join(PROC_DIR, "random_arm_exposures.pkl")
TAB_DIR = os.path.join(ROOT, "results", "tables")
FIG_DIR = os.path.join(ROOT, "results", "figures")
for _d in (PROC_DIR, TAB_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)


# ── fixed-effect absorption ──────────────────────────────────────────────────
def demean(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract group means from each column of a (2-D) or from a (1-D)."""
    codes, uniq = pd.factorize(groups)
    counts = np.bincount(codes).astype(float)
    a2 = np.asarray(a, dtype=float).reshape(len(a), -1)
    out = np.empty_like(a2)
    for j in range(a2.shape[1]):
        means = np.bincount(codes, weights=a2[:, j]) / counts
        out[:, j] = a2[:, j] - means[codes]
    return out.reshape(a.shape)


# ── OLS ──────────────────────────────────────────────────────────────────────
def ols(y, X, names, se="cluster", clusters=None, absorb=None,
        absorb_nested=True, add_const=True):
    """
    y        : (n,) outcome
    X        : (n, k) regressors (no constant needed)
    names    : list of k regressor names
    se       : "HC2" or "cluster"
    clusters : (n,) cluster ids when se="cluster"
    absorb   : (n,) group ids for one set of fixed effects, or None
    Returns a DataFrame: term, estimate, se, t, p, ci_lo, ci_hi, plus attrs n, G.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    names = list(names)
    n_fe = 0
    if absorb is not None:
        y = demean(y, absorb)
        X = demean(X, absorb)
        if not absorb_nested:
            n_fe = pd.Series(absorb).nunique()
    elif add_const:
        X = np.column_stack([np.ones(len(y)), X])
        names = ["(Intercept)"] + names

    # drop regressors with no variation (after absorbing FE), e.g. a flag that
    # is always 0 in a subsample; report them rather than failing
    keep = X.std(axis=0) > 1e-10
    if names and names[0] == "(Intercept)":
        keep[0] = True
    if not keep.all():
        print("  [ols] dropping constant regressors:",
              [nm for nm, kp in zip(names, keep) if not kp])
        X = X[:, keep]
        names = [nm for nm, kp in zip(names, keep) if kp]

    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta

    if se == "HC2":
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        w = u ** 2 / np.clip(1 - h, 1e-12, None)
        meat = (X * w[:, None]).T @ X
        V = XtX_inv @ meat @ XtX_inv
        G = None
    elif se == "cluster":
        codes, _ = pd.factorize(np.asarray(clusters))
        G = codes.max() + 1
        # per-cluster score sums, one column at a time (memory-light)
        S = np.column_stack([np.bincount(codes, weights=X[:, j] * u, minlength=G)
                             for j in range(k)])
        meat = S.T @ S
        kk = k + n_fe
        c = (G / (G - 1)) * ((n - 1) / (n - kk))
        V = c * XtX_inv @ meat @ XtX_inv
    else:
        raise ValueError("se must be 'HC2' or 'cluster'")

    sd = np.sqrt(np.diag(V))
    t = beta / sd
    from math import erf, sqrt
    p = np.array([2 * (1 - 0.5 * (1 + erf(abs(tt) / sqrt(2)))) for tt in t])
    out = pd.DataFrame({
        "term": names, "estimate": beta, "se": sd, "t": t, "p": p,
        "ci_lo": beta - 1.96 * sd, "ci_hi": beta + 1.96 * sd,
    })
    out.attrs["n"] = n
    out.attrs["G"] = G
    out.attrs["vcov"] = V
    return out


def own_controls(d):
    """Hotel characteristics used as controls (with missing-value indicators)."""
    star = d["prop_starrating"].astype(float)
    rev = d["prop_review_score"].astype(float).fillna(0)
    loc = d["prop_location_score1"].astype(float).fillna(0)
    return np.column_stack([
        star, d["star_unknown"], rev, d["review_missing"], d["review_none"],
        d["log_price_rel"], d["prop_brand_bool"], loc, d["promotion_flag"],
    ])


def lincom(res: pd.DataFrame, weights: dict):
    """Estimate and SE of a linear combination sum_j w_j * beta_j."""
    w = np.array([weights.get(t, 0.0) for t in res["term"]])
    est = float(w @ res["estimate"].values)
    se = float(np.sqrt(w @ res.attrs["vcov"] @ w))
    return est, se


# ── LaTeX table writer ───────────────────────────────────────────────────────
def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def fmt(x, digits=4):
    return f"{x:.{digits}f}"


def write_tex_table(path, columns, rows, caption, label, notes, extra_rows=None,
                    digits=4):
    """
    columns    : list of column headers
    rows       : list of (row_label, [ (est, se, p) or None per column ])
    extra_rows : list of (row_label, [str per column])
    """
    ncol = len(columns)
    lines = [
        "\\begin{table}[htbp]\\centering",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * ncol + "}",
        "\\toprule",
        " & " + " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for lab, cells in rows:
        est_line = [lab]
        se_line = [""]
        for c in cells:
            if c is None:
                est_line.append("")
                se_line.append("")
            else:
                e, s, p = c
                est_line.append(fmt(e, digits) + stars(p))
                se_line.append("(" + fmt(s, digits) + ")")
        lines.append(" & ".join(est_line) + " \\\\")
        lines.append(" & ".join(se_line) + " \\\\")
    if extra_rows:
        lines.append("\\midrule")
        for lab, vals in extra_rows:
            lines.append(" & ".join([lab] + [str(v) for v in vals]) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{minipage}{0.92\\textwidth}\\footnotesize",
        "\\vspace{0.3em}\\textit{Notes:} " + notes +
        " $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.",
        "\\end{minipage}",
        "\\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def nfmt(x):
    return f"{int(x):,}"
