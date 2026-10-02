# src/causal/robustness.R
# ─────────────────────────────────────────────────────────────────────────────
# Robustness checks:
#
#   Check 1 — Placebo outcome
#     Test whether random_bool predicts a hotel's PRICE (price_usd).
#     Price is a fixed characteristic — random_bool should have zero effect.
#     If it does, the randomisation is compromised.
#
#   Check 2 — Subsample stability (ITT)
#     Re-estimate the booking_bool ITT within:
#       (a) domestic vs. international searches
#       (b) short (<=3 nights) vs. long (>3 nights) stays
#       (c) branded vs. independent hotels
#     Results should be qualitatively consistent with the main estimate.
#
#   Check 3 — Same-position placebo (Exclusion restriction)
#     Among hotels where their position is the SAME under both conditions
#     (i.e., the instrument didn't move them), random_bool should have
#     no effect on booking. Tests whether random_bool affects outcomes
#     purely through position.
#
#   Check 4 — First-stage heterogeneity (Monotonicity)
#     Check whether the first stage (random_bool → position) is consistent
#     in direction across hotel quality tiers (star rating). If some hotels
#     move up under random ranking and others move down, monotonicity is
#     potentially violated.
#
# Outputs:
#   results/tables/robustness_placebo.csv
#   results/tables/robustness_subsamples.csv
#   results/tables/robustness_sameposition.csv
#   results/tables/robustness_monotonicity.csv
#   results/figures/robustness_subsamples.pdf
#
# Usage: Rscript src/causal/robustness.R
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(data.table)
  library(estimatr)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

root    <- getwd()
CLEAN   <- file.path(root, "data", "processed", "train_clean.csv")
FIG_DIR <- file.path(root, "results", "figures")
TAB_DIR <- file.path(root, "results", "tables")
for (d in c(FIG_DIR, TAB_DIR)) dir.create(d, recursive=TRUE, showWarnings=FALSE)

cat("Loading cleaned data...\n")
df  <- fread(CLEAN)
dat <- as.data.frame(df)
dat$Z <- 1L - dat$random_bool   # Z=1 → algorithmic

ALL_COVS <- c(
  "prop_starrating", "prop_review_score", "prop_brand_bool",
  "prop_location_score1", "prop_location_score2",
  "prop_log_historical_price", "promotion_flag",
  "srch_length_of_stay", "srch_booking_window",
  "srch_adults_count", "srch_children_count",
  "srch_room_count", "srch_saturday_night_bool",
  "visitor_hist_adr_usd", "no_purchase_history",
  "orig_destination_distance", "dist_missing"
)
ALL_COVS <- ALL_COVS[ALL_COVS %in% names(dat)]

# ── Check 1: Placebo outcome (price_usd) ─────────────────────────────────────
cat("\n══ Check 1: Placebo outcome (price_usd) ══\n")
cat("H0: random_bool has no effect on price_usd (a fixed hotel characteristic)\n")

m_placebo <- lm_robust(price_usd ~ Z, data = dat, se_type = "HC2")
ct <- summary(m_placebo)$coefficients["Z",]
placebo_res <- data.frame(
  check     = "Placebo: price_usd",
  estimate  = round(ct["Estimate"],   4),
  se        = round(ct["Std. Error"], 4),
  p_value   = round(ct["Pr(>|t|)"],   4),
  verdict   = ifelse(ct["Pr(>|t|)"] > 0.05, "PASS (p > 0.05)", "FAIL (p <= 0.05)")
)
print(placebo_res)
fwrite(placebo_res, file.path(TAB_DIR, "robustness_placebo.csv"))

# ── Check 2: Subsample stability ─────────────────────────────────────────────
cat("\n══ Check 2: Subsample stability ══\n")

run_subsample <- function(label, subset_dat) {
  m <- lm_robust(booking_bool ~ Z, data = subset_dat, se_type = "HC2")
  ct <- summary(m)$coefficients["Z",]
  data.frame(
    subsample = label,
    n_rows    = nrow(subset_dat),
    estimate  = round(ct["Estimate"],   5),
    se        = round(ct["Std. Error"], 5),
    ci_lo     = round(ct["CI Lower"],   5),
    ci_hi     = round(ct["CI Upper"],   5),
    p_value   = round(ct["Pr(>|t|)"],   4)
  )
}

# Define subgroups
dat$domestic <- as.integer(dat$visitor_location_country_id == dat$prop_country_id)
dat$short_stay <- as.integer(dat$srch_length_of_stay <= 3)

sub_res <- rbind(
  run_subsample("Full sample",              dat),
  run_subsample("Domestic searches",        dat[dat$domestic == 1, ]),
  run_subsample("International searches",   dat[dat$domestic == 0, ]),
  run_subsample("Short stay (<=3 nights)",  dat[dat$short_stay == 1, ]),
  run_subsample("Long stay (>3 nights)",    dat[dat$short_stay == 0, ]),
  run_subsample("Brand hotels",             dat[dat$prop_brand_bool == 1, ]),
  run_subsample("Independent hotels",       dat[dat$prop_brand_bool == 0, ])
)
print(sub_res)
fwrite(sub_res, file.path(TAB_DIR, "robustness_subsamples.csv"))

# Plot subsample estimates
sub_res$subsample <- factor(sub_res$subsample, levels = rev(sub_res$subsample))
p_sub <- ggplot(sub_res, aes(x = estimate, y = subsample)) +
  geom_point(size = 3, colour = "#2c7bb6") +
  geom_errorbar(aes(xmin = ci_lo, xmax = ci_hi),
                width = 0.15, linewidth = 0.8, orientation = "y") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  labs(
    title    = "ITT Stability Across Subsamples",
    subtitle = "Outcome: booking_bool | 95% CI, HC2 SE",
    x        = "Estimated ITT Effect",
    y        = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank())

ggsave(file.path(FIG_DIR, "robustness_subsamples.pdf"), p_sub, width=7, height=5)
cat(sprintf("Subsample plot saved → %s\n", file.path(FIG_DIR, "robustness_subsamples.pdf")))

# ── Check 3: Same-position placebo (Exclusion restriction) ───────────────────
cat("\n══ Check 3: Same-position placebo ══\n")
cat("Among hotels where modal position is the SAME under both conditions,\n")
cat("random_bool should not predict booking.\n")

prop_pos <- dat %>%
  group_by(prop_id, random_bool) %>%
  summarise(modal_pos = as.integer(names(sort(table(position), decreasing=TRUE))[1]),
            .groups = "drop")

same_pos_props <- prop_pos %>%
  tidyr::pivot_wider(names_from = random_bool, values_from = modal_pos,
                     names_prefix = "pos_") %>%
  filter(!is.na(pos_0) & !is.na(pos_1) & pos_0 == pos_1) %>%
  pull(prop_id)

dat_same <- dat[dat$prop_id %in% same_pos_props, ]
cat(sprintf("  Hotels with same modal position: %d\n", length(same_pos_props)))
cat(sprintf("  Rows in same-position subset: %d\n", nrow(dat_same)))

if (nrow(dat_same) > 100) {
  # Unadjusted
  m_same_raw <- lm_robust(booking_bool ~ Z, data = dat_same, se_type = "HC2")
  ct_raw <- summary(m_same_raw)$coefficients["Z", ]
  
  # Covariate-adjusted
  fml_adj <- as.formula(paste("booking_bool ~ Z +", paste(ALL_COVS, collapse = " + ")))
  m_same_adj <- lm_robust(fml_adj, data = dat_same, se_type = "HC2")
  ct_adj <- summary(m_same_adj)$coefficients["Z", ]
  
  same_res <- data.frame(
    check      = c("Same-position placebo (unadjusted)",
                   "Same-position placebo (covariate-adjusted)"),
    n_hotels   = length(same_pos_props),
    estimate   = round(c(ct_raw["Estimate"],   ct_adj["Estimate"]),   5),
    se         = round(c(ct_raw["Std. Error"],  ct_adj["Std. Error"]), 5),
    p_value    = round(c(ct_raw["Pr(>|t|)"],   ct_adj["Pr(>|t|)"]),   4),
    verdict    = ifelse(c(ct_raw["Pr(>|t|)"],  ct_adj["Pr(>|t|)"]) > 0.05,
                        "PASS — exclusion restriction plausible",
                        "CONCERN — effect persists at same position")
  )
  print(same_res)
  fwrite(same_res, file.path(TAB_DIR, "robustness_sameposition.csv"))
}else {
  cat("  Too few same-position hotels for reliable test.\n")
}

# ── Check 4: Monotonicity — first stage by hotel quality tier ─────────────────
cat("\n══ Check 4: Monotonicity check ══\n")
cat("First stage (random_bool → position) by hotel star rating tier.\n")
cat("Consistent sign = monotonicity plausible.\n")

mono_res <- dat %>%
  group_by(prop_starrating) %>%
  summarise(
    n = n(),
    mean_pos_algo   = round(mean(position[random_bool==0], na.rm=TRUE), 3),
    mean_pos_random = round(mean(position[random_bool==1], na.rm=TRUE), 3),
    .groups = "drop"
  ) %>%
  mutate(
    diff = round(mean_pos_algo - mean_pos_random, 3),
    direction = ifelse(diff < 0, "Algorithm ranks HIGHER (negative diff)",
                       "Algorithm ranks LOWER (positive diff)")
  )

print(mono_res)
fwrite(mono_res, file.path(TAB_DIR, "robustness_monotonicity.csv"))

n_pos <- sum(mono_res$diff > 0, na.rm=TRUE)
n_neg <- sum(mono_res$diff < 0, na.rm=TRUE)
if (n_pos == 0 | n_neg == 0) {
  cat("  Monotonicity looks plausible — first stage is consistent in direction.\n")
} else {
  cat(sprintf("  WARNING: %d tiers go positive, %d go negative — potential monotonicity violation.\n",
              n_pos, n_neg))
}

cat("\nAll robustness checks complete.\n")
