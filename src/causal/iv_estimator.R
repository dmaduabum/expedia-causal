# src/causal/iv_estimator.R
# ─────────────────────────────────────────────────────────────────────────────
# IV/LATE: Effect of position on booking_bool and click_bool.
# Instrument: random_bool (1 = random ranking)
#
# Table format (journal-style, one table four columns):
#   (1) Booking — IV unadjusted
#   (2) Booking — IV with controls
#   (3) Click   — IV unadjusted
#   (4) Click   — IV with controls
#
# Outputs:
#   results/tables/iv_results.tex
#   results/tables/first_stage.tex
#   results/tables/iv_results.csv
#   results/figures/iv_plot.pdf
#   results/estimates/iv_results.RDS
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(data.table)
  library(estimatr)
  library(modelsummary)
  library(lmtest)
  library(sandwich)
  library(dplyr)
  library(ggplot2)
})

root    <- getwd()
CLEAN   <- file.path(root, "data", "processed", "train_clean.csv")
FIG_DIR <- file.path(root, "results", "figures")
TAB_DIR <- file.path(root, "results", "tables")
EST_DIR <- file.path(root, "results", "estimates")
for (d in c(FIG_DIR, TAB_DIR, EST_DIR))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

cat("Loading cleaned data...\n")
df  <- fread(CLEAN)
dat <- as.data.frame(df)

ALL_COVS <- c(
  "prop_starrating", "prop_review_score", "prop_brand_bool",
  "prop_location_score1", 
  "prop_log_historical_price", "promotion_flag",
  "srch_length_of_stay", "srch_booking_window",
  "srch_adults_count", "srch_children_count",
  "srch_room_count", "srch_saturday_night_bool",
  "visitor_hist_adr_usd", "no_purchase_history",
  "orig_destination_distance", "dist_missing"
)
ALL_COVS <- ALL_COVS[ALL_COVS %in% names(dat)]
cov_str  <- paste(ALL_COVS, collapse = " + ")

# ── first stage ───────────────────────────────────────────────────────────────
cat("\n── First Stage ──\n")
fs1 <- lm_robust(position ~ random_bool, data = dat, se_type = "HC2")
fs2 <- lm_robust(as.formula(paste("position ~ random_bool +", cov_str)),
                 data = dat, se_type = "HC2")

# F-stats
f1 <- summary(lm(position ~ random_bool, data = dat))$fstatistic[1]
f2 <- (summary(fs2)$coefficients["random_bool", "Estimate"] /
         summary(fs2)$coefficients["random_bool", "Std. Error"])^2

cat(sprintf("First stage F (unadjusted)  = %.1f\n", f1))
cat(sprintf("First stage F (adjusted)    = %.1f\n", f2))

# First stage table
fs_rows <- data.frame(
  term              = c("Controls", "First-stage F"),
  `(1) Unadjusted`  = c("No",  round(f1, 1)),
  `(2) With controls` = c("Yes", round(f2, 1)),
  check.names = FALSE
)
attr(fs_rows, "position") <- c(3, 4)

modelsummary(
  list("(1) Unadjusted" = fs1, "(2) With controls" = fs2),
  coef_map = c("random_bool" = "Random ranking (instrument)"),
  gof_map  = list(
    list(raw = "nobs", clean = "N",
         fmt = function(x) formatC(x, format = "d", big.mark = ","))
  ),
  add_rows = fs_rows,
  stars    = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
  title    = "First Stage: Effect of Random Ranking on Hotel Position",
  notes    = "HC2 robust SE. Dependent variable is displayed position (1 = top). A negative coefficient means random ranking pushes hotels higher on the page on average.",
  output   = file.path(TAB_DIR, "first_stage.tex")
)
cat("First stage table saved → results/tables/first_stage.tex\n")

# ── IV models ─────────────────────────────────────────────────────────────────
cat("\n── IV Regression ──\n")

m1 <- iv_robust(booking_bool ~ position | random_bool,
                data = dat, se_type = "HC2")
m2 <- iv_robust(
  as.formula(paste("booking_bool ~ position +", cov_str,
                   "| random_bool +", cov_str)),
  data = dat, se_type = "HC2")
m3 <- iv_robust(click_bool ~ position | random_bool,
                data = dat, se_type = "HC2")
m4 <- iv_robust(
  as.formula(paste("click_bool ~ position +", cov_str,
                   "| random_bool +", cov_str)),
  data = dat, se_type = "HC2")

models <- list(
  "(1) Booking" = m1,
  "(2) Booking" = m2,
  "(3) Click"   = m3,
  "(4) Click"   = m4
)

rows <- data.frame(
  term          = c("Controls", "First-stage F"),
  `(1) Booking` = c("No",  round(f1, 1)),
  `(2) Booking` = c("Yes", round(f2, 1)),
  `(3) Click`   = c("No",  round(f1, 1)),
  `(4) Click`   = c("Yes", round(f2, 1)),
  check.names   = FALSE
)
attr(rows, "position") <- c(3, 4)

modelsummary(
  models,
  coef_map = c("position" = "Position (LATE)"),
  gof_map  = list(
    list(raw = "nobs", clean = "N",
         fmt = function(x) formatC(x, format = "d", big.mark = ","))
  ),
  add_rows = rows,
  stars    = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
  title    = "LATE: Causal Effect of Position on Booking and Click Probability",
  notes    = "HC2 robust SE. Instrument: random\\_bool. Position is the hotel's rank on the results page (1 = top). A negative LATE means being shown lower on the page reduces booking/click probability. Controls same as Table 2.",
  output   = file.path(TAB_DIR, "iv_results.tex")
)
cat("IV table saved → results/tables/iv_results.tex\n")

# raw CSV
raw <- bind_rows(lapply(seq_along(models), function(i) {
  m   <- models[[i]]
  ct  <- summary(m)$coefficients
  row <- ct["position", ]
  data.frame(
    model   = names(models)[i],
    estimate = round(row["Estimate"],   6),
    se       = round(row["Std. Error"], 6),
    ci_lo    = round(row["CI Lower"],   6),
    ci_hi    = round(row["CI Upper"],   6),
    p_value  = round(row["Pr(>|t|)"],   6),
    n        = nrow(dat)
  )
}))
fwrite(raw, file.path(TAB_DIR, "iv_results.csv"))
saveRDS(list(iv = raw, f1 = f1, f2 = f2), file.path(EST_DIR, "iv_results.RDS"))
print(raw)

# ── coefficient plot ──────────────────────────────────────────────────────────
raw$outcome  <- ifelse(grepl("Booking", raw$model), "Booking", "Click")
raw$controls <- factor(
  ifelse(grepl("\\(1\\)|\\(3\\)", raw$model), "No controls", "With controls"),
  levels = c("No controls", "With controls")
)

p <- ggplot(raw, aes(x = estimate, y = controls, colour = controls)) +
  geom_point(size = 3) +
  geom_errorbar(aes(xmin = ci_lo, xmax = ci_hi),
                width = 0.15, linewidth = 0.8,
                orientation = "y") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  facet_wrap(~ outcome, scales = "free_x") +
  scale_colour_manual(values = c("No controls"   = "#2c7bb6",
                                 "With controls" = "#d7191c")) +
  labs(
    title    = "LATE: Effect of Position on Booking/Click Probability",
    subtitle = "Instrument: random_bool | 95% CI, HC2 SE",
    x        = "Estimated Effect per Additional Position (lower on page)",
    y        = NULL, colour = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank())

ggsave(file.path(FIG_DIR, "iv_plot.pdf"), p, width = 8, height = 4)
cat("IV plot saved → results/figures/iv_plot.pdf\n")
cat("Done.\n")