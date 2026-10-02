# src/causal/itt_estimator.R
# ─────────────────────────────────────────────────────────────────────────────
# ITT: Effect of algorithmic ranking (Z=1) vs random ranking (Z=0)
# on booking_bool and click_bool.
#
# Table format (journal-style, one table four columns):
#   (1) Booking — No controls
#   (2) Booking — With controls
#   (3) Click   — No controls
#   (4) Click   — With controls
#
# Outputs:
#   results/tables/itt_results.tex   ← LaTeX table for paper
#   results/tables/itt_results.csv   ← raw numbers
#   results/figures/itt_plot.pdf
#   results/estimates/itt_results.RDS
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(data.table)
  library(estimatr)
  library(modelsummary)
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
dat$Z <- 1L - dat$random_bool  # Z=1 → algorithmic, Z=0 → random

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

# ── fit four models ───────────────────────────────────────────────────────────
m1 <- lm_robust(booking_bool ~ Z,
                data = dat, se_type = "HC2")
m2 <- lm_robust(as.formula(paste("booking_bool ~ Z +", cov_str)),
                data = dat, se_type = "HC2")
m3 <- lm_robust(click_bool ~ Z,
                data = dat, se_type = "HC2")
m4 <- lm_robust(as.formula(paste("click_bool ~ Z +", cov_str)),
                data = dat, se_type = "HC2")

# ── modelsummary table ────────────────────────────────────────────────────────
models <- list(
  "(1) Booking"  = m1,
  "(2) Booking"  = m2,
  "(3) Click"    = m3,
  "(4) Click"    = m4
)

# rows to add manually (Controls indicator)
rows <- data.frame(
  term = "Controls",
  `(1) Booking` = "No",
  `(2) Booking` = "Yes",
  `(3) Click`   = "No",
  `(4) Click`   = "Yes",
  check.names = FALSE
)
attr(rows, "position") <- 3  # insert after coefficients

modelsummary(
  models,
  coef_map  = c("Z" = "Algorithm (Z = 1)"),
  gof_map   = list(
    list(raw = "nobs",      clean = "N",         fmt = function(x) formatC(x, format="d", big.mark=",")),
    list(raw = "r.squared", clean = "R²",        fmt = 3)
  ),
  add_rows  = rows,
  stars     = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
  title     = "ITT: Effect of Algorithmic Ranking on Booking and Click Probability",
  notes     = "HC2 heteroskedasticity-robust standard errors in parentheses. Unit of observation is (search, hotel) pair. Z = 1 if search received algorithmic ranking, Z = 0 if random. Controls include hotel characteristics (star rating, review score, brand, location scores, price) and search characteristics (length of stay, booking window, party size, Saturday night, visitor history).",
  output    = file.path(TAB_DIR, "itt_results.tex")
)

# also save CSV of raw coefficients
raw <- bind_rows(
  lapply(seq_along(models), function(i) {
    m   <- models[[i]]
    ct  <- summary(m)$coefficients
    row <- ct["Z", ]
    data.frame(
      model    = names(models)[i],
      estimate = round(row["Estimate"],   6),
      se       = round(row["Std. Error"], 6),
      ci_lo    = round(row["CI Lower"],   6),
      ci_hi    = round(row["CI Upper"],   6),
      p_value  = round(row["Pr(>|t|)"],   6),
      n        = nrow(dat)
    )
  })
)
fwrite(raw, file.path(TAB_DIR, "itt_results.csv"))
saveRDS(raw, file.path(EST_DIR, "itt_results.RDS"))

cat("ITT table saved → results/tables/itt_results.tex\n")
print(raw)

# ── coefficient plot ──────────────────────────────────────────────────────────
raw$outcome   <- ifelse(grepl("Booking", raw$model), "Booking", "Click")
raw$controls  <- ifelse(grepl("(1)|\\(3\\)", raw$model), "No controls", "With controls")
raw$controls  <- factor(raw$controls, levels = c("No controls", "With controls"))

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
    title    = "ITT: Effect of Algorithmic vs. Random Ranking",
    subtitle = "Z = 1 (Algorithmic) | 95% CI, HC2 SE",
    x        = "Estimated Effect on Probability",
    y        = NULL, colour = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank())

ggsave(file.path(FIG_DIR, "itt_plot.pdf"), p, width = 8, height = 4)
cat("ITT plot saved → results/figures/itt_plot.pdf\n")
cat("Done.\n")