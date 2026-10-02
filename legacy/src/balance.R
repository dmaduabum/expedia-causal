# src/analysis/balance.R
# ─────────────────────────────────────────────────────────────────────────────
# Produces:
#   results/tables/balance_table.csv   — SMD for each covariate by random_bool
#   results/figures/love_plot.pdf      — Love plot (|SMD| before randomisation)
#
# Since random_bool is randomised, we EXPECT balance.
# This script CONFIRMS it and flags any imbalance as a warning.
#
# Usage: Rscript src/analysis/balance.R
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(dplyr)
  library(knitr)
  library(kableExtra)
})

root    <- here::here()  # project root (requires 'here' package or set manually)
# If 'here' not available, set manually:
# root <- "/path/to/expedia-causal"

CLEAN   <- file.path(root, "data", "processed", "train_clean.csv")
FIG_DIR <- file.path(root, "results", "figures")
TAB_DIR <- file.path(root, "results", "tables")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)

# ── covariates to check balance on ───────────────────────────────────────────
COVS <- c(
  "price_usd", "prop_starrating", "prop_review_score",
  "prop_brand_bool", "prop_location_score1", 
  "promotion_flag", "srch_length_of_stay", "srch_booking_window",
  "srch_adults_count", "srch_children_count", "srch_saturday_night_bool",
  "visitor_hist_adr_usd", "no_purchase_history", "orig_destination_distance",
  "dist_missing"
)

cat("Loading cleaned data...\n")
df <- fread(CLEAN)
cat(sprintf("  %s rows | %s searches\n",
            format(nrow(df), big.mark=","),
            format(uniqueN(df$srch_id), big.mark=",")))

# ── compute standardised mean differences ─────────────────────────────────────
smd <- function(x, z) {
  # SMD = (mean_treated - mean_control) / pooled SD
  # Here treated = random_bool=0 (algorithmic), control = random_bool=1 (random)
  x1 <- x[z == 0]; x0 <- x[z == 1]
  mu1 <- mean(x1, na.rm=TRUE); mu0 <- mean(x0, na.rm=TRUE)
  s1  <- var(x1,  na.rm=TRUE); s0  <- var(x0,  na.rm=TRUE)
  denom <- sqrt((s1 + s0) / 2)
  if (denom == 0) return(NA_real_)
  (mu1 - mu0) / denom
}

bal <- lapply(COVS, function(v) {
  if (!v %in% names(df)) return(NULL)
  data.frame(
    variable     = v,
    mean_algo    = round(mean(df[[v]][df$random_bool == 0], na.rm=TRUE), 4),
    mean_random  = round(mean(df[[v]][df$random_bool == 1], na.rm=TRUE), 4),
    smd          = round(smd(df[[v]], df$random_bool), 4),
    abs_smd      = round(abs(smd(df[[v]], df$random_bool)), 4)
  )
})
bal_df <- do.call(rbind, Filter(Negate(is.null), bal))

# ── flag imbalance (|SMD| > 0.1 is the conventional threshold) ───────────────
bal_df$imbalanced <- ifelse(bal_df$abs_smd > 0.1, "YES", "")
n_imbal <- sum(bal_df$imbalanced == "YES")
cat(sprintf("\nBalance check: %d/%d covariates have |SMD| > 0.1\n",
            n_imbal, nrow(bal_df)))
if (n_imbal > 0) {
  cat("  → Imbalanced vars:\n")
  print(bal_df[bal_df$imbalanced == "YES", c("variable","smd","abs_smd")])
  cat("  (Since random_bool is randomised, large SMDs suggest possible\n")
  cat("   stratification issues or finite-sample imbalance.)\n")
}

# ── save balance table ────────────────────────────────────────────────────────
out_tab <- file.path(TAB_DIR, "balance_table.csv")
fwrite(bal_df, out_tab)
cat(sprintf("Balance table saved → %s\n", out_tab))
print(bal_df)

# ── Love plot ─────────────────────────────────────────────────────────────────
bal_plot <- bal_df %>%
  arrange(abs_smd) %>%
  mutate(variable = factor(variable, levels = variable))

p <- ggplot(bal_plot, aes(x = abs_smd, y = variable)) +
  geom_point(size = 3, colour = "#2c7bb6") +
  geom_vline(xintercept = 0.1, linetype = "dashed", colour = "red", linewidth = 0.7) +
  labs(
    title    = "Love Plot: Absolute SMD by Covariate",
    subtitle = "Algorithmic (random_bool=0) vs. Random ranking (random_bool=1)\nDashed line = |SMD| = 0.1 threshold",
    x        = "Absolute Standardised Mean Difference",
    y        = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank())

out_fig <- file.path(FIG_DIR, "love_plot.pdf")
ggsave(out_fig, p, width = 7, height = 5)
cat(sprintf("Love plot saved → %s\n", out_fig))

#make a latex table....
bal_df %>%
  select(variable, mean_algo, mean_random, smd) %>%
  rename(Variable = variable,
         `Mean (Algo)` = mean_algo,
         `Mean (Random)` = mean_random,
         SMD = smd) %>%
  kable("latex", booktabs = TRUE, digits = 3,
        linesep = "") %>%
  kable_styling(latex_options = "hold_position") %>%
  save_kable(file.path(TAB_DIR, "balance_table_tex.tex"))