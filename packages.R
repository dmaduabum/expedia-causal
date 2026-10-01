pkgs <- c(
  "data.table",
  "dplyr",
  "tidyr",
  "ggplot2",
  "estimatr",
  "AER",
  "sandwich",
  "lmtest",
  "cobalt",
  "here",
  "knitr",
  "kableExtra",
  "modelsummary"
)

install.packages(pkgs[!pkgs %in% installed.packages()[, "Package"]],
                 repos = "https://cloud.r-project.org")