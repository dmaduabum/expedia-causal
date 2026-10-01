# ═══════════════════════════════════════════════════════════════════════════════
# Makefile — expedia-causal
# ═══════════════════════════════════════════════════════════════════════════════
# Usage:
#   make          → runs full pipeline (extract → clean → balance → itt → iv → robustness)
#   make extract  → unzip raw data only
#   make clean_data → extract + sample/clean
#   make balance  → balance table + Love plot
#   make itt      → ITT estimates (DiM + ANCOVA)
#   make iv       → IV estimates (Wald + iv_robust)
#   make robustness → all robustness checks
#   make paper    → compile LaTeX
#   make purge    → delete all generated files (keeps raw zip)
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all extract clean_data balance descriptive itt iv robustness paper purge

# ── file targets ───────────────────────────────────────────────────────────────
RAW_CSV   := data/raw/train.csv
CLEAN_CSV := data/processed/train_clean.csv

# ── default: full pipeline ─────────────────────────────────────────────────────
all: extract clean_data descriptive balance itt iv robustness
	@echo ""
	@echo "════════════════════════════════════════"
	@echo "  Pipeline complete."
	@echo "  Results in: results/tables/ and results/figures/"
	@echo "════════════════════════════════════════"

# ── Step 1: extract train.csv from zip ────────────────────────────────────────
extract: $(RAW_CSV)

$(RAW_CSV):
	@echo "── Step 1: Extracting train.csv from zip ──"
	python src/pipeline/extract.py

# ── Step 2: clean, drop, sample → train_clean.csv ─────────────────────────────
clean_data: $(CLEAN_CSV)

$(CLEAN_CSV): $(RAW_CSV)
	@echo "── Step 2: Cleaning and sampling data ──"
	python src/pipeline/clean_sample.py

# ── Step 3a: descriptive stats table ──────────────────────────────────────────
descriptive: $(CLEAN_CSV)
	@echo "── Step 3a: Descriptive statistics ──"
	python src/analysis/descriptive.py

# ── Step 3b: balance table + Love plot ────────────────────────────────────────
balance: $(CLEAN_CSV)
	@echo "── Step 3b: Balance check + Love plot ──"
	Rscript src/analysis/balance.R

# ── Step 4: ITT (DiM + ANCOVA) for booking_bool and click_bool ────────────────
itt: $(CLEAN_CSV)
	@echo "── Step 4: ITT estimator ──"
	Rscript src/causal/itt_estimator.R

# ── Step 5: IV (Wald + iv_robust) for position → booking_bool ─────────────────
iv: $(CLEAN_CSV)
	@echo "── Step 5: IV estimator ──"
	Rscript src/causal/iv_estimator.R

# ── Step 6: Robustness checks ─────────────────────────────────────────────────
robustness: $(CLEAN_CSV)
	@echo "── Step 6: Robustness checks ──"
	Rscript src/causal/robustness.R

# ── Compile paper ─────────────────────────────────────────────────────────────
paper:
	@echo "── Compiling LaTeX paper ──"
	cd paper && pdflatex main.tex && pdflatex main.tex

# ── Delete all generated files (keeps raw zip) ────────────────────────────────
purge:
	@echo "Deleting all generated files..."
	rm -f $(RAW_CSV)
	rm -f $(CLEAN_CSV)
	rm -f results/tables/*.csv
	rm -f results/tables/*.txt
	rm -f results/figures/*.pdf
	rm -f results/estimates/*.RDS
	@echo "Done. Raw zip preserved at data/raw/expedia-personalized-sort.zip"
