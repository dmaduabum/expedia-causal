# ═══════════════════════════════════════════════════════════════════════════════
# Makefile — expedia-causal (random-arm design)
# ═══════════════════════════════════════════════════════════════════════════════
#   make              full pipeline
#   make clean_data   build searches.csv + random_arm.pkl from train.csv
#   make exposures    competitor-exposure variables
#   make checks       sampling check (why arms aren't compared) + randomization check
#   make position     own-position effects
#   make interference competitor interference (exploratory, not in paper)
#   make hetero       who depends most on placement
#   make purge        delete generated data and results (keeps raw zip)
# ═══════════════════════════════════════════════════════════════════════════════

PY := python3

.PHONY: all extract clean_data exposures checks position interference hetero purge

RAW_CSV    := data/raw/train.csv
SEARCHES   := data/processed/searches.csv
RANDOM_ARM := data/processed/random_arm.pkl
EXPOSURES  := data/processed/random_arm_exposures.pkl

all: checks position hetero
	@echo ""
	@echo "Pipeline complete. Tables: results/tables/  Figures: results/figures/"

extract: $(RAW_CSV)
$(RAW_CSV):
	$(PY) src/pipeline/extract.py

clean_data: $(RANDOM_ARM)
$(RANDOM_ARM) $(SEARCHES): $(RAW_CSV) src/pipeline/clean_sample.py
	$(PY) src/pipeline/clean_sample.py

exposures: $(EXPOSURES)
$(EXPOSURES): $(RANDOM_ARM) src/pipeline/build_exposures.py
	$(PY) src/pipeline/build_exposures.py

checks: $(RANDOM_ARM)
	$(PY) src/analysis/sampling_check.py
	$(PY) src/analysis/randomization_check.py

position: $(RANDOM_ARM)
	$(PY) src/causal/position_effects.py

interference: $(EXPOSURES)
	$(PY) src/causal/interference.py

hetero: $(RANDOM_ARM)
	$(PY) src/causal/heterogeneity.py

purge:
	rm -f $(SEARCHES) $(RANDOM_ARM) $(EXPOSURES)
	rm -f results/tables/*.csv results/tables/*.tex results/tables/*.txt results/figures/*.pdf
