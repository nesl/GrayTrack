# GrayTrack tracking experiments.
#
#   make all        run all four tracking experiments
#   make tracking_value robustness estimator_comparison multivehicle
#   make smoke      quick end-to-end check with synthetic corruption
#
PYTHON ?= python3
WORKERS ?= 4

RESULTS := results
FIGURES := figures

PROFILE ?= dnn_event_v3
PROFILE_ARG := $(if $(PROFILE),--detector_profile $(PROFILE),)

.PHONY: all tracking_value robustness estimator_comparison multivehicle smoke dirs

all: tracking_value robustness estimator_comparison multivehicle
	@echo
	@echo "All experiments complete."

dirs:
	@mkdir -p $(RESULTS) $(FIGURES)

tracking_value: dirs
	$(PYTHON) experiments/tracking_value/run_tracking_value.py --n_workers $(WORKERS) $(PROFILE_ARG)

robustness: dirs
	$(PYTHON) experiments/robustness/run_robustness.py --n_workers $(WORKERS) $(PROFILE_ARG)

estimator_comparison: dirs
	$(PYTHON) experiments/estimator_comparison/run_estimator_comparison.py --n_workers $(WORKERS) $(PROFILE_ARG)

multivehicle: dirs
	$(PYTHON) experiments/multivehicle/run_multivehicle.py --n_scenarios 15 $(PROFILE_ARG)

smoke: dirs
	@mkdir -p $(RESULTS)/smoke $(FIGURES)/smoke
	$(PYTHON) experiments/tracking_value/run_tracking_value.py \
	    --n_trajectories 6 --n_seeds 1 --num_particles 300 \
	    --dev_corruption --n_workers $(WORKERS) \
	    --out $(RESULTS)/smoke/tracking_value_smoke.csv \
	    --figures_dir $(FIGURES)/smoke
	$(PYTHON) experiments/multivehicle/run_multivehicle.py \
	    --n_scenarios 2 --estimators road_pf --num_particles 300 \
	    --dev_corruption --out $(RESULTS)/smoke/mv_smoke.csv \
	    --figures_dir $(FIGURES)/smoke
	@echo "smoke OK"
