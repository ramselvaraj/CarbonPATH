# Makefile

PYTHON := python -m main
TEST_PYTHON ?= .venv/bin/python

# Default values (can be overridden on command line)
WORKLOAD ?= 1
RUN_MODE ?= run_sim_anneal
COST_PROFILE ?= t1
ITERATION ?= 1
CALIBRATION_ITERATIONS ?= 10
RUN_NAME ?=
SEED ?=
INITIAL_TEMP ?= 40
FREEZING_TEMP ?= 5e-4
MAX_MOVE_PER_TEMP_STEP ?= 5
COOLING_RATE ?= 0.3
CACHE_FILE ?=

# Rule for running with run_mode and optional cost_profile
run:
	$(PYTHON) --workload $(WORKLOAD) --run_mode $(RUN_MODE) --iteration $(ITERATION) --calibration_iterations $(CALIBRATION_ITERATIONS) --initial_temp $(INITIAL_TEMP) --freezing_temp $(FREEZING_TEMP) --max_move_per_temp_step $(MAX_MOVE_PER_TEMP_STEP) --cooling_rate $(COOLING_RATE) $(if $(COST_PROFILE),--cost_profile $(COST_PROFILE)) $(if $(RUN_NAME),--run_name $(RUN_NAME)) $(if $(SEED),--seed $(SEED)) $(if $(CACHE_FILE),--cache_file $(CACHE_FILE))

# Run one specific workload  
sim_anneal:
	$(MAKE) run WORKLOAD=$(WORKLOAD) RUN_MODE=run_sim_anneal COST_PROFILE=$(COST_PROFILE)

# Modular ATLAS annealing: architecture + evaluation-profile search.
ATLAS_GRAPH ?= cfg/examples/atlas/dense_relu_funnel.graph_dump.json
EVALUATION_PROFILE ?= cfg/profiles/legacy_sa_fpga_v1.json
ATLAS_SEARCH_SPACE ?= cfg/experiments/atlas_modular_search_space.json
ARCHITECTURE_FILE ?= cfg/examples/sa_fpga_architecture.json

sim_anneal_atlas:
	$(PYTHON) --run_mode run_sim_anneal --atlas_graph $(ATLAS_GRAPH) --evaluation_profile $(EVALUATION_PROFILE) --atlas_search_space $(ATLAS_SEARCH_SPACE) --architecture_file $(ARCHITECTURE_FILE) --initial_temp $(INITIAL_TEMP) --freezing_temp $(FREEZING_TEMP) --max_move_per_temp_step $(MAX_MOVE_PER_TEMP_STEP) --cooling_rate $(COOLING_RATE) $(if $(RUN_NAME),--run_name $(RUN_NAME)) $(if $(SEED),--seed $(SEED)) $(if $(CACHE_FILE),--cache_file $(CACHE_FILE))

validate_atlas_policy_scores:
	$(TEST_PYTHON) script/validate_atlas_policy_scores.py --graph $(ATLAS_GRAPH) --architecture $(ARCHITECTURE_FILE)

calibration:
	$(MAKE) run WORKLOAD=$(WORKLOAD) RUN_MODE=run_calibration

test:
	$(TEST_PYTHON) -m unittest discover -s tests -v

clean_calibration:
	python script/clean_calibration.py 


# Run parallel jobs for all workloads 
parallel_calibration:
	./script/run_parallel_calibration.sh

parallel_carbonpath:
	./script/run_parallel_carbonpath.sh 

clean_all:
	rm log_wl*.log job_s*.log
	rm -rf wl*_carbon_path*
