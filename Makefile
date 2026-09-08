# Makefile

PYTHON := python -m main
TEST_PYTHON ?= .venv/bin/python

# Default values (can be overridden on command line)
WORKLOAD ?= 1
RUN_MODE ?= run_sim_anneal
COST_PROFILE ?= t1

# Rule for running with run_mode and optional cost_profile
run:
	$(PYTHON) --workload $(WORKLOAD) --run_mode $(RUN_MODE) $(if $(COST_PROFILE),--cost_profile $(COST_PROFILE))

# Run one specific workload  
sim_anneal:
	$(MAKE) run WORKLOAD=$(WORKLOAD) RUN_MODE=run_sim_anneal COST_PROFILE=$(COST_PROFILE)

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
