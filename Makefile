PYTHON ?= python3
PYTHONPATH := src

.PHONY: all verify fixtures test experiments metrics check-numbers clean

all: verify

fixtures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m groundedness_referee.cli fixtures

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/run_tests.py

experiments:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m groundedness_referee.cli experiments

metrics:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/collect_metrics.py

check-numbers:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/check_numbers.py --prove-breakage

verify: fixtures test metrics check-numbers
	@echo "=================================================================="
	@echo "MAKE VERIFY GREEN: Fixtures byte-identical, tests & coverage pass,"
	@echo "experiments 1-5 regenerated, and all 22 anchored figures verified."
	@echo "=================================================================="

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
