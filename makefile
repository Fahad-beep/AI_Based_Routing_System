# Network Routing Simulator Build System

SIMULATOR = main.py
CONFIG_DIR = config

.PHONY: run clean setup

setup:
	@echo "Installing dependencies..."
	pip install torch
	pip install numpy 
	pip install 
	pip install matplotlib
	pip install tqdm
	python -m venv venv
	./venv/Scripts/activate
	@echo "To run: \"make run\""

run:
	python $(SIMULATOR) $(CONFIG_DIR)

clean:
	@echo "Cleaning outputs..."
	rm -rf models/ plots/ __pycache__/