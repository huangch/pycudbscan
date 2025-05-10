.PHONY: all clean build install test

# Default target
all: build

# Create build directory and run cmake
build:
	mkdir -p build
	cd build && cmake .. && make -j

# Install the package
install: build
	pip install -e .

# Run tests
test: install
	cd tests && python -m unittest discover

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf dist/
	find . -name "*.so" -delete
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Create conda environment
conda-env:
	conda env create -f environment.yml

# Activate the conda environment with a message
conda-activate:
	@echo "Run this command to activate the conda environment:"
	@echo "conda activate pycudbscan"

# Run the example
example: install
	python examples/basic_usage.py