.PHONY: help install dev test lint format clean

help:
	@echo "Targets:"
	@echo "  install   Install package"
	@echo "  dev       Install with dev dependencies"
	@echo "  test      Run pytest"
	@echo "  lint      Run ruff"
	@echo "  format    Run ruff --fix"
	@echo "  clean     Remove caches and build artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check src tests

format:
	ruff check --fix src tests

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
