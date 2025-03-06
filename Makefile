# Makefile for Enhanced Trading Strategy System

# Variables
PYTHON := python3
VENV := venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PYTEST := $(BIN)/pytest
BLACK := $(BIN)/black
FLAKE8 := $(BIN)/flake8
MYPY := $(BIN)/mypy
ISORT := $(BIN)/isort

# Directories
SRC_DIR := src
TEST_DIR := tests
DATA_DIR := data
FINTSB_DIR := FinTSB
MATH_DIR := mathematricks

.PHONY: all setup validate test lint format clean help

# Default target
all: help

# Help
help:
	@echo "Enhanced Trading Strategy System Development Commands:"
	@echo ""
	@echo "Setup and Validation:"
	@echo "  make setup      - Set up development environment"
	@echo "  make validate   - Run project validation"
	@echo ""
	@echo "Development:"
	@echo "  make test       - Run tests"
	@echo "  make coverage   - Run tests with coverage"
	@echo "  make lint       - Run linting checks"
	@echo "  make format     - Format code"
	@echo "  make all-checks - Run all checks (format, lint, test)"
	@echo ""
	@echo "Data and Training:"
	@echo "  make download   - Download EastMoney data"
	@echo "  make train      - Train LSTM model"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Clean temporary files"
	@echo "  make update-deps- Update dependencies"
	@echo "  make check-env  - Check environment"
	@echo ""

# Setup development environment
setup: $(VENV)
	$(PIP) install -r requirements.txt
	$(PYTHON) -m pip install -e .
	mkdir -p $(DATA_DIR)/ibkr/1d $(DATA_DIR)/kraken/1d
	chmod +x *.py

$(VENV):
	$(PYTHON) -m venv $(VENV)

# Validation
validate:
	./validate_setup.py

# Testing
test:
	$(PYTEST) $(TEST_DIR)

coverage:
	$(PYTEST) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing $(TEST_DIR)

# Code quality
lint:
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(MYPY) $(SRC_DIR)

format:
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(ISORT) $(SRC_DIR) $(TEST_DIR)

# Download data
download:
	./download_eastmoney_data.py

# Training
train:
	$(PYTHON) src/training/train_fintsb.py $(FINTSB_DIR)/configs/fintsb_lstm.yaml

# Run all checks
all-checks: format lint test

# Cleaning
clean:
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf $(TEST_DIR)/__pycache__
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +

# Dependencies
requirements:
	$(PIP) freeze > requirements.txt

# Additional utility targets
.PHONY: check-env
check-env:
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Virtual env: $(VENV)"
	@echo "Packages:"
	@$(PIP) list

.PHONY: update-deps
update-deps:
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt

.PHONY: setup-hooks
setup-hooks:
	cp pre-commit-check.sh .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit