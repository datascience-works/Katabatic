.PHONY: clear-cache install-core install-ganblr install-great install-all setup-dev help ci lint format security test build integration hooks

# Core installation (minimal dependencies)
install-core:
	@echo "Installing core Katabatic dependencies..."
	poetry install

# Install GANBLR model dependencies
install-ganblr:
	@echo "Installing GANBLR model dependencies..."
	poetry install -E ganblr

# Install GReaT model dependencies
install-great:
	@echo "Installing GReaT model dependencies..."
	poetry install -E great

# Install all model dependencies
install-all:
	@echo "Installing all model dependencies..."
	poetry install -E all

# Setup development environment for specific model
setup-ganblr-dev:
	@echo "Setting up GANBLR development environment..."
	@chmod +x scripts/setup_ganblr.sh
	@./scripts/setup_ganblr.sh

setup-great-dev:
	@echo "Setting up GReaT development environment..."
	@chmod +x scripts/setup_great.sh
	@./scripts/setup_great.sh

# Setup full development environment
setup-dev:
	@echo "Setting up full development environment..."
	poetry install -E dev -E all
	python dev_deps.py install all

clear-cache:
	@echo "Clearing Python cache directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cache cleared successfully!"

# Quality checks (mirrors CI lint-and-test job)
# --- Quality checks (mirror the CI lint-and-test job) ---

# Run the fast CI checks locally before pushing / opening a PR
ci: lint security test build
	@echo "All local CI checks passed."

lint:
	@echo "Running ruff..."
	poetry run ruff check katabatic tests

format:
	@echo "Auto-formatting with ruff..."
	poetry run ruff format katabatic tests
	poetry run ruff check --fix katabatic tests

security:
	@echo "Running bandit security scan..."
	poetry run bandit -r katabatic -ll

test:
	@echo "Running fast tests with coverage..."
	poetry run pytest -q --cov=katabatic --cov-report=term-missing

build:
	@echo "Building wheel..."
	poetry build

# Run an integration test for a specific model.
integration:
	@echo "Running integration tests for $(MODEL)..."
	poetry install --with dev -E $(MODEL)
	poetry run pytest -m "integration and $(MODEL)" -q

# Install and activate pre-commit hooks.
hooks:
	@echo "Installing pre-commit hooks..."
	poetry run pre-commit install
	poetry run pre-commit run --all-files

# Show help
help:
	@echo "Katabatic Development Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install-core       Install core dependencies only"
	@echo "  make install-ganblr     Install with GANBLR dependencies"
	@echo "  make install-great      Install with GReaT dependencies"
	@echo "  make install-all        Install all model dependencies"
	@echo ""
	@echo "Development Setup:"
	@echo "  make setup-ganblr-dev   Setup isolated GANBLR dev environment"
	@echo "  make setup-great-dev    Setup isolated GReaT dev environment"
	@echo "  make setup-dev          Setup full development environment"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clear-cache        Clear Python cache files"
	@echo "  make help               Show this help message"
	@echo ""
	@echo "Quality / CI:"
	@echo "  make ci                 Run all local CI checks (lint, security, test, build)"
	@echo "  make lint               Run ruff lint + format check"
	@echo "  make format             Auto-fix formatting and lint issues"
	@echo "  make test               Run fast tests with coverage"
	@echo "  make integration MODEL=ctgan   Run integration tests for a model"
	@echo "  make hooks              Install pre-commit hooks"
