.PHONY: clear-cache install-core install-model setup-dev help ci lint format security test build integration hooks contract

# Fails a target that needs MODEL when none was given.
require-model = @if [ -z "$(MODEL)" ]; then echo "Error: specify a model, e.g. make $@ MODEL=ganblr"; exit 1; fi

# Core installation (minimal dependencies)
install-core:
	@echo "Installing core Katabatic dependencies..."
	poetry install

# Install one or more models, e.g. MODEL="ganblr ctgan"
install-model:
	$(require-model)
	@echo "Installing $(MODEL) model dependencies..."
	poetry install $(addprefix -E ,$(MODEL))

# Setup development environment, plus any models given in MODEL
setup-dev:
	@echo "Setting up development environment..."
	poetry install --with dev $(addprefix -E ,$(MODEL))
	poetry run pre-commit install

clear-cache:
	@echo "Clearing Python cache directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cache cleared successfully!"

# Quality checks (mirrors CI lint-and-test job)
# Run the fast CI checks locally before pushing / opening a PR
ci: lint security test build
	@echo "All local CI checks passed."

lint:
	@echo "Running pre-commit hooks (ruff check, format, etc.)..."
	poetry run pre-commit run --all-files

format:
	@echo "Auto-formatting with ruff..."
	poetry run ruff format katabatic tests
	poetry run ruff check --fix katabatic tests

security:
	@echo "Running bandit security scan..."
	poetry run bandit -r katabatic -ll

test:
	@echo "Running fast tests with coverage..."
	poetry run pytest -q --deselect tests/test_model_registry.py::test_model_promotion_contract --cov=katabatic --cov-report=term-missing
	@echo "Checking core coverage floor (mirrors CI's lint-and-test 'Core coverage floor' step)..."
	poetry run coverage report --include="katabatic/pipeline/*,katabatic/utils/*,katabatic/datasets/*,katabatic/artifacts/*,katabatic/evaluate/*,katabatic/models/registry.py,katabatic/models/base_model.py" --fail-under=70

build:
	@echo "Building wheel..."
	poetry build

# Run the model promotion contract for a specific model (mirrors CI).
contract:
	$(require-model)
	@echo "Running model promotion contract test for $(MODEL)..."
	poetry install --with dev -E $(MODEL)
	poetry run pytest tests/test_model_registry.py -k "$(MODEL)" -v

# Run an integration test for a specific model.
integration:
	$(require-model)
	@echo "Running integration tests for $(MODEL)..."
	poetry install --with dev -E $(MODEL)
	poetry run pytest -m "integration and $(MODEL)" -q

# Install and activate pre-commit hooks.
hooks:
	@echo "Installing pre-commit hooks..."
	poetry run pre-commit install
	poetry run pre-commit install --hook-type commit-msg
	poetry run pre-commit run --all-files

# Show help
help:
	@echo "Katabatic Development Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install-core       Install core dependencies only"
	@echo "  make install-model MODEL=x   Install model deps (e.g. MODEL=ctgan or MODEL=\"ganblr ctgan\")"
	@echo ""
	@echo "Development Setup:"
	@echo "  make setup-dev          Setup dev environment + hooks (add MODEL=... for model extras)"
	@echo "  make hooks              Install and run pre-commit hooks"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clear-cache        Clear Python cache files"
	@echo "  make help               Show this help message"
	@echo ""
	@echo "Quality / CI:"
	@echo "  make ci                 Run all local CI checks (lint, security, test, build)"
	@echo "  make format             Auto-fix formatting and lint issues"
	@echo "  make test               Run fast tests with coverage"
	@echo "  make integration MODEL=ctgan   Run integration tests for a model"
	@echo "  make contract MODEL=ctgan      Run the model promotion contract for a model"
	@echo "  make hooks              Install pre-commit hooks"
