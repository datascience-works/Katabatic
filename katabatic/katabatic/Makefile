.PHONY: clear-cache install-core install-ganblr install-great install-all setup-dev help

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