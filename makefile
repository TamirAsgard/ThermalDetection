# Makefile for Thermal Detection System
# Provides convenient commands for development, testing, and deployment

.PHONY: help install test build run clean docker lint format docs deploy

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := thermal-detection-system
VERSION := $(shell grep -E '^__version__' src/__init__.py | cut -d'"' -f2 2>/dev/null || echo "1.0.0")
DOCKER_IMAGE := thermal-detection
DOCKER_TAG := $(VERSION)
COMPOSE_FILE := docker-compose.yml

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Helper function to print colored output
define print_color
	@echo -e "$(1)[$(PROJECT_NAME)]$(NC) $(2)"
endef

# =============================================================================
# Help and Information
# =============================================================================

help: ## Show this help message
	@echo -e "$(BLUE)Thermal Detection System - Makefile Commands$(NC)"
	@echo -e "$(BLUE)===============================================$(NC)"
	@echo ""
	@echo -e "$(GREEN)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $1, $2}' $(MAKEFILE_LIST)
	@echo ""
	@echo -e "$(GREEN)Usage examples:$(NC)"
	@echo "  make install          # Install all dependencies"
	@echo "  make test             # Run all tests"
	@echo "  make docker-build     # Build Docker image"
	@echo "  make docker-run       # Run with Docker"
	@echo "  make dev              # Start development environment"

info: ## Show project information
	$(call print_color,$(BLUE),Project: $(PROJECT_NAME))
	$(call print_color,$(BLUE),Version: $(VERSION))
	$(call print_color,$(BLUE),Python: $(shell $(PYTHON) --version))
	$(call print_color,$(BLUE),Docker: $(shell docker --version 2>/dev/null || echo "Not installed"))

# =============================================================================
# Development Environment
# =============================================================================

install: ## Install all dependencies
	$(call print_color,$(GREEN),Installing dependencies...)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r tests/test_requirements.txt
	$(call print_color,$(GREEN),Dependencies installed successfully)

install-dev: install ## Install development dependencies
	$(call print_color,$(GREEN),Installing development tools...)
	$(PIP) install black flake8 mypy pre-commit jupyter
	pre-commit install
	$(call print_color,$(GREEN),Development environment ready)

setup: ## Setup project directories and download models
	$(call print_color,$(GREEN),Setting up project structure...)
	mkdir -p logs data backups test_data/images model sounds config/backups
	$(PYTHON) setup.py develop
	$(call print_color,$(GREEN),Project setup complete)

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	$(call print_color,$(GREEN),Running tests...)
	$(PYTHON) -m pytest tests/ -v --tb=short
	$(call print_color,$(GREEN),All tests completed)

test-unit: ## Run unit tests only
	$(call print_color,$(GREEN),Running unit tests...)
	$(PYTHON) -m pytest tests/ -v -m "unit" --tb=short

test-integration: ## Run integration tests
	$(call print_color,$(GREEN),Running integration tests...)
	$(PYTHON) -m pytest tests/ -v -m "integration" --tb=short

test-coverage: ## Run tests with coverage
	$(call print_color,$(GREEN),Running tests with coverage...)
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term
	$(call print_color,$(GREEN),Coverage report generated in htmlcov/)

test-quick: ## Run quick tests (excluding slow tests)
	$(call print_color,$(GREEN),Running quick tests...)
	$(PYTHON) tests/test_runner.py --quick

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run code linting
	$(call print_color,$(GREEN),Running linters...)
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports
	$(call print_color,$(GREEN),Linting completed)

format: ## Format code with black
	$(call print_color,$(GREEN),Formatting code...)
	black src/ tests/ --line-length=100
	isort src/ tests/
	$(call print_color,$(GREEN),Code formatted)

format-check: ## Check code formatting
	$(call print_color,$(GREEN),Checking code format...)
	black --check src/ tests/ --line-length=100
	isort --check-only src/ tests/

# =============================================================================
# Application Commands
# =============================================================================

run: ## Run the thermal detection system
	$(call print_color,$(GREEN),Starting thermal detection system...)
	$(PYTHON) -m src.main

run-api: ## Run the API server only
	$(call print_color,$(GREEN),Starting API server...)
	$(PYTHON) -m src.api.app

run-dev: ## Run in development mode with auto-reload
	$(call print_color,$(GREEN),Starting development server...)
	$(PYTHON) -m src.api.app --reload --log-level debug

dashboard: ## Open the dashboard in browser
	$(call print_color,$(GREEN),Opening dashboard...)
	python -m webbrowser http://localhost:8000/dashboard

# =============================================================================
# Docker Commands
# =============================================================================

docker-build: ## Build Docker image
	$(call print_color,$(GREEN),Building Docker image...)
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -t $(DOCKER_IMAGE):latest .
	$(call print_color,$(GREEN),Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG))

docker-build-dev: ## Build development Docker image
	$(call print_color,$(GREEN),Building development Docker image...)
	docker build --target development -t $(DOCKER_IMAGE):dev .

docker-run: ## Run with Docker
	$(call print_color,$(GREEN),Starting Docker container...)
	docker-compose up -d
	$(call print_color,$(GREEN),Container started. Access: http://localhost:8000)

docker-run-dev: ## Run development environment with Docker
	$(call print_color,$(GREEN),Starting development container...)
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

docker-stop: ## Stop Docker containers
	$(call print_color,$(GREEN),Stopping Docker containers...)
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f thermal-detection

docker-shell: ## Open shell in running container
	docker-compose exec thermal-detection bash

docker-clean: ## Clean Docker images and containers
	$(call print_color,$(YELLOW),Cleaning Docker resources...)
	docker-compose down -v
	docker system prune -f
	$(call print_color,$(GREEN),Docker cleanup completed)

# =============================================================================
# Development Shortcuts
# =============================================================================

dev: docker-run-dev ## Start development environment

demo: ## Run demo mode
	$(call print_color,$(GREEN),Starting demo mode...)
	THERMAL_DEMO_MODE=true $(PYTHON) -m src.api.app

pi: ## Run Raspberry Pi configuration
	$(call print_color,$(GREEN),Starting Raspberry Pi mode...)
	docker-compose -f docker-compose.yml -f docker-compose.pi.yml up

# =============================================================================
# Deployment
# =============================================================================

build: docker-build ## Build for deployment

deploy: ## Deploy to production (customize as needed)
	$(call print_color,$(GREEN),Deploying to production...)
	docker-compose -f docker-compose.yml up -d
	$(call print_color,$(GREEN),Deployment completed)

# =============================================================================
# Documentation
# =============================================================================

docs: ## Generate documentation
	$(call print_color,$(GREEN),Generating documentation...)
	sphinx-build -b html docs/ docs/_build/html/
	$(call print_color,$(GREEN),Documentation generated in docs/_build/html/)

docs-serve: ## Serve documentation locally
	$(call print_color,$(GREEN),Serving documentation...)
	cd docs/_build/html && $(PYTHON) -m http.server 8080

# =============================================================================
# Maintenance
# =============================================================================

clean: ## Clean temporary files
	$(call print_color,$(GREEN),Cleaning temporary files...)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	$(call print_color,$(GREEN),Cleanup completed)

clean-all: clean docker-clean ## Clean everything including Docker

update: ## Update dependencies
	$(call print_color,$(GREEN),Updating dependencies...)
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r tests/test_requirements.txt
	$(call print_color,$(GREEN),Dependencies updated)

backup: ## Create backup of data and logs
	$(call print_color,$(GREEN),Creating backup...)
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ logs/ config/
	$(call print_color,$(GREEN),Backup created)

# =============================================================================
# Quick Start Commands
# =============================================================================

quickstart: install setup test docker-build ## Complete setup and test
	$(call print_color,$(GREEN),Quick start completed!)
	$(call print_color,$(BLUE),Next steps:)
	$(call print_color,$(BLUE),  1. Run 'make run' to start the system)
	$(call print_color,$(BLUE),  2. Open http://localhost:8000/dashboard)
	$(call print_color,$(BLUE),  3. Or run 'make docker-run' to use Docker)

init: ## Initialize new installation
	$(call print_color,$(GREEN),Initializing thermal detection system...)
	make install
	make setup
	make test-quick
	$(call print_color,$(GREEN),Initialization completed!)

# =============================================================================
# System Information
# =============================================================================

status: ## Show system status
	$(call print_color,$(BLUE),System Status:)
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "OpenCV: $(shell $(PYTHON) -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Git: $(shell git --version 2>/dev/null || echo 'Not installed')"
	@echo "Cameras: $(shell ls /dev/video* 2>/dev/null | wc -l) detected"

health-check: ## Check system health
	$(call print_color,$(GREEN),Checking system health...)
	$(PYTHON) -c "
import sys
import importlib
required_modules = ['cv2', 'numpy', 'fastapi', 'uvicorn', 'yaml']
for module in required_modules:
    try:
        importlib.import_module(module)
        print(f'✓ {module}')
    except ImportError:
        print(f'✗ {module} - MISSING')
        sys.exit(1)
print('All required modules are available')
"
	$(call print_color,$(GREEN),System health check passed)

# =============================================================================
# Special targets
# =============================================================================

# Create .env file if it doesn't exist
.env:
	$(call print_color,$(GREEN),Creating .env file...)
	echo "THERMAL_LOG_LEVEL=INFO" > .env
	echo "THERMAL_DEMO_MODE=true" >> .env
	echo "THERMAL_API_HOST=0.0.0.0" >> .env
	echo "THERMAL_API_PORT=8000" >> .env

# Install git hooks
.git/hooks/pre-commit:
	$(call print_color,$(GREEN),Installing git hooks...)
	pre-commit install

# Declare phony targets
.PHONY: $(MAKECMDGOALS)