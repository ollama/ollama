.PHONY: help local-start local-stop local-restart local-logs \
        test test-unit test-integration test-coverage \
        type-check lint format security-audit \
        db-migrate db-reset \
        run-dev

.DEFAULT_GOAL := help

# ============================================================================
# LOCAL DEVELOPMENT
# ============================================================================

help: ## Display this help message
	@echo "Ollama Development Tasks"
	@echo ""
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

local-start: ## Start the complete local development stack
	@echo "Starting local development stack..."
	@bash scripts/local-start.sh

local-stop: ## Stop all local services (preserves data)
	@echo "Stopping local services..."
	@bash scripts/local-stop.sh

local-restart: local-stop local-start ## Restart all local services

local-logs: ## Show logs from all services (use Ctrl+C to exit)
	docker-compose -f docker-compose.local.yml logs -f

local-logs-api: ## Show logs from API service only
	docker-compose -f docker-compose.local.yml logs -f api

local-logs-db: ## Show logs from PostgreSQL
	docker-compose -f docker-compose.local.yml logs -f postgres

local-logs-redis: ## Show logs from Redis
	docker-compose -f docker-compose.local.yml logs -f redis

local-logs-ollama: ## Show logs from Ollama
	docker-compose -f docker-compose.local.yml logs -f ollama

local-status: ## Show status of all local services
	docker-compose -f docker-compose.local.yml ps

local-shell-api: ## Open shell in API container
	docker-compose -f docker-compose.local.yml exec api /bin/bash

local-shell-db: ## Open psql shell in PostgreSQL
	docker-compose -f docker-compose.local.yml exec postgres psql -U ollama -d ollama

local-shell-redis: ## Open redis-cli in Redis
	docker-compose -f docker-compose.local.yml exec redis redis-cli

# ============================================================================
# TESTING
# ============================================================================

test: ## Run all tests with coverage
	pytest tests/ -v --cov=ollama --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests (requires services running)
	pytest tests/integration/ -v

test-coverage: ## Run tests and generate coverage report
	pytest tests/ -v --cov=ollama --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw tests/ -v

test-quick: ## Run fast tests only (skip slow tests)
	pytest tests/ -v -m "not slow"

# ============================================================================
# CODE QUALITY
# ============================================================================

type-check: ## Run mypy type checker
	mypy ollama/ --strict

lint: ## Run linting with ruff
	ruff check ollama/ tests/

format: ## Format code with black
	black ollama/ tests/ --line-length=100
	isort ollama/ tests/

format-check: ## Check code formatting without changes
	black ollama/ tests/ --line-length=100 --check
	isort ollama/ tests/ --check-only

security-audit: ## Run security audit with pip-audit
	pip-audit

all-checks: type-check lint test security-audit ## Run all quality checks

fix-all: format lint type-check ## Format, lint, and type-check all code

# ============================================================================
# DATABASE MIGRATIONS
# ============================================================================

db-migrate: ## Run database migrations
	alembic upgrade head

db-makemigration: ## Create a new database migration
	@read -p "Migration name: " name; \
	alembic revision -m "$$name"

db-rollback: ## Rollback to previous migration
	alembic downgrade -1

db-reset: ## Reset database (WARNING: deletes all data)
	docker-compose -f docker-compose.local.yml down -v postgres
	docker-compose -f docker-compose.local.yml up -d postgres
	sleep 10
	$(MAKE) db-migrate

db-shell: ## Open database shell
	docker-compose -f docker-compose.local.yml exec postgres psql -U ollama -d ollama

# ============================================================================
# RUNNING THE APPLICATION
# ============================================================================

run-dev: ## Run API server locally with auto-reload (requires dependencies)
	uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000

run-docker: ## Run API server in Docker (with live reload)
	docker-compose -f docker-compose.local.yml up api

build-docker: ## Build Docker image
	docker-compose -f docker-compose.local.yml build api

# ============================================================================
# UTILITIES
# ============================================================================

clean: ## Remove build artifacts and cache files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name *.egg-info -exec rm -rf {} +
	find . -type f -name .coverage -delete
	rm -rf build/ dist/ htmlcov/

clean-docker: ## Remove Docker containers and images
	docker-compose -f docker-compose.local.yml down
	docker-compose -f docker-compose.local.yml rm -f

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

pre-commit: format lint type-check ## Run pre-commit checks

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs: ## Build documentation
	cd docs && make html
	@echo "Documentation: docs/_build/html/index.html"

docs-serve: ## Serve documentation locally
	python -m http.server --directory docs/_build/html 8080

# ============================================================================
# GIT
# ============================================================================

git-status: ## Show git status
	git status

git-commit-all: ## Stage and commit all changes
	@read -p "Commit message: " message; \
	git add -A && git commit -S -m "$$message"

# ============================================================================
# EXAMPLES
# ============================================================================

examples: ## Show common examples
	@echo "COMMON TASKS:"
	@echo ""
	@echo "Start development stack:"
	@echo "  make local-start"
	@echo ""
	@echo "Run tests:"
	@echo "  make test"
	@echo ""
	@echo "Check code quality:"
	@echo "  make all-checks"
	@echo ""
	@echo "Format code:"
	@echo "  make format"
	@echo ""
	@echo "View logs:"
	@echo "  make local-logs"
	@echo ""
	@echo "Open database shell:"
	@echo "  make db-shell"
	@echo ""
	@echo "Reset everything:"
	@echo "  make local-stop"
	@echo "  make clean-docker"
	@echo "  make local-start"
