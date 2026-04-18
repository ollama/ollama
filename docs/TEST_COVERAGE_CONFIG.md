"""Test Coverage Baseline & Configuration

This file documents the test coverage targets and baseline metrics for Ollama.

Generated: January 13, 2026
Target Coverage: ≥90% overall, 100% for critical paths
"""

# Coverage targets by module
COVERAGE_TARGETS = {
    # Critical paths (100% coverage required)
    "ollama/services/auth": 100,  # Authentication is security-critical
    "ollama/middleware/rate_limit": 100,  # Rate limiting must be reliable
    "ollama/exceptions": 100,  # Exception hierarchy must be complete
    "ollama/repositories": 95,  # Data access layer critical
    "ollama/middleware": 95,  # Middleware must be reliable
    # API layer (90%+ required)
    "ollama/api": 90,
    "ollama/api/routes": 90,
    "ollama/api/schemas": 85,  # Many small schemas
    "ollama/api/dependencies": 95,
    # Services (90%+ required)
    "ollama/services": 90,
    "ollama/services/inference": 90,
    "ollama/services/models": 90,
    "ollama/services/embeddings": 90,
    "ollama/services/cache": 90,
    # Monitoring (85%+ acceptable)
    "ollama/monitoring": 85,
    # Config (80%+ acceptable - many env vars)
    "ollama/config": 80,
    # Overall target
    "overall": 90,
}

# Critical paths that MUST have 100% coverage
CRITICAL_PATHS = [
    "ollama/services/auth.py",
    "ollama/middleware/rate_limit.py",
    "ollama/exceptions.py",
]

# Modules/functions to focus coverage on
FOCUS_AREAS = {
    "authentication": [
        "ollama/services/auth.py::generate_api_key",
        "ollama/services/auth.py::verify_api_key",
        "ollama/api/dependencies.py::get_current_user",
    ],
    "rate_limiting": [
        "ollama/middleware/rate_limit.py::RateLimitMiddleware",
        "ollama/middleware/rate_limit.py::RedisRateLimiter",
        "ollama/middleware/rate_limit.py::RateLimiter",
    ],
    "error_handling": [
        "ollama/api/routes/health.py",
        "ollama/api/routes/models.py",
        "ollama/api/routes/generate.py",
    ],
    "database": [
        "ollama/repositories/",
        "ollama/models.py",
    ],
    "caching": [
        "ollama/services/cache.py",
    ],
}

# Known uncoverable code (excluded from coverage)
UNCOVERABLE = {
    "pragma: no cover": "Explicitly marked as uncoverable",
    "__main__": "Entry point code",
    "TYPE_CHECKING": "Type-checking only imports",
    "NotImplementedError": "Future implementations",
    "except KeyboardInterrupt": "User interruption",
}

# Test organization
TEST_STRUCTURE = {
    "unit": {
        "description": "Fast, isolated tests (no external dependencies)",
        "location": "tests/unit/",
        "target_coverage": "85%+",
        "timeout": 1000,  # milliseconds
    },
    "integration": {
        "description": "Tests with real services (Redis, PostgreSQL, etc)",
        "location": "tests/integration/",
        "target_coverage": "70%+",
        "timeout": 5000,  # milliseconds
    },
    "e2e": {
        "description": "End-to-end tests (full application stack)",
        "location": "tests/e2e/",
        "target_coverage": "60%+",
        "timeout": 30000,  # milliseconds
    },
}

# Coverage gaps to address (as of January 13, 2026)
COVERAGE_GAPS = {
    "inference_engine": {
        "module": "ollama/services/inference",
        "description": "Model inference logic coverage incomplete",
        "priority": "HIGH",
        "estimate": "2-3 hours",
        "status": "TODO",
    },
    "embedding_service": {
        "module": "ollama/services/embeddings",
        "description": "Embedding generation not fully tested",
        "priority": "HIGH",
        "estimate": "1-2 hours",
        "status": "TODO",
    },
    "error_scenarios": {
        "module": "ollama/api/routes",
        "description": "Error handling in API routes needs more tests",
        "priority": "MEDIUM",
        "estimate": "3-4 hours",
        "status": "TODO",
    },
    "edge_cases": {
        "module": "ollama/services/cache",
        "description": "Cache eviction and edge cases not fully covered",
        "priority": "MEDIUM",
        "estimate": "2 hours",
        "status": "TODO",
    },
}

# How to measure coverage
MEASUREMENT_GUIDE = """
Generate coverage report:
    pytest tests/ --cov=ollama --cov-report=html --cov-report=term-missing

View HTML report:
    open htmlcov/index.html

Show uncovered lines:
    pytest tests/ --cov=ollama --cov-report=term-missing | grep -E "TOTAL|[0-9]%"

Coverage by module:
    pytest tests/ --cov=ollama --cov-report=term-missing:skip-covered

Exclude specific paths:
    pytest tests/ --cov=ollama --cov=!ollama/migrations --cov-report=html
"""

# Pytest configuration (in pyproject.toml)
PYTEST_CONFIG = """
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=ollama --cov-report=html --cov-report=term-missing"
asyncio_mode = "auto"
python_files = ["test_*.py", "*_test.py"]

[tool.coverage.run]
branch = true
source = ["ollama"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__main__.py",
    "*/migrations/*",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "if typing.TYPE_CHECKING:",
]
"""

# Testing best practices
TESTING_BEST_PRACTICES = """
1. Unit Tests (tests/unit/)
   - Test single functions/methods in isolation
   - Mock external dependencies (DB, Redis, etc)
   - Keep tests fast (<1s per test)
   - Use fixtures for common setup
   - Test happy path, edge cases, and errors

2. Integration Tests (tests/integration/)
   - Test multiple components together
   - Use real services (Docker, test databases)
   - Test realistic workflows
   - Can be slower (up to 5s per test)
   - Use pytest markers: @pytest.mark.integration

3. End-to-End Tests (tests/e2e/)
   - Test full application stack
   - Simulate real user workflows
   - Can be slow (up to 30s per test)
   - Use pytest markers: @pytest.mark.e2e
   - Run on staging/production-like environments

4. Coverage Measurement
   - Always measure coverage locally before pushing
   - GitHub Actions runs coverage on every PR
   - Coverage report available as artifact
   - Codecov integration for tracking trends
   - Fail CI if coverage drops below threshold

5. Common Patterns
   - Use pytest.mark for test categorization
   - Use parametrize for testing multiple inputs
   - Use fixtures for setup/teardown
   - Mock time-dependent operations
   - Test logging output with caplog fixture
   - Test async code with pytest-asyncio
"""

# Continuous integration
CI_COVERAGE_SETTINGS = """
GitHub Actions Integration:

1. Tests workflow (.github/workflows/tests.yml)
   - Runs on: Python 3.11, 3.12
   - Generates: HTML coverage report, XML for Codecov
   - Fails if: Overall coverage < 90% (configurable)

2. Coverage reporting:
   - Artifact: htmlcov/ (available for download)
   - Upload to Codecov: Automatic
   - GitHub status check: PASS/FAIL based on threshold

3. Branch protection:
   - Require all checks to pass
   - Require coverage threshold met
   - Require code review before merge
"""

print(__doc__)
