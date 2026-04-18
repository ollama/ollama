#!/usr/bin/env bash
# scripts/onboard.sh — convenience wrapper to setup development environment
# Usage: ./scripts/onboard.sh [--dry-run] [--yes]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.local.yml"

compose_cmd() {
  if command -v docker-compose &>/dev/null; then
    echo docker-compose
    return
  fi

  if docker compose version &>/dev/null; then
    echo docker compose
    return
  fi

  echo "Docker Compose is not installed"
  exit 1
}

DRY_RUN=false
ASSUME_YES=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true; shift ;;
    --yes)
      ASSUME_YES=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--dry-run] [--yes]"; exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 2 ;;
  esac
done

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = false ]; then
    eval "$@"
  fi
}

confirm() {
  if [ "$ASSUME_YES" = true ]; then
    return 0
  fi
  read -r -p "$1 [y/N]: " resp
  case "$resp" in
    [Yy]*) return 0 ;;
    *) return 1 ;;
  esac
}

echo "== Ollama Onboarding Helper =="

# 1) Virtualenv and dependencies
cd "$PROJECT_ROOT"
if [ ! -d "venv" ]; then
  run "python3 -m venv venv"
fi
run "source venv/bin/activate"

# prefer setup-dev.sh if present
if [ -f "./scripts/setup-dev.sh" ]; then
  run "bash ./scripts/setup-dev.sh"
else
  if [ -f "requirements/dev.txt" ]; then
    run "pip install -r requirements/dev.txt"
  else
    run "pip install -e .[dev] || pip install -r requirements.txt"
  fi
fi

# 2) Pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
  run "pre-commit install || true"
  run "pre-commit run --all-files || true"
fi

# 3) Env file setup
cd "$PROJECT_ROOT"

if [ ! -f ".env.dev" ]; then
  run "cp .env.example .env.dev"
  REAL_IP=$(hostname -I | awk '{print $1}')
  run "sed -i \"s|PUBLIC_API_URL=.*|PUBLIC_API_URL=http://$REAL_IP:8000|\" .env.dev || true"
  echo "Created .env.dev and set PUBLIC_API_URL to http://$REAL_IP:8000"
fi

# 4) Docker stack (optional)
if confirm "Start local Docker compose stack (docker-compose.local.yml)?"; then
  if [ -f "$COMPOSE_FILE" ]; then
    DOCKER_COMPOSE="$(compose_cmd)"
    run "$DOCKER_COMPOSE -f \"$COMPOSE_FILE\" up -d --remove-orphans --build"
  else
    echo "docker-compose.local.yml not found, skipping"
  fi
fi

# 5) DB migrations
if confirm "Run DB migrations (alembic upgrade head)?"; then
  if [ -f "alembic.ini" ] || [ -d "alembic" ]; then
    DOCKER_COMPOSE="$(compose_cmd)"
    run "$DOCKER_COMPOSE -f \"$COMPOSE_FILE\" exec -T api alembic upgrade head || alembic upgrade head"
  else
    echo "No alembic configuration found, skipping migrations"
  fi
fi

# 6) Run verification checks
if [ -f "./scripts/verify-elite-setup.sh" ]; then
  run "bash ./scripts/verify-elite-setup.sh"
else
  echo "verify script not found; recommended checks: pytest, mypy, ruff, pip-audit"
fi

# 7) Final instructions
cat <<EOF

Onboarding steps complete (dry-run=$DRY_RUN).
Next recommended steps:
  - Activate venv: source venv/bin/activate
  - Run tests: pytest tests/ -v --cov=ollama
  - Type check: mypy ollama/ --strict
  - Lint: ruff check ollama/ --fix
  - Make your first feature branch and PR

Make the script executable: chmod +x scripts/onboard.sh
EOF
