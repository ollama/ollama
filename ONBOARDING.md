# Onboarding: Developer setup & contribution checklist

Welcome — this file collects the exact steps to get a developer productive quickly in this repository.

## Quick start (5–10 minutes) ✅
1. Clone the repo and change directory:

```bash
git clone https://github.com/kushin77/ollama.git
cd ollama
```

2. Run the automated bootstrap (creates venv, installs dev deps, installs git hooks):

```bash
# Preferred
./scripts/bootstrap.sh
# Or explicit dev setup
./scripts/setup-dev.sh
```

3. Activate the venv:

```bash
source venv/bin/activate
```

4. Copy and configure env files (use REAL_IP for local dev):

```bash
cp .env.example .env.dev
export REAL_IP=$(hostname -I | awk '{print $1}')
sed -i "s|PUBLIC_API_URL=.*|PUBLIC_API_URL=http://$REAL_IP:8000|" .env.dev
```

5. Start the local stack (development):

```bash
docker-compose -f docker-compose.local.yml up -d
```

6. Apply DB migrations & seed (optional):

```bash
docker-compose exec api alembic upgrade head
# Seed database (if needed)
docker-compose exec api python -m scripts.seed_database
```

7. Start the API server (if not using docker):

```bash
uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000
```

8. Frontend (if working on UI):

```bash
cd frontend
npm ci
cp .env.example .env.local # edit for Firebase etc.
npm run dev
```

---

## Tests & quality (required before PR) 🧪
- Run tests: `pytest tests/ -v --cov=ollama`
- Type checks: `mypy ollama/ --strict`
- Lint & format: `ruff check ollama/ --fix` and `black ollama/ tests/`
- Security scan: `pip-audit`
- Run verification: `bash scripts/verify-elite-setup.sh` and `./scripts/run-all-checks.sh`
- Pre-commit hooks: `pre-commit install`

---

## Making changes & PRs ✍️
- Branch naming: `feature/..`, `bugfix/..`, `refactor/..` (see `CONTRIBUTING.md`)
- GPG-signed commits required for `main`/`develop`: `git commit -S -m "type(scope): description"`
- Include tests & docs for user-visible changes
- Run checks locally before creating PR

---

## Quick verification checklist (for maintainers)
- [ ] New files documented in `ONBOARDING.md` and referenced where appropriate
- [ ] `scripts/onboard.sh` added and documented
- [ ] CI passes on PR
- [ ] PR reviewed and merged
- [ ] Any related issues referenced and closed

---

## Using `scripts/onboard.sh` (recommended)
- Make it executable: `chmod +x scripts/onboard.sh`
- Run it locally (it will prompt):

```bash
./scripts/onboard.sh
```

If you prefer unattended runs use `--yes` to skip prompts or `--dry-run` to preview the steps.

---

If anything is missing or you want the script to perform additional tasks (e.g., run e2e tests or seed a demo dataset), open a PR on this branch or request a follow-up and I will extend the automation.

Welcome aboard! 🎉
