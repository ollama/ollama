#!/usr/bin/env bash
set -euo pipefail
# Helper to clone the git-rca-workspace tooling into `tools/git-rca-workspace`.
# Usage: ./scripts/setup-git-rca-workspace.sh [--submodule]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools/git-rca-workspace"

if [ "${1-}" = "--submodule" ]; then
  echo "Adding as a git submodule..."
  git submodule add https://github.com/kushin77/git-rca-workspace.git "$TOOLS_DIR"
  git submodule update --init --recursive
  echo "Submodule added at $TOOLS_DIR"
  exit 0
fi

if [ -d "$TOOLS_DIR/.git" ]; then
  echo "git-rca-workspace already present at $TOOLS_DIR"
  exit 0
fi

mkdir -p "$(dirname "$TOOLS_DIR")"
echo "Cloning git-rca-workspace into $TOOLS_DIR (shallow)..."
git clone --depth=1 https://github.com/kushin77/git-rca-workspace.git "$TOOLS_DIR"
echo "Clone complete. Review the contents of $TOOLS_DIR before running any scripts."

echo "Recommended next steps:"
echo "  - Open the workspace and add $TOOLS_DIR as a folder in VS Code."
echo "  - Review tools and optional scripts in $TOOLS_DIR before use."
echo "  - To convert this to a submodule later, run:"
echo "      git submodule add https://github.com/kushin77/git-rca-workspace.git tools/git-rca-workspace"

exit 0
