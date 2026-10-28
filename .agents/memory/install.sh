#!/usr/bin/env bash
#
# install.sh puts agent-mem in ~/.agents/memory/bin, links it onto PATH for
# shell-driven agents like pi, and prints the Zed wiring for this machine.

set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)/agent_memory.py"
home="${AGENT_MEMORY_HOME:-$HOME/.agents/memory}"
bin="$home/bin/agent-mem"
link="$HOME/.local/bin/agent-mem"

if ! command -v python3 >/dev/null 2>&1; then
	echo "python3 is required; xcode-select --install provides it" >&2
	exit 1
fi

mkdir -p "$home/bin" "$(dirname "$link")"
install -m 0755 "$src" "$bin"
ln -sf "$bin" "$link"
"$bin" --version

case ":$PATH:" in
*":$HOME/.local/bin:"*) path_note="" ;;
*) path_note="
~/.local/bin is not on PATH. Add this to ~/.zshrc so pi's shell finds agent-mem:

  export PATH=\"\$HOME/.local/bin:\$PATH\"
" ;;
esac

cat <<EOF

Installed $bin
Linked    $link
$path_note
Zed: add this to ~/.config/zed/settings.json, then reopen the agent panel.

  "context_servers": {
    "agent-memory": {
      "source": "custom",
      "command": "$bin",
      "args": ["mcp"],
      "env": {}
    }
  }

Each project: run once from the repository root.

  agent-mem init --agents-md

That creates .agents/memory/sessions/ (git-ignored) and adds the session
protocol to AGENTS.md, which Zed and pi both read.
EOF
