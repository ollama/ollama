export interface LaunchIntegration {
  id: string;
  name: string;
  command: string;
  description: string;
  icon?: string;
  darkIcon?: string;
  iconClassName?: string;
}

const launchIntegrations = [
  {
    "id": "openclaw",
    "name": "OpenClaw",
    "command": "ollama launch openclaw",
    "description": "Personal AI with 100+ skills",
    "icon": "/launch-icons/openclaw.svg"
  },
  {
    "id": "claude",
    "name": "Claude Code",
    "command": "ollama launch claude",
    "description": "Anthropic's coding tool with subagents",
    "icon": "/launch-icons/claude.svg",
    "iconClassName": "h-7 w-7"
  },
  {
    "id": "opencode",
    "name": "OpenCode",
    "command": "ollama launch opencode",
    "description": "Anomaly's open-source coding agent",
    "icon": "/launch-icons/opencode.svg",
    "iconClassName": "h-7 w-7 rounded"
  },
  {
    "id": "hermes",
    "name": "Hermes Agent",
    "command": "ollama launch hermes",
    "description": "Self-improving AI agent built by Nous Research",
    "icon": "/launch-icons/hermes-agent.svg",
    "iconClassName": "h-7 w-7"
  },
  {
    "id": "codex",
    "name": "Codex",
    "command": "ollama launch codex",
    "description": "OpenAI's open-source coding agent",
    "icon": "/launch-icons/codex.svg",
    "darkIcon": "/launch-icons/codex-dark.svg",
    "iconClassName": "h-7 w-7"
  },
  {
    "id": "copilot",
    "name": "Copilot CLI",
    "command": "ollama launch copilot",
    "description": "GitHub's AI coding agent for the terminal",
    "icon": "/launch-icons/copilot.svg",
    "iconClassName": "h-7 w-7"
  },
  {
    "id": "droid",
    "name": "Droid",
    "command": "ollama launch droid",
    "description": "Factory's coding agent across terminal and IDEs",
    "icon": "/launch-icons/droid.svg"
  },
  {
    "id": "pi",
    "name": "Pi",
    "command": "ollama launch pi",
    "description": "Minimal AI agent toolkit with plugin support",
    "icon": "/launch-icons/pi.svg",
    "darkIcon": "/launch-icons/pi-dark.svg",
    "iconClassName": "h-7 w-7"
  }
] satisfies LaunchIntegration[];

export default launchIntegrations;
