export interface IntegrationIcon {
  src: string;
  darkSrc?: string;
  className?: string;
}

export const INTEGRATION_ICONS: Record<string, IntegrationIcon> = {
  "claude-desktop": { src: "/launch-icons/claude.svg" },
  claude: { src: "/launch-icons/claude-code.svg" },
  hermes: { src: "/launch-icons/hermes-agent.svg" },
  "hermes-desktop": { src: "/launch-icons/hermes-agent.svg" },
  openclaw: { src: "/launch-icons/openclaw.svg" },
  opencode: {
    src: "/launch-icons/opencode.svg",
    className: "h-7 w-7 rounded",
  },
  codex: {
    src: "/launch-icons/codex.svg",
    darkSrc: "/launch-icons/codex-dark.svg",
  },
  copilot: {
    src: "/launch-icons/copilot.svg",
    darkSrc: "/launch-icons/copilot-dark.svg",
  },
  droid: { src: "/launch-icons/droid.svg" },
  dsh: { src: "/launch-icons/deepseek-harness.svg" },
  pi: { src: "/launch-icons/pi.svg" },
  cline: {
    src: "/launch-icons/cline.svg",
    className: "h-7 w-7 dark:invert",
  },
  omp: { src: "/launch-icons/oh-my-pi.svg" },
  pool: { src: "/launch-icons/poolside.svg" },
  qwen: { src: "/launch-icons/qwen-code.svg" },
};
