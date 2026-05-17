import { useSettings } from "@/hooks/useSettings";
import CopyButton from "@/components/CopyButton";

interface LaunchCommand {
  id: string;
  name: string;
  command: string;
  description: string;
  icon: string;
  darkIcon?: string;
  iconClassName?: string;
  borderless?: boolean;
}

const LAUNCH_COMMANDS: LaunchCommand[] = [
  {
    id: "claude",
    name: "Claude Code",
    command: "ollama launch claude",
    description: "Anthropic's coding tool with subagents",
    icon: "/launch-icons/claude-code.svg",
    iconClassName: "h-7 w-7",
  },
  {
    id: "codex-app",
    name: "Codex App",
    command: "ollama launch codex-app",
    description: "An AI agent you can delegate real work to, by OpenAI",
    icon: "/launch-icons/codex-app.png",
    iconClassName: "h-full w-full",
  },
  {
    id: "hermes",
    name: "Hermes Agent",
    command: "ollama launch hermes",
    description: "Self-improving AI agent built by Nous Research",
    icon: "/launch-icons/hermes-agent.svg",
    iconClassName: "h-7 w-7",
  },
  {
    id: "openclaw",
    name: "OpenClaw",
    command: "ollama launch openclaw",
    description: "Personal AI with 100+ skills",
    icon: "/launch-icons/openclaw.svg",
  },
  {
    id: "opencode",
    name: "OpenCode",
    command: "ollama launch opencode",
    description: "Anomaly's open-source coding agent",
    icon: "/launch-icons/opencode.svg",
    iconClassName: "h-7 w-7 rounded",
  },
  {
    id: "codex",
    name: "Codex",
    command: "ollama launch codex",
    description: "OpenAI's open-source coding agent",
    icon: "/launch-icons/codex.svg",
    darkIcon: "/launch-icons/codex-dark.svg",
    iconClassName: "h-7 w-7",
  },
  {
    id: "copilot",
    name: "Copilot CLI",
    command: "ollama launch copilot",
    description: "GitHub's AI coding agent for the terminal",
    icon: "/launch-icons/copilot.svg",
    darkIcon: "/launch-icons/copilot-dark.svg",
    iconClassName: "h-7 w-7",
  },
  {
    id: "droid",
    name: "Droid",
    command: "ollama launch droid",
    description: "Factory's coding agent across terminal and IDEs",
    icon: "/launch-icons/droid.svg",
  },
  {
    id: "pi",
    name: "Pi",
    command: "ollama launch pi",
    description: "Minimal AI agent toolkit with plugin support",
    icon: "/launch-icons/pi.svg",
    darkIcon: "/launch-icons/pi-dark.svg",
    iconClassName: "h-7 w-7",
  },
];

export default function LaunchCommands() {
  const isWindows = navigator.platform.toLowerCase().includes("win");
  const { setSettings } = useSettings();

  const renderCommandCard = (item: LaunchCommand) => (
    <div key={item.command} className="w-full text-left">
      <div className="flex items-start gap-4 sm:gap-5">
        <div
          aria-hidden="true"
          className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg overflow-hidden ${item.borderless ? "" : "border border-neutral-200 bg-white dark:border-neutral-700 dark:bg-neutral-900"}`}
        >
          {item.darkIcon ? (
            <picture>
              <source srcSet={item.darkIcon} media="(prefers-color-scheme: dark)" />
              <img src={item.icon} alt="" className={`${item.iconClassName ?? "h-8 w-8"} rounded-sm`} />
            </picture>
          ) : (
            <img src={item.icon} alt="" className={item.borderless ? "h-full w-full rounded-xl" : `${item.iconClassName ?? "h-8 w-8"} rounded-sm`} />
          )}
        </div>

        <div className="min-w-0 flex-1">
          <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
            {item.name}
          </span>
          <p className="mt-0.5 text-xs text-neutral-500 dark:text-neutral-400">
            {item.description}
          </p>
          <div className="mt-2 flex items-center gap-2 rounded-xl border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800 px-3 py-2">
            <code className="min-w-0 flex-1 truncate text-xs text-neutral-600 dark:text-neutral-300">
              {item.command}
            </code>
            <CopyButton
              content={item.command}
              size="md"
              title="Copy command to clipboard"
              className="text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-200 hover:bg-neutral-200/60 dark:hover:bg-neutral-700/70"
              onCopy={() => {
                setSettings({ LastHomeView: item.id }).catch(() => { });
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <main className="flex h-screen w-full flex-col relative">
      <section
        className={`flex-1 overflow-y-auto overscroll-contain relative min-h-0 ${isWindows ? "xl:pt-4" : "xl:pt-8"}`}
      >
        <div className="max-w-[730px] mx-auto w-full px-4 pt-4 pb-20 sm:px-6 sm:pt-6 sm:pb-24 lg:px-8 lg:pt-8 lg:pb-28">
          <h1 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
            Launch
          </h1>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            Copy a command and run it in your terminal.
          </p>

          <div className="mt-6 grid gap-7">
            {LAUNCH_COMMANDS.map(renderCommandCard)}
          </div>
        </div>
      </section>
    </main>
  );
}
