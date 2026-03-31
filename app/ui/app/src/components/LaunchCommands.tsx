import { useState } from "react";
import { copyTextToClipboard } from "@/utils/clipboard";

interface LaunchCommand {
  name: string;
  command: string;
  description: string;
}

const FEATURED_COMMANDS: LaunchCommand[] = [
  {
    name: "OpenClaw",
    command: "ollama launch openclaw",
    description: "Personal AI with 100+ skills",
  },
  {
    name: "Claude",
    command: "ollama launch claude",
    description: "Anthropic's coding tool with subagents",
  },
];

const MORE_COMMANDS: LaunchCommand[] = [
  {
    name: "Codex",
    command: "ollama launch codex",
    description: "OpenAI's open-source coding agent",
  },
  {
    name: "OpenCode",
    command: "ollama launch opencode",
    description: "Anomaly's open-source coding agent",
  },
  {
    name: "Droid",
    command: "ollama launch droid",
    description: "Factory's coding agent across terminal and IDEs",
  },
  {
    name: "Pi",
    command: "ollama launch pi",
    description: "Minimal AI agent toolkit with plugin support",
  },
];

export default function LaunchCommands() {
  const isWindows = navigator.platform.toLowerCase().includes("win");
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null);
  const [copyError, setCopyError] = useState<string | null>(null);

  const copyCommand = async (command: string) => {
    const copied = await copyTextToClipboard(command);
    if (!copied) {
      setCopyError("Unable to copy command to clipboard.");
      setTimeout(() => setCopyError(null), 2500);
      return;
    }

    setCopyError(null);
    setCopiedCommand(command);
    setTimeout(() => setCopiedCommand(null), 2000);
  };

  const renderCommandCard = (item: LaunchCommand) => (
    <div
      key={item.command}
      className="w-full rounded-xl border border-neutral-200 dark:border-neutral-700 px-4 py-3 text-left"
    >
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
          {item.name}
        </span>
        <button
          type="button"
          onClick={() => copyCommand(item.command)}
          className="text-xs text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-200 cursor-pointer"
        >
          {copiedCommand === item.command ? "Copied" : "Copy"}
        </button>
      </div>
      <p className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
        {item.description}
      </p>
      <code className="mt-2 block text-xs text-neutral-500 dark:text-neutral-400">
        {item.command}
      </code>
    </div>
  );

  return (
    <main className="flex h-screen w-full flex-col relative">
      <section
        className={`flex-1 overflow-y-auto overscroll-contain relative min-h-0 ${isWindows ? "xl:pt-4" : "xl:pt-8"}`}
      >
        <div className="max-w-[730px] mx-auto w-full px-4 pt-4 pb-20 sm:px-6 sm:pt-6 sm:pb-24 lg:px-8 lg:pt-8 lg:pb-28">
          <h1 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
            Launch an App
          </h1>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            Copy a command and run it in your terminal.
          </p>

          <div className="mt-6 grid gap-3">
            {FEATURED_COMMANDS.map(renderCommandCard)}
          </div>

          <details className="mt-6">
            <summary className="cursor-pointer select-none text-sm font-medium text-neutral-900 dark:text-neutral-100">
              More apps
            </summary>
            <div className="mt-3 grid gap-3">
              {MORE_COMMANDS.map(renderCommandCard)}
            </div>
          </details>

          {copyError && (
            <p className="mt-4 text-sm text-red-600 dark:text-red-400">
              {copyError}
            </p>
          )}
        </div>
      </section>
    </main>
  );
}
