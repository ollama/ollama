import { useState } from "react";
import {
  CheckIcon,
  DocumentDuplicateIcon,
} from "@heroicons/react/24/outline";
import { copyTextToClipboard } from "@/utils/clipboard";
import { useSettings } from "@/hooks/useSettings";

interface LaunchCommand {
  id: string;
  name: string;
  command: string;
  description: string;
}

const FEATURED_COMMANDS: LaunchCommand[] = [
  {
    id: "openclaw",
    name: "OpenClaw",
    command: "ollama launch openclaw",
    description: "Personal AI with 100+ skills",
  },
  {
    id: "claude",
    name: "Claude",
    command: "ollama launch claude",
    description: "Anthropic's coding tool with subagents",
  },
];

const MORE_COMMANDS: LaunchCommand[] = [
  {
    id: "codex",
    name: "Codex",
    command: "ollama launch codex",
    description: "OpenAI's open-source coding agent",
  },
  {
    id: "opencode",
    name: "OpenCode",
    command: "ollama launch opencode",
    description: "Anomaly's open-source coding agent",
  },
  {
    id: "droid",
    name: "Droid",
    command: "ollama launch droid",
    description: "Factory's coding agent across terminal and IDEs",
  },
  {
    id: "pi",
    name: "Pi",
    command: "ollama launch pi",
    description: "Minimal AI agent toolkit with plugin support",
  },
];

export default function LaunchCommands() {
  const isWindows = navigator.platform.toLowerCase().includes("win");
  const { setSettings } = useSettings();
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null);
  const [copyError, setCopyError] = useState<string | null>(null);

  const copyCommand = async (item: LaunchCommand) => {
    const copied = await copyTextToClipboard(item.command);
    if (!copied) {
      setCopyError("Unable to copy command to clipboard.");
      setTimeout(() => setCopyError(null), 2500);
      return;
    }

    setCopyError(null);
    setCopiedCommand(item.command);
    setSettings({ LastHomeView: item.id }).catch(() => {
      // Best effort persistence for launch integration preference.
    });
    setTimeout(() => setCopiedCommand(null), 2000);
  };

  const renderCommandCard = (item: LaunchCommand) => (
    <div key={item.command} className="w-full text-left">
      <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
        {item.name}
      </span>
      <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
        {item.description}
      </p>
      <div className="mt-2 flex items-center gap-2 rounded-xl border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800 px-3 py-2">
        <code className="min-w-0 flex-1 truncate text-xs text-neutral-600 dark:text-neutral-300">
          {item.command}
        </code>
        <div className="relative group">
          <button
            type="button"
            onClick={() => copyCommand(item)}
            aria-label="Copy command to clipboard"
            className="rounded-md p-1.5 text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-200 hover:bg-neutral-200/60 dark:hover:bg-neutral-700/70 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300 dark:focus-visible:ring-neutral-600"
          >
            {copiedCommand === item.command ? (
              <CheckIcon className="h-4 w-4" />
            ) : (
              <DocumentDuplicateIcon className="h-4 w-4" />
            )}
          </button>
          <span className="pointer-events-none absolute left-1/2 top-0 z-10 -translate-x-1/2 -translate-y-[calc(100%+8px)] whitespace-nowrap rounded-md border border-neutral-200 bg-white px-2 py-1 text-[11px] text-neutral-700 opacity-0 shadow-sm transition-opacity duration-100 group-hover:opacity-100 group-focus-within:opacity-100 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-200">
            Copy command to clipboard
          </span>
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
            Launch an App
          </h1>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            Copy a command and run it in your terminal.
          </p>

          <div className="mt-6 grid gap-5">
            {FEATURED_COMMANDS.map(renderCommandCard)}
          </div>

          <details className="mt-6">
            <summary className="cursor-pointer select-none text-sm font-medium text-neutral-900 dark:text-neutral-100 outline-none focus:outline-none focus-visible:outline-none ring-0 focus:ring-0 focus-visible:ring-0">
              More apps
            </summary>
            <div className="mt-3 grid gap-5">
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
