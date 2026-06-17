import { CommandLineIcon } from "@heroicons/react/24/outline";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { runOllamaInTerminal, skipFirstRun } from "@/api";
import type { SettingsResponse } from "@/api";
import CopyButton from "@/components/CopyButton";

interface FirstRunPromptProps {
  open: boolean;
}

export function FirstRunPrompt({ open }: FirstRunPromptProps) {
  const queryClient = useQueryClient();

  const dismiss = () => {
    queryClient.setQueryData<SettingsResponse | undefined>(
      ["settings"],
      (current) => {
        if (!current) return current;
        return { ...current, hasCompletedFirstRun: true };
      },
    );
    queryClient.invalidateQueries({ queryKey: ["settings"] });
  };

  const runMutation = useMutation({
    mutationFn: runOllamaInTerminal,
    onSuccess: dismiss,
  });

  const skipMutation = useMutation({
    mutationFn: skipFirstRun,
    onSuccess: dismiss,
  });

  if (!open) return null;

  const pending = runMutation.isPending || skipMutation.isPending;
  const error = runMutation.error ?? skipMutation.error;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-white px-5 py-10 dark:bg-neutral-900">
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Run Ollama in terminal"
        aria-describedby="first-run-description"
        className="flex min-h-[620px] w-full max-w-[500px] flex-col justify-center text-neutral-950 dark:text-white"
      >
        <div>
          <div className="flex items-center justify-center gap-4">
            <img
              src="/hello.png"
              alt=""
              draggable={false}
              className="h-100 w-100 select-none dark:invert"
            />
            <span className="-mt-40 font-rounded text-4xl font-semibold text-neutral-950 dark:text-neutral-50">
              hi!
            </span>
          </div>

          <div className="-mt-5 flex min-h-15 items-center gap-4 max-w-s rounded-[28px] bg-neutral-100 py-4 pr-4 pl-6 dark:bg-neutral-800">
            <CommandLineIcon
              aria-hidden="true"
              className="h-5 w-5 shrink-0 text-neutral-400 dark:text-neutral-500"
            />
            <code className="min-w-0 truncate font-mono text-xl font-normal tracking-normal text-neutral-700 dark:text-neutral-200">
              ollama
            </code>
            <CopyButton
              content="ollama"
              size="md"
              title="Copy command to clipboard"
              className="ml-auto text-neutral-500 hover:bg-neutral-200/60 hover:text-neutral-700 dark:text-neutral-400 dark:hover:bg-neutral-700/70 dark:hover:text-neutral-200"
            />
          </div>

          <p
            id="first-run-description"
            className="mt-5 text-center text-sm/6 text-neutral-500 dark:text-neutral-400"
          >
            Run Ollama in your terminal to get started.
          </p>

          {error ? (
            <p className="mt-6 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-center text-sm/6 text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200">
              Could not open a terminal. Run{" "}
              <span className="font-mono">ollama</span>.
            </p>
          ) : null}
        </div>

        <div className="mt-8 flex flex-col-reverse items-stretch justify-center gap-3 sm:flex-row sm:items-center">
          <button
            type="button"
            onClick={() => skipMutation.mutate()}
            disabled={pending}
            className="inline-flex h-12 items-center justify-center rounded-full bg-neutral-100 px-8 text-lg font-normal text-neutral-950 transition hover:bg-neutral-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300 disabled:pointer-events-none disabled:opacity-50 dark:bg-neutral-800 dark:text-neutral-50 dark:hover:bg-neutral-700 dark:focus-visible:ring-neutral-600"
          >
            Skip
          </button>
          <button
            type="button"
            onClick={() => runMutation.mutate()}
            disabled={pending}
            className="inline-flex h-12 items-center justify-center gap-3 rounded-full bg-black px-8 text-lg font-normal text-white shadow-sm transition hover:bg-neutral-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-400 disabled:pointer-events-none disabled:opacity-50 dark:bg-white dark:text-black dark:hover:bg-neutral-200 dark:focus-visible:ring-neutral-500"
          >
            Open in terminal
          </button>
        </div>
      </div>
    </div>
  );
}
