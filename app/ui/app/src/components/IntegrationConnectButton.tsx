import { ArrowPathIcon } from "@heroicons/react/24/outline";
import type { ComponentProps } from "react";

export function IntegrationConnectButton({
  connected,
  busy,
  progress,
  label,
  ...props
}: Pick<ComponentProps<"button">, "onClick" | "disabled" | "title"> & {
  connected: boolean;
  label: string;
  busy: boolean;
  progress?: string | null;
}) {
  return (
    <button
      {...props}
      type="button"
      aria-label={label}
      aria-pressed={connected}
      aria-busy={busy || undefined}
      className="inline-flex min-h-10 shrink-0 items-center justify-center gap-2 rounded-full bg-neutral-900 px-6 text-sm font-normal text-white transition-colors hover:bg-neutral-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-wait disabled:opacity-60 dark:bg-white dark:text-neutral-950 dark:hover:bg-neutral-200"
    >
      {busy ? (
        <>
          <ArrowPathIcon className="h-3.5 w-3.5 animate-spin" />
          <span role="status" aria-live="polite">
            {progress ?? "Checking…"}
          </span>
        </>
      ) : connected ? (
        "Disconnect"
      ) : (
        "Connect"
      )}
    </button>
  );
}
