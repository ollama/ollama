import type { IntegrationStatus } from "@/api";
import { INTEGRATION_ICONS } from "@/lib/launchCommands";
import type {
  CodexDesktopActionResult,
  CodexDesktopStatus,
} from "@/types/webview";
import { ArrowPathIcon, CommandLineIcon } from "@heroicons/react/24/outline";
import { useCallback, useEffect, useState } from "react";

interface CodexDesktopRowProps {
  integration: IntegrationStatus;
  initialStatus?: CodexDesktopStatus;
}

function CodexIcon({ integration }: { integration: IntegrationStatus }) {
  const icon = INTEGRATION_ICONS[integration.id];
  return (
    <div className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-xl bg-transparent">
      {icon ? (
        <>
          <img
            src={icon.src}
            alt=""
            className={`${icon.className ?? "h-7 w-7"} rounded-sm object-contain ${icon.darkSrc ? "dark:hidden" : ""}`}
          />
          {icon.darkSrc && (
            <img
              src={icon.darkSrc}
              alt=""
              className={`${icon.className ?? "h-7 w-7"} hidden rounded-sm object-contain dark:block`}
            />
          )}
        </>
      ) : (
        <CommandLineIcon className="h-6 w-6 stroke-[1.5] text-neutral-700 dark:text-neutral-300" />
      )}
    </div>
  );
}

export function codexDesktopDescription(
  status: CodexDesktopStatus | null,
  integration: IntegrationStatus,
): string {
  if (!status?.installed && !status?.connected) {
    return "Install ChatGPT to use Ollama models in the Codex app.";
  }
  if (!status.connected) return integration.description;
  const modelCount = status.models?.length ?? (status.model ? 1 : 0);
  const requestCount = status.requests ?? 0;
  const requests = `${requestCount} ${requestCount === 1 ? "request" : "requests"} this session`;
  if (modelCount > 0) {
    return `Using Ollama · ${modelCount} ${modelCount === 1 ? "model" : "models"} · ${requests}`;
  }
  return `Using Ollama · ${requests}`;
}

export function CodexDesktopRow({
  integration,
  initialStatus,
}: CodexDesktopRowProps) {
  const [status, setStatus] = useState<CodexDesktopStatus | null>(
    initialStatus ?? null,
  );
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    if (!window.getCodexDesktopStatus) return;
    try {
      const next = await window.getCodexDesktopStatus();
      setStatus(next);
      setError(null);
    } catch {
      setError("Ollama could not read the ChatGPT connection status.");
    }
  }, []);

  useEffect(() => {
    if (!initialStatus) void refreshStatus();
    const onFocus = () => void refreshStatus();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [initialStatus, refreshStatus]);

  useEffect(() => {
    if (!status?.connected || !window.getCodexDesktopRequestCount) return;

    let active = true;
    let checking = false;
    const refreshRequestCount = async () => {
      if (!active || checking || document.visibilityState === "hidden") return;
      checking = true;
      try {
        const requests = await window.getCodexDesktopRequestCount?.();
        if (!active || requests === undefined) return;
        setStatus((current) => {
          if (!current || current.requests === requests) return current;
          return { ...current, requests };
        });
      } catch {
        // The next interval or window-focus refresh can recover the count.
      } finally {
        checking = false;
      }
    };

    void refreshRequestCount();
    const interval = window.setInterval(refreshRequestCount, 1000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [status?.connected]);

  const connected = status?.connected ?? false;
  const installed = status?.installed ?? integration.installed ?? false;

  const toggleConnection = async () => {
    if (pending || !installed) return;
    if (!window.setCodexDesktopConnected) {
      setError("The ChatGPT integration is unavailable.");
      return;
    }

    const enabled = !connected;
    if (
      status?.running &&
      !window.confirm(
        enabled
          ? "Restart ChatGPT to use Ollama models? Your account, chats, plugins, and skills will stay in the same profile. Codex CLI and IDE will also use this shared configuration while Ollama is enabled. Any running task will stop."
          : "Restart ChatGPT to restore its previous OpenAI provider and model list? Any running task will stop.",
      )
    ) {
      return;
    }
    setPending(true);
    setError(null);
    try {
      const result: CodexDesktopActionResult =
        await window.setCodexDesktopConnected(enabled);

      setStatus(result.status);
      if (result.error) {
        setError(result.error);
        return;
      }
    } catch {
      setError(
        enabled
          ? "Ollama could not switch ChatGPT to Ollama models."
          : "Ollama could not restore ChatGPT's OpenAI models.",
      );
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex min-h-18 items-center justify-between gap-4 bg-white px-4 py-3 dark:bg-neutral-900">
      <div className="flex min-w-0 items-center gap-3">
        <CodexIcon integration={integration} />
        <div className="min-w-0">
          <p className="text-sm font-medium text-neutral-950 dark:text-neutral-100">
            {integration.name}
          </p>
          <p
            role={error ? "alert" : undefined}
            className="truncate text-xs leading-5 text-neutral-500 dark:text-neutral-400"
          >
            {error ?? codexDesktopDescription(status, integration)}
          </p>
        </div>
      </div>
      <div className="ml-auto flex shrink-0 items-center gap-2.5">
        {pending && (
          <span
            role="status"
            aria-live="polite"
            className="inline-flex items-center gap-1.5 whitespace-nowrap text-xs text-neutral-500 dark:text-neutral-400"
          >
            <ArrowPathIcon className="h-3.5 w-3.5 animate-spin" />
            {connected ? "Closing…" : "Opening…"}
          </span>
        )}
        <button
          type="button"
          role="switch"
          aria-checked={connected}
          aria-busy={pending || undefined}
          aria-label={
            connected
              ? "Restore OpenAI models in ChatGPT"
              : "Use Ollama models in ChatGPT"
          }
          title={
            connected
              ? "Restore OpenAI"
              : installed
                ? "Use Ollama"
                : "Not installed"
          }
          disabled={pending || (!installed && !connected)}
          onClick={() => void toggleConnection()}
          className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-not-allowed disabled:opacity-50 ${connected ? "bg-neutral-950 dark:bg-white" : "bg-neutral-300 dark:bg-neutral-700"}`}
        >
          <span
            aria-hidden="true"
            className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${pending ? "animate-pulse" : ""} ${connected ? "translate-x-4.5 dark:bg-neutral-900" : "translate-x-0.5"}`}
          />
        </button>
      </div>
    </div>
  );
}
