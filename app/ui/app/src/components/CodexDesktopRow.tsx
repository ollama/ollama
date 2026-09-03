import type { IntegrationStatus } from "@/api";
import { INTEGRATION_ICONS } from "@/lib/launchCommands";
import type {
  CodexDesktopActionResult,
  CodexDesktopInstallResult,
  CodexDesktopStatus,
} from "@/types/webview";
import { ArrowPathIcon, CommandLineIcon } from "@heroicons/react/24/outline";
import { useCallback, useEffect, useRef, useState } from "react";

export const CODEX_DESKTOP_INSTALL_TIMEOUT_MS = 120_000;

type CodexConnectPhase =
  | "idle"
  | "installing"
  | "waiting-for-install"
  | "connecting"
  | "disconnecting";

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

function codexDesktopDescription(
  status: CodexDesktopStatus | null,
  defaultDescription: string,
): string {
  if (!status?.connected) return defaultDescription;
  const modelCount = status.models?.length ?? (status.model ? 1 : 0);
  const requestCount = status.requests ?? 0;
  const requests = `${requestCount} Ollama ${requestCount === 1 ? "request" : "requests"} this session`;
  if (modelCount > 0) {
    return `Codex + Ollama · ${modelCount} Ollama ${modelCount === 1 ? "model" : "models"} · ${requests}`;
  }
  return `Codex + Ollama · ${requests}`;
}

export function CodexDesktopRow({
  integration,
  initialStatus,
}: CodexDesktopRowProps) {
  const [status, setStatus] = useState<CodexDesktopStatus | null>(
    initialStatus ?? null,
  );
  const [phase, setPhase] = useState<CodexConnectPhase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const mounted = useRef(true);
  const operationInFlight = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refreshStatus = useCallback(async () => {
    if (operationInFlight.current || !window.getCodexDesktopStatus) return;
    try {
      const next = await window.getCodexDesktopStatus();
      setStatus(next);
      setError(null);
      setNotice(null);
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

  useEffect(() => {
    if (phase !== "waiting-for-install") return;

    let active = true;
    let checking = false;
    let completing = false;
    const checkForInstall = async () => {
      if (
        !active ||
        checking ||
        completing ||
        !window.getCodexDesktopStatus ||
        !window.setCodexDesktopConnected
      ) {
        return;
      }
      checking = true;
      try {
        const next = await window.getCodexDesktopStatus();
        if (!active || !mounted.current) return;
        setStatus(next);
        if (!next.installed) return;
        completing = true;

        if (next.running) {
          setPhase("idle");
          setError(
            "ChatGPT is installed. Turn on the switch to restart it with Ollama models.",
          );
          return;
        }

        setPhase("connecting");
        const result = await window.setCodexDesktopConnected(true, false);
        if (!mounted.current) return;
        setStatus(result.status);
        if (result.restartConfirmationRequired) {
          setError(
            "ChatGPT is installed. Turn on the switch to restart it with Ollama models.",
          );
        } else if (result.error || !result.status.connected) {
          setError(
            result.error || "Ollama could not add its models to ChatGPT.",
          );
        } else {
          setNotice("Ollama models added alongside Codex models");
        }
        setPhase("idle");
      } catch {
        if (!mounted.current) return;
        setPhase("idle");
        setError("Ollama could not finish connecting ChatGPT.");
      } finally {
        checking = false;
      }
    };

    void checkForInstall();
    const interval = window.setInterval(checkForInstall, 1000);
    const timeout = window.setTimeout(() => {
      if (!active || completing) return;
      setPhase("idle");
      setError("ChatGPT installation wasn’t detected. Try again.");
    }, CODEX_DESKTOP_INSTALL_TIMEOUT_MS);
    return () => {
      active = false;
      window.clearInterval(interval);
      window.clearTimeout(timeout);
    };
  }, [phase]);

  const connected = status?.connected ?? false;
  const installed = status?.installed ?? integration.installed ?? false;
  const pending = phase !== "idle";
  const displayedConnected =
    phase === "disconnecting"
      ? false
      : connected ||
        phase === "installing" ||
        phase === "waiting-for-install" ||
        phase === "connecting";
  const isConnecting = phase !== "idle";
  const statusLabel =
    phase === "installing"
      ? "Downloading…"
      : phase === "waiting-for-install"
        ? "Finish installing…"
        : phase === "connecting"
          ? "Connecting…"
          : phase === "disconnecting"
            ? "Disconnecting…"
            : !connected && !installed
              ? "Download & connect"
              : null;
  const description =
    error ??
    notice ??
    (phase === "installing"
      ? "Ollama is downloading the ChatGPT installer…"
      : phase === "waiting-for-install"
        ? "Finish installing ChatGPT. Ollama will connect it automatically."
        : phase === "connecting"
          ? "Connecting ChatGPT to Ollama…"
          : phase === "disconnecting"
            ? "Restoring ChatGPT’s usual connection…"
            : codexDesktopDescription(status, integration.description));

  const toggleConnection = async () => {
    if (pending || operationInFlight.current) return;
    if (!window.setCodexDesktopConnected) {
      setError("The ChatGPT integration is unavailable.");
      return;
    }

    const enabled = !connected;
    if (enabled && !installed) {
      if (!window.installCodexDesktop || !window.getCodexDesktopStatus) {
        setError("Ollama could not install ChatGPT.");
        return;
      }
      setPhase("installing");
      setError(null);
      setNotice(null);
      let installResult: CodexDesktopInstallResult = "failed";
      try {
        installResult = await window.installCodexDesktop();
      } catch {
        // The shared failure message below covers a rejected native request.
      }
      if (installResult === "cancelled") {
        setPhase("idle");
        return;
      }
      if (installResult !== "opened") {
        setPhase("idle");
        setError("Ollama could not install ChatGPT.");
        return;
      }
      setPhase("waiting-for-install");
      return;
    }

    const nextPhase = enabled ? "connecting" : "disconnecting";
    operationInFlight.current = true;
    setPhase(nextPhase);
    setError(null);
    setNotice(null);
    try {
      let result: CodexDesktopActionResult =
        await window.setCodexDesktopConnected(enabled, false);

      setStatus(result.status);
      if (result.restartConfirmationRequired) {
        // Keep focus-driven status refreshes from discarding this operation
        // while the native confirmation dialog temporarily owns focus.
        if (
          !window.confirm(
            enabled
              ? "Restart ChatGPT to add Ollama models? Any running task will stop."
              : "Restart ChatGPT to remove Ollama models? Any running task will stop.",
          )
        ) {
          return;
        }
        setPhase(nextPhase);
        result = await window.setCodexDesktopConnected(enabled, true);
        setStatus(result.status);
      }

      if (result.error) {
        setError(result.error);
        return;
      }
      if (result.status.connected !== enabled) {
        setError(
          enabled
            ? "Ollama could not add its models to ChatGPT."
            : "Ollama could not remove its models from ChatGPT.",
        );
        return;
      }
      setNotice(
        enabled
          ? "Ollama models added alongside Codex models"
          : "Ollama models removed · Codex models remain available",
      );
    } catch {
      setError(
        enabled
          ? "Ollama could not add its models to ChatGPT."
          : "Ollama could not remove its models from ChatGPT.",
      );
    } finally {
      operationInFlight.current = false;
      if (mounted.current) setPhase("idle");
    }
  };

  return (
    <div className="flex min-h-18 items-center justify-between gap-4 bg-white px-4 py-3 dark:bg-neutral-900">
      <div className="flex min-w-0 items-center gap-3">
        <CodexIcon integration={integration} />
        <div className="min-w-0">
          <p className="text-sm font-medium text-neutral-950 dark:text-neutral-100">
            ChatGPT (Desktop)
          </p>
          <p
            role={error ? "alert" : notice ? "status" : undefined}
            className="truncate text-xs leading-5 text-neutral-500 dark:text-neutral-400"
          >
            {description}
          </p>
        </div>
      </div>
      <div className="ml-auto flex shrink-0 items-center gap-2.5">
        {statusLabel && (
          <span
            role="status"
            aria-live="polite"
            className="inline-flex items-center gap-1.5 whitespace-nowrap text-xs text-neutral-500 dark:text-neutral-400"
          >
            {isConnecting && (
              <ArrowPathIcon className="h-3.5 w-3.5 animate-spin" />
            )}
            {statusLabel}
          </span>
        )}
        <button
          type="button"
          role="switch"
          aria-checked={displayedConnected}
          aria-busy={pending || undefined}
          aria-label={
            connected
              ? "Remove Ollama models from ChatGPT"
              : isConnecting
                ? "Connecting ChatGPT"
                : "Add Ollama models to ChatGPT"
          }
          title={
            connected
              ? "Remove Ollama models"
              : installed
                ? "Add Ollama models"
                : "Install ChatGPT and add Ollama models"
          }
          disabled={pending}
          onClick={() => void toggleConnection()}
          className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-wait disabled:opacity-50 ${displayedConnected ? "bg-neutral-950 dark:bg-white" : "bg-neutral-300 dark:bg-neutral-700"}`}
        >
          <span
            aria-hidden="true"
            className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${pending ? "animate-pulse" : ""} ${displayedConnected ? "translate-x-4.5 dark:bg-neutral-900" : "translate-x-0.5"}`}
          />
        </button>
      </div>
    </div>
  );
}
