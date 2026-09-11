import { CodexConnectedIntro } from "./CodexConnectedIntro";
import type { IntegrationStatus } from "@/api";
import { INTEGRATION_ICONS } from "@/lib/launchCommands";
import type {
  CodexDesktopActionResult,
  CodexDesktopStatus,
} from "@/types/webview";
import { ArrowPathIcon, CommandLineIcon } from "@heroicons/react/24/outline";
import {
  useMutation,
  useMutationState,
  useQueryClient,
} from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";

export const CODEX_DESKTOP_INSTALL_TIMEOUT_MS = 120_000;
const acknowledgmentKey = ["codex-desktop-acknowledgment"];

const connectionProgress = {
  idle: null,
  installing: {
    label: "Downloading…",
    description: "Ollama is downloading the ChatGPT installer…",
  },
  "waiting-for-install": {
    label: "Finish installing…",
    description:
      "Finish installing ChatGPT. Ollama will connect it automatically.",
  },
  connecting: {
    label: "Connecting…",
    description: "Connecting ChatGPT to Ollama…",
  },
  saving: {
    label: "Saving…",
    description: "Saving your progress…",
  },
  disconnecting: {
    label: "Disconnecting…",
    description: "Restoring ChatGPT’s usual connection…",
  },
} as const;

type CodexConnectPhase = keyof typeof connectionProgress;

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
  const requestCount = status.requests ?? 0;
  return `Connected to Ollama · ${requestCount} ${requestCount === 1 ? "request" : "requests"} this session`;
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
  const [showIntro, setShowIntro] = useState(false);
  const introRestartConfirmed = useRef(false);
  const used = useRef(false);
  const mounted = useRef(true);
  const operationInFlight = useRef(false);
  const statusRequest = useRef(0);
  const queryClient = useQueryClient();
  const acknowledgment = useMutation({
    mutationKey: acknowledgmentKey,
    // Keep the save and its retry available across Apps page navigation.
    gcTime: Infinity,
    retry: false,
    networkMode: "always",
    mutationFn: async () => {
      if (!window.markCodexDesktopIntegrationUsed)
        throw new Error("Acknowledgment is unavailable");
      const saveError = await window.markCodexDesktopIntegrationUsed();
      if (saveError) throw new Error(saveError);
    },
  });
  const acknowledgmentStates = useMutationState({
    filters: { mutationKey: acknowledgmentKey, exact: true },
    select: (mutation) => mutation.state.status,
  });
  const acknowledgmentStatus =
    acknowledgmentStates[acknowledgmentStates.length - 1];
  const savingAcknowledgment = acknowledgmentStates.includes("pending");
  const acknowledgmentFailed =
    acknowledgmentStates.includes("error") &&
    acknowledgmentStatus !== "success" &&
    !status?.used &&
    !used.current;

  const beginOperation = useCallback(
    (nextPhase: CodexConnectPhase) => {
      if (
        !mounted.current ||
        operationInFlight.current ||
        queryClient.isMutating({ mutationKey: acknowledgmentKey })
      )
        return false;
      operationInFlight.current = true;
      ++statusRequest.current;
      setPhase(nextPhase);
      setError(null);
      setNotice(null);
      return true;
    },
    [queryClient],
  );

  const finishOperation = useCallback(
    (nextPhase: CodexConnectPhase = "idle") => {
      operationInFlight.current = false;
      if (mounted.current) setPhase(nextPhase);
    },
    [],
  );

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (acknowledgmentStatus !== "success") return;
    used.current = true;
    setStatus((current) =>
      current && !current.used ? { ...current, used: true } : current,
    );
  }, [acknowledgmentStatus]);

  const refreshStatus = useCallback(async () => {
    if (operationInFlight.current || !window.getCodexDesktopStatus) return;
    const request = ++statusRequest.current;
    const isCurrent = () =>
      mounted.current &&
      request === statusRequest.current &&
      !operationInFlight.current;
    try {
      const next = await window.getCodexDesktopStatus();
      if (!isCurrent()) return;
      setStatus(next);
      if (next.used) {
        used.current = true;
      }
      setError(null);
      setNotice(null);
    } catch {
      if (isCurrent())
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
        operationInFlight.current ||
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
        if (!beginOperation("connecting")) return;
        completing = true;

        if (next.running) {
          setError(
            "ChatGPT is installed. Turn on the switch to restart it with Ollama models.",
          );
          return;
        }

        if (!next.used && !used.current) {
          introRestartConfirmed.current = false;
          setShowIntro(true);
          return;
        }

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
      } catch {
        if (!mounted.current || (!active && !completing)) return;
        setPhase("idle");
        setError("Ollama could not finish connecting ChatGPT.");
      } finally {
        checking = false;
        if (completing) finishOperation();
      }
    };

    void checkForInstall();
    const interval = window.setInterval(checkForInstall, 1000);
    const timeout = window.setTimeout(() => {
      if (!active || completing) return;
      active = false;
      setPhase("idle");
      setError("ChatGPT installation wasn’t detected. Try again.");
    }, CODEX_DESKTOP_INSTALL_TIMEOUT_MS);
    return () => {
      active = false;
      window.clearInterval(interval);
      window.clearTimeout(timeout);
    };
  }, [phase, beginOperation, finishOperation]);

  const connected = status?.connected ?? false;
  const installed = status?.installed ?? integration.installed ?? false;
  const pending = phase !== "idle" || savingAcknowledgment;
  const displayedConnected =
    phase === "disconnecting"
      ? false
      : connected ||
        showIntro ||
        phase === "installing" ||
        phase === "waiting-for-install" ||
        phase === "connecting";
  const progress = connectionProgress[savingAcknowledgment ? "saving" : phase];
  const statusLabel =
    progress?.label ?? (!connected && !installed ? "Download & connect" : null);
  const actionError =
    error ??
    (acknowledgmentFailed
      ? "Ollama couldn’t save your progress. Please try again."
      : null);
  const description =
    actionError ??
    notice ??
    progress?.description ??
    codexDesktopDescription(status, integration.description);

  const saveAcknowledgment = async (): Promise<boolean> => {
    if (queryClient.isMutating({ mutationKey: acknowledgmentKey }))
      return false;
    try {
      await acknowledgment.mutateAsync();
      used.current = true;
      if (mounted.current) {
        setStatus((current) =>
          current ? { ...current, used: true } : current,
        );
      }
      return true;
    } catch {
      return false;
    }
  };

  const retryAcknowledgment = async () => {
    if (pending || !acknowledgmentFailed || !beginOperation("saving")) return;
    try {
      await saveAcknowledgment();
    } finally {
      finishOperation();
      if (mounted.current) void refreshStatus();
    }
  };

  const toggleConnection = async (fromIntro = false) => {
    const enabled = fromIntro || !connected;
    const nextPhase = enabled
      ? installed
        ? "connecting"
        : "installing"
      : "disconnecting";
    if (pending || (showIntro && !fromIntro) || !beginOperation(nextPhase))
      return;
    let finalPhase: CodexConnectPhase = "idle";
    let restartConfirmed = fromIntro && introRestartConfirmed.current;
    if (fromIntro) {
      setShowIntro(false);
      introRestartConfirmed.current = false;
    }
    try {
      if (!window.setCodexDesktopConnected) {
        setError("The ChatGPT integration is unavailable.");
        return;
      }
      if (enabled && !installed) {
        if (!window.installCodexDesktop || !window.getCodexDesktopStatus) {
          setError("Ollama could not install ChatGPT.");
          return;
        }
        const installResult = await window.installCodexDesktop();
        if (!mounted.current) return;
        if (installResult === "opened") finalPhase = "waiting-for-install";
        else if (installResult !== "cancelled")
          setError("Ollama could not install ChatGPT.");
        return;
      }
      if (fromIntro || (enabled && !status?.used && !used.current)) {
        if (!window.getCodexDesktopStatus) {
          setError("Ollama could not read the ChatGPT connection status.");
          return;
        }
        const liveStatus = await window.getCodexDesktopStatus();
        if (!mounted.current) return;
        setStatus(liveStatus);
        if (liveStatus.running && !restartConfirmed) {
          restartConfirmed = window.confirm(
            "Restart ChatGPT to add Ollama models? Any running task will stop.",
          );
          if (!restartConfirmed) return;
        }

        if (!fromIntro && !liveStatus.used && !used.current) {
          introRestartConfirmed.current = restartConfirmed;
          setShowIntro(true);
          return;
        }
      }

      let result: CodexDesktopActionResult =
        await window.setCodexDesktopConnected(enabled, restartConfirmed);

      setStatus(result.status);
      if (result.restartConfirmationRequired) {
        if (!mounted.current) return;
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
        result = await window.setCodexDesktopConnected(enabled, true);
        setStatus(result.status);
      }

      if (result.restartConfirmationRequired) return;
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
      if (fromIntro) {
        setPhase("saving");
        if (!(await saveAcknowledgment())) return;
      }
      if (enabled) {
        setNotice("Ollama models added alongside Codex models");
      } else {
        setNotice("Ollama models removed · Codex models remain available");
      }
    } catch {
      setError(
        nextPhase === "installing"
          ? "Ollama could not install ChatGPT."
          : enabled
            ? "Ollama could not add its models to ChatGPT."
            : "Ollama could not remove its models from ChatGPT.",
      );
    } finally {
      finishOperation(finalPhase);
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
            role={actionError ? "alert" : notice ? "status" : undefined}
            className="truncate text-xs leading-5 text-neutral-500 dark:text-neutral-400"
          >
            {description}
          </p>
        </div>
      </div>
      <div className="ml-auto flex shrink-0 items-center gap-2.5">
        {acknowledgmentFailed && (
          <button
            type="button"
            aria-label="Retry saving progress"
            disabled={pending}
            onClick={() => void retryAcknowledgment()}
            className="text-xs font-medium text-neutral-700 hover:underline disabled:cursor-wait disabled:opacity-50 dark:text-neutral-300"
          >
            Retry
          </button>
        )}
        {statusLabel && (
          <span
            role="status"
            aria-live="polite"
            className="inline-flex items-center gap-1.5 whitespace-nowrap text-xs text-neutral-500 dark:text-neutral-400"
          >
            {pending && <ArrowPathIcon className="h-3.5 w-3.5 animate-spin" />}
            {statusLabel}
          </span>
        )}
        <button
          type="button"
          role="switch"
          aria-checked={displayedConnected}
          aria-busy={pending || undefined}
          aria-label={
            showIntro
              ? "Finish connecting ChatGPT"
              : connected
                ? "Remove Ollama models from ChatGPT"
                : pending
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
          disabled={pending || showIntro}
          onClick={() => void toggleConnection()}
          className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-wait disabled:opacity-50 ${displayedConnected ? "bg-neutral-950 dark:bg-white" : "bg-neutral-300 dark:bg-neutral-700"}`}
        >
          <span
            aria-hidden="true"
            className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${pending ? "animate-pulse" : ""} ${displayedConnected ? "translate-x-4.5 dark:bg-neutral-900" : "translate-x-0.5"}`}
          />
        </button>
      </div>
      {showIntro && (
        <CodexConnectedIntro onDone={() => void toggleConnection(true)} />
      )}
    </div>
  );
}
