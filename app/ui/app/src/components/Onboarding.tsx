import CopyButton from "@/components/CopyButton";
import Logo from "@/components/Logo";
import { nextOnboardingStep, type OnboardingStep } from "@/lib/onboarding";
import {
  getIntegrationStatuses,
  type IntegrationStatus,
  type IntegrationStatuses,
} from "@/api";
import { INTEGRATION_ICONS } from "@/lib/launchCommands";
import {
  CLAUDE_CONNECTION_TIMEOUT_MS,
  ClaudeConnectionTimeoutError,
  claudeDesktopRecoveryMessage,
  claudeDesktopRequestCountLabel,
  isClaudeConfigured,
  isClaudeConnectionComplete,
  optimisticClaudeConnectionState,
  scheduleClaudeInstallTimeout,
  withClaudeConnectionTimeout,
} from "@/lib/claudeDesktop";
import { isWindowsPlatform } from "@/lib/platform";
import type {
  ClaudeDesktopActionResult,
  ClaudeDesktopStatus,
} from "@/types/webview";
import { copyTextToClipboard } from "@/utils/clipboard";
import {
  ArrowPathIcon,
  ArrowsRightLeftIcon,
  CommandLineIcon,
  ShieldCheckIcon,
  Square2StackIcon,
} from "@heroicons/react/24/outline";
import { CheckIcon } from "@heroicons/react/20/solid";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

export const FIRST_MODEL_COMMAND = "ollama";

type ClaudeConnectPhase =
  | "idle"
  | "installing"
  | "waiting-for-install"
  | "connecting"
  | "launching"
  | "disconnecting";

export function shouldShowClaudeConnectedIntro(status: ClaudeDesktopStatus) {
  return status.connected && !status.startFailed && !status.used;
}

function setClaudeConnection(
  enabled: boolean,
  deferLaunch = false,
  restartConfirmed = false,
) {
  if (enabled && deferLaunch && window.prepareClaudeDesktopConnection) {
    return window.prepareClaudeDesktopConnection();
  }
  if (!window.setClaudeDesktopConnected) {
    throw new Error("Claude Desktop connection is unavailable");
  }
  return window.setClaudeDesktopConnected(enabled, restartConfirmed);
}

function getClaudeConnectionSummary() {
  const getStatus =
    window.getClaudeDesktopConnectionSummary ?? window.getClaudeDesktopStatus;
  return getStatus?.() ?? Promise.resolve(null);
}

interface ScreenProps {
  isSigningIn: boolean;
  signInError: string | null;
  onSignIn: () => void;
}

interface WelcomeScreenProps extends ScreenProps {
  isAuthenticated: boolean;
  completionError?: string | null;
  onRetryCompletion?: () => void;
  onLocal: () => void;
  onSignUp: () => void;
}

interface RunOllamaScreenProps {
  completionError: string | null;
  onRetryCompletion: () => void;
}

interface ConnectAppsScreenProps {
  initialIntegrations?: IntegrationStatuses;
  initialClaudeStatus?: ClaudeDesktopStatus;
}

function TitleBar({ onSignIn }: { onSignIn?: () => void }) {
  const isMacOS =
    typeof navigator !== "undefined" &&
    navigator.platform.toLowerCase().includes("mac");

  return (
    <header
      className={`relative flex shrink-0 items-center justify-center bg-white ${isMacOS ? "h-[52px]" : "h-10"}`}
      onDoubleClick={() => window.doubleClick?.()}
      onMouseDown={() => window.drag?.()}
    >
      {onSignIn && (
        <button
          type="button"
          className="absolute inset-y-0 right-5 flex cursor-pointer items-center rounded-md px-2 text-sm font-normal leading-none text-neutral-500 hover:text-neutral-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
          onClick={onSignIn}
          onMouseDown={(event) => event.stopPropagation()}
        >
          Sign in
        </button>
      )}
    </header>
  );
}

function OnboardingIcon({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center justify-center">
      <Logo
        size={compact ? 42 : 54}
        containerClassName="mb-0"
        showBackground={false}
      />
    </div>
  );
}

function OnboardingCard({ children }: { children: ReactNode }) {
  return (
    <section className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto bg-white px-6 pb-10 pt-0">
      <div className="flex w-full max-w-[760px] flex-col items-center justify-center bg-white px-10 py-6 text-center">
        {children}
      </div>
    </section>
  );
}

const OLLAMA_FEATURES = [
  {
    title: "Connect your apps",
    description: "Power your existing coding apps with open models",
    icon: CommandLineIcon,
  },
  {
    title: "Easily switch models",
    description: "Swap between frontier models in one click.",
    icon: ArrowsRightLeftIcon,
  },
  {
    title: "Your data stays yours",
    description: "Your prompt data is never logged or trained on.",
    icon: ShieldCheckIcon,
  },
];

export function IntroScreen({
  completionError = null,
  onContinue,
  onRetryCompletion,
}: {
  completionError?: string | null;
  onContinue: () => void;
  onRetryCompletion?: () => void;
}) {
  return (
    <main className="light-only flex h-screen w-full flex-col overflow-hidden bg-white text-neutral-950">
      <TitleBar />

      <section className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto px-6 pb-10">
        <div className="mx-auto flex min-h-full w-full max-w-[620px] flex-col items-center justify-center py-4 text-center">
          <div className="flex flex-col items-center justify-center gap-2">
            <img
              src="/hello.png"
              alt="Ollama waving"
              className="h-[72px] w-[72px] select-none object-contain"
              draggable={false}
            />
            <h1 className="font-rounded text-2xl font-medium leading-8">
              Welcome to Ollama!
            </h1>
          </div>
          <p className="mt-4 max-w-[380px] text-sm leading-6 text-neutral-500">
            Run open models with your coding agents so you can spend less while
            keeping your data private.
          </p>
          <div className="mx-auto mt-8 flex w-fit max-w-full flex-col gap-6 text-left">
            {OLLAMA_FEATURES.map((feature) => {
              const Icon = feature.icon;

              return (
                <div key={feature.title} className="flex items-start gap-4">
                  <Icon className="mt-0.5 h-7 w-7 shrink-0 stroke-[1.5] text-neutral-700" />
                  <div>
                    <h2 className="text-sm font-medium text-neutral-950">
                      {feature.title}
                    </h2>
                    <p className="mt-0.5 text-[13px] leading-5 text-neutral-500">
                      {feature.description}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>

          <button
            type="button"
            className="mt-8 flex h-11 w-full max-w-[240px] cursor-pointer items-center justify-center rounded-full bg-neutral-900 px-5 font-sans text-sm font-normal text-white transition-colors hover:bg-neutral-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
            onClick={onContinue}
          >
            Continue
          </button>
          <InlineError message={completionError} />
          {completionError && onRetryCompletion && (
            <button
              type="button"
              className="mt-2 cursor-pointer rounded-md px-3 py-1 text-sm font-normal text-neutral-600 underline decoration-neutral-300 underline-offset-4 hover:text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
              onClick={onRetryCompletion}
            >
              Try again
            </button>
          )}
        </div>
      </section>
    </main>
  );
}

function InlineError({
  message,
  className = "mt-3 text-sm",
}: {
  message: string | null;
  className?: string;
}) {
  if (!message) return null;

  return (
    <p role="alert" className={`${className} text-red-600`}>
      {message}
    </p>
  );
}

export function WelcomeScreen({
  isAuthenticated,
  isSigningIn,
  signInError,
  completionError = null,
  onSignIn,
  onSignUp,
  onLocal,
  onRetryCompletion,
}: WelcomeScreenProps) {
  return (
    <main className="light-only flex min-h-screen w-full flex-col bg-white text-neutral-950">
      <TitleBar onSignIn={isAuthenticated ? undefined : onSignIn} />
      <OnboardingCard>
        <OnboardingIcon />
        <h1 className="mt-7 font-rounded text-2xl font-medium leading-8">
          Create an account
        </h1>
        <p className="mt-3 max-w-[400px] text-sm leading-6 text-neutral-400">
          Create your account for access to faster, larger open models.
        </p>
        <p className="mt-1 max-w-[400px] text-sm leading-6 text-neutral-400">
          Your data is never logged or trained on.
        </p>

        <div className="mt-7 flex w-full max-w-[240px] flex-col items-center">
          <button
            type="button"
            className="flex h-11 w-full cursor-pointer items-center justify-center rounded-full bg-neutral-900 px-5 font-sans text-sm font-normal text-white transition-colors hover:bg-neutral-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-wait disabled:opacity-70"
            onClick={onSignUp}
            disabled={isSigningIn}
            aria-busy={isSigningIn}
          >
            {isSigningIn ? "Finish in your browser…" : "Sign up"}
          </button>
          <button
            type="button"
            className="mt-2 cursor-pointer rounded-md px-3 py-2 text-sm font-normal text-neutral-600 underline decoration-neutral-300 underline-offset-4 hover:text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
            onClick={onLocal}
          >
            No thanks, I&apos;ll use Ollama locally
          </button>
          <InlineError message={signInError ?? completionError} />
          {completionError && onRetryCompletion && (
            <button
              type="button"
              className="mt-2 cursor-pointer rounded-md px-3 py-1 text-sm font-normal text-neutral-600 underline decoration-neutral-300 underline-offset-4 hover:text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
              onClick={onRetryCompletion}
            >
              Try again
            </button>
          )}
        </div>
      </OnboardingCard>
    </main>
  );
}

export function RunOllamaScreen({
  completionError,
  onRetryCompletion,
}: RunOllamaScreenProps) {
  return (
    <main className="light-only flex min-h-screen w-full flex-col bg-white text-neutral-950">
      <TitleBar />
      <OnboardingCard>
        <OnboardingIcon compact />
        <h1 className="mt-6 font-rounded text-[22px] font-medium leading-7">
          Run Ollama
        </h1>

        <div className="mt-6 grid h-12 w-full max-w-[330px] grid-cols-[minmax(0,1fr)_32px] items-center rounded-full bg-neutral-100 px-4 pr-3">
          <code className="min-w-0 truncate text-left font-mono text-sm">
            {FIRST_MODEL_COMMAND}
          </code>
          <CopyButton
            content={FIRST_MODEL_COMMAND}
            size="md"
            title="Copy command to clipboard"
            className="shrink-0 text-neutral-400 hover:!bg-transparent hover:!text-neutral-400"
          />
        </div>

        <p className="mt-3 max-w-xs text-[13px] leading-5 text-neutral-400">
          Run this command in your terminal to get started.
        </p>

        <InlineError message={completionError} />
        {completionError && (
          <button
            type="button"
            className="mt-2 cursor-pointer rounded-md px-3 py-1 text-sm font-normal text-neutral-600 underline decoration-neutral-300 underline-offset-4 hover:text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
            onClick={onRetryCompletion}
          >
            Try again
          </button>
        )}
      </OnboardingCard>
    </main>
  );
}

function LaunchCommandIcon({ item }: { item: IntegrationStatus }) {
  const icon = INTEGRATION_ICONS[item.id];

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

export function ClaudeConnectedIntro({ onDone }: { onDone: () => void }) {
  return (
    <div className="claude-connected-backdrop fixed inset-0 z-50 flex items-center justify-center bg-black/20 p-6 dark:bg-black/50">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="claude-connected-title"
        aria-describedby="claude-connected-description"
        className="claude-connected-dialog relative w-full max-w-md overflow-hidden rounded-2xl bg-white font-sans shadow-2xl ring-1 ring-black/10 dark:bg-neutral-800 dark:ring-white/10"
      >
        <img
          src="/claude-connected.png"
          alt="Ollama models in the Claude model picker"
          width={900}
          height={761}
          className="h-auto w-full object-contain"
          draggable={false}
        />
        <div className="p-6">
          <h2
            id="claude-connected-title"
            className="font-rounded text-lg font-medium leading-6 text-neutral-950 dark:text-neutral-100"
          >
            Easily access Ollama models in your Claude
          </h2>
          <p
            id="claude-connected-description"
            className="mt-2 text-[13px] leading-5 text-neutral-500 dark:text-neutral-400"
          >
            Ollama models now show up in Claude so you can pick the right model
            for the task.
          </p>
          <div className="mt-5 flex justify-end">
            <button
              type="button"
              autoFocus
              className="rounded-full bg-neutral-100 px-6 py-2 text-sm font-normal text-neutral-950 transition-colors hover:bg-neutral-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 dark:bg-white dark:hover:bg-neutral-100"
              onClick={onDone}
            >
              Continue
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

export function ConnectAppsScreen({
  initialIntegrations,
  initialClaudeStatus,
}: ConnectAppsScreenProps) {
  const isWindows = isWindowsPlatform();
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null);
  const [claudeError, setClaudeError] = useState<string | null>(null);
  const [claudeStatus, setClaudeStatus] = useState<ClaudeDesktopStatus | null>(
    initialClaudeStatus ?? null,
  );
  const [claudePhase, setClaudePhase] = useState<ClaudeConnectPhase>("idle");
  const [showClaudeConnectedIntro, setShowClaudeConnectedIntro] =
    useState(false);
  const claudeConnectedIntroPending = useRef(false);
  const claudeRestartConfirmed = useRef(false);
  const screenMounted = useRef(true);
  const [integrationStatuses, setIntegrationStatuses] =
    useState<IntegrationStatuses | null>(initialIntegrations ?? null);
  const [statusError, setStatusError] = useState(false);

  useEffect(() => {
    screenMounted.current = true;
    return () => {
      screenMounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (!copiedCommand) return;

    const timeout = window.setTimeout(() => {
      setCopiedCommand(null);
    }, 5000);

    return () => window.clearTimeout(timeout);
  }, [copiedCommand]);

  const refreshClaudeStatus = useCallback(async () => {
    if (isWindows) return null;
    if (
      !window.getClaudeDesktopConnectionSummary &&
      !window.getClaudeDesktopStatus
    ) {
      return null;
    }
    try {
      const status = await getClaudeConnectionSummary();
      if (!status || !screenMounted.current) return null;
      setClaudeStatus(status);
      setClaudeError(null);
      return status;
    } catch {
      if (screenMounted.current) {
        setClaudeError("Ollama could not read the Claude connection status.");
      }
      return null;
    }
  }, [isWindows]);

  useEffect(() => {
    let active = true;
    const integrations = initialIntegrations
      ? Promise.resolve(initialIntegrations)
      : getIntegrationStatuses();
    const claude = initialClaudeStatus
      ? Promise.resolve(initialClaudeStatus)
      : getClaudeConnectionSummary();

    void Promise.allSettled([integrations, claude]).then(
      ([integrationResult, claudeResult]) => {
        if (!active) return;
        if (integrationResult.status === "fulfilled") {
          setIntegrationStatuses(integrationResult.value);
        } else {
          setStatusError(true);
        }
        if (claudeResult.status === "fulfilled") {
          setClaudeStatus(claudeResult.value);
        } else {
          setClaudeError("Ollama could not read the Claude connection status.");
        }
      },
    );

    return () => {
      active = false;
    };
  }, [initialClaudeStatus, initialIntegrations]);

  const openConnectedClaude = useCallback(
    async (status: ClaudeDesktopStatus) => {
      if (!status.connected || status.running || !window.openClaudeDesktop) {
        return null;
      }
      try {
        return (await window.openClaudeDesktop()) || null;
      } catch {
        return "Ollama connected Claude, but could not open the app.";
      }
    },
    [],
  );

  const finishClaudeConnection = useCallback(
    async (status: ClaudeDesktopStatus) => {
      if (claudeConnectedIntroPending.current) return null;
      if (shouldShowClaudeConnectedIntro(status)) {
        claudeConnectedIntroPending.current = true;
        setShowClaudeConnectedIntro(true);
        window.activateOllama?.();
        return null;
      }

      return openConnectedClaude(status);
    },
    [openConnectedClaude],
  );

  const reconcileLateClaudeAction = useCallback(
    (enabled: boolean) =>
      (settled: PromiseSettledResult<ClaudeDesktopActionResult>) => {
        if (!screenMounted.current) return;

        if (settled.status === "fulfilled") {
          const result = settled.value;
          setClaudeStatus(result.status);
          setClaudeError(result.error || null);
          if (result.error || !enabled) return;

          void finishClaudeConnection(result.status).then(
            (completionError) => {
              if (completionError && screenMounted.current) {
                setClaudeError(completionError);
              }
            },
            () => {
              if (screenMounted.current) {
                setClaudeError(
                  "Ollama connected Claude, but could not open the app.",
                );
              }
            },
          );
          return;
        }

        void refreshClaudeStatus().then(() => {
          if (screenMounted.current) {
            setClaudeError(
              enabled
                ? "Ollama could not connect to Claude."
                : "Ollama could not disconnect from Claude.",
            );
          }
        });
      },
    [finishClaudeConnection, refreshClaudeStatus],
  );

  const dismissClaudeConnectedIntro = async () => {
    if (!window.setClaudeDesktopConnected) return;
    setClaudePhase("launching");
    try {
      const liveStatus = await withClaudeConnectionTimeout(
        getClaudeConnectionSummary(),
      );
      if (!liveStatus) {
        throw new Error("Claude Desktop connection status is unavailable");
      }
      setClaudeStatus(liveStatus);

      let restartConfirmed = claudeRestartConfirmed.current;
      if (liveStatus.running && !restartConfirmed) {
        restartConfirmed = window.confirm(
          "Restart Claude Desktop to use Ollama? Any running task will stop.",
        );
        if (!restartConfirmed) {
          setClaudePhase("idle");
          return;
        }
      }

      claudeConnectedIntroPending.current = false;
      claudeRestartConfirmed.current = false;
      setShowClaudeConnectedIntro(false);
      const result = await withClaudeConnectionTimeout(
        window.setClaudeDesktopConnected(true, restartConfirmed),
        CLAUDE_CONNECTION_TIMEOUT_MS,
        reconcileLateClaudeAction(true),
      );
      setClaudeStatus(result.status);
      setClaudeError(result.error || null);
    } catch (error) {
      setClaudeError(
        error instanceof ClaudeConnectionTimeoutError
          ? "Claude is taking too long to launch. Check Claude and try again."
          : "Ollama connected Claude, but could not open the app.",
      );
    } finally {
      setClaudePhase("idle");
    }
  };

  useEffect(() => {
    const handleFocus = () => void refreshClaudeStatus();
    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, [refreshClaudeStatus]);

  useEffect(() => {
    if (!claudeStatus?.connected || !window.getClaudeDesktopRequestCount) {
      return;
    }

    let active = true;
    let checking = false;
    const refreshRequestCount = async () => {
      if (!active || checking || document.visibilityState === "hidden") return;
      checking = true;
      try {
        const routedRequests = await window.getClaudeDesktopRequestCount?.();
        if (!active || routedRequests === undefined) return;
        setClaudeStatus((current) => {
          if (!current || current.routedRequests === routedRequests) {
            return current;
          }
          return { ...current, routedRequests };
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
  }, [claudeStatus?.connected]);

  useEffect(() => {
    if (claudePhase !== "waiting-for-install") return;

    let active = true;
    let completing = false;
    let checking = false;
    const checkForInstall = async () => {
      if (checking) return;
      if (
        (!window.getClaudeDesktopConnectionSummary &&
          !window.getClaudeDesktopStatus) ||
        !window.setClaudeDesktopConnected
      ) {
        return;
      }
      checking = true;
      try {
        const status = await withClaudeConnectionTimeout(
          getClaudeConnectionSummary(),
        );
        if (!status) return;
        if (!active) return;
        setClaudeStatus(status);
        if (!status.installed || completing) return;
        completing = true;

        if (status.running) {
          setClaudePhase("idle");
          setClaudeError(
            "Claude is installed. Turn on Connect to restart it with Ollama.",
          );
          return;
        }

        setClaudePhase("connecting");
        claudeRestartConfirmed.current = false;
        const result = await withClaudeConnectionTimeout(
          setClaudeConnection(true, !status.used, false),
          CLAUDE_CONNECTION_TIMEOUT_MS,
          reconcileLateClaudeAction(true),
        );
        if (!screenMounted.current) return;
        setClaudeStatus(result.status);
        let actionError = result.error || null;
        if (!actionError && result.status.connected) {
          actionError = await finishClaudeConnection(result.status);
        } else if (!actionError) {
          actionError = "Ollama could not connect to Claude.";
        }
        setClaudeError(actionError);
        setClaudePhase("idle");
      } catch (error) {
        if (!screenMounted.current) return;
        setClaudePhase("idle");
        setClaudeError(
          error instanceof ClaudeConnectionTimeoutError
            ? "Claude is taking too long to connect. Check Claude and try again."
            : "Ollama could not finish connecting Claude.",
        );
      } finally {
        checking = false;
      }
    };

    void checkForInstall();
    const interval = window.setInterval(checkForInstall, 1000);
    const timeout = scheduleClaudeInstallTimeout(() => {
      if (!active) return;
      setClaudePhase("idle");
      setClaudeError("Claude installation wasn’t detected. Try again.");
    });
    return () => {
      active = false;
      window.clearInterval(interval);
      window.clearTimeout(timeout);
    };
  }, [claudePhase, finishClaudeConnection, reconcileLateClaudeAction]);

  const copyLaunchCommand = async (item: IntegrationStatus) => {
    if (item.command && (await copyTextToClipboard(item.command))) {
      setClaudeError(null);
      setCopiedCommand(item.command);
    }
  };

  const connectClaude = async () => {
    if (claudePhase !== "idle") return;
    if (
      (!window.getClaudeDesktopConnectionSummary &&
        !window.getClaudeDesktopStatus) ||
      !window.setClaudeDesktopConnected
    ) {
      setClaudeError("Claude connection is available in the Ollama macOS app.");
      return;
    }

    setCopiedCommand(null);
    setClaudeError(null);
    const enabling = claudeStatus ? !isClaudeConfigured(claudeStatus) : true;
    setClaudePhase(enabling ? "connecting" : "disconnecting");

    let status: ClaudeDesktopStatus | null;
    try {
      status = await withClaudeConnectionTimeout(getClaudeConnectionSummary());
    } catch (error) {
      setClaudePhase("idle");
      setClaudeError(
        error instanceof ClaudeConnectionTimeoutError
          ? `Claude is taking too long to ${enabling ? "connect" : "disconnect"}. Try again.`
          : "Ollama could not read the Claude connection status.",
      );
      return;
    }
    if (!status) {
      setClaudePhase("idle");
      setClaudeError("Ollama could not read the Claude connection status.");
      return;
    }
    setClaudeStatus(status);

    // The menu-bar control may have reached this target since the app last
    // refreshed. Sync the row without repeating the profile change or
    // restarting Claude, while preserving first-use completion behavior.
    if (isClaudeConnectionComplete(enabling, status)) {
      claudeRestartConfirmed.current = false;
      const completionError = enabling
        ? await finishClaudeConnection(status)
        : null;
      setClaudeError(completionError);
      setClaudePhase("idle");
      return;
    }

    if (enabling && !status.installed) {
      if (!window.installClaudeDesktop) {
        setClaudePhase("idle");
        setClaudeError("Ollama could not open the Claude installer.");
        return;
      }
      setClaudePhase("installing");
      let installResult: "opened" | "cancelled" | "failed" = "failed";
      try {
        installResult = await window.installClaudeDesktop();
      } catch {
        // The error below is shared with a rejected native install request.
      }
      if (installResult === "cancelled") {
        setClaudePhase("idle");
        return;
      }
      if (installResult !== "opened") {
        setClaudePhase("idle");
        setClaudeError("Ollama could not open the Claude installer.");
        return;
      }
      setClaudePhase("waiting-for-install");
      return;
    }

    let restartConfirmed = false;
    if (status.running) {
      restartConfirmed = window.confirm(
        enabling
          ? "Restart Claude Desktop to use Ollama? Any running task will stop."
          : "Restart Claude Desktop to remove Ollama? Any running task will stop.",
      );
      if (!restartConfirmed) {
        setClaudePhase("idle");
        return;
      }
    }

    claudeRestartConfirmed.current = restartConfirmed;
    try {
      const result = await withClaudeConnectionTimeout(
        setClaudeConnection(
          enabling,
          enabling && !status.used,
          restartConfirmed,
        ),
        CLAUDE_CONNECTION_TIMEOUT_MS,
        reconcileLateClaudeAction(enabling),
      );
      setClaudeStatus(result.status);
      let actionError = result.error || null;
      if (!actionError && enabling && result.status.connected) {
        const openError = await finishClaudeConnection(result.status);
        actionError = openError;
      } else if (
        !actionError &&
        (enabling ? !result.status.connected : result.status.configured)
      ) {
        actionError = enabling
          ? "Ollama could not connect to Claude."
          : "Ollama could not disconnect from Claude.";
      }
      setClaudeError(actionError);
    } catch (error) {
      setClaudeError(
        error instanceof ClaudeConnectionTimeoutError
          ? `Claude is taking too long to ${enabling ? "connect" : "disconnect"}. Try again.`
          : enabling
            ? "Ollama could not connect to Claude."
            : "Ollama could not disconnect from Claude.",
      );
    } finally {
      if (!claudeConnectedIntroPending.current) {
        claudeRestartConfirmed.current = false;
      }
      setClaudePhase("idle");
    }
  };

  const claudeIntegration = isWindows
    ? undefined
    : integrationStatuses?.find((item) => item.id === "claude-desktop");
  const launchIntegrations =
    integrationStatuses?.filter(
      (item) => item.id !== "claude-desktop" && item.command,
    ) ?? [];
  const claudeConnected = claudeStatus?.connected ?? false;
  const claudeConfigured = claudeStatus
    ? isClaudeConfigured(claudeStatus)
    : false;
  const pendingClaudeConnection =
    claudePhase === "installing" ||
    claudePhase === "waiting-for-install" ||
    claudePhase === "connecting" ||
    claudePhase === "launching"
      ? true
      : claudePhase === "disconnecting"
        ? false
        : null;
  const claudeToggleConfigured = optimisticClaudeConnectionState(
    claudeConfigured,
    pendingClaudeConnection,
  );
  const claudeInstalled =
    claudeStatus?.installed ?? claudeIntegration?.installed ?? false;
  const isConnectingClaude = claudePhase !== "idle";
  const claudeStatusLabel =
    claudePhase === "installing"
      ? "Downloading…"
      : claudePhase === "waiting-for-install"
        ? "Finish installing…"
        : claudePhase === "connecting"
          ? "Connecting…"
          : claudePhase === "launching"
            ? "Opening…"
            : claudePhase === "disconnecting"
              ? "Disconnecting…"
              : !claudeConfigured && !claudeInstalled
                ? "Download & connect"
                : null;
  const claudeGuidance = claudeDesktopRecoveryMessage(
    claudeStatus?.error,
    claudeError,
  );
  const launchIntegrationRow = (item: IntegrationStatus) => {
    const copied = copiedCommand === item.command;
    return (
      <div
        key={item.id}
        className="flex min-h-18 items-center justify-between gap-4 px-4 py-3"
      >
        <div className="flex min-w-0 items-center gap-3">
          <LaunchCommandIcon item={item} />
          <div className="min-w-0">
            <p className="text-sm font-medium text-neutral-950 dark:text-neutral-100">
              {item.name}
            </p>
            <p className="truncate text-xs leading-5 text-neutral-500 dark:text-neutral-400">
              {item.description}
            </p>
          </div>
        </div>
        <div className="ml-auto flex min-w-0 shrink-0 items-center overflow-hidden rounded-lg bg-neutral-100 pl-3 dark:bg-neutral-800">
          <code className="block flex-1 whitespace-nowrap py-2 pr-2 font-mono text-[13px] text-neutral-500 dark:text-neutral-400">
            {item.command}
          </code>
          <button
            type="button"
            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-neutral-500 transition-colors hover:text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 dark:text-neutral-400 dark:hover:text-neutral-100"
            onClick={() => copyLaunchCommand(item)}
            aria-label={
              copied
                ? `${item.name} command copied`
                : `Copy ${item.name} command`
            }
            title={copied ? "Copied" : "Copy command"}
          >
            {copied ? (
              <CheckIcon className="h-4 w-4 text-green-600" />
            ) : (
              <Square2StackIcon className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
    );
  };
  const claudeRow = claudeIntegration ? (
    <div className="flex min-h-18 items-center justify-between gap-4 bg-white px-4 py-3 dark:bg-neutral-900">
      <div className="flex min-w-0 items-center gap-3">
        <LaunchCommandIcon item={claudeIntegration} />
        <div className="min-w-0">
          <p className="text-sm font-medium text-neutral-950 dark:text-neutral-100">
            {claudeIntegration.name}
          </p>
          <p
            role={claudeGuidance ? "alert" : undefined}
            className="truncate text-xs leading-5 text-neutral-500 dark:text-neutral-400"
          >
            {claudeGuidance ??
              (claudeConnected
                ? `Connected to Ollama · ${claudeDesktopRequestCountLabel(claudeStatus?.routedRequests ?? 0)}`
                : claudePhase === "installing"
                  ? "Ollama is downloading the Claude installer…"
                  : claudePhase === "waiting-for-install"
                    ? "Finish installing Claude. Ollama will connect it automatically."
                    : claudePhase === "connecting"
                      ? "Connecting Claude to Ollama…"
                      : claudePhase === "launching"
                        ? "Opening Claude…"
                        : claudePhase === "disconnecting"
                          ? "Restoring Claude’s usual connection…"
                          : claudeIntegration.description)}
          </p>
        </div>
      </div>
      <div className="ml-auto flex shrink-0 items-center gap-2.5">
        {claudeStatusLabel && (
          <span
            role="status"
            aria-live="polite"
            className="inline-flex items-center gap-1.5 whitespace-nowrap text-xs text-neutral-500 dark:text-neutral-400"
          >
            {isConnectingClaude && (
              <ArrowPathIcon className="h-3.5 w-3.5 animate-spin" />
            )}
            {claudeStatusLabel}
          </span>
        )}
        <button
          type="button"
          role="switch"
          aria-checked={claudeToggleConfigured}
          aria-busy={isConnectingClaude || undefined}
          aria-label={
            claudeConfigured
              ? "Disconnect Claude"
              : isConnectingClaude
                ? "Connecting Claude"
                : "Connect Claude"
          }
          title={claudeConfigured ? "Disconnect" : "Connect"}
          disabled={isConnectingClaude}
          onClick={connectClaude}
          className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-wait disabled:opacity-50 ${claudeToggleConfigured ? "bg-neutral-950 dark:bg-white" : "bg-neutral-300 dark:bg-neutral-700"}`}
        >
          <span
            aria-hidden="true"
            className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${isConnectingClaude ? "animate-pulse" : ""} ${claudeToggleConfigured ? "translate-x-4.5 dark:bg-neutral-900" : "translate-x-0.5"}`}
          />
        </button>
      </div>
    </div>
  ) : null;

  return (
    <main className="flex min-h-0 w-full flex-1 flex-col overflow-hidden bg-white text-neutral-950 dark:bg-neutral-900 dark:text-neutral-100">
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain p-6">
        <section className="min-h-0 flex-1">
          <div className="mx-auto w-full max-w-4xl text-left">
            {integrationStatuses ? (
              <div className="space-y-7 pb-4 pt-2">
                {claudeIntegration && (
                  <section aria-labelledby="desktop-heading">
                    <h2
                      id="desktop-heading"
                      className="px-4 text-xs font-medium uppercase tracking-wider text-neutral-400 dark:text-neutral-500"
                    >
                      Desktop
                    </h2>
                    <div className="mt-2 bg-white dark:bg-neutral-900">
                      {claudeRow}
                    </div>
                  </section>
                )}

                {launchIntegrations.length > 0 && (
                  <section aria-labelledby="terminal-heading">
                    <h2
                      id="terminal-heading"
                      className="px-4 text-xs font-medium uppercase tracking-wider text-neutral-400 dark:text-neutral-500"
                    >
                      Terminal
                    </h2>
                    <div className="mt-2 bg-white dark:bg-neutral-900">
                      <div className="space-y-2">
                        {launchIntegrations.map(launchIntegrationRow)}
                      </div>
                    </div>
                  </section>
                )}

                {!claudeIntegration && launchIntegrations.length === 0 && (
                  <p className="py-12 text-center text-sm text-neutral-400 dark:text-neutral-500">
                    No apps found.
                  </p>
                )}
              </div>
            ) : statusError ? (
              <p role="alert" className="mt-8 text-sm text-red-600">
                Couldn&apos;t load integrations.
              </p>
            ) : (
              <p className="mt-8 text-sm text-neutral-400 dark:text-neutral-500">
                Checking integrations…
              </p>
            )}
          </div>
        </section>
      </div>
      {showClaudeConnectedIntro && (
        <ClaudeConnectedIntro
          onDone={() => void dismissClaudeConnectedIntro()}
        />
      )}
    </main>
  );
}

interface OnboardingProps extends ScreenProps {
  completionError: string | null;
  isAuthenticated: boolean;
  onOpenApps: () => Promise<boolean>;
  onRetryCompletion: () => void;
  onSignUp: () => void;
  onUseLocal: () => void;
}

export default function Onboarding(props: OnboardingProps) {
  const [step, setStep] = useState<OnboardingStep>("intro");
  const appsOpeningRef = useRef(false);
  const { onOpenApps } = props;

  const openApps = useCallback(async () => {
    if (appsOpeningRef.current) return;
    appsOpeningRef.current = true;
    const opened = await onOpenApps();
    if (!opened) appsOpeningRef.current = false;
  }, [onOpenApps]);

  useEffect(() => {
    window.setOnboardingWindow?.(true);
    return () => window.setOnboardingWindow?.(false);
  }, []);

  useEffect(() => {
    if (!props.isAuthenticated) return;
    const nextStep = nextOnboardingStep(step, "authenticated", true);
    if (nextStep === "apps") {
      void openApps();
      return;
    }
    setStep(nextStep);
  }, [openApps, props.isAuthenticated, step]);

  if (step === "run") {
    return (
      <RunOllamaScreen
        completionError={props.completionError}
        onRetryCompletion={props.onRetryCompletion}
      />
    );
  }

  if (step === "intro") {
    return (
      <IntroScreen
        completionError={props.completionError}
        onRetryCompletion={() => void openApps()}
        onContinue={() => {
          const nextStep = nextOnboardingStep(
            step,
            "continue",
            props.isAuthenticated,
          );
          if (nextStep === "apps") {
            void openApps();
            return;
          }
          setStep(nextStep);
        }}
      />
    );
  }

  return (
    <WelcomeScreen
      {...props}
      onRetryCompletion={() => void openApps()}
      onLocal={() => {
        props.onUseLocal();
        setStep((current) =>
          nextOnboardingStep(current, "local", props.isAuthenticated),
        );
      }}
    />
  );
}
