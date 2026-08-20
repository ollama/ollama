import CopyButton from "@/components/CopyButton";
import Logo from "@/components/Logo";
import {
  isOnboardingZoomShortcut,
  nextOnboardingStep,
  type OnboardingStep,
} from "@/lib/onboarding";
import {
  getIntegrationStatuses,
  type IntegrationStatus,
  type IntegrationStatuses,
} from "@/api";
import { INTEGRATION_ICONS } from "@/lib/launchCommands";
import type { ClaudeDesktopStatus } from "@/types/webview";
import { copyTextToClipboard } from "@/utils/clipboard";
import {
  ArrowPathIcon,
  ArrowsRightLeftIcon,
  CommandLineIcon,
  ShieldCheckIcon,
  Square2StackIcon,
} from "@heroicons/react/24/outline";
import {
  CheckIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from "@heroicons/react/20/solid";
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
  | "disconnecting";

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
  completionError: string | null;
  onRetryCompletion: () => void;
  initialIntegrations?: IntegrationStatuses;
  initialClaudeStatus?: ClaudeDesktopStatus;
  embedded?: boolean;
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
    <main className="flex h-screen w-full flex-col overflow-hidden bg-white text-neutral-950">
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
    <main className="flex min-h-screen w-full flex-col bg-white text-neutral-950">
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
    <main className="flex min-h-screen w-full flex-col bg-white text-neutral-950">
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
            className="shrink-0 text-neutral-400 hover:!bg-transparent hover:!text-neutral-400 dark:hover:!bg-transparent"
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
        <img
          src={icon.src}
          alt=""
          className={`${icon.className ?? "h-7 w-7"} rounded-sm object-contain`}
        />
      ) : (
        <CommandLineIcon className="h-6 w-6 stroke-[1.5] text-neutral-700" />
      )}
    </div>
  );
}

export function ConnectAppsScreen({
  completionError,
  onRetryCompletion,
  initialIntegrations,
  initialClaudeStatus,
  embedded = false,
}: ConnectAppsScreenProps) {
  const isWindows = navigator.platform.toLowerCase().includes("win");
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null);
  const [claudeError, setClaudeError] = useState<string | null>(null);
  const [claudeStatus, setClaudeStatus] = useState<ClaudeDesktopStatus | null>(
    initialClaudeStatus ?? null,
  );
  const [claudePhase, setClaudePhase] = useState<ClaudeConnectPhase>("idle");
  const [integrationStatuses, setIntegrationStatuses] =
    useState<IntegrationStatuses | null>(initialIntegrations ?? null);
  const [showAllIntegrations, setShowAllIntegrations] = useState(false);
  const [statusError, setStatusError] = useState(false);

  useEffect(() => {
    if (initialIntegrations) return;

    let active = true;
    getIntegrationStatuses()
      .then((statuses) => {
        if (active) setIntegrationStatuses(statuses);
      })
      .catch(() => {
        if (active) setStatusError(true);
      });

    return () => {
      active = false;
    };
  }, [initialIntegrations]);

  useEffect(() => {
    if (!copiedCommand && !claudeError) return;

    const timeout = window.setTimeout(() => {
      setCopiedCommand(null);
      setClaudeError(null);
    }, 5000);

    return () => window.clearTimeout(timeout);
  }, [claudeError, copiedCommand]);

  const refreshClaudeStatus = useCallback(async () => {
    if (!window.getClaudeDesktopStatus) return null;
    try {
      const status = await window.getClaudeDesktopStatus();
      setClaudeStatus(status);
      return status;
    } catch {
      setClaudeError("Ollama could not read the Claude connection status.");
      return null;
    }
  }, []);

  useEffect(() => {
    void refreshClaudeStatus();
    const handleFocus = () => void refreshClaudeStatus();
    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, [refreshClaudeStatus]);

  useEffect(() => {
    if (claudePhase !== "waiting-for-install") return;

    let active = true;
    let completing = false;
    const checkForInstall = async () => {
      if (!window.getClaudeDesktopStatus || !window.setClaudeDesktopConnected) {
        return;
      }
      try {
        const status = await window.getClaudeDesktopStatus();
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
        const result = await window.setClaudeDesktopConnected(true);
        if (!active) return;
        setClaudeStatus(result.status);
        setClaudeError(
          result.error ||
            (!result.status.connected
              ? "Ollama could not connect to Claude."
              : null),
        );
        setClaudePhase("idle");
      } catch {
        if (!active) return;
        setClaudePhase("idle");
        setClaudeError("Ollama could not finish connecting Claude.");
      }
    };

    void checkForInstall();
    const interval = window.setInterval(checkForInstall, 1000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [claudePhase]);

  const copyLaunchCommand = async (item: IntegrationStatus) => {
    if (item.command && (await copyTextToClipboard(item.command))) {
      setClaudeError(null);
      setCopiedCommand(item.command);
    }
  };

  const connectClaude = async () => {
    if (claudePhase !== "idle") return;
    if (!window.getClaudeDesktopStatus || !window.setClaudeDesktopConnected) {
      setClaudeError("Claude connection is available in the Ollama macOS app.");
      return;
    }

    setCopiedCommand(null);
    setClaudeError(null);
    const status = await refreshClaudeStatus();
    if (!status) return;
    if (!status.supported) {
      setClaudeError("Claude connection is currently available on macOS.");
      return;
    }

    const enabling = !status.connected;
    if (enabling && !status.installed) {
      if (!window.installClaudeDesktop) {
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

    if (status.running) {
      const confirmed = window.confirm(
        enabling
          ? "Restart Claude Desktop to use Ollama? Any running task will stop."
          : "Restart Claude Desktop to remove Ollama? Any running task will stop.",
      );
      if (!confirmed) return;
    }

    setClaudePhase(enabling ? "connecting" : "disconnecting");
    try {
      const result = await window.setClaudeDesktopConnected(enabling);
      setClaudeStatus(result.status);
      if (result.error) {
        setClaudeError(result.error);
      } else if (result.status.connected !== enabling) {
        setClaudeError(
          enabling
            ? "Ollama could not connect to Claude."
            : "Ollama could not disconnect from Claude.",
        );
      }
    } catch {
      setClaudeError(
        enabling
          ? "Ollama could not connect to Claude."
          : "Ollama could not disconnect from Claude.",
      );
    } finally {
      setClaudePhase("idle");
    }
  };

  const claudeIntegration = integrationStatuses?.find(
    (item) => item.id === "claude-desktop",
  );
  const launchIntegrations =
    integrationStatuses?.filter(
      (item) => item.id !== "claude-desktop" && item.command,
    ) ?? [];
  const claudeConnected = claudeStatus?.connected ?? false;
  const claudeInstalled =
    claudeStatus?.installed ?? claudeIntegration?.installed ?? false;
  const claudeSupported = claudeStatus?.supported ?? true;
  const isConnectingClaude = claudePhase !== "idle";
  const disconnectedClaudeCount = claudeIntegration && !claudeConnected ? 1 : 0;
  const collapsedLaunchCount = Math.max(0, 5 - disconnectedClaudeCount);
  const initialLaunchIntegrations = launchIntegrations.slice(
    0,
    collapsedLaunchCount,
  );
  const additionalLaunchIntegrations =
    launchIntegrations.slice(collapsedLaunchCount);
  const canToggleIntegrations =
    launchIntegrations.length + disconnectedClaudeCount > 5;
  const launchIntegrationRow = (item: IntegrationStatus) => {
    const copied = copiedCommand === item.command;
    return (
      <div
        key={item.id}
        className="flex min-h-18 items-center justify-between gap-4 px-4 py-3 max-[1050px]:flex-col max-[1050px]:items-stretch max-[1050px]:gap-2"
      >
        <div className="flex min-w-0 items-center gap-3">
          <LaunchCommandIcon item={item} />
          <div className="min-w-0">
            <p className="text-sm font-medium text-neutral-950">{item.name}</p>
            <p className="truncate text-xs leading-5 text-neutral-500">
              {item.description}
            </p>
          </div>
        </div>
        <div className="ml-auto flex min-w-0 shrink-0 items-center overflow-hidden rounded-lg bg-neutral-100 pl-3 max-[1050px]:ml-0 max-[1050px]:w-full">
          <code className="block flex-1 whitespace-nowrap py-2 pr-2 font-mono text-[13px] text-neutral-500 max-[1050px]:min-w-0">
            {item.command}
          </code>
          <button
            type="button"
            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-neutral-500 transition-colors hover:text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
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
    <div
      className={`flex min-h-18 items-center justify-between gap-4 bg-white px-4 py-3 ${claudeConnected ? "mt-2 rounded-xl border border-neutral-200" : ""}`}
    >
      <div className="flex min-w-0 items-center gap-3">
        <LaunchCommandIcon item={claudeIntegration} />
        <div className="min-w-0">
          <p className="text-sm font-medium text-neutral-950">
            {claudeIntegration.name}
          </p>
          <p
            role={claudeError ? "alert" : undefined}
            className={`truncate text-xs leading-5 ${claudeError ? "text-red-600" : "text-neutral-500"}`}
          >
            {claudeError ??
              (claudeConnected
                ? "Connected to Ollama"
                : claudePhase === "installing"
                  ? "Ollama is downloading the Claude installer…"
                  : claudePhase === "waiting-for-install"
                    ? "Finish installing Claude. Ollama will connect it automatically."
                    : claudePhase === "connecting"
                      ? "Connecting Claude to Ollama…"
                      : claudePhase === "disconnecting"
                        ? "Restoring Claude’s usual connection…"
                        : !claudeSupported
                          ? "Claude connection is currently available on macOS."
                          : claudeIntegration.description)}
          </p>
        </div>
      </div>
      <div className="ml-auto flex shrink-0 items-center gap-2.5">
        <span
          role="status"
          aria-live="polite"
          className={`inline-flex items-center gap-1.5 whitespace-nowrap text-xs ${claudeConnected && !isConnectingClaude ? "text-green-700" : "text-neutral-500"}`}
        >
          {isConnectingClaude && (
            <ArrowPathIcon className="h-3.5 w-3.5 animate-spin" />
          )}
          {claudeConnected && !isConnectingClaude && (
            <span
              aria-hidden="true"
              className="h-1.5 w-1.5 rounded-full bg-green-500"
            />
          )}
          {claudePhase === "installing"
            ? "Downloading…"
            : claudePhase === "waiting-for-install"
              ? "Finish installing…"
              : claudePhase === "connecting"
                ? "Connecting…"
                : claudePhase === "disconnecting"
                  ? "Disconnecting…"
                  : claudeConnected
                    ? "Active"
                    : !claudeSupported
                      ? "Unavailable"
                      : claudeInstalled
                        ? "Inactive"
                        : "Download & connect"}
        </span>
        <button
          type="button"
          role="switch"
          aria-checked={claudeConnected}
          aria-busy={isConnectingClaude || undefined}
          aria-label={
            claudeConnected
              ? "Disconnect Claude"
              : isConnectingClaude
                ? "Connecting Claude"
                : "Connect Claude"
          }
          title={claudeConnected ? "Connected" : "Connect"}
          disabled={isConnectingClaude || !claudeSupported}
          onClick={connectClaude}
          className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-wait ${claudeConnected ? "bg-neutral-950" : "bg-neutral-300"}`}
        >
          <span
            aria-hidden="true"
            className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${isConnectingClaude ? "animate-pulse" : ""} ${claudeConnected ? "translate-x-4.5" : "translate-x-0.5"}`}
          />
        </button>
      </div>
    </div>
  ) : null;

  return (
    <main
      className={`flex w-full flex-col overflow-hidden bg-white text-neutral-950 ${embedded ? "min-h-0 flex-1" : "h-screen"}`}
    >
      {!embedded && (
        <header
          className="flex h-[52px] w-full flex-none select-none items-center justify-between border-b border-neutral-200 py-2.5"
          onMouseDown={() => window.drag?.()}
          onDoubleClick={() => window.doubleClick?.()}
        >
          <h1
            className={`${isWindows ? "pl-4" : "pl-24"} font-rounded text-md flex items-center font-medium`}
          >
            Connect your apps
          </h1>
        </header>
      )}
      <div className="flex min-h-0 flex-1 flex-col p-6">
        <section className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-5xl text-left">
            {integrationStatuses ? (
              <div className="space-y-7 pb-4 pt-2">
                {claudeIntegration && claudeConnected && (
                  <section aria-labelledby="claude-apps-heading">
                    <h2
                      id="claude-apps-heading"
                      className="px-1 text-xs font-medium uppercase tracking-wider text-neutral-400"
                    >
                      Connected
                    </h2>
                    {claudeRow}
                  </section>
                )}

                {(launchIntegrations.length > 0 ||
                  (claudeIntegration && !claudeConnected)) && (
                  <section aria-labelledby="launch-apps-heading">
                    <h2
                      id="launch-apps-heading"
                      className="px-1 text-xs font-medium uppercase tracking-wider text-neutral-400"
                    >
                      Ready to launch
                    </h2>
                    <div className="mt-2 overflow-hidden bg-white">
                      {!claudeConnected && claudeRow && (
                        <div className="mb-2">{claudeRow}</div>
                      )}
                      <div className="space-y-2">
                        {initialLaunchIntegrations.map(launchIntegrationRow)}
                      </div>
                      {canToggleIntegrations && (
                        <div
                          aria-hidden={!showAllIntegrations}
                          className={`grid transition-[grid-template-rows,opacity,visibility] duration-300 ease-out ${
                            showAllIntegrations
                              ? "visible grid-rows-[1fr] opacity-100"
                              : "invisible grid-rows-[0fr] opacity-0"
                          }`}
                        >
                          <div className="min-h-0 overflow-hidden">
                            <div className="space-y-2 pt-2">
                              {additionalLaunchIntegrations.map(
                                launchIntegrationRow,
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                      {canToggleIntegrations && (
                        <button
                          type="button"
                          aria-expanded={showAllIntegrations}
                          className="mt-2 flex h-9 w-full items-center justify-center gap-1.5 rounded-lg text-xs text-neutral-500 transition-colors hover:text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
                          onMouseEnter={() => setShowAllIntegrations(true)}
                          onFocus={() => setShowAllIntegrations(true)}
                          onClick={() =>
                            setShowAllIntegrations((current) => !current)
                          }
                        >
                          {showAllIntegrations ? "Collapse" : "More"}
                          {showAllIntegrations ? (
                            <ChevronUpIcon className="h-3.5 w-3.5" />
                          ) : (
                            <ChevronDownIcon className="h-3.5 w-3.5" />
                          )}
                        </button>
                      )}
                    </div>
                  </section>
                )}

                {!claudeIntegration && launchIntegrations.length === 0 && (
                  <p className="py-12 text-center text-sm text-neutral-400">
                    No apps found.
                  </p>
                )}
              </div>
            ) : statusError ? (
              <p role="alert" className="mt-8 text-sm text-red-600">
                Couldn&apos;t load integrations.
              </p>
            ) : (
              <p className="mt-8 text-sm text-neutral-400">
                Checking integrations…
              </p>
            )}

            <InlineError message={completionError} />
            {completionError && (
              <button
                type="button"
                className="mt-2 cursor-pointer rounded-md px-1 py-1 text-sm font-normal text-neutral-600 underline decoration-neutral-300 underline-offset-4 hover:text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
                onClick={onRetryCompletion}
              >
                Try again
              </button>
            )}
          </div>
        </section>
      </div>
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
    const preventZoom = (event: KeyboardEvent) => {
      if (isOnboardingZoomShortcut(event)) event.preventDefault();
    };

    window.addEventListener("keydown", preventZoom, true);
    return () => window.removeEventListener("keydown", preventZoom, true);
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
