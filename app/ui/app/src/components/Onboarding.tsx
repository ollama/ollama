import CopyButton from "@/components/CopyButton";
import Logo from "@/components/Logo";
import { nextOnboardingStep, type OnboardingStep } from "@/lib/onboarding";
import {
  ArrowsRightLeftIcon,
  CommandLineIcon,
  ShieldCheckIcon,
} from "@heroicons/react/24/outline";
import { useEffect, useState, type ReactNode } from "react";

export const FIRST_MODEL_COMMAND = "ollama";

interface ScreenProps {
  isSigningIn: boolean;
  signInError: string | null;
  onSignIn: () => void;
}

interface WelcomeScreenProps extends ScreenProps {
  isAuthenticated: boolean;
  onLocal: () => void;
  onSignUp: () => void;
}

interface RunOllamaScreenProps {
  completionError: string | null;
  onRetryCompletion: () => void;
}

function TitleBar({ onSignIn }: { onSignIn?: () => void }) {
  return (
    <header
      className="relative flex h-10 shrink-0 items-center justify-center bg-white"
      onDoubleClick={() => window.doubleClick?.()}
      onMouseDown={() => window.drag?.()}
    >
      {onSignIn && (
        <button
          type="button"
          className="absolute right-5 cursor-pointer rounded-md px-2 py-1 text-sm font-normal text-neutral-500 hover:text-neutral-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
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
    title: "Keep your setup",
    description: "Run Ollama with the agents you already use.",
    icon: CommandLineIcon,
  },
  {
    title: "Your data stays yours",
    description: "Your data is never logged or trained on.",
    icon: ShieldCheckIcon,
  },
  {
    title: "Easily switch models",
    description: "Swap between frontier models in one click.",
    icon: ArrowsRightLeftIcon,
  },
];

export function IntroScreen({ onContinue }: { onContinue: () => void }) {
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
          <p className="mt-4 max-w-[500px] text-sm leading-6 text-neutral-500">
            Run open models locally or in the cloud.
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
  onSignIn,
  onSignUp,
  onLocal,
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
          <InlineError message={signInError} />
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

interface OnboardingProps extends ScreenProps {
  completionError: string | null;
  isAuthenticated: boolean;
  onRetryCompletion: () => void;
  onSignUp: () => void;
  onUseLocal: () => void;
  showRun: boolean;
}

export default function Onboarding(props: OnboardingProps) {
  const [step, setStep] = useState<OnboardingStep>("intro");

  useEffect(() => {
    window.setOnboardingWindow?.(true);
    return () => window.setOnboardingWindow?.(false);
  }, []);

  if (props.showRun || step === "run") {
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
        onContinue={() => {
          const nextStep = nextOnboardingStep(
            step,
            "continue",
            props.isAuthenticated,
          );
          if (nextStep === "run") props.onUseLocal();
          setStep(nextStep);
        }}
      />
    );
  }

  return (
    <WelcomeScreen
      {...props}
      onLocal={() => {
        props.onUseLocal();
        setStep((current) =>
          nextOnboardingStep(current, "local", props.isAuthenticated),
        );
      }}
    />
  );
}
