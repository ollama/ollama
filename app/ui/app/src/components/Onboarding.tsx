import CopyButton from "@/components/CopyButton";
import Logo from "@/components/Logo";
import { nextOnboardingStep, type OnboardingStep } from "@/lib/onboarding";
import {
  CommandLineIcon,
  CubeTransparentIcon,
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
  onLocal: () => void;
}

interface RunOllamaScreenProps extends ScreenProps {
  completionError: string | null;
  onFinish: () => void;
  showSignIn: boolean;
}

function TitleBar({ onSkip }: { onSkip?: () => void }) {
  return (
    <header
      className="relative flex h-10 shrink-0 items-center justify-center bg-white"
      onDoubleClick={() => window.doubleClick?.()}
      onMouseDown={() => window.drag?.()}
    >
      {onSkip && (
        <button
          type="button"
          className="absolute right-5 cursor-pointer rounded-md px-2 py-1 text-sm font-normal text-neutral-500 hover:text-neutral-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
          onClick={onSkip}
          onMouseDown={(event) => event.stopPropagation()}
        >
          Skip
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
    title: "Open models, anywhere",
    description: "Run open models on your computer or in the cloud.",
    icon: CubeTransparentIcon,
  },
  {
    title: "Works with your tools",
    description: "Use Ollama with Claude, ChatGPT, Codex, and more.",
    icon: CommandLineIcon,
  },
  {
    title: "Your data stays yours",
    description: "Choose where your models run and keep control of your data.",
    icon: ShieldCheckIcon,
  },
];

export function IntroScreen({ onContinue }: { onContinue: () => void }) {
  return (
    <main className="flex h-screen w-full flex-col overflow-hidden bg-white text-neutral-950">
      <TitleBar />

      <section className="min-h-0 flex-1 overflow-y-auto px-6">
        <div className="mx-auto flex min-h-full w-full max-w-[620px] flex-col items-center justify-center py-4 text-center">
          <img
            src="/hello.png"
            alt="Ollama waving"
            className="h-[72px] w-[72px] select-none object-contain"
            draggable={false}
          />
          <h1 className="mt-6 font-rounded text-2xl font-medium leading-8">
            What is Ollama?
          </h1>
          <p className="mt-3 max-w-[440px] text-sm leading-6 text-neutral-500">
            Run AI models on your own computer, or connect to Ollama for cloud
            models.
          </p>

          <div className="mt-8 w-full max-w-[480px] space-y-5 text-left">
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
  isSigningIn,
  signInError,
  onSignIn,
  onLocal,
}: WelcomeScreenProps) {
  return (
    <main className="flex min-h-screen w-full flex-col bg-white text-neutral-950">
      <TitleBar onSkip={onLocal} />
      <OnboardingCard>
        <OnboardingIcon />
        <h1 className="mt-7 font-rounded text-2xl font-medium leading-8">
          Welcome to Ollama
        </h1>
        <p className="mt-3 max-w-[400px] text-sm leading-6 text-neutral-400">
          Sign in to use the latest open models in the cloud, or continue with
          local models.
        </p>

        <div className="mt-7 flex w-full max-w-[240px] flex-col items-center">
          <button
            type="button"
            className="flex h-11 w-full cursor-pointer items-center justify-center rounded-full bg-neutral-900 px-5 font-sans text-sm font-normal text-white transition-colors hover:bg-neutral-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-wait disabled:opacity-70"
            onClick={onSignIn}
            disabled={isSigningIn}
            aria-busy={isSigningIn}
          >
            {isSigningIn
              ? "Waiting to connect…"
              : "Sign in or create an account"}
          </button>
          <button
            type="button"
            className="mt-2 cursor-pointer rounded-md px-3 py-2 text-sm font-normal text-neutral-600 underline decoration-neutral-300 underline-offset-4 hover:text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
            onClick={onLocal}
          >
            Local only
          </button>
          <InlineError message={signInError} />
        </div>
      </OnboardingCard>
    </main>
  );
}

export function RunOllamaScreen({
  isSigningIn,
  signInError,
  completionError,
  onSignIn,
  onFinish,
  showSignIn,
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
            className="shrink-0 text-neutral-400 hover:bg-neutral-200 hover:text-neutral-700"
          />
        </div>

        <p className="mt-3 max-w-xs text-[13px] leading-5 text-neutral-400">
          Run this command in your terminal to get started.
        </p>

        <button
          type="button"
          className="mt-7 h-10 w-full max-w-[180px] cursor-pointer rounded-full bg-neutral-900 px-5 font-sans text-sm font-normal text-white transition-colors hover:bg-neutral-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500"
          onClick={onFinish}
        >
          Finish
        </button>
        <InlineError message={completionError} />

        {showSignIn && (
          <div className="mt-2 max-w-[220px] text-center">
            <button
              type="button"
              className="cursor-pointer rounded-md px-3 py-2 text-[13px] font-normal text-neutral-500 underline decoration-neutral-300 underline-offset-4 hover:text-neutral-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-wait disabled:opacity-70"
              onClick={onSignIn}
              disabled={isSigningIn}
              aria-busy={isSigningIn}
            >
              {isSigningIn
                ? "Waiting to connect…"
                : "Sign in or create an account"}
            </button>
            <InlineError message={signInError} />
          </div>
        )}
      </OnboardingCard>
    </main>
  );
}

interface OnboardingProps extends ScreenProps {
  completionError: string | null;
  isAuthenticated: boolean;
  onFinish: () => void;
  onUseLocal: () => void;
  showRun: boolean;
}

export default function Onboarding(props: OnboardingProps) {
  const [step, setStep] = useState<OnboardingStep>("intro");

  useEffect(() => {
    window.setOnboardingWindow?.(true);
    return () => window.setOnboardingWindow?.(false);
  }, []);

  if (
    props.showRun ||
    step === "run" ||
    (step === "welcome" && props.isAuthenticated)
  ) {
    return (
      <RunOllamaScreen
        isSigningIn={props.isSigningIn}
        signInError={props.signInError}
        completionError={props.completionError}
        onSignIn={props.onSignIn}
        onFinish={props.onFinish}
        showSignIn={!props.isAuthenticated}
      />
    );
  }

  if (step === "intro") {
    return (
      <IntroScreen
        onContinue={() =>
          setStep((current) =>
            nextOnboardingStep(current, "continue", props.isAuthenticated),
          )
        }
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
