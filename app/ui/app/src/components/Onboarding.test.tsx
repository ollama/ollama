import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import {
  FIRST_MODEL_COMMAND,
  IntroScreen,
  default as Onboarding,
  RunOllamaScreen,
  WelcomeScreen,
} from "./Onboarding";
import {
  authenticationTimeoutAction,
  nextOnboardingStep,
} from "@/lib/onboarding";

describe("Onboarding", () => {
  it("explains what Ollama is before asking the user to choose a path", () => {
    const html = renderToStaticMarkup(
      <IntroScreen onContinue={vi.fn()} onSkip={vi.fn()} />,
    );

    expect(html).toContain("What is Ollama?");
    expect(html).toContain("Run models locally");
    expect(html).toContain("Connect your tools");
    expect(html).toContain("Choose the right model");
    expect(html).toContain("Continue");
    expect(html).toContain("Skip");
  });

  it("routes signed-out and signed-in users without an extra connected page", () => {
    expect(nextOnboardingStep("intro", "continue", false)).toBe("welcome");
    expect(nextOnboardingStep("intro", "continue", true)).toBe("run");
    expect(nextOnboardingStep("intro", "skip", false)).toBe("run");
    expect(nextOnboardingStep("welcome", "local", false)).toBe("run");
  });

  it("lets an in-flight authentication check finish before timing out", () => {
    expect(authenticationTimeoutAction(false, true)).toBe("defer");
    expect(authenticationTimeoutAction(false, false)).toBe("fail");
    expect(authenticationTimeoutAction(true, true)).toBe("ignore");
  });

  it("shows Run Ollama after a successful connection and hides sign in", () => {
    const html = renderToStaticMarkup(
      <Onboarding
        isAuthenticated
        isSigningIn={false}
        signInError={null}
        completionError={null}
        onSignIn={vi.fn()}
        onFinish={vi.fn()}
        showRun
      />,
    );

    expect(html).toContain("Run Ollama");
    expect(html).not.toContain("Welcome to Ollama");
    expect(html).not.toContain("Sign in or create an account");
  });

  it("offers sign in, local only, and skip on the welcome screen", () => {
    const html = renderToStaticMarkup(
      <WelcomeScreen
        isSigningIn={false}
        signInError={null}
        onSignIn={vi.fn()}
        onLocal={vi.fn()}
      />,
    );

    expect(html).toContain("Welcome to Ollama");
    expect(html).toContain("Sign in or create an account");
    expect(html).toContain("Local only");
    expect(html).toContain("Skip");
    expect(html).not.toContain("Use local only");
  });

  it("shows the local command, finish action, and a way back to sign in", () => {
    const html = renderToStaticMarkup(
      <RunOllamaScreen
        isSigningIn={false}
        signInError={null}
        completionError={null}
        onSignIn={vi.fn()}
        onFinish={vi.fn()}
        showSignIn
      />,
    );

    expect(html).toContain("Run Ollama");
    expect(html).toContain(FIRST_MODEL_COMMAND);
    expect(html).toContain("Finish");
    expect(html).toContain("Sign in or create an account");
  });

  it("shows the connecting state on the welcome action", () => {
    const html = renderToStaticMarkup(
      <WelcomeScreen
        isSigningIn
        signInError={null}
        onSignIn={vi.fn()}
        onLocal={vi.fn()}
      />,
    );

    expect(html).toContain("Waiting to connect…");
    expect(html).not.toContain("Waiting for sign in…");
  });

  it("shows a retryable error when onboarding completion cannot be saved", () => {
    const html = renderToStaticMarkup(
      <RunOllamaScreen
        isSigningIn={false}
        signInError={null}
        completionError="Unable to save setup. Please try again."
        onSignIn={vi.fn()}
        onFinish={vi.fn()}
        showSignIn
      />,
    );

    expect(html).toContain("Unable to save setup. Please try again.");
    expect(html).toContain('role="alert"');
  });
});
