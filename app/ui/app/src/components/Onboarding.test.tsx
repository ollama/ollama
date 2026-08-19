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
  onboardingAuthUrl,
} from "@/lib/onboarding";

describe("Onboarding", () => {
  it("explains what Ollama is before asking the user to choose a path", () => {
    const html = renderToStaticMarkup(<IntroScreen onContinue={vi.fn()} />);

    expect(html).toContain("Welcome to Ollama!");
    expect(html).toContain(
      "Ollama lets you use open models with your coding agents so you can spend less while keeping your data private.",
    );
    expect(html).toContain("Keep your setup");
    expect(html).toContain("Your data stays yours");
    expect(html).toContain("Easily switch models");
    expect(html).toContain("Run Ollama with the agents you already use.");
    expect(html).toContain("Your prompts are never trained on or tracked.");
    expect(html).toContain("Swap between frontier models in one click.");
    expect(html).toContain("Continue");
    expect(html).not.toContain("Skip");
  });

  it("shows the cloud choice to signed-out and signed-in users", () => {
    expect(nextOnboardingStep("intro", "continue", false)).toBe("welcome");
    expect(nextOnboardingStep("intro", "continue", true)).toBe("welcome");
    expect(nextOnboardingStep("intro", "skip", false)).toBe("run");
    expect(nextOnboardingStep("welcome", "local", false)).toBe("run");
  });

  it("lets an in-flight authentication check finish before timing out", () => {
    expect(authenticationTimeoutAction(false, true)).toBe("defer");
    expect(authenticationTimeoutAction(false, false)).toBe("fail");
    expect(authenticationTimeoutAction(true, true)).toBe("ignore");
  });

  it("keeps the onboarding app in control of sign-up and sign-in returns", () => {
    expect(
      onboardingAuthUrl(
        "https://ollama.com/connect?name=MacBook&key=public-key&launch=true",
        "signup",
      ),
    ).toBe(
      "https://ollama.com/signup?next=%2Fconnect%3Fname%3DMacBook%26key%3Dpublic-key",
    );
    expect(
      onboardingAuthUrl(
        "https://ollama.com/connect?name=MacBook&key=public-key&launch=true",
        "signin",
      ),
    ).toBe(
      "https://ollama.com/signin?next=%2Fconnect%3Fname%3DMacBook%26key%3Dpublic-key",
    );
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
        onUseLocal={vi.fn()}
        showRun
      />,
    );

    expect(html).toContain("Run Ollama");
    expect(html).not.toContain("Welcome to Ollama");
    expect(html).not.toContain("Get started with Ollama Cloud");
  });

  it("offers cloud sign-up, local setup, and sign in on the welcome screen", () => {
    const html = renderToStaticMarkup(
      <WelcomeScreen
        isAuthenticated={false}
        isSigningIn={false}
        signInError={null}
        onSignIn={vi.fn()}
        onSignUp={vi.fn()}
        onLocal={vi.fn()}
      />,
    );

    expect(html).toContain("Run powerful models with Ollama Cloud");
    expect(html).toContain(
      "Sign up to power your agents with frontier models without having to buy frontier hardware.",
    );
    expect(html).toContain("Get started with Ollama Cloud");
    expect(html).toContain("Get started locally");
    expect(html).toContain("Sign in");
    expect(html).not.toContain("Skip");
  });

  it("shows the cloud choice without a sign-in link for authenticated users", () => {
    const html = renderToStaticMarkup(
      <WelcomeScreen
        isAuthenticated
        isSigningIn={false}
        signInError={null}
        onSignIn={vi.fn()}
        onSignUp={vi.fn()}
        onLocal={vi.fn()}
      />,
    );

    expect(html).toContain("Run powerful models with Ollama Cloud");
    expect(html).toContain(
      "Power your agents with frontier models without having to buy frontier hardware.",
    );
    expect(html).not.toContain("Sign up to power your agents");
    expect(html).not.toContain(">Sign in<");
  });

  it("shows only the local command and finish action on the final page", () => {
    const html = renderToStaticMarkup(
      <RunOllamaScreen
        completionError={null}
        onFinish={vi.fn()}
      />,
    );

    expect(html).toContain("Run Ollama");
    expect(html).toContain(FIRST_MODEL_COMMAND);
    expect(html).toContain("Finish");
    expect(html).not.toContain("Sign in");
    expect(html).not.toContain("create an account");
  });

  it("shows the connecting state on the welcome action", () => {
    const html = renderToStaticMarkup(
      <WelcomeScreen
        isAuthenticated={false}
        isSigningIn
        signInError={null}
        onSignIn={vi.fn()}
        onSignUp={vi.fn()}
        onLocal={vi.fn()}
      />,
    );

    expect(html).toContain("Finish in your browser…");
    expect(html).not.toContain("Waiting for sign in…");
  });

  it("shows a retryable error when onboarding completion cannot be saved", () => {
    const html = renderToStaticMarkup(
      <RunOllamaScreen
        completionError="Unable to save setup. Please try again."
        onFinish={vi.fn()}
      />,
    );

    expect(html).toContain("Unable to save setup. Please try again.");
    expect(html).toContain('role="alert"');
  });
});
