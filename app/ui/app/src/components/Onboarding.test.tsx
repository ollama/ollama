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
  onboardingConnectUrl,
} from "@/lib/onboarding";

describe("Onboarding", () => {
  it("explains what Ollama is before asking the user to choose a path", () => {
    const html = renderToStaticMarkup(<IntroScreen onContinue={vi.fn()} />);

    expect(html).toContain("Welcome to Ollama!");
    expect(html.indexOf('alt="Ollama waving"')).toBeLessThan(
      html.indexOf("Welcome to Ollama!"),
    );
    expect(html).toContain("Run open models locally or in the cloud.");
    expect(html).toContain("Keep your setup");
    expect(html).toContain("Your data stays yours");
    expect(html).toContain("Easily switch models");
    expect(html).toContain("Run Ollama with the agents you already use.");
    expect(html).toContain("Your data is never logged or trained on.");
    expect(html).toContain("Swap between frontier models in one click.");
    expect(html).toContain("Continue");
    expect(html).not.toContain("Skip");
  });

  it("shows the account choice only to signed-out users", () => {
    expect(nextOnboardingStep("intro", "continue", false)).toBe("welcome");
    expect(nextOnboardingStep("intro", "continue", true)).toBe("run");
    expect(nextOnboardingStep("welcome", "local", false)).toBe("run");
  });

  it("lets an in-flight authentication check finish before timing out", () => {
    expect(authenticationTimeoutAction(false, true)).toBe("defer");
    expect(authenticationTimeoutAction(false, false)).toBe("fail");
    expect(authenticationTimeoutAction(true, true)).toBe("ignore");
  });

  it("opens the device connection flow and asks it to return to the app", () => {
    expect(
      onboardingConnectUrl(
        "https://ollama.com/connect?name=MacBook&key=public-key&launch=true",
        "signin",
      ),
    ).toBe(
      "https://ollama.com/connect?name=MacBook&key=public-key&launch=true",
    );
    expect(
      onboardingConnectUrl(
        "https://ollama.com/connect?name=MacBook&key=public-key",
        "signup",
      ),
    ).toBe(
      "https://ollama.com/connect?name=MacBook&key=public-key&launch=true&signup=true",
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
        onRetryCompletion={vi.fn()}
        onUseLocal={vi.fn()}
        showRun
      />,
    );

    expect(html).toContain("Run Ollama");
    expect(html).not.toContain("Welcome to Ollama");
    expect(html).not.toContain("Sign up");
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

    expect(html).toContain("Create an account");
    expect(html).toContain(
      "Create your account for access to faster, larger open models.",
    );
    expect(html).toContain("Your data is never logged or trained on.");
    expect(html).toContain("Sign up");
    expect(html).toContain("No thanks, I&#x27;ll use Ollama locally");
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

    expect(html).toContain("Create an account");
    expect(html).toContain(
      "Create your account for access to faster, larger open models.",
    );
    expect(html).toContain("Your data is never logged or trained on.");
    expect(html).not.toContain(">Sign in<");
  });

  it("shows only the local command on the final page", () => {
    const html = renderToStaticMarkup(
      <RunOllamaScreen completionError={null} onRetryCompletion={vi.fn()} />,
    );

    expect(html).toContain("Run Ollama");
    expect(html).toContain(FIRST_MODEL_COMMAND);
    expect(html).not.toContain("Finish");
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
    const onRetryCompletion = vi.fn();
    const html = renderToStaticMarkup(
      <RunOllamaScreen
        completionError="Unable to save setup. Please try again."
        onRetryCompletion={onRetryCompletion}
      />,
    );

    expect(html).toContain("Unable to save setup. Please try again.");
    expect(html).toContain('role="alert"');
    expect(html).toContain("Try again");
  });
});
