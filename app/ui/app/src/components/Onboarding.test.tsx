import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import {
  FIRST_MODEL_COMMAND,
  ConnectAppsScreen,
  IntroScreen,
  default as Onboarding,
  RunOllamaScreen,
  terminalRowsForWindowHeight,
  WelcomeScreen,
} from "./Onboarding";
import {
  CLAUDE_INSTALL_TIMEOUT_MS,
  isClaudeConnectionComplete,
  scheduleClaudeInstallTimeout,
} from "@/lib/claudeDesktop";
import { isWindowsPlatform } from "@/lib/platform";
import {
  authenticationTimeoutAction,
  nextOnboardingStep,
  onboardingConnectUrl,
} from "@/lib/onboarding";
import type { IntegrationStatuses } from "@/api";

describe("Onboarding", () => {
  it("explains what Ollama is before asking the user to choose a path", () => {
    const html = renderToStaticMarkup(<IntroScreen onContinue={vi.fn()} />);

    expect(html).toContain("Welcome to Ollama!");
    expect(html.indexOf('alt="Ollama waving"')).toBeLessThan(
      html.indexOf("Welcome to Ollama!"),
    );
    expect(html).toContain(
      "Run open models with your coding agents so you can spend less while keeping your data private.",
    );
    expect(html.indexOf("Connect your apps")).toBeLessThan(
      html.indexOf("Easily switch models"),
    );
    expect(html.indexOf("Easily switch models")).toBeLessThan(
      html.indexOf("Your data stays yours"),
    );
    expect(html).toContain("Power your existing coding apps with open models");
    expect(html).toContain("Swap between frontier models in one click.");
    expect(html).toContain("Your prompt data is never logged or trained on.");
    expect(html).toContain("Continue");
    expect(html).not.toContain("Skip");
  });

  it("shows more terminal integrations as the window gets taller", () => {
    expect(terminalRowsForWindowHeight(400)).toBe(1);
    expect(terminalRowsForWindowHeight(660)).toBe(4);
    expect(terminalRowsForWindowHeight(960)).toBe(8);
  });

  it("renders the apps screen without browser platform globals", () => {
    vi.stubGlobal("navigator", undefined);
    try {
      expect(() =>
        renderToStaticMarkup(<ConnectAppsScreen initialIntegrations={[]} />),
      ).not.toThrow();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("hides the Claude application on Windows", () => {
    vi.stubGlobal("window", {
      OLLAMA_PLATFORM: "windows",
      innerHeight: 660,
    });
    vi.stubGlobal("navigator", { platform: "MacIntel" });
    try {
      expect(isWindowsPlatform()).toBe(true);
      const html = renderToStaticMarkup(
        <ConnectAppsScreen
          initialIntegrations={[
            {
              id: "claude-desktop",
              name: "Claude",
              description: "Use Ollama models in Claude Desktop",
              installed: true,
            },
            {
              id: "claude",
              name: "Claude Code",
              description: "Anthropic's coding tool with subagents",
              command: "ollama launch claude",
            },
          ]}
        />,
      );

      expect(html).not.toContain('id="applications-heading"');
      expect(html).not.toContain("Use Ollama models in Claude Desktop");
      expect(html).toContain('id="terminal-heading"');
      expect(html).toContain("ollama launch claude");
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("shows the account choice only to signed-out users", () => {
    expect(nextOnboardingStep("intro", "continue", false)).toBe("welcome");
    expect(nextOnboardingStep("intro", "continue", true)).toBe("apps");
    expect(nextOnboardingStep("welcome", "authenticated", true)).toBe("apps");
    expect(nextOnboardingStep("apps", "continue", true)).toBe("apps");
    expect(nextOnboardingStep("welcome", "local", false)).toBe("run");
  });

  it("lets an in-flight authentication check finish before timing out", () => {
    expect(authenticationTimeoutAction(false, true)).toBe("defer");
    expect(authenticationTimeoutAction(false, false)).toBe("fail");
    expect(authenticationTimeoutAction(true, true)).toBe("ignore");
  });

  it("finishes Claude connection states from the native status hook", () => {
    const status = {
      installed: true,
      configured: true,
      connected: true,
      running: false,
      startFailed: false,
    };

    expect(isClaudeConnectionComplete(true, status)).toBe(true);
    expect(
      isClaudeConnectionComplete(true, { ...status, connected: false }),
    ).toBe(false);
    expect(
      isClaudeConnectionComplete(true, { ...status, startFailed: true }),
    ).toBe(false);
    expect(
      isClaudeConnectionComplete(false, {
        ...status,
        configured: false,
        connected: false,
      }),
    ).toBe(true);
  });

  it("bounds the Claude installer wait", () => {
    vi.useFakeTimers();
    vi.stubGlobal("window", { setTimeout: globalThis.setTimeout });
    const onTimeout = vi.fn();

    try {
      scheduleClaudeInstallTimeout(onTimeout);
      vi.advanceTimersByTime(CLAUDE_INSTALL_TIMEOUT_MS - 1);
      expect(onTimeout).not.toHaveBeenCalled();
      vi.advanceTimersByTime(1);
      expect(onTimeout).toHaveBeenCalledOnce();
    } finally {
      vi.useRealTimers();
      vi.unstubAllGlobals();
    }
  });

  it("opens the device connection flow without relaunching the app", () => {
    expect(
      onboardingConnectUrl(
        "https://ollama.com/connect?name=MacBook&key=public-key&launch=true",
        "signin",
      ),
    ).toBe("https://ollama.com/connect?name=MacBook&key=public-key");
    expect(
      onboardingConnectUrl(
        "https://ollama.com/connect?name=MacBook&key=public-key",
        "signup",
      ),
    ).toBe(
      "https://ollama.com/connect?name=MacBook&key=public-key&signup=true",
    );
  });

  it("preserves the intro for a device that is already connected", () => {
    const html = renderToStaticMarkup(
      <Onboarding
        isAuthenticated
        isSigningIn={false}
        signInError={null}
        completionError={null}
        onOpenApps={vi.fn().mockResolvedValue(true)}
        onSignIn={vi.fn()}
        onSignUp={vi.fn()}
        onRetryCompletion={vi.fn()}
        onUseLocal={vi.fn()}
      />,
    );

    expect(html).toContain("Welcome to Ollama");
    expect(html).not.toContain("Run Ollama");
    expect(html).not.toContain("Sign up");
  });

  it("groups disconnected Claude with applications and terminal separately", () => {
    const integrations: IntegrationStatuses = [
      {
        id: "claude-desktop",
        name: "Claude",
        description: "Use Ollama models in Claude Desktop",
        installed: true,
      },
      {
        id: "claude",
        name: "Claude Code",
        description: "Anthropic's coding tool with subagents",
        command: "ollama launch claude",
      },
      {
        id: "codex",
        name: "Codex",
        description: "OpenAI's open-source coding agent",
        command: "ollama launch codex",
      },
      {
        id: "openclaw",
        name: "OpenClaw",
        description: "Personal AI with 100+ skills",
        command: "ollama launch openclaw",
      },
      {
        id: "opencode",
        name: "OpenCode",
        description: "Anomaly's open-source coding agent",
        command: "ollama launch opencode",
      },
      {
        id: "droid",
        name: "Droid",
        description: "AI software engineering agent",
        command: "ollama launch droid",
      },
      {
        id: "terminal",
        name: "Terminal",
        description: "Run local models from your terminal",
        command: "ollama",
      },
    ];
    const html = renderToStaticMarkup(
      <ConnectAppsScreen initialIntegrations={integrations} />,
    );

    expect(html).toContain('id="applications-heading"');
    expect(html).toContain('id="terminal-heading"');
    expect(html.indexOf("Application")).toBeLessThan(
      html.indexOf("Use Ollama models in Claude Desktop"),
    );
    expect(html).toContain('aria-label="Connect Claude"');
    expect(html).toContain('role="switch"');
    expect(html).toContain('aria-checked="false"');
    expect(html).not.toContain("Inactive");
    expect(html).not.toContain("Active");
    expect(html).toContain('aria-label="Copy OpenCode command"');
    expect(html).toContain('aria-label="Copy Terminal command"');
    expect(html).not.toContain("ChatGPT");
    expect(html).toContain('aria-label="Show more apps"');
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain("grid-rows-[0fr]");
    expect(html).toContain("/launch-icons/claude.svg");
    expect(html).toContain("/launch-icons/claude-code.svg");
    expect(html).toContain("Run local models from your terminal");
  });

  it("keeps connected Claude in Application without an idle status", () => {
    const html = renderToStaticMarkup(
      <ConnectAppsScreen
        initialClaudeStatus={{
          installed: true,
          configured: true,
          connected: true,
          running: false,
          startFailed: false,
        }}
        initialIntegrations={[
          {
            id: "claude-desktop",
            name: "Claude",
            description: "Use Ollama models in Claude Desktop",
            installed: true,
          },
          {
            id: "codex",
            name: "Codex",
            description: "OpenAI's open-source coding agent",
            command: "ollama launch codex",
          },
        ]}
      />,
    );

    expect(html).toContain('id="applications-heading"');
    expect(html).not.toContain('id="claude-apps-heading"');
    expect(html).not.toContain("Ready to launch");
    expect(html).not.toContain("Active");
    expect(html).not.toContain("Inactive");
    expect(html).toContain('aria-checked="true"');
    expect(html).toContain('aria-label="Disconnect Claude"');
  });

  it("keeps failed configured Claude disconnectable", () => {
    const html = renderToStaticMarkup(
      <ConnectAppsScreen
        initialClaudeStatus={{
          installed: true,
          configured: true,
          connected: false,
          running: false,
          startFailed: true,
        }}
        initialIntegrations={[
          {
            id: "claude-desktop",
            name: "Claude",
            description: "Use Ollama models in Claude Desktop",
            installed: true,
          },
        ]}
      />,
    );

    expect(html).toContain('aria-checked="true"');
    expect(html).toContain('aria-label="Disconnect Claude"');
    expect(html).toContain("Ollama couldn’t start the Claude connection.");
  });

  it("keeps Claude available without a separate not-installed group", () => {
    const html = renderToStaticMarkup(
      <ConnectAppsScreen
        initialIntegrations={[
          {
            id: "claude-desktop",
            name: "Claude",
            description: "Use Ollama models in Claude Desktop",
            installed: false,
          },
        ]}
      />,
    );

    expect(html).toContain("Use Ollama models in Claude Desktop");
    expect(html).toContain('aria-label="Connect Claude"');
    expect(html).toContain("Download &amp; connect");
    expect(html).not.toContain("Inactive");
    expect(html).not.toContain("Not installed");
    expect(html).not.toContain('disabled=""');
  });

  it("uses branded icons for the remaining launcher integrations", () => {
    const html = renderToStaticMarkup(
      <ConnectAppsScreen
        initialIntegrations={[
          {
            id: "cline",
            name: "Cline",
            description: "Autonomous coding agent",
            command: "ollama launch cline",
          },
          {
            id: "omp",
            name: "Oh My Pi",
            description: "AI coding agent",
            command: "ollama launch omp",
          },
          {
            id: "pool",
            name: "Poolside",
            description: "Poolside's coding agent",
            command: "ollama launch pool",
          },
          {
            id: "qwen",
            name: "Qwen Code",
            description: "Qwen's coding agent",
            command: "ollama launch qwen",
          },
        ]}
      />,
    );

    expect(html).toContain("/launch-icons/cline.svg");
    expect(html).toContain("/launch-icons/oh-my-pi.svg");
    expect(html).toContain("/launch-icons/poolside.svg");
    expect(html).toContain("/launch-icons/qwen-code.svg");
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
