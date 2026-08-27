import { renderToStaticMarkup } from "react-dom/server";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { describe, expect, it, vi } from "vitest";
import {
  ClaudeConnectedIntro,
  FIRST_MODEL_COMMAND,
  ConnectAppsScreen,
  IntroScreen,
  default as Onboarding,
  RunOllamaScreen,
  shouldShowClaudeConnectedIntro,
  WelcomeScreen,
} from "./Onboarding";
import {
  CLAUDE_CONNECTION_TIMEOUT_MS,
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
    expect(html).toMatch(/<main class="light-only [^"]*bg-white/);
    expect(html).not.toMatch(/alt="Ollama waving" class="[^"]*dark:/);
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

      expect(html).not.toContain('id="desktop-heading"');
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

  it("detects when the menu bar already reached the requested Claude state", () => {
    const status = {
      supported: true,
      installed: true,
      configured: true,
      connected: true,
      running: false,
      startFailed: false,
      portConflict: false,
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
    expect(
      isClaudeConnectionComplete(false, { ...status, connected: false }),
    ).toBe(false);
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

  it("keeps the Claude switch on and busy through installer detection", async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

    const disconnectedStatus = {
      supported: true,
      used: false,
      installed: false,
      configured: false,
      connected: false,
      running: false,
      startFailed: false,
      portConflict: false,
    };
    let finishInstall!: (result: "opened") => void;
    const install = new Promise<"opened">((resolve) => {
      finishInstall = resolve;
    });

    vi.stubGlobal("navigator", { platform: "MacIntel" });
    vi.stubGlobal("window", {
      OLLAMA_PLATFORM: "darwin",
      innerHeight: 660,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout,
      setInterval: globalThis.setInterval,
      clearInterval: globalThis.clearInterval,
      getClaudeDesktopConnectionSummary: vi
        .fn()
        .mockResolvedValue(disconnectedStatus),
      setClaudeDesktopConnected: vi.fn(),
      installClaudeDesktop: vi.fn().mockReturnValue(install),
    });

    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(
          <ConnectAppsScreen
            initialClaudeStatus={disconnectedStatus}
            initialIntegrations={[
              {
                id: "claude-desktop",
                name: "Claude",
                description: "Use Ollama models in Claude Desktop",
                installed: false,
                action: "connect",
              },
            ]}
          />,
        );
        await Promise.resolve();
      });

      const claudeSwitch = () => renderer!.root.findByProps({ role: "switch" });
      let clickResult!: Promise<void>;
      await act(async () => {
        clickResult = claudeSwitch().props.onClick();
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(claudeSwitch().props["aria-checked"]).toBe(true);
      expect(claudeSwitch().props["aria-busy"]).toBe(true);
      expect(claudeSwitch().props.disabled).toBe(true);
      expect(claudeSwitch().props.className).toContain("disabled:opacity-50");
      expect(renderer.root.findByProps({ role: "status" }).children).toContain(
        "Downloading…",
      );
      expect(
        renderer.root.findAll(
          (node) =>
            typeof node.props.className === "string" &&
            node.props.className.includes("animate-spin"),
        ),
      ).not.toHaveLength(0);

      await act(async () => {
        finishInstall("opened");
        await clickResult;
        await Promise.resolve();
      });

      expect(claudeSwitch().props["aria-checked"]).toBe(true);
      expect(claudeSwitch().props["aria-busy"]).toBe(true);
      expect(claudeSwitch().props.disabled).toBe(true);
      expect(claudeSwitch().props.className).toContain("disabled:opacity-50");
      expect(renderer.root.findByProps({ role: "status" }).children).toContain(
        "Finish installing…",
      );
      expect(
        renderer.root.findAll(
          (node) =>
            typeof node.props.className === "string" &&
            node.props.className.includes("animate-spin"),
        ),
      ).not.toHaveLength(0);
    } finally {
      if (renderer) {
        act(() => renderer?.unmount());
      }
      vi.unstubAllGlobals();
    }
  });

  it("preserves a late native error after the Connect Apps action times out", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

    const disconnectedStatus = {
      supported: true,
      used: false,
      installed: true,
      configured: false,
      connected: false,
      running: false,
      startFailed: false,
      portConflict: false,
    };
    const connectedStatus = {
      ...disconnectedStatus,
      configured: true,
      connected: true,
    };
    let finishNativeAction!: (result: {
      status: typeof connectedStatus;
      error?: string;
    }) => void;
    const nativeAction = new Promise<{
      status: typeof connectedStatus;
      error?: string;
    }>((resolve) => {
      finishNativeAction = resolve;
    });
    const getClaudeStatus = vi
      .fn()
      .mockResolvedValueOnce(disconnectedStatus)
      .mockResolvedValue(connectedStatus);
    const setClaudeConnected = vi.fn().mockReturnValue(nativeAction);
    vi.stubGlobal("navigator", { platform: "MacIntel" });
    vi.stubGlobal("window", {
      OLLAMA_PLATFORM: "darwin",
      innerHeight: 660,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout,
      setInterval: globalThis.setInterval,
      clearInterval: globalThis.clearInterval,
      getClaudeDesktopConnectionSummary: getClaudeStatus,
      setClaudeDesktopConnected: setClaudeConnected,
    });

    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(
          <ConnectAppsScreen
            initialClaudeStatus={disconnectedStatus}
            initialIntegrations={[
              {
                id: "claude-desktop",
                name: "Claude",
                description: "Use Ollama models in Claude Desktop",
                installed: true,
                action: "connect",
              },
            ]}
          />,
        );
        await Promise.resolve();
      });

      const claudeSwitch = () => renderer!.root.findByProps({ role: "switch" });
      expect(claudeSwitch().props["aria-checked"]).toBe(false);
      expect(claudeSwitch().props["aria-busy"]).toBeUndefined();
      expect(claudeSwitch().props.disabled).toBe(false);

      let clickResult!: Promise<void>;
      await act(async () => {
        clickResult = claudeSwitch().props.onClick();
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(setClaudeConnected).toHaveBeenCalledWith(true, false);
      expect(claudeSwitch().props["aria-checked"]).toBe(true);
      expect(claudeSwitch().props["aria-busy"]).toBe(true);
      expect(claudeSwitch().props.disabled).toBe(true);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(CLAUDE_CONNECTION_TIMEOUT_MS);
        await clickResult;
      });

      expect(claudeSwitch().props["aria-checked"]).toBe(false);
      expect(claudeSwitch().props["aria-busy"]).toBeUndefined();
      expect(claudeSwitch().props.disabled).toBe(false);
      expect(
        renderer.root.findByProps({ role: "alert" }).children.join(""),
      ).toContain("Claude is taking too long to connect");

      await act(async () => {
        finishNativeAction({
          status: connectedStatus,
          error: "Claude failed to restart.",
        });
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(getClaudeStatus).toHaveBeenCalledOnce();
      expect(claudeSwitch().props["aria-checked"]).toBe(true);
      expect(claudeSwitch().props["aria-busy"]).toBeUndefined();
      expect(claudeSwitch().props.disabled).toBe(false);
      expect(
        renderer.root.findByProps({ role: "alert" }).children.join(""),
      ).toContain("Claude failed to restart.");
      expect(
        renderer.root.findAllByProps({ id: "claude-connected-title" }),
      ).toHaveLength(0);
    } finally {
      if (renderer) {
        act(() => renderer?.unmount());
      }
      vi.useRealTimers();
      vi.unstubAllGlobals();
    }
  });

  it("shows the Claude intro only before the integration has been used", () => {
    const firstConnection = {
      supported: true,
      used: false,
      installed: true,
      configured: true,
      connected: true,
      running: false,
      startFailed: false,
      portConflict: false,
    };

    expect(shouldShowClaudeConnectedIntro(firstConnection)).toBe(true);
    expect(
      shouldShowClaudeConnectedIntro({ ...firstConnection, used: true }),
    ).toBe(false);
    expect(
      shouldShowClaudeConnectedIntro({
        ...firstConnection,
        connected: false,
      }),
    ).toBe(false);
    expect(
      shouldShowClaudeConnectedIntro({
        ...firstConnection,
        startFailed: true,
      }),
    ).toBe(false);
  });

  it("uses Continue as the only Claude intro action", () => {
    const html = renderToStaticMarkup(
      <ClaudeConnectedIntro onDone={vi.fn()} />,
    );

    expect(html).toContain(">Continue</button>");
    expect(html).not.toContain('aria-label="Close"');
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

  it("groups disconnected Claude with a scrollable terminal list", () => {
    const integrations: IntegrationStatuses = [
      {
        id: "claude-desktop",
        name: "Claude",
        description: "Use Ollama models in Claude Desktop",
        installed: true,
        action: "connect",
      },
      {
        id: "claude",
        name: "Claude Code",
        description: "Anthropic's coding tool with subagents",
        installed: true,
        action: "copy",
        command: "ollama launch claude",
      },
      {
        id: "codex",
        name: "Codex",
        description: "OpenAI's open-source coding agent",
        installed: true,
        action: "copy",
        command: "ollama launch codex",
      },
      {
        id: "openclaw",
        name: "OpenClaw",
        description: "Personal AI with 100+ skills",
        installed: true,
        action: "copy",
        command: "ollama launch openclaw",
      },
      {
        id: "opencode",
        name: "OpenCode",
        description: "Anomaly's open-source coding agent",
        installed: false,
        action: "copy",
        command: "ollama launch opencode",
      },
      {
        id: "droid",
        name: "Droid",
        description: "AI software engineering agent",
        installed: false,
        action: "copy",
        command: "ollama launch droid",
      },
      {
        id: "dsh",
        name: "DeepSeek Harness",
        description: "DeepSeek's open-source agent harness",
        installed: false,
        action: "copy",
        command: "ollama launch dsh",
      },
      {
        id: "cline",
        name: "Cline",
        description: "Autonomous coding agent",
        installed: false,
        action: "copy",
        command: "ollama launch cline",
      },
      {
        id: "terminal",
        name: "Terminal",
        description: "Run local models from your terminal",
        action: "copy",
        command: "ollama",
      },
    ];
    const html = renderToStaticMarkup(
      <ConnectAppsScreen
        completionError={null}
        onRetryCompletion={vi.fn()}
        initialIntegrations={integrations}
      />,
    );

    expect(html).not.toContain(
      "Connect Claude, or copy a command to run in your terminal.",
    );
    expect(html).toContain("Claude");
    expect(html).toContain("Use Ollama models in Claude Desktop");
    expect(html).toContain("Claude Code");
    expect(html).not.toContain("Search apps");
    expect(html).not.toContain('type="search"');
    expect(html).toContain("Desktop");
    expect(html).toContain('id="desktop-heading"');
    expect(html).toContain('id="terminal-heading"');
    expect(html).not.toContain("Ready to launch");
    expect(html).not.toContain('id="claude-apps-heading"');
    expect(html.indexOf("Desktop")).toBeLessThan(
      html.indexOf("Use Ollama models in Claude Desktop"),
    );
    expect(html).not.toContain(">Command</th>");
    expect(html).toContain("ollama launch claude");
    expect(html).not.toContain("Installed");
    expect(html).not.toContain("Not installed");
    expect(html).toContain('aria-label="Connect Claude"');
    expect(html).toContain('role="switch"');
    expect(html).toContain('aria-checked="false"');
    expect(html).not.toContain("Inactive");
    expect(html).not.toContain("Download &amp; connect");
    expect(html).not.toContain("Active");
    expect(html).toContain("bg-transparent");
    expect(html).toContain('aria-label="Copy OpenCode command"');
    expect(html).toContain('aria-label="Copy Terminal command"');
    expect(html).not.toContain(">Copy command</button>");
    expect(html).not.toContain("ChatGPT");
    expect(html).toContain("OpenCode");
    expect(html).toContain("Terminal");
    expect(html).toContain("overflow-y-auto");
    expect(html).not.toContain('aria-label="Show more apps"');
    expect(html).not.toContain("aria-expanded");
    expect(html).not.toContain("grid-rows-[0fr]");
    expect(html).not.toContain("inert");
    expect(html).toContain("/launch-icons/claude.svg");
    expect(html).toContain("/launch-icons/claude-code.svg");
    expect(html).toMatch(
      /src="\/launch-icons\/cline\.svg"[^>]*class="[^"]*dark:invert/,
    );
    expect(html).toContain("/launch-icons/deepseek-harness.svg");
    expect(html).not.toMatch(
      /src="\/launch-icons\/deepseek-harness\.svg"[^>]*class="[^"]*dark:invert/,
    );
    expect(html).not.toContain("<table");
    expect(html).not.toContain("<footer");
    expect(html).not.toContain("Command copied. Run it in your terminal.");
    expect(html).toContain("Run local models from your terminal");
    expect(html).not.toContain("Launch command");
    expect(html).not.toContain('aria-pressed="true"');
    expect(html).not.toContain("Continue");
    expect(html).not.toContain("Run Ollama");
    expect(html).not.toContain('viewBox="0 0 3400 3400"');
  });

  it("keeps connected Claude in Desktop without an idle status", () => {
    const html = renderToStaticMarkup(
      <ConnectAppsScreen
        completionError={null}
        onRetryCompletion={vi.fn()}
        initialClaudeStatus={{
          supported: true,
          used: true,
          installed: true,
          connected: true,
          running: false,
          startFailed: false,
          portConflict: false,
          routedRequests: 12,
        }}
        initialIntegrations={[
          {
            id: "claude-desktop",
            name: "Claude",
            description: "Use Ollama models in Claude Desktop",
            installed: true,
            action: "connect",
          },
          {
            id: "codex",
            name: "Codex",
            description: "OpenAI's open-source coding agent",
            installed: true,
            action: "copy",
            command: "ollama launch codex",
          },
        ]}
      />,
    );

    expect(html).toContain('id="desktop-heading"');
    expect(html).not.toContain('id="claude-apps-heading"');
    expect(html).not.toContain("Ready to launch");
    expect(html).not.toContain("Active");
    expect(html).not.toContain("Inactive");
    expect(html).toContain('aria-checked="true"');
    expect(html).toContain('aria-label="Disconnect Claude"');
    expect(html).toContain("Connected to Ollama · 12 requests this session");
  });

  it("shows initial Claude recovery guidance without error styling", () => {
    const html = renderToStaticMarkup(
      <ConnectAppsScreen
        completionError={null}
        onRetryCompletion={vi.fn()}
        initialClaudeStatus={{
          supported: true,
          used: true,
          installed: true,
          configured: true,
          connected: false,
          running: false,
          startFailed: true,
          portConflict: false,
          error: "Cloud models are off. Select an installed model in Settings.",
        }}
        initialIntegrations={[
          {
            id: "claude-desktop",
            name: "Claude",
            description: "Use Ollama models in Claude Desktop",
            installed: true,
            action: "connect",
          },
        ]}
      />,
    );

    expect(html).toContain(
      "Cloud models are off. Select an installed model in Settings.",
    );
    expect(html).toContain('role="alert"');
    expect(html).not.toContain("text-red");
    expect(html).toContain('aria-checked="true"');
    expect(html).toContain('aria-label="Disconnect Claude"');
  });

  it("keeps Claude model management off the Connect Apps page", () => {
    const html = renderToStaticMarkup(
      <ConnectAppsScreen
        completionError={null}
        onRetryCompletion={vi.fn()}
        initialClaudeStatus={{
          supported: true,
          used: true,
          installed: true,
          connected: true,
          running: true,
          startFailed: false,
          portConflict: false,
          modelSource: "endpoint",
          models: [
            {
              name: "glm-5.2:cloud",
              displayName: "GLM 5.2",
              description: "Long-horizon coding",
              selected: true,
            },
            {
              name: "qwen3.8:27b",
              displayName: "Qwen 3.8 27B",
              description: "Local coding",
              selected: false,
            },
          ],
        }}
        initialIntegrations={[
          {
            id: "claude-desktop",
            name: "Claude",
            description: "Use Ollama models in Claude Desktop",
            installed: true,
            action: "connect",
          },
        ]}
      />,
    );

    expect(html).not.toContain("Models in Claude");
    expect(html).not.toContain("GLM 5.2");
    expect(html).not.toContain("Qwen 3.8 27B");
    expect(html).not.toContain('type="checkbox"');
    expect(html).not.toContain("Restart Claude");
    expect(html).not.toContain("Built-in defaults");
  });

  it("keeps Claude available without a separate not-installed group", () => {
    const html = renderToStaticMarkup(
      <ConnectAppsScreen
        completionError={null}
        onRetryCompletion={vi.fn()}
        initialIntegrations={[
          {
            id: "claude-desktop",
            name: "Claude",
            description: "Use Ollama models in Claude Desktop",
            installed: false,
            action: "connect",
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
        completionError={null}
        onRetryCompletion={vi.fn()}
        initialIntegrations={[
          {
            id: "cline",
            name: "Cline",
            description: "Autonomous coding agent",
            action: "copy",
            command: "ollama launch cline",
          },
          {
            id: "omp",
            name: "Oh My Pi",
            description: "AI coding agent",
            action: "copy",
            command: "ollama launch omp",
          },
          {
            id: "pool",
            name: "Poolside",
            description: "Poolside's coding agent",
            action: "copy",
            command: "ollama launch pool",
          },
          {
            id: "qwen",
            name: "Qwen Code",
            description: "Qwen's coding agent",
            action: "copy",
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
    expect(html).toMatch(/<main class="light-only [^"]*bg-white/);
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
    expect(html).toMatch(/<main class="light-only [^"]*bg-white/);
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
