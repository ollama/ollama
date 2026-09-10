import { StrictMode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { QueryClient } from "@tanstack/react-query";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ClaudeConnectedIntro,
  FIRST_MODEL_COMMAND,
  ConnectAppsScreen,
  INTEGRATION_SCROLL_MS,
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
  onboardingConnectUrl,
} from "@/lib/onboarding";
import type { IntegrationStatuses } from "@/api";
import * as clipboard from "@/utils/clipboard";

// Transition timing is checked in the browser; these tests cover copy-notice
// lifetimes with the real component logic.
vi.mock("@headlessui/react", async (importOriginal) => {
  const original = await importOriginal<typeof import("@headlessui/react")>();
  return Object.assign({}, original, {
    Transition: ({
      show,
      children,
    }: {
      show: boolean;
      children: React.ReactNode;
    }) => (show ? <div>{children}</div> : null),
  });
});

let queryClient: QueryClient;
beforeEach(() => {
  queryClient = new QueryClient();
});
afterEach(() => queryClient.clear());

vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>();
  return Object.assign({}, actual, {
    useQueryClient: () => queryClient,
    useMutation: (options: Parameters<typeof actual.useMutation>[0]) =>
      actual.useMutation(options, queryClient),
    useMutationState: (
      options: Parameters<typeof actual.useMutationState>[0],
    ) => actual.useMutationState(options, queryClient),
  });
});

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

  it("hides the Claude and ChatGPT desktop integrations on Windows", () => {
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
            {
              id: "chatgpt",
              name: "ChatGPT",
              description: "Use Ollama models in ChatGPT",
              installed: true,
              command: "ollama launch chatgpt",
            },
          ]}
        />,
      );

      expect(html).not.toContain('id="recommended-heading"');
      expect(html).not.toContain("Use Ollama models in Claude Desktop");
      expect(html).not.toContain("Use Ollama models in ChatGPT");
      expect(html).not.toContain("ollama launch chatgpt");
      expect(html).toContain('id="terminal-heading"');
      expect(html).toContain('aria-label="Copy Claude Code command"');
    } finally {
      vi.unstubAllGlobals();
    }
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

      const claudeSwitch = () =>
        renderer!.root
          .findAll(
            (node) =>
              node.type === "button" &&
              typeof node.props["aria-pressed"] === "boolean",
          )
          .find((node) => String(node.props["aria-label"]).endsWith("Claude"))!;
      let clickResult!: Promise<void>;
      await act(async () => {
        clickResult = claudeSwitch().props.onClick();
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(claudeSwitch().props["aria-pressed"]).toBe(true);
      expect(claudeSwitch().props["aria-busy"]).toBe(true);
      expect(claudeSwitch().props.disabled).toBe(true);
      expect(claudeSwitch().props.className).toContain("disabled:opacity-60");
      expect(
        renderer.root
          .findAllByProps({ role: "status" })
          .some((node) => node.children.includes("Downloading…")),
      ).toBe(true);
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

      expect(claudeSwitch().props["aria-pressed"]).toBe(true);
      expect(claudeSwitch().props["aria-busy"]).toBe(true);
      expect(claudeSwitch().props.disabled).toBe(true);
      expect(claudeSwitch().props.className).toContain("disabled:opacity-60");
      expect(
        renderer.root
          .findAllByProps({ role: "status" })
          .some((node) => node.children.includes("Finish installing…")),
      ).toBe(true);
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

      const claudeSwitch = () =>
        renderer!.root
          .findAll(
            (node) =>
              node.type === "button" &&
              typeof node.props["aria-pressed"] === "boolean",
          )
          .find((node) => String(node.props["aria-label"]).endsWith("Claude"))!;
      expect(claudeSwitch().props["aria-pressed"]).toBe(false);
      expect(claudeSwitch().props["aria-busy"]).toBeUndefined();
      expect(claudeSwitch().props.disabled).toBe(false);

      let clickResult!: Promise<void>;
      await act(async () => {
        clickResult = claudeSwitch().props.onClick();
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(setClaudeConnected).toHaveBeenCalledWith(true, false);
      expect(claudeSwitch().props["aria-pressed"]).toBe(true);
      expect(claudeSwitch().props["aria-busy"]).toBe(true);
      expect(claudeSwitch().props.disabled).toBe(true);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(CLAUDE_CONNECTION_TIMEOUT_MS);
        await clickResult;
      });

      expect(claudeSwitch().props["aria-pressed"]).toBe(false);
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
      expect(claudeSwitch().props["aria-pressed"]).toBe(true);
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

  it("shows recommended connections and every launcher in a scrollable grid", () => {
    const integrations: IntegrationStatuses = [
      {
        id: "claude-desktop",
        name: "Claude Code (Desktop)",
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
        name: "Codex CLI",
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

    expect(html).toContain('id="recommended-heading"');
    expect(html).toContain("Recommended");
    expect(html).toContain("Other apps");
    expect(html).toContain("Use Ollama models in your Claude Code.");
    expect(html).toContain('aria-label="Connect Claude"');
    expect(html).toContain('aria-pressed="false"');
    expect(html).not.toContain('role="switch"');
    for (const integration of integrations.filter((item) => item.command)) {
      expect(html).toContain(`id="integration-${integration.id}"`);
      expect(html).toContain(`aria-label="Copy ${integration.name} command"`);
    }
    expect(html.match(/aria-label="Copy [^"]+ command"/g)).toHaveLength(8);
    expect(html).toContain("sm:grid-cols-2");
    expect(html).toContain("overflow-y-auto");
    expect(html).not.toContain("<h1");
    expect(html).not.toContain("Connect a coding agent");
    expect(html).not.toContain("Skip for now");
    expect(html).not.toContain("Search apps");
    expect(html).not.toContain("<table");
    expect(html).toContain("/launch-icons/claude.svg");
    expect(html).toContain("/launch-icons/claude-code.svg");
    expect(html).toContain("/launch-icons/codex-color.svg");
  });

  it("places ChatGPT directly below Claude instead of in Terminal", () => {
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
            id: "codex",
            name: "Codex CLI",
            description: "OpenAI's coding agent",
            command: "ollama launch codex",
          },
        ]}
        initialCodexStatus={{
          supported: true,
          installed: true,
          connected: false,
          running: false,
        }}
      />,
    );

    expect(html.indexOf('id="integration-claude-desktop"')).toBeLessThan(
      html.indexOf('id="integration-chatgpt"'),
    );
    expect(html).toContain('aria-label="Add Ollama models to ChatGPT"');
    expect(html).not.toContain('aria-label="Copy ChatGPT command"');
    expect(html).toContain('aria-label="Copy Codex CLI command"');
  });

  it("keeps connected Claude in Recommended without an idle status", () => {
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
            name: "Codex CLI",
            description: "OpenAI's open-source coding agent",
            installed: true,
            action: "copy",
            command: "ollama launch codex",
          },
        ]}
      />,
    );

    expect(html).toContain('id="recommended-heading"');
    expect(html).not.toContain('id="claude-apps-heading"');
    expect(html).not.toContain("Ready to launch");
    expect(html).not.toContain("Active");
    expect(html).not.toContain("Inactive");
    expect(html).toContain('aria-pressed="true"');
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
    expect(html).toContain('aria-pressed="true"');
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
        initialClaudeStatus={{ ...DISCONNECTED_CLAUDE, installed: false }}
        initialIntegrations={[
          {
            id: "claude-desktop",
            name: "Claude",
            description: "We’ll download Claude and connect it to Ollama.",
            installed: false,
            action: "connect",
          },
        ]}
      />,
    );

    expect(html).toContain("We’ll download Claude and connect it to Ollama.");
    expect(html).toContain('aria-label="Connect Claude"');
    expect(html).toContain(">Connect</button>");
    expect(html).not.toContain("Inactive");
    const claudeButton = html.match(
      /<button[^>]*aria-label="Connect Claude"[^>]*>/,
    )?.[0];
    expect(claudeButton).toBeDefined();
    expect(claudeButton).not.toContain('disabled=""');
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

function appsIntegrations(claudeInstalled: boolean): IntegrationStatuses {
  const launcher = (id: string, name: string) => ({
    id,
    name,
    description: `${name} description`,
    installed: false,
    command: `ollama launch ${id}`,
  });
  return [
    {
      id: "claude-desktop",
      name: "Claude",
      description: "Use Ollama models in Claude Desktop",
      installed: claudeInstalled,
    },
    launcher("claude", "Claude Code"),
    launcher("codex", "Codex CLI"),
    launcher("opencode", "OpenCode"),
    launcher("pi", "Pi"),
    launcher("hermes", "Hermes Agent"),
    {
      id: "terminal",
      name: "Terminal",
      description: "Run local models from your terminal",
      command: "ollama",
    },
  ];
}

function onboardingProps(onOpenApps: () => Promise<boolean>) {
  return {
    isAuthenticated: true,
    isSigningIn: false,
    signInError: null,
    completionError: null,
    onOpenApps,
    onSignIn: vi.fn(),
    onSignUp: vi.fn(),
    onRetryCompletion: vi.fn(),
    onUseLocal: vi.fn(),
  };
}

const DISCONNECTED_CLAUDE = {
  supported: true,
  used: false,
  installed: true,
  configured: false,
  connected: false,
  running: false,
  startFailed: false,
  portConflict: false,
};

async function settle(ticks = 8) {
  for (let i = 0; i < ticks; i++) {
    await Promise.resolve();
  }
}

function stubOnboardingWindow(platform = "darwin") {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("navigator", { platform: "MacIntel" });
  vi.stubGlobal("window", {
    OLLAMA_PLATFORM: platform,
    setOnboardingWindow: vi.fn(),
  });
}

describe("Onboarding handoff", () => {
  it("preserves local setup without opening Apps", async () => {
    stubOnboardingWindow();
    const props = { ...onboardingProps(vi.fn()), isAuthenticated: false };
    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(<Onboarding {...props} />);
      });
      await act(async () => {
        renderer!.root.findByType(IntroScreen).props.onContinue();
      });
      expect(props.onOpenApps).not.toHaveBeenCalled();
      await act(async () => {
        renderer!.root.findByType(WelcomeScreen).props.onLocal();
      });
      expect(renderer!.root.findByType(RunOllamaScreen)).toBeTruthy();
      expect(props.onUseLocal).toHaveBeenCalledOnce();
      expect(props.onOpenApps).not.toHaveBeenCalled();
    } finally {
      if (renderer) act(() => renderer?.unmount());
      vi.unstubAllGlobals();
    }
  });
});

describe("ConnectAppsScreen loading and deep links", () => {
  function stubAppsWindow(overrides: Record<string, unknown> = {}) {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
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
      ...overrides,
    });
  }

  it.each([false, true])(
    "honors the native Claude disconnect confirmation: %s",
    async (confirmed) => {
      const connected = {
        ...DISCONNECTED_CLAUDE,
        used: true,
        running: true,
        configured: true,
        connected: true,
      };
      const confirm = vi.fn(() => confirmed);
      const disconnect = vi.fn().mockResolvedValue({
        status: { ...connected, configured: false, connected: false },
      });
      stubAppsWindow({
        getClaudeDesktopConnectionSummary: vi.fn().mockResolvedValue(connected),
        setClaudeDesktopConnected: disconnect,
        confirm,
      });
      let renderer: ReactTestRenderer | undefined;
      try {
        await act(async () => {
          renderer = create(
            <ConnectAppsScreen
              initialIntegrations={appsIntegrations(true)}
              initialClaudeStatus={connected}
            />,
          );
          await settle();
        });
        const toggle = renderer!.root.findByProps({
          "aria-label": "Disconnect Claude",
        });
        await act(async () => {
          await toggle.props.onClick();
        });
        expect(confirm).toHaveBeenCalledExactlyOnceWith(
          "Restart Claude Desktop to remove Ollama? Any running task will stop.",
        );
        if (confirmed) {
          expect(disconnect).toHaveBeenCalledExactlyOnceWith(false, true);
        } else {
          expect(disconnect).not.toHaveBeenCalled();
          expect(toggle.props.disabled).toBe(false);
          expect(toggle.props["aria-pressed"]).toBe(true);
        }
      } finally {
        await act(async () => renderer?.unmount());
        vi.unstubAllGlobals();
      }
    },
  );

  // Rows sit 500px below the top of a 660px-tall scroll container, so
  // centering a 72px row lands the container at 206px.
  function scrollMocks() {
    const container = {
      scrollTop: 0,
      clientHeight: 660,
      scrollHeight: 2000,
      getBoundingClientRect: () => ({ top: 0 }),
    };
    const row = { getBoundingClientRect: () => ({ top: 500, height: 72 }) };
    const createNodeMock = (element: { props: { className?: string } }) =>
      element.props.className?.includes("overflow-y-auto") ? container : row;
    return { container, createNodeMock };
  }
  const CENTERED_SCROLL_TOP = 206;

  it.each(["darwin", "windows"])(
    "keeps the full catalog in the %s grid",
    (platform) => {
      stubAppsWindow({ OLLAMA_PLATFORM: platform });
      const integrations = [
        ...appsIntegrations(true),
        ...Array.from({ length: 20 }, (_, index) => ({
          id: `extra-${index}`,
          name: `Extra app ${index}`,
          description: "Another supported integration",
          installed: false,
          command: `ollama launch extra-${index}`,
        })),
      ];
      try {
        const html = renderToStaticMarkup(
          <ConnectAppsScreen initialIntegrations={integrations} />,
        );
        for (const integration of integrations.filter((item) => item.command)) {
          expect(html).toContain(
            `aria-label="Copy ${integration.name} command"`,
          );
        }
        expect(html.includes('id="recommended-heading"')).toBe(
          platform === "darwin",
        );
        expect(html.includes('id="integration-chatgpt"')).toBe(
          platform === "darwin",
        );
        expect(html).not.toContain("Connect a coding agent");
        expect(html).not.toContain("Skip for now");
      } finally {
        vi.unstubAllGlobals();
      }
    },
  );

  it.each(["darwin", "windows"])(
    "copies from a card and renews the temporary %s hint on every click",
    async (platform) => {
      vi.useFakeTimers();
      const copyCommand = vi
        .spyOn(clipboard, "copyTextToClipboard")
        .mockResolvedValue(true);
      stubAppsWindow({ OLLAMA_PLATFORM: platform });
      let renderer: ReactTestRenderer | undefined;
      try {
        await act(async () => {
          renderer = create(
            <ConnectAppsScreen
              initialIntegrations={appsIntegrations(true)}
              initialClaudeStatus={DISCONNECTED_CLAUDE}
            />,
          );
          await settle();
        });
        const card = () =>
          renderer!.root.findByProps({ id: "integration-codex" });
        expect(card().type).toBe("button");
        await act(async () => {
          await card().props.onClick();
        });
        expect(copyCommand).toHaveBeenCalledExactlyOnceWith(
          "ollama launch codex",
        );
        const notice = () => renderer!.root.findByProps({ role: "status" });
        expect(
          notice()
            .findAllByType("p")
            .map((node) => node.children.join(""))
            .join(" "),
        ).toContain("Launch command copied. Paste it into your terminal");
        act(() => vi.advanceTimersByTime(5000));
        await act(async () => {
          await card().props.onClick();
        });
        act(() => vi.advanceTimersByTime(1001));
        expect(notice()).toBeTruthy();
        act(() => vi.advanceTimersByTime(5000));
        expect(renderer!.root.findAllByProps({ role: "status" })).toHaveLength(
          0,
        );
        expect(card().props["aria-label"]).toBe("Copy Codex CLI command");
        expect(copyCommand).toHaveBeenCalledTimes(2);
      } finally {
        if (renderer) act(() => renderer?.unmount());
        copyCommand.mockRestore();
        vi.unstubAllGlobals();
        vi.useRealTimers();
      }
    },
  );

  it.each(["denied", "throws"])(
    "offers manual copying instead of success when clipboard access %s",
    async (outcome) => {
      vi.useFakeTimers();
      const copyCommand = vi
        .spyOn(clipboard, "copyTextToClipboard")
        .mockImplementationOnce(() =>
          outcome === "throws"
            ? Promise.reject(new Error("clipboard unavailable"))
            : Promise.resolve(false),
        )
        .mockResolvedValue(true);
      stubAppsWindow();
      let renderer: ReactTestRenderer | undefined;
      try {
        await act(async () => {
          renderer = create(
            <ConnectAppsScreen
              initialIntegrations={appsIntegrations(true)}
              initialClaudeStatus={DISCONNECTED_CLAUDE}
            />,
          );
          await settle();
        });
        const card = () =>
          renderer!.root.findByProps({ id: "integration-codex" });
        await act(async () => {
          await card().props.onClick();
        });
        expect(card().props["aria-label"]).toBe("Copy Codex CLI command");
        act(() => vi.advanceTimersByTime(20_000));
        expect(
          renderer!.root.findByProps({ role: "alert" }).findByType("code")
            .children,
        ).toEqual(["ollama launch codex"]);
        expect(renderer!.root.findAllByProps({ role: "status" })).toHaveLength(
          0,
        );
        await act(async () => {
          await card().props.onClick();
        });
        expect(renderer!.root.findAllByProps({ role: "alert" })).toHaveLength(
          0,
        );
        expect(renderer!.root.findByProps({ role: "status" })).toBeTruthy();
        expect(card().props["aria-label"]).toBe("Codex CLI command copied");
      } finally {
        if (renderer) act(() => renderer?.unmount());
        copyCommand.mockRestore();
        vi.unstubAllGlobals();
        vi.useRealTimers();
      }
    },
  );

  function animationFrames() {
    let nextId = 0;
    const pending = new Map<number, FrameRequestCallback>();
    const requestAnimationFrame = vi.fn((callback: FrameRequestCallback) => {
      const id = ++nextId;
      pending.set(id, callback);
      return id;
    });
    const cancelAnimationFrame = vi.fn((id: number) => pending.delete(id));
    const paint = (now: number) => {
      const callbacks = [...pending.values()];
      pending.clear();
      act(() => callbacks.forEach((callback) => callback(now)));
    };
    return { requestAnimationFrame, cancelAnimationFrame, paint, pending };
  }

  it("paints the Apps page before scrolling, even when the first frames are delayed", async () => {
    const frames = animationFrames();
    const { container, createNodeMock } = scrollMocks();
    const integrations = appsIntegrations(true);
    const onDeepLinkHandled = vi.fn();
    stubAppsWindow({
      ...frames,
      getClaudeDesktopConnectionSummary: vi
        .fn()
        .mockReturnValue(new Promise(() => {})),
    });

    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(
          <StrictMode>
            <ConnectAppsScreen
              initialIntegrations={integrations}
              highlightIntegrationId="codex"
              onDeepLinkHandled={onDeepLinkHandled}
            />
          </StrictMode>,
          { createNodeMock },
        );
        await settle();
      });
      expect(onDeepLinkHandled).toHaveBeenCalledOnce();
      expect(container.scrollTop).toBe(0);

      // Clearing the handled URL must not cancel the in-progress handoff.
      await act(async () => {
        renderer!.update(
          <StrictMode>
            <ConnectAppsScreen
              initialIntegrations={integrations}
              onDeepLinkHandled={onDeepLinkHandled}
            />
          </StrictMode>,
        );
      });

      const firstPaint = performance.now() + 2000;
      frames.paint(firstPaint);
      expect(container.scrollTop).toBe(0);
      expect(frames.pending.size).toBe(1);

      // Native work can delay either frame; neither delay consumes the scroll.
      const scrollStart = firstPaint + 2000;
      frames.paint(scrollStart);
      expect(container.scrollTop).toBe(0);
      frames.paint(scrollStart + INTEGRATION_SCROLL_MS / 2);
      expect(container.scrollTop).toBeGreaterThan(0);
      expect(container.scrollTop).toBeLessThan(CENTERED_SCROLL_TOP);
      frames.paint(scrollStart + INTEGRATION_SCROLL_MS);
      expect(container.scrollTop).toBe(CENTERED_SCROLL_TOP);
      expect(frames.pending.size).toBe(0);
      expect(
        renderer!.root.findByProps({ id: "integration-codex" }).props.className,
      ).toContain("bg-neutral-100");
      expect(onDeepLinkHandled).toHaveBeenCalledOnce();
    } finally {
      if (renderer) act(() => renderer?.unmount());
      vi.unstubAllGlobals();
    }
  });

  it.each([0, 1, 3])(
    "stops the handoff scroll when leaving Apps after %s frames",
    async (paintedFrames) => {
      const frames = animationFrames();
      const { container, createNodeMock } = scrollMocks();
      stubAppsWindow(frames);

      let renderer: ReactTestRenderer | undefined;
      try {
        await act(async () => {
          renderer = create(
            <ConnectAppsScreen
              initialIntegrations={appsIntegrations(true)}
              highlightIntegrationId="codex"
            />,
            { createNodeMock },
          );
          await settle();
        });
        for (let i = 0; i < paintedFrames; i++) frames.paint(i * 16);
        const scrollTop = container.scrollTop;
        act(() => renderer!.unmount());
        renderer = undefined;
        expect(frames.pending.size).toBe(0);
        frames.paint(INTEGRATION_SCROLL_MS + 1000);
        expect(container.scrollTop).toBe(scrollTop);
      } finally {
        if (renderer) act(() => renderer?.unmount());
        vi.unstubAllGlobals();
      }
    },
  );

  it.each(["connected", "failed"])(
    "shows and highlights apps while Claude status is still loading (%s)",
    async (outcome) => {
      let resolveClaude!: (status: typeof DISCONNECTED_CLAUDE) => void;
      let rejectClaude!: (error: Error) => void;
      const claude = new Promise<typeof DISCONNECTED_CLAUDE>(
        (resolve, reject) => {
          resolveClaude = resolve;
          rejectClaude = reject;
        },
      );
      const copyCommand = vi
        .spyOn(clipboard, "copyTextToClipboard")
        .mockResolvedValue(true);
      const onDeepLinkHandled = vi.fn();
      const { container, createNodeMock } = scrollMocks();
      stubAppsWindow({
        getClaudeDesktopConnectionSummary: vi.fn().mockReturnValue(claude),
        matchMedia: vi.fn().mockReturnValue({ matches: true }),
      });
      vi.stubGlobal(
        "fetch",
        vi
          .fn()
          .mockResolvedValue(
            new Response(JSON.stringify(appsIntegrations(true))),
          ),
      );

      let renderer: ReactTestRenderer | undefined;
      try {
        await act(async () => {
          renderer = create(
            <ConnectAppsScreen
              highlightIntegrationId="codex"
              onDeepLinkHandled={onDeepLinkHandled}
            />,
            { createNodeMock },
          );
          await settle();
        });

        const row = () =>
          renderer!.root.findByProps({ id: "integration-codex" });
        const claudeToggle = () =>
          renderer!.root
            .findByProps({ id: "integration-claude-desktop" })
            .find(
              (node) =>
                node.type === "button" &&
                typeof node.props["aria-pressed"] === "boolean",
            );
        expect(row().props.className).toContain("bg-neutral-100");
        expect(container.scrollTop).toBe(CENTERED_SCROLL_TOP);
        expect(onDeepLinkHandled).toHaveBeenCalledOnce();
        expect(claudeToggle().props.disabled).toBe(true);
        expect(claudeToggle().props["aria-busy"]).toBe(true);

        await act(async () => {
          await renderer!.root
            .findByProps({ "aria-label": "Copy Codex CLI command" })
            .props.onClick();
        });
        expect(copyCommand).toHaveBeenCalledWith("ollama launch codex");
        expect(row().props.className).not.toContain("bg-neutral-100");

        await act(async () => {
          if (outcome === "connected") {
            resolveClaude({
              ...DISCONNECTED_CLAUDE,
              used: true,
              configured: true,
              connected: true,
            });
          } else {
            rejectClaude(new Error("Claude status unavailable"));
          }
          await settle();
        });
        expect(claudeToggle().props.disabled).toBe(false);
        expect(claudeToggle().props["aria-pressed"]).toBe(
          outcome === "connected",
        );
        if (outcome === "failed") {
          expect(
            renderer!.root.findByProps({ role: "alert" }).children,
          ).toContain("Ollama could not read the Claude connection status.");
        }
        expect(row().props.className).not.toContain("bg-neutral-100");
        expect(onDeepLinkHandled).toHaveBeenCalledOnce();
      } finally {
        if (renderer) act(() => renderer?.unmount());
        copyCommand.mockRestore();
        vi.unstubAllGlobals();
      }
    },
  );

  it("shows an app-list error without waiting for Claude status", async () => {
    stubAppsWindow({
      getClaudeDesktopConnectionSummary: vi
        .fn()
        .mockReturnValue(new Promise(() => {})),
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 500 })),
    );

    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(<ConnectAppsScreen />);
        await settle();
      });
      expect(renderer!.root.findByProps({ role: "alert" }).children).toContain(
        "Couldn't load integrations.",
      );
    } finally {
      if (renderer) act(() => renderer?.unmount());
      vi.unstubAllGlobals();
    }
  });

  it.each([false, true])(
    "waits for Claude status before handling its onboarding connection (connected: %s)",
    async (connected) => {
      const status = {
        ...DISCONNECTED_CLAUDE,
        used: true,
        configured: connected,
        connected,
      };
      let resolveClaude!: (value: typeof status) => void;
      const claude = new Promise<typeof status>((resolve) => {
        resolveClaude = resolve;
      });
      const getStatus = vi
        .fn()
        .mockReturnValueOnce(claude)
        .mockResolvedValue(status);
      const setClaudeDesktopConnected = vi.fn().mockResolvedValue({
        status: { ...status, configured: true, connected: true },
      });
      const onDeepLinkHandled = vi.fn();
      stubAppsWindow({
        getClaudeDesktopConnectionSummary: getStatus,
        setClaudeDesktopConnected,
        matchMedia: vi.fn().mockReturnValue({ matches: true }),
      });

      let renderer: ReactTestRenderer | undefined;
      try {
        await act(async () => {
          renderer = create(
            <ConnectAppsScreen
              autoConnectClaude
              initialIntegrations={appsIntegrations(true)}
              onDeepLinkHandled={onDeepLinkHandled}
            />,
            { createNodeMock: scrollMocks().createNodeMock },
          );
          await settle();
        });
        expect(
          renderer!.root.findByProps({
            "aria-label": "Copy Codex CLI command",
          }),
        ).toBeTruthy();
        expect(onDeepLinkHandled).not.toHaveBeenCalled();
        expect(getStatus).toHaveBeenCalledOnce();
        expect(setClaudeDesktopConnected).not.toHaveBeenCalled();

        await act(async () => {
          resolveClaude(status);
          await settle();
        });
        expect(onDeepLinkHandled).toHaveBeenCalledOnce();
        if (connected) {
          expect(setClaudeDesktopConnected).not.toHaveBeenCalled();
        } else {
          expect(setClaudeDesktopConnected).toHaveBeenCalledExactlyOnceWith(
            true,
            false,
          );
        }
        expect(
          renderer!.root
            .findByProps({ id: "integration-claude-desktop" })
            .find(
              (node) =>
                node.type === "button" &&
                typeof node.props["aria-pressed"] === "boolean",
            ).props["aria-pressed"],
        ).toBe(true);
      } finally {
        if (renderer) act(() => renderer?.unmount());
        vi.unstubAllGlobals();
      }
    },
  );

  it("starts the Claude toggle once when opened from onboarding", async () => {
    const connectedStatus = {
      ...DISCONNECTED_CLAUDE,
      configured: true,
      connected: true,
    };
    const setClaudeDesktopConnected = vi
      .fn()
      .mockResolvedValue({ status: connectedStatus });
    const { container, createNodeMock } = scrollMocks();
    const frames = animationFrames();
    const onDeepLinkHandled = vi.fn();
    stubAppsWindow({
      getClaudeDesktopConnectionSummary: vi
        .fn()
        .mockResolvedValue(DISCONNECTED_CLAUDE),
      setClaudeDesktopConnected,
      activateOllama: vi.fn(),
      ...frames,
    });

    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(
          <StrictMode>
            <ConnectAppsScreen
              autoConnectClaude
              initialClaudeStatus={DISCONNECTED_CLAUDE}
              initialIntegrations={appsIntegrations(true)}
              onDeepLinkHandled={onDeepLinkHandled}
            />
          </StrictMode>,
          { createNodeMock },
        );
        await settle();
      });
      await act(async () => {
        await settle();
      });

      expect(setClaudeDesktopConnected).toHaveBeenCalledTimes(1);
      expect(setClaudeDesktopConnected).toHaveBeenCalledWith(true, false);
      frames.paint(0);
      frames.paint(16);
      frames.paint(16 + INTEGRATION_SCROLL_MS);
      expect(container.scrollTop).toBe(CENTERED_SCROLL_TOP);
      expect(onDeepLinkHandled).toHaveBeenCalledTimes(1);
      expect(
        renderer!.root.findByProps({ id: "integration-claude-desktop" }).props
          .className,
      ).toContain("bg-neutral-100 dark:bg-neutral-700/60");
      expect(
        renderer!.root
          .findByProps({ id: "integration-claude-desktop" })
          .find(
            (node) =>
              node.type === "button" &&
              typeof node.props["aria-pressed"] === "boolean",
          ).props["aria-pressed"],
      ).toBe(true);
      expect(renderer!.root.findByType(ClaudeConnectedIntro)).toBeTruthy();
    } finally {
      if (renderer) act(() => renderer?.unmount());
      vi.unstubAllGlobals();
    }
  });

  it("does not disconnect a Claude that is already connected", async () => {
    const connectedStatus = {
      ...DISCONNECTED_CLAUDE,
      used: true,
      configured: true,
      connected: true,
    };
    const setClaudeDesktopConnected = vi.fn();
    const onDeepLinkHandled = vi.fn();
    stubAppsWindow({
      getClaudeDesktopConnectionSummary: vi
        .fn()
        .mockResolvedValue(connectedStatus),
      setClaudeDesktopConnected,
    });

    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(
          <ConnectAppsScreen
            autoConnectClaude
            initialClaudeStatus={connectedStatus}
            initialIntegrations={appsIntegrations(true)}
            onDeepLinkHandled={onDeepLinkHandled}
          />,
          { createNodeMock: scrollMocks().createNodeMock },
        );
        await settle();
      });

      expect(setClaudeDesktopConnected).not.toHaveBeenCalled();
      expect(onDeepLinkHandled).toHaveBeenCalledTimes(1);
      expect(
        renderer!.root.findByProps({ id: "integration-claude-desktop" }).props
          .className,
      ).toContain("bg-neutral-100 dark:bg-neutral-700/60");
    } finally {
      if (renderer) act(() => renderer?.unmount());
      vi.unstubAllGlobals();
    }
  });

  it("keeps an app highlighted until its command is successfully copied", async () => {
    vi.useFakeTimers();
    const copyCommand = vi
      .spyOn(clipboard, "copyTextToClipboard")
      .mockResolvedValueOnce(false)
      .mockResolvedValue(true);
    const { container, createNodeMock } = scrollMocks();
    const requestAnimationFrame = vi.fn();
    const onDeepLinkHandled = vi.fn();
    stubAppsWindow({
      getClaudeDesktopConnectionSummary: vi
        .fn()
        .mockResolvedValue(DISCONNECTED_CLAUDE),
      matchMedia: vi.fn().mockReturnValue({ matches: true }),
      requestAnimationFrame,
    });

    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(
          <ConnectAppsScreen
            highlightIntegrationId="codex"
            initialClaudeStatus={DISCONNECTED_CLAUDE}
            initialIntegrations={appsIntegrations(true)}
            onDeepLinkHandled={onDeepLinkHandled}
          />,
          { createNodeMock },
        );
        await settle();
      });
      const row = () => renderer!.root.findByProps({ id: "integration-codex" });

      expect(row().props.className).toContain(
        "bg-neutral-100 dark:bg-neutral-700/60",
      );
      // Reduced motion jumps straight to the row without animating.
      expect(container.scrollTop).toBe(CENTERED_SCROLL_TOP);
      expect(requestAnimationFrame).not.toHaveBeenCalled();
      expect(onDeepLinkHandled).toHaveBeenCalledTimes(1);

      // Handling the deep link clears the URL, but the row stays highlighted.
      await act(async () => {
        renderer!.update(
          <ConnectAppsScreen
            initialClaudeStatus={DISCONNECTED_CLAUDE}
            initialIntegrations={appsIntegrations(true)}
            onDeepLinkHandled={onDeepLinkHandled}
          />,
        );
      });
      await act(async () => {
        vi.advanceTimersByTime(60_000);
      });
      expect(row().props.className).toContain("bg-neutral-100");

      // A failed copy leaves the user's selected app highlighted for retry.
      const copyCodex = () =>
        renderer!.root
          .findByProps({ "aria-label": "Copy Codex CLI command" })
          .props.onClick();
      await act(async () => {
        await copyCodex();
      });
      expect(copyCommand).toHaveBeenNthCalledWith(1, "ollama launch codex");
      expect(row().props.className).toContain("bg-neutral-100");

      await act(async () => {
        await renderer!.root
          .findByProps({ "aria-label": "Copy OpenCode command" })
          .props.onClick();
      });
      expect(copyCommand).toHaveBeenNthCalledWith(2, "ollama launch opencode");
      expect(row().props.className).toContain("bg-neutral-100");

      await act(async () => {
        await copyCodex();
      });
      expect(copyCommand).toHaveBeenNthCalledWith(3, "ollama launch codex");
      expect(row().props.className).toContain("bg-white dark:bg-neutral-900");
      expect(row().props.className).not.toContain("bg-neutral-100");
      expect(
        renderer!.root.findByProps({
          "aria-label": "Codex CLI command copied",
        }),
      ).toBeTruthy();
    } finally {
      if (renderer) act(() => renderer?.unmount());
      copyCommand.mockRestore();
      vi.unstubAllGlobals();
      vi.useRealTimers();
    }
  });
});
