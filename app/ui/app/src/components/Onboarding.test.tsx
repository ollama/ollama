import { StrictMode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { QueryClient } from "@tanstack/react-query";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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

function claudeConnectionButton(renderer: ReactTestRenderer) {
  return renderer.root
    .findByProps({ id: "integration-claude-desktop" })
    .find(
      (node) =>
        node.type === "button" &&
        typeof node.props["aria-pressed"] === "boolean",
    );
}

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

  it("keeps the Claude connection busy through installer detection", async () => {
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

      const claudeSwitch = () => claudeConnectionButton(renderer!);
      expect(claudeSwitch().props["aria-pressed"]).toBe(false);
      expect(claudeSwitch().props.disabled).toBe(false);
      let clickResult!: Promise<void>;
      await act(async () => {
        clickResult = claudeSwitch().props.onClick();
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(window.installClaudeDesktop).toHaveBeenCalledOnce();
      expect(claudeSwitch().props["aria-pressed"]).toBe(true);
      expect(claudeSwitch().props["aria-busy"]).toBe(true);
      expect(claudeSwitch().props.disabled).toBe(true);
      expect(claudeSwitch().findByProps({ role: "status" })).toBeTruthy();

      await act(async () => {
        finishInstall("opened");
        await clickResult;
        await Promise.resolve();
      });

      expect(claudeSwitch().props["aria-pressed"]).toBe(true);
      expect(claudeSwitch().props["aria-busy"]).toBe(true);
      expect(claudeSwitch().props.disabled).toBe(true);
      expect(claudeSwitch().findByProps({ role: "status" })).toBeTruthy();
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

      const claudeSwitch = () => claudeConnectionButton(renderer!);
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

  it("offers ChatGPT when the catalog has no desktop metadata", () => {
    const html = renderToStaticMarkup(
      <ConnectAppsScreen initialIntegrations={appsIntegrations(true)} />,
    );
    expect(html).toContain('id="integration-chatgpt"');
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

describe("ConnectAppsScreen interactions", () => {
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
        const toggle = claudeConnectionButton(renderer!);
        await act(async () => {
          await toggle.props.onClick();
        });
        expect(confirm).toHaveBeenCalledOnce();
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

  it("lets the user retry Connect after cancelling the native confirmation", async () => {
    const running = { ...DISCONNECTED_CLAUDE, running: true, used: true };
    const confirm = vi.fn().mockReturnValueOnce(false).mockReturnValue(true);
    const connect = vi.fn().mockResolvedValue({
      status: { ...running, configured: true, connected: true },
    });
    stubAppsWindow({
      getClaudeDesktopConnectionSummary: vi.fn().mockResolvedValue(running),
      setClaudeDesktopConnected: connect,
      confirm,
    });
    let renderer: ReactTestRenderer | undefined;
    const clickConnect = () =>
      claudeConnectionButton(renderer!).props.onClick();
    try {
      await act(async () => {
        renderer = create(
          <StrictMode>
            <ConnectAppsScreen
              initialIntegrations={appsIntegrations(true)}
              initialClaudeStatus={running}
            />
          </StrictMode>,
        );
        await settle();
      });
      await act(async () => {
        await clickConnect();
      });
      expect(confirm).toHaveBeenCalledTimes(1);
      expect(connect).not.toHaveBeenCalled();
      await act(async () => {
        await clickConnect();
      });
      expect(confirm).toHaveBeenCalledTimes(2);
      expect(connect).toHaveBeenCalledExactlyOnceWith(true, true);
    } finally {
      if (renderer) act(() => renderer?.unmount());
      vi.unstubAllGlobals();
    }
  });

  it.each(["darwin", "windows"])(
    "copies every catalog command and limits desktop connections to macOS (%s)",
    async (platform) => {
      const copyCommand = vi
        .spyOn(clipboard, "copyTextToClipboard")
        .mockResolvedValue(true);
      stubAppsWindow({ OLLAMA_PLATFORM: platform });
      const integrations = [
        ...appsIntegrations(true),
        ...Array.from({ length: 20 }, (_, index) => ({
          id: `extra-${index}`,
          name: `Extra app ${index}`,
          description: "Another supported integration",
          command: `ollama launch extra-${index}`,
        })),
      ];
      const launchers = integrations.filter((item) => item.command);
      let renderer: ReactTestRenderer | undefined;
      try {
        await act(async () => {
          renderer = create(
            <ConnectAppsScreen
              initialIntegrations={[
                ...integrations,
                {
                  id: "chatgpt",
                  name: "ChatGPT",
                  description: "Desktop integration",
                  command: "ollama launch chatgpt",
                },
              ]}
              initialClaudeStatus={DISCONNECTED_CLAUDE}
              initialCodexStatus={{
                supported: true,
                installed: true,
                connected: false,
                running: false,
              }}
            />,
          );
        });
        const cards = renderer!.root.findAll(
          (node) =>
            node.type === "button" && node.props.id?.startsWith("integration-"),
        );
        expect(new Set(cards.map((card) => card.props.id))).toEqual(
          new Set(launchers.map((item) => `integration-${item.id}`)),
        );
        expect(
          renderer!.root.findAll(
            (node) =>
              node.type === "button" &&
              typeof node.props["aria-pressed"] === "boolean",
          ),
        ).toHaveLength(platform === "darwin" ? 2 : 0);
        for (const item of launchers) {
          await act(async () => {
            await renderer!.root
              .findByProps({ id: `integration-${item.id}` })
              .props.onClick();
          });
          expect(copyCommand).toHaveBeenLastCalledWith(item.command);
        }
        expect(copyCommand).toHaveBeenCalledTimes(launchers.length);
      } finally {
        if (renderer) act(() => renderer?.unmount());
        copyCommand.mockRestore();
        vi.unstubAllGlobals();
      }
    },
  );

  it("renews the copy notification on each click and dismisses it after inactivity", async () => {
    vi.useFakeTimers();
    const copyCommand = vi
      .spyOn(clipboard, "copyTextToClipboard")
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
      expect(copyCommand).toHaveBeenCalledExactlyOnceWith(
        "ollama launch codex",
      );
      const notice = () => renderer!.root.findByProps({ role: "status" });
      expect(notice()).toBeTruthy();
      act(() => vi.advanceTimersByTime(5000));
      await act(async () => {
        await card().props.onClick();
      });
      act(() => vi.advanceTimersByTime(1001));
      expect(notice()).toBeTruthy();
      act(() => vi.advanceTimersByTime(5000));
      expect(renderer!.root.findAllByProps({ role: "status" })).toHaveLength(0);
      expect(copyCommand).toHaveBeenCalledTimes(2);
    } finally {
      if (renderer) act(() => renderer?.unmount());
      copyCommand.mockRestore();
      vi.unstubAllGlobals();
      vi.useRealTimers();
    }
  });

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
      } finally {
        if (renderer) act(() => renderer?.unmount());
        copyCommand.mockRestore();
        vi.unstubAllGlobals();
        vi.useRealTimers();
      }
    },
  );

  it.each(["Escape", "outside pointer"])(
    "dismisses a copy error with %s while preserving manual copying",
    async (dismissal) => {
      const events = new EventTarget();
      const commandNode = {};
      const noticeNode = {
        contains: (target: unknown) => target === commandNode,
      };
      const copyCommand = vi
        .spyOn(clipboard, "copyTextToClipboard")
        .mockResolvedValueOnce(false)
        .mockResolvedValue(true);
      stubAppsWindow({
        addEventListener: events.addEventListener.bind(events),
        removeEventListener: events.removeEventListener.bind(events),
      });
      let renderer: ReactTestRenderer | undefined;
      try {
        await act(async () => {
          renderer = create(
            <ConnectAppsScreen
              initialIntegrations={appsIntegrations(true)}
              initialClaudeStatus={DISCONNECTED_CLAUDE}
            />,
            {
              createNodeMock: (element) =>
                element.props.role === "alert" ? noticeNode : null,
            },
          );
          await settle();
        });
        const card = () =>
          renderer!.root.findByProps({ id: "integration-codex" });
        await act(async () => {
          await card().props.onClick();
        });

        // Selecting the command and unrelated keys must keep it available.
        const selection = new Event("pointerdown");
        Object.defineProperty(selection, "target", { value: commandNode });
        act(() => {
          events.dispatchEvent(selection);
          events.dispatchEvent(
            Object.assign(new Event("keydown"), { key: "c" }),
          );
        });
        expect(
          renderer!.root.findByProps({ role: "alert" }).findByType("code")
            .children,
        ).toEqual(["ollama launch codex"]);

        act(() => {
          events.dispatchEvent(
            dismissal === "Escape"
              ? Object.assign(new Event("keydown"), { key: "Escape" })
              : new Event("pointerdown"),
          );
        });
        expect(renderer!.root.findAllByProps({ role: "alert" })).toHaveLength(
          0,
        );

        // A later successful copy keeps its usual notification lifetime.
        await act(async () => {
          await card().props.onClick();
        });
        act(() => {
          events.dispatchEvent(
            Object.assign(new Event("keydown"), { key: "Escape" }),
          );
          events.dispatchEvent(new Event("pointerdown"));
        });
        expect(renderer!.root.findByProps({ role: "status" })).toBeTruthy();
      } finally {
        if (renderer) act(() => renderer?.unmount());
        copyCommand.mockRestore();
        vi.unstubAllGlobals();
      }
    },
  );

  it.each(["connected", "disconnected", "failed"])(
    "lets users copy commands while Claude status is still loading (%s)",
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
      const connect = vi.fn();
      stubAppsWindow({
        getClaudeDesktopConnectionSummary: vi.fn().mockReturnValue(claude),
        setClaudeDesktopConnected: connect,
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
          renderer = create(<ConnectAppsScreen />);
          await settle();
        });

        const claudeToggle = () => claudeConnectionButton(renderer!);
        expect(claudeToggle().props.disabled).toBe(true);
        expect(claudeToggle().props["aria-busy"]).toBe(true);

        await act(async () => {
          await renderer!.root
            .findByProps({ id: "integration-codex" })
            .props.onClick();
        });
        expect(copyCommand).toHaveBeenCalledWith("ollama launch codex");
        expect(connect).not.toHaveBeenCalled();

        await act(async () => {
          if (outcome === "failed") {
            rejectClaude(new Error("Claude status unavailable"));
          } else {
            resolveClaude({
              ...DISCONNECTED_CLAUDE,
              used: true,
              configured: outcome === "connected",
              connected: outcome === "connected",
            });
          }
          await settle();
        });
        expect(claudeToggle().props.disabled).toBe(false);
        expect(claudeToggle().props["aria-pressed"]).toBe(
          outcome === "connected",
        );
        if (outcome === "failed") {
          expect(renderer!.root.findByProps({ role: "alert" })).toBeTruthy();
        }
        expect(connect).not.toHaveBeenCalled();
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
      expect(renderer!.root.findByProps({ role: "alert" })).toBeTruthy();
    } finally {
      if (renderer) act(() => renderer?.unmount());
      vi.unstubAllGlobals();
    }
  });

  it("shows the first-use intro after the user connects Claude", async () => {
    const connectedStatus = {
      ...DISCONNECTED_CLAUDE,
      configured: true,
      connected: true,
    };
    const setClaudeDesktopConnected = vi
      .fn()
      .mockResolvedValue({ status: connectedStatus });
    stubAppsWindow({
      getClaudeDesktopConnectionSummary: vi
        .fn()
        .mockResolvedValue(DISCONNECTED_CLAUDE),
      setClaudeDesktopConnected,
      activateOllama: vi.fn(),
    });

    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(
          <StrictMode>
            <ConnectAppsScreen
              initialClaudeStatus={DISCONNECTED_CLAUDE}
              initialIntegrations={appsIntegrations(true)}
            />
          </StrictMode>,
        );
        await settle();
      });
      expect(setClaudeDesktopConnected).not.toHaveBeenCalled();
      await act(async () => {
        await claudeConnectionButton(renderer!).props.onClick();
      });

      expect(setClaudeDesktopConnected).toHaveBeenCalledTimes(1);
      expect(setClaudeDesktopConnected).toHaveBeenCalledWith(true, false);
      expect(claudeConnectionButton(renderer!).props["aria-pressed"]).toBe(
        true,
      );
      expect(renderer!.root.findByType(ClaudeConnectedIntro)).toBeTruthy();
    } finally {
      if (renderer) act(() => renderer?.unmount());
      vi.unstubAllGlobals();
    }
  });
});
