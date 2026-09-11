import type { IntegrationStatus } from "@/api";
import type { CodexDesktopStatus } from "@/types/webview";
import {
  QueryClient,
  defaultScheduler,
  notifyManager,
} from "@tanstack/react-query";
import { renderToStaticMarkup } from "react-dom/server";
import { act, create } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CodexConnectedIntro } from "./CodexConnectedIntro";
import {
  CODEX_DESKTOP_INSTALL_TIMEOUT_MS,
  CodexDesktopRow,
} from "./CodexDesktopRow";

vi.mock("./CodexConnectedIntro", () => ({
  CodexConnectedIntro: () => null,
}));

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

let queryClient: QueryClient;
beforeEach(() => {
  queryClient = new QueryClient();
  notifyManager.setScheduler(queueMicrotask);
});
afterEach(() => {
  queryClient.clear();
  notifyManager.setScheduler(defaultScheduler);
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const integration: IntegrationStatus = {
  id: "chatgpt",
  name: "ChatGPT",
  description: "Use Ollama models in ChatGPT",
  installed: true,
  command: "ollama launch chatgpt",
};

function status(
  overrides: Partial<CodexDesktopStatus> = {},
): CodexDesktopStatus {
  return {
    used: true,
    supported: true,
    installed: true,
    connected: false,
    running: false,
    ...overrides,
  };
}

describe("CodexDesktopRow", () => {
  it("renders a disconnected ChatGPT toggle", () => {
    const html = renderToStaticMarkup(
      <CodexDesktopRow integration={integration} initialStatus={status()} />,
    );

    expect(html).toContain(">ChatGPT (Desktop)</p>");
    expect(html).toContain("Use Ollama models in ChatGPT");
    expect(html).toContain('aria-label="Add Ollama models to ChatGPT"');
    expect(html).toContain('aria-checked="false"');
  });

  it("matches Claude's connected copy before the first request", () => {
    const html = renderToStaticMarkup(
      <CodexDesktopRow
        integration={integration}
        initialStatus={status({
          connected: true,
          model: "qwen3:8b",
          models: ["qwen3:8b", "glm-5.3-flash:cloud", "kimi-k2.7-code:cloud"],
        })}
      />,
    );

    expect(html).toContain("Connected to Ollama · 0 requests this session");
    expect(html).not.toContain("Codex + Ollama");
    expect(html).not.toContain("3 Ollama models");
    expect(html).toContain('aria-label="Remove Ollama models from ChatGPT"');
    expect(html).toContain('aria-checked="true"');
  });

  it.each([
    { requests: 1, expected: "Connected to Ollama · 1 request this session" },
    {
      requests: 12,
      expected: "Connected to Ollama · 12 requests this session",
    },
  ])(
    "matches Claude's connected copy for $requests requests",
    ({ requests, expected }) => {
      const html = renderToStaticMarkup(
        <CodexDesktopRow
          integration={integration}
          initialStatus={status({
            connected: true,
            model: "qwen3:8b",
            requests,
          })}
        />,
      );

      expect(html).toContain(expected);
    },
  );

  it("offers installation when ChatGPT is not installed", () => {
    const html = renderToStaticMarkup(
      <CodexDesktopRow
        integration={{ ...integration, installed: false }}
        initialStatus={status({ installed: false })}
      />,
    );

    expect(html).toContain("Use Ollama models in ChatGPT");
    expect(html).not.toContain('disabled=""');
    expect(html).toContain('title="Install ChatGPT and add Ollama models"');
    expect(html).toContain("Download &amp; connect");
  });

  it("matches Claude's download and install progress states", async () => {
    const notInstalled = status({ installed: false });
    let finishInstall!: (result: "opened") => void;
    const install = new Promise<"opened">((resolve) => {
      finishInstall = resolve;
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setInterval: globalThis.setInterval,
      clearInterval: globalThis.clearInterval,
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout,
      getCodexDesktopStatus: vi.fn().mockResolvedValue(notInstalled),
      setCodexDesktopConnected: vi.fn(),
      installCodexDesktop: vi.fn().mockReturnValue(install),
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={{ ...integration, installed: false }}
            initialStatus={notInstalled}
          />,
        );
      });
      const toggle = renderer!.root.findByProps({ role: "switch" });
      await act(async () => {
        toggle.props.onClick();
        await Promise.resolve();
      });

      expect(toggle.props["aria-checked"]).toBe(true);
      expect(toggle.props["aria-busy"]).toBe(true);
      expect(toggle.props.disabled).toBe(true);
      expect(toggle.props.className).toContain("disabled:cursor-wait");
      expect(renderer!.root.findByProps({ role: "status" }).children).toContain(
        "Downloading…",
      );
      expect(
        renderer!.root.findAll((node) =>
          node.children.includes(
            "Ollama is downloading the ChatGPT installer…",
          ),
        ),
      ).toHaveLength(1);

      await act(async () => {
        finishInstall("opened");
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(toggle.props["aria-checked"]).toBe(true);
      expect(toggle.props["aria-busy"]).toBe(true);
      expect(toggle.props.disabled).toBe(true);
      expect(renderer!.root.findByProps({ role: "status" }).children).toContain(
        "Finish installing…",
      );
      expect(
        renderer!.root.findAll((node) =>
          node.children.includes(
            "Finish installing ChatGPT. Ollama will connect it automatically.",
          ),
        ),
      ).toHaveLength(1);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("matches Claude's connecting state", async () => {
    let finishConnect!: (result: { status: CodexDesktopStatus }) => void;
    const connect = new Promise<{ status: CodexDesktopStatus }>((resolve) => {
      finishConnect = resolve;
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setCodexDesktopConnected: vi.fn().mockReturnValue(connect),
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={integration}
            initialStatus={status()}
          />,
        );
      });
      const toggle = renderer!.root.findByProps({ role: "switch" });
      await act(async () => {
        toggle.props.onClick();
        await Promise.resolve();
      });

      expect(toggle.props["aria-checked"]).toBe(true);
      expect(toggle.props["aria-busy"]).toBe(true);
      expect(toggle.props.disabled).toBe(true);
      expect(renderer!.root.findByProps({ role: "status" }).children).toContain(
        "Connecting…",
      );
      expect(
        renderer!.root.findAll((node) =>
          node.children.includes("Connecting ChatGPT to Ollama…"),
        ),
      ).toHaveLength(1);

      await act(async () => {
        finishConnect({ status: status({ connected: true }) });
        await connect;
      });
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("opens the installer and connects after ChatGPT is detected", async () => {
    const installedStatus = status({ installed: true });
    const connectedStatus = status({
      installed: true,
      connected: true,
      models: ["glm-5.3-flash:cloud"],
    });
    const openInstaller = vi.fn().mockResolvedValue("opened");
    const getStatus = vi.fn().mockResolvedValue(installedStatus);
    const connect = vi.fn().mockResolvedValue({ status: connectedStatus });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setInterval: globalThis.setInterval,
      clearInterval: globalThis.clearInterval,
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout,
      getCodexDesktopStatus: getStatus,
      setCodexDesktopConnected: connect,
      installCodexDesktop: openInstaller,
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={{ ...integration, installed: false }}
            initialStatus={status({ installed: false })}
          />,
        );
      });
      const toggle = renderer!.root.findByProps({
        "aria-label": "Add Ollama models to ChatGPT",
      });
      await act(async () => {
        await toggle.props.onClick();
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(openInstaller).toHaveBeenCalledOnce();
      expect(getStatus).toHaveBeenCalled();
      expect(connect).toHaveBeenCalledWith(true, false);
      expect(
        renderer!.root.findByProps({
          "aria-label": "Remove Ollama models from ChatGPT",
        }).props["aria-checked"],
      ).toBe(true);
      expect(renderer!.root.findByProps({ role: "status" }).children).toContain(
        "Ollama models added alongside Codex models",
      );
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it.each([
    { outcome: "resolved", retry: false },
    { outcome: "rejected", retry: false },
    { outcome: "resolved", retry: true },
    { outcome: "rejected", retry: true },
  ])(
    "ignores an expired installation check ($outcome, retried: $retry)",
    async ({ outcome, retry }) => {
      vi.useFakeTimers();
      const notInstalled = status({ installed: false });
      const stale = deferred<CodexDesktopStatus>();
      const getStatus = vi.fn().mockResolvedValue(notInstalled);
      const install = vi.fn().mockResolvedValue("opened");
      const connect = vi.fn().mockResolvedValue({
        status: status({ connected: true }),
      });
      vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
      vi.stubGlobal("window", {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        setInterval: globalThis.setInterval,
        clearInterval: globalThis.clearInterval,
        setTimeout: globalThis.setTimeout,
        clearTimeout: globalThis.clearTimeout,
        getCodexDesktopStatus: getStatus,
        setCodexDesktopConnected: connect,
        installCodexDesktop: install,
      });

      const settleOldCheck = () => {
        if (outcome === "resolved") stale.resolve(status());
        else stale.reject(new Error("expired status request failed"));
      };
      let renderer;
      try {
        await act(async () => {
          renderer = create(
            <CodexDesktopRow
              integration={integration}
              initialStatus={notInstalled}
            />,
          );
        });
        const toggle = renderer!.root.findByProps({ role: "switch" });
        await act(async () => toggle.props.onClick());
        await act(async () =>
          vi.advanceTimersByTimeAsync(CODEX_DESKTOP_INSTALL_TIMEOUT_MS - 2000),
        );
        getStatus.mockReturnValueOnce(stale.promise);
        await act(async () => vi.advanceTimersByTimeAsync(1000));
        await act(async () => {
          vi.advanceTimersByTime(1000);
          if (!retry) settleOldCheck();
        });
        expect(toggle.props.disabled).toBe(false);
        expect(toggle.props["aria-checked"]).toBe(false);
        expect(
          renderer!.root.findByProps({ role: "alert" }).children,
        ).toContain("ChatGPT installation wasn’t detected. Try again.");
        expect(connect).not.toHaveBeenCalled();

        if (retry) {
          await act(async () => toggle.props.onClick());
          expect(install).toHaveBeenCalledTimes(2);
          await act(async () => settleOldCheck());
          expect(toggle.props.disabled).toBe(true);
          expect(renderer!.root.findAllByProps({ role: "alert" })).toHaveLength(
            0,
          );
          expect(
            renderer!.root.findByProps({ role: "status" }).children,
          ).toContain("Finish installing…");

          getStatus.mockResolvedValue(status());
          await act(async () => vi.advanceTimersByTimeAsync(1000));
          expect(connect).toHaveBeenCalledOnce();
          expect(toggle.props.disabled).toBe(false);
          expect(toggle.props["aria-checked"]).toBe(true);
        }
      } finally {
        await act(async () => renderer?.unmount());
      }
    },
  );

  it.each(["status", "connection"])(
    "keeps an active installation's %s failure visible",
    async (step) => {
      const check = deferred<CodexDesktopStatus>();
      const connection = deferred<{ status: CodexDesktopStatus }>();
      vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
      vi.stubGlobal("window", {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        setInterval: globalThis.setInterval,
        clearInterval: globalThis.clearInterval,
        setTimeout: globalThis.setTimeout,
        clearTimeout: globalThis.clearTimeout,
        getCodexDesktopStatus: vi.fn().mockReturnValue(check.promise),
        setCodexDesktopConnected: vi.fn().mockReturnValue(connection.promise),
        installCodexDesktop: vi.fn().mockResolvedValue("opened"),
      });
      let renderer;
      try {
        await act(async () => {
          renderer = create(
            <CodexDesktopRow
              integration={integration}
              initialStatus={status({ installed: false })}
            />,
          );
        });
        const toggle = renderer!.root.findByProps({ role: "switch" });
        await act(async () => toggle.props.onClick());
        if (step === "connection") {
          await act(async () => check.resolve(status()));
          expect(toggle.props.disabled).toBe(true);
        }
        await act(async () => {
          if (step === "status") check.reject(new Error("status failed"));
          else connection.reject(new Error("connection failed"));
        });
        expect(toggle.props.disabled).toBe(false);
        expect(
          renderer!.root.findByProps({ role: "alert" }).children,
        ).toContain("Ollama could not finish connecting ChatGPT.");
      } finally {
        await act(async () => renderer?.unmount());
      }
    },
  );

  it.each([false, true])(
    "does not restart ChatGPT automatically when installation detection finds it running (used: %s)",
    async (used) => {
      const installedAndRunning = status({
        installed: true,
        running: true,
        used,
      });
      const connect = vi.fn();
      vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
      vi.stubGlobal("window", {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        setInterval: globalThis.setInterval,
        clearInterval: globalThis.clearInterval,
        setTimeout: globalThis.setTimeout,
        clearTimeout: globalThis.clearTimeout,
        getCodexDesktopStatus: vi.fn().mockResolvedValue(installedAndRunning),
        setCodexDesktopConnected: connect,
        installCodexDesktop: vi.fn().mockResolvedValue("opened"),
      });

      let renderer;
      try {
        await act(async () => {
          renderer = create(
            <CodexDesktopRow
              integration={{ ...integration, installed: false }}
              initialStatus={status({ installed: false, used })}
            />,
          );
        });
        const toggle = renderer!.root.findByProps({
          "aria-label": "Add Ollama models to ChatGPT",
        });
        await act(async () => {
          await toggle.props.onClick();
          await Promise.resolve();
          await Promise.resolve();
        });

        expect(connect).not.toHaveBeenCalled();
        expect(renderer!.root.findAllByType(CodexConnectedIntro)).toHaveLength(
          0,
        );
        expect(
          renderer!.root.findByProps({ role: "alert" }).children,
        ).toContain(
          "ChatGPT is installed. Turn on the switch to restart it with Ollama models.",
        );
        expect(
          renderer!.root.findByProps({
            "aria-label": "Add Ollama models to ChatGPT",
          }).props["aria-checked"],
        ).toBe(false);
      } finally {
        await act(async () => renderer?.unmount());
      }
    },
  );

  it("returns to the disconnected state when installation is cancelled", async () => {
    const getStatus = vi.fn();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getCodexDesktopStatus: getStatus,
      setCodexDesktopConnected: vi.fn(),
      installCodexDesktop: vi.fn().mockResolvedValue("cancelled"),
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={{ ...integration, installed: false }}
            initialStatus={status({ installed: false })}
          />,
        );
      });
      const toggle = renderer!.root.findByProps({
        "aria-label": "Add Ollama models to ChatGPT",
      });
      await act(async () => {
        await toggle.props.onClick();
      });

      expect(getStatus).not.toHaveBeenCalled();
      expect(toggle.props["aria-checked"]).toBe(false);
      expect(toggle.props.disabled).toBe(false);
      expect(renderer!.root.findAllByProps({ role: "alert" })).toHaveLength(0);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("uses concise restart copy when adding Ollama models", async () => {
    const confirm = vi.fn(() => false);
    const runningStatus = status({ running: true });
    const connect = vi.fn().mockResolvedValue({
      status: runningStatus,
      restartConfirmationRequired: true,
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setCodexDesktopConnected: connect,
      confirm,
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={integration}
            initialStatus={status({ running: true })}
          />,
        );
      });
      const toggle = renderer!.root.findByProps({
        "aria-label": "Add Ollama models to ChatGPT",
      });
      await act(async () => {
        await toggle.props.onClick();
      });

      expect(confirm).toHaveBeenCalledWith(
        "Restart ChatGPT to add Ollama models? Any running task will stop.",
      );
      expect(connect).toHaveBeenCalledOnce();
      expect(connect).toHaveBeenCalledWith(true, false);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("adds Ollama models after the restart is confirmed", async () => {
    const confirm = vi.fn(() => true);
    const connectedStatus = status({
      connected: true,
      running: true,
      models: ["glm-5.3-flash:cloud"],
    });
    const connect = vi
      .fn()
      .mockResolvedValueOnce({
        status: status({ running: true }),
        restartConfirmationRequired: true,
      })
      .mockResolvedValueOnce({ status: connectedStatus });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setInterval: globalThis.setInterval,
      clearInterval: globalThis.clearInterval,
      setCodexDesktopConnected: connect,
      confirm,
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={integration}
            initialStatus={status({ running: true })}
          />,
        );
      });
      const toggle = renderer!.root.findByProps({
        "aria-label": "Add Ollama models to ChatGPT",
      });
      await act(async () => {
        await toggle.props.onClick();
      });

      expect(confirm).toHaveBeenCalledWith(
        "Restart ChatGPT to add Ollama models? Any running task will stop.",
      );
      expect(connect).toHaveBeenNthCalledWith(1, true, false);
      expect(connect).toHaveBeenNthCalledWith(2, true, true);
      expect(
        renderer!.root.findByProps({
          "aria-label": "Remove Ollama models from ChatGPT",
        }).props["aria-checked"],
      ).toBe(true);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("keeps a native restart failure visible after confirmation", async () => {
    let onFocus: (() => void) | undefined;
    const addEventListener = vi.fn((event: string, handler: () => void) => {
      if (event === "focus") onFocus = handler;
    });
    const getStatus = vi.fn().mockResolvedValue(status({ running: true }));
    const confirm = vi.fn(() => {
      onFocus?.();
      return true;
    });
    const connect = vi
      .fn()
      .mockResolvedValueOnce({
        status: status({ running: true }),
        restartConfirmationRequired: true,
      })
      .mockResolvedValueOnce({
        status: status({ running: true }),
        error: "quit ChatGPT: timed out waiting for ChatGPT to exit",
      });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener,
      removeEventListener: vi.fn(),
      getCodexDesktopStatus: getStatus,
      setCodexDesktopConnected: connect,
      confirm,
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={integration}
            initialStatus={status({ running: true })}
          />,
        );
      });
      const toggle = renderer!.root.findByProps({
        "aria-label": "Add Ollama models to ChatGPT",
      });
      await act(async () => {
        await toggle.props.onClick();
      });

      expect(getStatus).not.toHaveBeenCalled();
      expect(renderer!.root.findByProps({ role: "alert" }).children).toContain(
        "quit ChatGPT: timed out waiting for ChatGPT to exit",
      );
      expect(toggle.props["aria-checked"]).toBe(false);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("allows the normal profile to be restored if ChatGPT is removed", async () => {
    const html = renderToStaticMarkup(
      <CodexDesktopRow
        integration={{ ...integration, installed: false }}
        initialStatus={status({ installed: false, connected: true })}
      />,
    );

    expect(html).toContain('aria-label="Remove Ollama models from ChatGPT"');
    expect(html).not.toContain('disabled=""');

    const restore = vi.fn().mockResolvedValue({
      status: status({ installed: false, connected: false }),
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setCodexDesktopConnected: restore,
      confirm: vi.fn(() => true),
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={{ ...integration, installed: false }}
            initialStatus={status({ installed: false, connected: true })}
          />,
        );
      });
      const restoreButton = renderer!.root.findByProps({
        "aria-label": "Remove Ollama models from ChatGPT",
      });
      await act(async () => {
        await restoreButton.props.onClick();
      });

      expect(restore).toHaveBeenCalledWith(false, false);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });
});

describe("ChatGPT first connection intro", () => {
  it.each([false, true])(
    "confirms before the intro and waits for Continue to launch (running: %s)",
    async (running) => {
      const firstUseStatus = status({ running, used: false });
      const confirm = vi.fn(() => true);
      const save = vi.fn().mockResolvedValue("");
      const connect = vi
        .fn()
        .mockResolvedValue({ status: status({ connected: true }) });
      vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
      vi.stubGlobal("window", {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getCodexDesktopStatus: vi.fn().mockResolvedValue(firstUseStatus),
        setCodexDesktopConnected: connect,
        markCodexDesktopIntegrationUsed: save,
        confirm,
      });
      let renderer;
      try {
        await act(async () => {
          renderer = create(
            <CodexDesktopRow
              integration={integration}
              initialStatus={firstUseStatus}
            />,
          );
        });
        await act(async () => {
          renderer!.root.findByProps({ role: "switch" }).props.onClick();
        });
        expect(connect).not.toHaveBeenCalled();
        expect(save).not.toHaveBeenCalled();
        expect(confirm).toHaveBeenCalledTimes(running ? 1 : 0);
        const toggle = renderer!.root.findByProps({ role: "switch" });
        expect(toggle.props["aria-checked"]).toBe(true);
        expect(toggle.props.disabled).toBe(true);
        const intro = renderer!.root.findByType(CodexConnectedIntro);
        await act(async () => {
          intro.props.onDone();
        });
        expect(connect).toHaveBeenCalledOnce();
        expect(connect).toHaveBeenCalledWith(true, running);
        expect(confirm).toHaveBeenCalledTimes(running ? 1 : 0);
        expect(save).toHaveBeenCalledOnce();
        expect(toggle.props.disabled).toBe(false);
        expect(toggle.props["aria-checked"]).toBe(true);
        expect(renderer!.root.findAllByType(CodexConnectedIntro)).toHaveLength(
          0,
        );
      } finally {
        await act(async () => renderer?.unmount());
      }
    },
  );

  it("waits for Continue after detecting a first-time install", async () => {
    const connect = vi.fn();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setInterval: globalThis.setInterval,
      clearInterval: globalThis.clearInterval,
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout,
      getCodexDesktopStatus: vi.fn().mockResolvedValue(status({ used: false })),
      installCodexDesktop: vi.fn().mockResolvedValue("opened"),
      setCodexDesktopConnected: connect,
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={integration}
            initialStatus={status({
              installed: false,
              used: false,
            })}
          />,
        );
      });
      await act(async () => {
        renderer!.root.findByProps({ role: "switch" }).props.onClick();
      });
      expect(connect).not.toHaveBeenCalled();
      expect(renderer!.root.findAllByType(CodexConnectedIntro)).toHaveLength(1);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });
});

it("leaves the Apps page usable when the initial restart is cancelled", async () => {
  const firstUseStatus = status({ running: true, used: false });
  const connect = vi.fn();
  const save = vi.fn();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getCodexDesktopStatus: vi.fn().mockResolvedValue(firstUseStatus),
    setCodexDesktopConnected: connect,
    markCodexDesktopIntegrationUsed: save,
    confirm: vi.fn(() => false),
  });
  let renderer;
  try {
    await act(async () => {
      renderer = create(
        <CodexDesktopRow
          integration={integration}
          initialStatus={firstUseStatus}
        />,
      );
    });
    const toggle = renderer!.root.findByProps({ role: "switch" });
    await act(async () => toggle.props.onClick());
    expect(renderer!.root.findAllByType(CodexConnectedIntro)).toHaveLength(0);
    expect(toggle.props["aria-checked"]).toBe(false);
    expect(toggle.props.disabled).toBe(false);
    expect(connect).not.toHaveBeenCalled();
    expect(save).not.toHaveBeenCalled();
  } finally {
    await act(async () => renderer?.unmount());
  }
});

it.each(["failed", "rejected", "save failed"])(
  "shows errors on the Apps page after Continue (%s)",
  async (outcome) => {
    const firstUseStatus = status({ used: false });
    const connect = vi.fn().mockResolvedValue({
      status: status({ connected: outcome === "save failed", used: false }),
      error: outcome === "failed" ? "launch failed" : undefined,
    });
    if (outcome === "rejected")
      connect.mockRejectedValue(new Error("launch failed"));
    const save = vi.fn().mockResolvedValue("disk error");
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getCodexDesktopStatus: vi.fn().mockResolvedValue(firstUseStatus),
      setCodexDesktopConnected: connect,
      markCodexDesktopIntegrationUsed: save,
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={integration}
            initialStatus={firstUseStatus}
          />,
        );
      });
      const toggle = renderer!.root.findByProps({ role: "switch" });
      await act(async () => {
        toggle.props.onClick();
      });
      expect(toggle.props["aria-checked"]).toBe(true);
      expect(toggle.props.disabled).toBe(true);
      const intro = renderer!.root.findByType(CodexConnectedIntro);
      await act(async () => {
        intro.props.onDone();
      });
      expect(renderer!.root.findAllByType(CodexConnectedIntro)).toHaveLength(0);
      expect(toggle.props["aria-checked"]).toBe(outcome === "save failed");
      expect(toggle.props.disabled).toBe(false);
      expect(renderer!.root.findByProps({ role: "alert" }).children).toContain(
        outcome === "failed"
          ? "launch failed"
          : outcome === "save failed"
            ? "Ollama couldn’t save your progress. Please try again."
            : "Ollama could not add its models to ChatGPT.",
      );
      expect(save).toHaveBeenCalledTimes(outcome === "save failed" ? 1 : 0);
    } finally {
      await act(async () => renderer?.unmount());
    }
  },
);

it("dismisses before launch and prevents duplicate Continue requests", async () => {
  let finishConnect!: (result: { status: CodexDesktopStatus }) => void;
  const connect = vi.fn(
    () =>
      new Promise<{ status: CodexDesktopStatus }>((resolve) => {
        finishConnect = resolve;
      }),
  );
  const save = vi.fn().mockResolvedValue("");
  const firstUseStatus = status({ used: false });
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getCodexDesktopStatus: vi.fn().mockResolvedValue(firstUseStatus),
    setCodexDesktopConnected: connect,
    markCodexDesktopIntegrationUsed: save,
  });
  let renderer;
  try {
    await act(async () => {
      renderer = create(
        <CodexDesktopRow
          integration={integration}
          initialStatus={firstUseStatus}
        />,
      );
    });
    const toggle = renderer!.root.findByProps({ role: "switch" });
    await act(async () => toggle.props.onClick());
    const intro = renderer!.root.findByType(CodexConnectedIntro);
    await act(async () => {
      intro.props.onDone();
      intro.props.onDone();
    });
    expect(renderer!.root.findAllByType(CodexConnectedIntro)).toHaveLength(0);
    expect(connect).toHaveBeenCalledOnce();
    expect(save).not.toHaveBeenCalled();
    expect(toggle.props.disabled).toBe(true);
    await act(async () =>
      finishConnect({ status: status({ connected: true }) }),
    );
    expect(save).toHaveBeenCalledOnce();
    expect(toggle.props.disabled).toBe(false);
  } finally {
    await act(async () => renderer?.unmount());
  }
});

it.each(["cancelled", "status failed"])(
  "closes the intro and restores the switch when Continue is %s",
  async (outcome) => {
    const firstUseStatus = status({ used: false });
    const getStatus = vi.fn().mockResolvedValueOnce(firstUseStatus);
    if (outcome === "cancelled") {
      getStatus.mockResolvedValue(status({ used: false, running: true }));
    } else {
      getStatus.mockRejectedValue(new Error("status unavailable"));
    }
    const connect = vi.fn();
    const save = vi.fn();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getCodexDesktopStatus: getStatus,
      setCodexDesktopConnected: connect,
      markCodexDesktopIntegrationUsed: save,
      confirm: vi.fn(() => false),
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={integration}
            initialStatus={firstUseStatus}
          />,
        );
      });
      await act(async () =>
        renderer!.root.findByProps({ role: "switch" }).props.onClick(),
      );
      await act(async () =>
        renderer!.root.findByType(CodexConnectedIntro).props.onDone(),
      );
      expect(connect).not.toHaveBeenCalled();
      expect(save).not.toHaveBeenCalled();
      expect(renderer!.root.findAllByType(CodexConnectedIntro)).toHaveLength(0);
      const toggle = renderer!.root.findByProps({ role: "switch" });
      expect(toggle.props.disabled).toBe(false);
      expect(toggle.props["aria-checked"]).toBe(false);
    } finally {
      await act(async () => renderer?.unmount());
    }
  },
);

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

it("does not open a restart prompt after leaving the Apps page", async () => {
  const action = deferred<{
    status: CodexDesktopStatus;
    restartConfirmationRequired: boolean;
  }>();
  const connect = vi.fn().mockReturnValue(action.promise);
  const confirm = vi.fn(() => true);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    setCodexDesktopConnected: connect,
    confirm,
  });
  let renderer;
  try {
    await act(async () => {
      renderer = create(
        <CodexDesktopRow integration={integration} initialStatus={status()} />,
      );
    });
    await act(async () =>
      renderer!.root.findByProps({ role: "switch" }).props.onClick(),
    );
    await act(async () => renderer!.unmount());
    await act(async () =>
      action.resolve({
        status: status({ running: true }),
        restartConfirmationRequired: true,
      }),
    );
    expect(confirm).not.toHaveBeenCalled();
    expect(connect).toHaveBeenCalledOnce();
  } finally {
    await act(async () => renderer?.unmount());
  }
});

it("keeps the latest status when focus refreshes complete out of order", async () => {
  const older = deferred<CodexDesktopStatus>();
  const newer = deferred<CodexDesktopStatus>();
  const getStatus = vi
    .fn()
    .mockReturnValueOnce(older.promise)
    .mockReturnValueOnce(newer.promise);
  let onFocus: (() => void) | undefined;
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn((event: string, handler: () => void) => {
      if (event === "focus") onFocus = handler;
    }),
    removeEventListener: vi.fn(),
    getCodexDesktopStatus: getStatus,
  });
  let renderer;
  try {
    await act(async () => {
      renderer = create(
        <CodexDesktopRow integration={integration} initialStatus={status()} />,
      );
    });
    await act(async () => {
      onFocus?.();
      onFocus?.();
    });
    await act(async () => newer.resolve(status({ connected: true })));
    await act(async () => older.resolve(status()));
    expect(
      renderer!.root.findByProps({ role: "switch" }).props["aria-checked"],
    ).toBe(true);
  } finally {
    await act(async () => renderer?.unmount());
  }
});

it("does not start overlapping installers before the switch rerenders", async () => {
  const result = deferred<"cancelled">();
  const install = vi.fn().mockReturnValue(result.promise);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getCodexDesktopStatus: vi.fn(),
    setCodexDesktopConnected: vi.fn(),
    installCodexDesktop: install,
  });
  let renderer;
  try {
    await act(async () => {
      renderer = create(
        <CodexDesktopRow
          integration={integration}
          initialStatus={status({ installed: false })}
        />,
      );
    });
    const toggle = renderer!.root.findByProps({ role: "switch" });
    await act(async () => {
      toggle.props.onClick();
      toggle.props.onClick();
    });
    expect(install).toHaveBeenCalledOnce();
    await act(async () => result.resolve("cancelled"));
    expect(toggle.props.disabled).toBe(false);
    expect(toggle.props["aria-checked"]).toBe(false);
  } finally {
    await act(async () => renderer?.unmount());
  }
});

it("does not retry acknowledgment during a disconnect, or reconnect to save afterward", async () => {
  const firstUseStatus = status({ used: false });
  const disconnect = deferred<{ status: CodexDesktopStatus }>();
  const connect = vi
    .fn()
    .mockReturnValue(disconnect.promise)
    .mockResolvedValueOnce({
      status: status({ used: false, connected: true }),
    });
  const save = vi
    .fn()
    .mockResolvedValueOnce("disk error")
    .mockResolvedValue("");
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getCodexDesktopStatus: vi.fn().mockResolvedValue(firstUseStatus),
    setCodexDesktopConnected: connect,
    markCodexDesktopIntegrationUsed: save,
  });
  let renderer;
  try {
    await act(async () => {
      renderer = create(
        <CodexDesktopRow
          integration={integration}
          initialStatus={firstUseStatus}
        />,
      );
    });
    const toggle = renderer!.root.findByProps({ role: "switch" });
    await act(async () => toggle.props.onClick());
    await act(async () =>
      renderer!.root.findByType(CodexConnectedIntro).props.onDone(),
    );
    const retry = renderer!.root.findByProps({
      "aria-label": "Retry saving progress",
    });
    await act(async () => {
      toggle.props.onClick();
      retry.props.onClick();
    });
    expect(connect).toHaveBeenCalledTimes(2);
    expect(connect).toHaveBeenLastCalledWith(false, false);
    expect(save).toHaveBeenCalledOnce();
    await act(async () => disconnect.resolve({ status: firstUseStatus }));
    await act(async () => retry.props.onClick());
    expect(connect).toHaveBeenCalledTimes(2);
    expect(save).toHaveBeenCalledTimes(2);
    expect(toggle.props["aria-checked"]).toBe(false);
    expect(
      renderer!.root.findAllByProps({ "aria-label": "Retry saving progress" }),
    ).toHaveLength(0);
  } finally {
    await act(async () => renderer?.unmount());
  }
});

it.each(["returned", "rejected"])(
  "retries only acknowledgment after a %s save failure, including overlapping clicks",
  async (failure) => {
    const firstUseStatus = status({ used: false });
    const connectedStatus = status({ used: false, connected: true });
    const connect = vi.fn().mockResolvedValue({ status: connectedStatus });
    const getStatus = vi.fn().mockResolvedValue(firstUseStatus);
    const retry = deferred<string>();
    const save = vi.fn().mockReturnValue(retry.promise);
    if (failure === "returned") save.mockResolvedValueOnce("disk error");
    else save.mockRejectedValueOnce(new Error("disk error"));
    let onFocus: (() => void) | undefined;
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn((event: string, handler: () => void) => {
        if (event === "focus") onFocus = handler;
      }),
      removeEventListener: vi.fn(),
      getCodexDesktopStatus: getStatus,
      setCodexDesktopConnected: connect,
      markCodexDesktopIntegrationUsed: save,
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={integration}
            initialStatus={firstUseStatus}
          />,
        );
      });
      await act(async () =>
        renderer!.root.findByProps({ role: "switch" }).props.onClick(),
      );
      await act(async () =>
        renderer!.root.findByType(CodexConnectedIntro).props.onDone(),
      );
      expect(connect).toHaveBeenCalledOnce();
      expect(save).toHaveBeenCalledOnce();
      expect(renderer!.root.findAllByType(CodexConnectedIntro)).toHaveLength(0);

      getStatus.mockResolvedValue(connectedStatus);
      await act(async () => onFocus?.());
      expect(renderer!.root.findByProps({ role: "alert" }).children).toContain(
        "Ollama couldn’t save your progress. Please try again.",
      );
      const retryButton = renderer!.root.findByProps({
        "aria-label": "Retry saving progress",
      });
      const toggle = renderer!.root.findByProps({ role: "switch" });
      await act(async () => {
        retryButton.props.onClick();
        retryButton.props.onClick();
        toggle.props.onClick();
      });
      expect(connect).toHaveBeenCalledOnce();
      expect(save).toHaveBeenCalledTimes(2);
      expect(retryButton.props.disabled).toBe(true);
      expect(toggle.props.disabled).toBe(true);
      expect(toggle.props["aria-checked"]).toBe(true);
      await act(async () => retry.resolve(""));
      expect(
        renderer!.root.findAllByProps({
          "aria-label": "Retry saving progress",
        }),
      ).toHaveLength(0);
      expect(renderer!.root.findAllByProps({ role: "alert" })).toHaveLength(0);
      expect(toggle.props.disabled).toBe(false);
      expect(toggle.props["aria-checked"]).toBe(true);
      expect(connect).toHaveBeenCalledOnce();
      expect(save).toHaveBeenCalledTimes(2);
    } finally {
      await act(async () => renderer?.unmount());
    }
  },
);

it.each([
  { failure: "returned", timing: "before leaving" },
  { failure: "rejected", timing: "before leaving" },
  { failure: "returned", timing: "while away" },
  { failure: "rejected", timing: "while away" },
  { failure: "returned", timing: "after returning" },
  { failure: "rejected", timing: "after returning" },
])(
  "preserves save-only recovery across navigation ($failure failure $timing)",
  async ({ failure, timing }) => {
    vi.useFakeTimers();
    const firstUseStatus = status({ used: false });
    const connectedStatus = status({ used: false, connected: true });
    const getStatus = vi.fn().mockResolvedValue(firstUseStatus);
    const connect = vi.fn().mockResolvedValue({ status: connectedStatus });
    const firstSave = deferred<string>();
    const retrySave = deferred<string>();
    const save = vi
      .fn()
      .mockReturnValueOnce(firstSave.promise)
      .mockReturnValue(retrySave.promise);
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getCodexDesktopStatus: getStatus,
      setCodexDesktopConnected: connect,
      markCodexDesktopIntegrationUsed: save,
    });
    const failSave = () => {
      if (failure === "returned") firstSave.resolve("disk error");
      else firstSave.reject(new Error("disk error"));
    };
    let renderer;
    try {
      await act(async () => {
        renderer = create(<CodexDesktopRow integration={integration} />);
      });
      await act(async () =>
        renderer!.root.findByProps({ role: "switch" }).props.onClick(),
      );
      await act(async () =>
        renderer!.root.findByType(CodexConnectedIntro).props.onDone(),
      );
      if (timing === "before leaving") await act(async () => failSave());
      await act(async () => renderer!.unmount());
      if (timing === "while away") await act(async () => failSave());
      await act(async () => vi.advanceTimersByTimeAsync(6 * 60_000));
      getStatus.mockResolvedValue(connectedStatus);
      await act(async () => {
        renderer = create(<CodexDesktopRow integration={integration} />);
      });
      const toggle = renderer!.root.findByProps({ role: "switch" });
      if (timing === "after returning") {
        expect(toggle.props.disabled).toBe(true);
        await act(async () => toggle.props.onClick());
        expect(connect).toHaveBeenCalledOnce();
        expect(save).toHaveBeenCalledOnce();
        await act(async () => failSave());
      }
      expect(toggle.props.disabled).toBe(false);
      const retry = renderer!.root.findByProps({
        "aria-label": "Retry saving progress",
      });
      await act(async () => {
        retry.props.onClick();
        retry.props.onClick();
        toggle.props.onClick();
      });
      expect(connect).toHaveBeenCalledOnce();
      expect(save).toHaveBeenCalledTimes(2);
      expect(toggle.props.disabled).toBe(true);
      await act(async () => renderer!.unmount());
      await act(async () => {
        renderer = create(<CodexDesktopRow integration={integration} />);
      });
      expect(
        renderer!.root.findByProps({ role: "switch" }).props.disabled,
      ).toBe(true);
      await act(async () =>
        renderer!.root
          .findByProps({ "aria-label": "Retry saving progress" })
          .props.onClick(),
      );
      expect(save).toHaveBeenCalledTimes(2);
      await act(async () => retrySave.resolve(""));
      expect(
        renderer!.root.findAllByProps({
          "aria-label": "Retry saving progress",
        }),
      ).toHaveLength(0);
      expect(renderer!.root.findAllByProps({ role: "alert" })).toHaveLength(0);
      expect(
        renderer!.root.findByProps({ role: "switch" }).props.disabled,
      ).toBe(false);
      expect(
        renderer!.root.findByProps({ role: "switch" }).props["aria-checked"],
      ).toBe(true);
      expect(connect).toHaveBeenCalledOnce();
      expect(save).toHaveBeenCalledTimes(2);
    } finally {
      await act(async () => renderer?.unmount());
    }
  },
);

it("refreshes connection status when Retry overtakes the returning page's status check", async () => {
  const firstUseStatus = status({ used: false });
  const connectedStatus = status({ used: false, connected: true });
  const stale = deferred<CodexDesktopStatus>();
  const retrySave = deferred<string>();
  const getStatus = vi.fn().mockResolvedValue(firstUseStatus);
  const connect = vi.fn().mockResolvedValue({ status: connectedStatus });
  const save = vi
    .fn()
    .mockResolvedValueOnce("disk error")
    .mockReturnValue(retrySave.promise);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getCodexDesktopStatus: getStatus,
    setCodexDesktopConnected: connect,
    markCodexDesktopIntegrationUsed: save,
  });
  let renderer;
  try {
    await act(async () => {
      renderer = create(<CodexDesktopRow integration={integration} />);
    });
    await act(async () =>
      renderer!.root.findByProps({ role: "switch" }).props.onClick(),
    );
    await act(async () =>
      renderer!.root.findByType(CodexConnectedIntro).props.onDone(),
    );
    await act(async () => renderer!.unmount());
    getStatus
      .mockReturnValueOnce(stale.promise)
      .mockResolvedValue(connectedStatus);
    await act(async () => {
      renderer = create(<CodexDesktopRow integration={integration} />);
    });
    await act(async () =>
      renderer!.root
        .findByProps({ "aria-label": "Retry saving progress" })
        .props.onClick(),
    );
    await act(async () => stale.resolve(connectedStatus));
    await act(async () => retrySave.resolve(""));
    expect(
      renderer!.root.findByProps({ role: "switch" }).props["aria-checked"],
    ).toBe(true);
    expect(renderer!.root.findByProps({ role: "switch" }).props.disabled).toBe(
      false,
    );
    expect(connect).toHaveBeenCalledOnce();
    expect(save).toHaveBeenCalledTimes(2);
  } finally {
    await act(async () => renderer?.unmount());
  }
});

it("observes a successful pending save after returning without saving again", async () => {
  const firstUseStatus = status({ used: false });
  const connectedStatus = status({ used: false, connected: true });
  const pendingSave = deferred<string>();
  const getStatus = vi.fn().mockResolvedValue(firstUseStatus);
  const connect = vi.fn().mockResolvedValue({ status: connectedStatus });
  const save = vi.fn().mockReturnValue(pendingSave.promise);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getCodexDesktopStatus: getStatus,
    setCodexDesktopConnected: connect,
    markCodexDesktopIntegrationUsed: save,
  });
  let renderer;
  try {
    await act(async () => {
      renderer = create(<CodexDesktopRow integration={integration} />);
    });
    await act(async () =>
      renderer!.root.findByProps({ role: "switch" }).props.onClick(),
    );
    await act(async () =>
      renderer!.root.findByType(CodexConnectedIntro).props.onDone(),
    );
    await act(async () => renderer!.unmount());
    getStatus.mockResolvedValue(connectedStatus);
    await act(async () => {
      renderer = create(<CodexDesktopRow integration={integration} />);
    });
    expect(renderer!.root.findByProps({ role: "switch" }).props.disabled).toBe(
      true,
    );
    await act(async () => pendingSave.resolve(""));
    expect(
      renderer!.root.findAllByProps({ "aria-label": "Retry saving progress" }),
    ).toHaveLength(0);
    expect(renderer!.root.findAllByType(CodexConnectedIntro)).toHaveLength(0);
    expect(renderer!.root.findByProps({ role: "switch" }).props.disabled).toBe(
      false,
    );
    expect(connect).toHaveBeenCalledOnce();
    expect(save).toHaveBeenCalledOnce();
  } finally {
    await act(async () => renderer?.unmount());
  }
});

it.each([false, true])(
  "does not infer acknowledgment from connection status (connected: %s)",
  async (connected) => {
    const save = vi.fn();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getCodexDesktopStatus: vi
        .fn()
        .mockResolvedValue(status({ used: false, connected })),
      markCodexDesktopIntegrationUsed: save,
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(<CodexDesktopRow integration={integration} />);
      });
      expect(save).not.toHaveBeenCalled();
      expect(
        renderer!.root.findAllByProps({
          "aria-label": "Retry saving progress",
        }),
      ).toHaveLength(0);
    } finally {
      await act(async () => renderer?.unmount());
    }
  },
);

it.each(["resolved", "rejected"])(
  "ignores a stale focus refresh that is %s after the connection and save failure",
  async (outcome) => {
    const firstUseStatus = status({ used: false });
    const stale = deferred<CodexDesktopStatus>();
    const getStatus = vi
      .fn()
      .mockReturnValueOnce(stale.promise)
      .mockResolvedValue(firstUseStatus);
    const connect = vi.fn().mockResolvedValue({
      status: status({ used: false, connected: true }),
    });
    const save = vi.fn().mockResolvedValue("disk error");
    let onFocus: (() => void) | undefined;
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn((event: string, handler: () => void) => {
        if (event === "focus") onFocus = handler;
      }),
      removeEventListener: vi.fn(),
      getCodexDesktopStatus: getStatus,
      setCodexDesktopConnected: connect,
      markCodexDesktopIntegrationUsed: save,
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopRow
            integration={integration}
            initialStatus={firstUseStatus}
          />,
        );
      });
      await act(async () => onFocus?.());
      await act(async () =>
        renderer!.root.findByProps({ role: "switch" }).props.onClick(),
      );
      await act(async () =>
        renderer!.root.findByType(CodexConnectedIntro).props.onDone(),
      );
      await act(async () => {
        if (outcome === "resolved") stale.resolve(firstUseStatus);
        else stale.reject(new Error("stale status failure"));
      });
      expect(
        renderer!.root.findByProps({ role: "switch" }).props["aria-checked"],
      ).toBe(true);
      expect(renderer!.root.findByProps({ role: "alert" }).children).toContain(
        "Ollama couldn’t save your progress. Please try again.",
      );
      expect(
        renderer!.root.findByProps({ "aria-label": "Retry saving progress" })
          .props.disabled,
      ).toBe(false);
    } finally {
      await act(async () => renderer?.unmount());
    }
  },
);
