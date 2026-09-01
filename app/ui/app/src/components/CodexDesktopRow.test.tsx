import type { IntegrationStatus } from "@/api";
import type { CodexDesktopStatus } from "@/types/webview";
import { renderToStaticMarkup } from "react-dom/server";
import { act, create } from "react-test-renderer";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CodexDesktopRow } from "./CodexDesktopRow";

afterEach(() => {
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

    expect(html).toContain(">ChatGPT</p>");
    expect(html).toContain("Codex models · Ollama models not added");
    expect(html).toContain('aria-label="Add Ollama models to ChatGPT"');
    expect(html).toContain('aria-checked="false"');
  });

  it("shows how many Ollama models are available in ChatGPT", () => {
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

    expect(html).toContain(
      "Codex + Ollama · 3 Ollama models · 0 Ollama requests this session",
    );
    expect(html).toContain('aria-label="Remove Ollama models from ChatGPT"');
    expect(html).toContain('aria-checked="true"');
  });

  it("uses singular copy for one Ollama model", () => {
    const html = renderToStaticMarkup(
      <CodexDesktopRow
        integration={integration}
        initialStatus={status({
          connected: true,
          model: "qwen3:8b",
        })}
      />,
    );

    expect(html).toContain(
      "Codex + Ollama · 1 Ollama model · 0 Ollama requests this session",
    );
  });

  it("shows the Ollama request count with singular copy", () => {
    const html = renderToStaticMarkup(
      <CodexDesktopRow
        integration={integration}
        initialStatus={status({
          connected: true,
          model: "qwen3:8b",
          requests: 1,
        })}
      />,
    );

    expect(html).toContain(
      "Codex + Ollama · 1 Ollama model · 1 Ollama request this session",
    );
  });

  it("offers installation when ChatGPT is not installed", () => {
    const html = renderToStaticMarkup(
      <CodexDesktopRow
        integration={{ ...integration, installed: false }}
        initialStatus={status({ installed: false })}
      />,
    );

    expect(html).toContain(
      "Install ChatGPT to use Ollama models in the Codex app.",
    );
    expect(html).not.toContain('disabled=""');
    expect(html).toContain('title="Install ChatGPT and add Ollama models"');
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
      expect(connect).toHaveBeenCalledWith(true);
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

  it("does not restart ChatGPT automatically when installation detection finds it running", async () => {
    const installedAndRunning = status({ installed: true, running: true });
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

      expect(connect).not.toHaveBeenCalled();
      expect(renderer!.root.findByProps({ role: "alert" }).children).toContain(
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
  });

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

      expect(restore).toHaveBeenCalledWith(false);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });
});
