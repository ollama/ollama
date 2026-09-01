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

  it("disables the toggle when ChatGPT is not installed", () => {
    const html = renderToStaticMarkup(
      <CodexDesktopRow
        integration={{ ...integration, installed: false }}
        initialStatus={status({ installed: false })}
      />,
    );

    expect(html).toContain(
      "Install ChatGPT to use Ollama models in the Codex app.",
    );
    expect(html).toContain('disabled=""');
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
