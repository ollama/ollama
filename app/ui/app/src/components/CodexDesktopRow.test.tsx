import type { IntegrationStatus } from "@/api";
import type { CodexDesktopStatus } from "@/types/webview";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CodexDesktopRow } from "./CodexDesktopRow";

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
    expect(html).toContain("Use Ollama models in ChatGPT");
    expect(html).toContain('aria-label="Use Ollama models in ChatGPT"');
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

    expect(html).toContain("Using Ollama · 3 models · 0 requests this session");
    expect(html).toContain('aria-label="Restore OpenAI models in ChatGPT"');
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

    expect(html).toContain("Using Ollama · 1 model · 0 requests this session");
  });

  it("shows the ChatGPT request count with singular copy", () => {
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

    expect(html).toContain("Using Ollama · 1 model · 1 request this session");
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

  it("allows the normal profile to be restored if ChatGPT is removed", () => {
    const html = renderToStaticMarkup(
      <CodexDesktopRow
        integration={{ ...integration, installed: false }}
        initialStatus={status({ installed: false, connected: true })}
      />,
    );

    expect(html).toContain('aria-label="Restore OpenAI models in ChatGPT"');
    expect(html).not.toContain('disabled=""');
  });
});
