import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ClaudeDesktopModelsSettings } from "./ClaudeDesktopModelsSettings";

describe("ClaudeDesktopModelsSettings", () => {
  it("shows Claude recommendations and an installed-model search in Settings", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        initialLocalModels={["llama3.2", "qwen3:8b"]}
        includeCloudModels
        initialStatus={{
          supported: true,
          used: true,
          installed: true,
          connected: true,
          running: false,
          startFailed: false,
          portConflict: false,
          modelSource: "endpoint",
          models: [
            {
              name: "glm-5.2:cloud",
              displayName: "glm-5.2:cloud",
              cloud: true,
              selected: true,
            },
            {
              name: "deepseek-v4-flash:cloud",
              displayName: "deepseek-v4-flash:cloud",
              cloud: true,
              selected: false,
            },
          ],
        }}
      />,
    );

    expect(html).toContain(">Claude<");
    expect(html).toContain(">Apps<");
    expect(html).toContain('id="apps-settings-heading"');
    expect(html).toContain('src="/launch-icons/claude.svg"');
    expect(html).not.toContain("Models in Claude");
    expect(html).toContain("glm-5.2:cloud");
    expect(html).toContain("deepseek-v4-flash:cloud");
    expect(html).toContain("Search Ollama models");
    expect(html).not.toContain("Add any Ollama model");
    expect(html).toContain("Restart Claude");
  });

  it("does not show the invalid Ollama Cloud sentinel", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        includeCloudModels
        initialStatus={{
          supported: true,
          used: true,
          installed: true,
          connected: true,
          running: false,
          startFailed: false,
          portConflict: false,
          modelSource: "user",
          models: [
            {
              name: "Ollama Cloud",
              displayName: "Ollama Cloud",
              selected: true,
            },
            {
              name: "qwen3:8b",
              displayName: "qwen3:8b",
              selected: true,
            },
          ],
        }}
      />,
    );

    expect(html).not.toContain("Ollama Cloud");
    expect(html).toContain("qwen3:8b");
  });

  it("labels the built-in fallback without exposing MLX", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        includeCloudModels
        initialStatus={{
          supported: true,
          used: true,
          installed: true,
          connected: true,
          running: false,
          startFailed: false,
          portConflict: false,
          modelSource: "fallback",
          models: [
            {
              name: "deepseek-v4-flash:0731:cloud",
              displayName: "deepseek-v4-flash:0731:cloud",
              cloud: true,
              selected: true,
            },
          ],
        }}
      />,
    );

    expect(html).toContain("Built-in defaults");
    expect(html).toContain("deepseek-v4-flash:0731:cloud");
    expect(html).not.toContain("MLX");
  });

  it("prevents a sixth selection when the literal Claude slots are full", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        includeCloudModels
        initialStatus={{
          supported: true,
          used: true,
          installed: true,
          connected: true,
          running: false,
          startFailed: false,
          portConflict: false,
          modelSource: "endpoint",
          maxModels: 5,
          models: [
            {
              name: "glm-5.2:cloud",
              displayName: "glm-5.2:cloud",
              cloud: true,
              selected: true,
            },
            {
              name: "kimi-k3:cloud",
              displayName: "kimi-k3:cloud",
              cloud: true,
              selected: true,
            },
            {
              name: "deepseek-v4-pro:cloud",
              displayName: "deepseek-v4-pro:cloud",
              cloud: true,
              selected: true,
            },
            {
              name: "deepseek-v4-flash:cloud",
              displayName: "deepseek-v4-flash:cloud",
              cloud: true,
              selected: true,
            },
            {
              name: "gemma4:26b:cloud",
              displayName: "gemma4:26b:cloud",
              cloud: true,
              selected: true,
            },
            {
              name: "qwen3:8b",
              displayName: "qwen3:8b",
              selected: false,
            },
          ],
        }}
      />,
    );

    const index = html.indexOf(">qwen3:8b</span>");
    expect(index).toBeGreaterThan(-1);
    const label = html.slice(html.lastIndexOf("<label", index), index);
    expect(label).toContain("disabled");
    expect((html.match(/disabled=""/g) ?? []).length).toBe(1);
    expect(html).toContain(
      "Claude supports up to 5 models. Deselect one to add another.",
    );
  });

  it("honors a smaller maxModels limit from the status", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        includeCloudModels
        initialStatus={{
          supported: true,
          used: true,
          installed: true,
          connected: true,
          running: false,
          startFailed: false,
          portConflict: false,
          modelSource: "endpoint",
          maxModels: 1,
          models: [
            {
              name: "kimi-k3:cloud",
              displayName: "kimi-k3:cloud",
              cloud: true,
              selected: true,
            },
            {
              name: "qwen3:8b",
              displayName: "qwen3:8b",
              selected: false,
            },
          ],
        }}
      />,
    );

    const index = html.indexOf(">qwen3:8b</span>");
    expect(index).toBeGreaterThan(-1);
    const label = html.slice(html.lastIndexOf("<label", index), index);
    expect(label).toContain("disabled");
  });

  it("hides recommendation and selected cloud models when cloud is disabled", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        initialLocalModels={["qwen3:8b"]}
        initialStatus={{
          supported: true,
          used: true,
          installed: true,
          connected: false,
          running: false,
          startFailed: false,
          portConflict: false,
          modelSource: "endpoint",
          models: [
            {
              name: "glm-5.2:cloud",
              displayName: "glm-5.2:cloud",
              cloud: true,
              selected: true,
            },
          ],
        }}
      />,
    );

    expect(html).not.toContain("glm-5.2:cloud");
    expect(html).toContain("Search Ollama models");
  });

  it("remains visible after Claude has been disconnected", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        initialStatus={{
          supported: true,
          used: true,
          installed: true,
          connected: false,
          running: false,
          startFailed: false,
          portConflict: false,
          modelSource: "user",
          models: [
            {
              name: "qwen3:8b",
              displayName: "qwen3:8b",
              selected: true,
            },
          ],
        }}
      />,
    );

    expect(html).toContain(">Claude<");
    expect(html).toContain("qwen3:8b");
    expect(html).toContain(
      "These models will be available when Claude starts.",
    );
    expect(html).toContain("Start Claude");
  });

  it("stays hidden until Claude has been enabled once", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        initialStatus={{
          supported: true,
          used: false,
          installed: true,
          connected: false,
          running: false,
          startFailed: false,
          portConflict: false,
        }}
      />,
    );

    expect(html).toBe("");
  });
});
