import type { ClaudeDesktopStatus } from "@/types/webview";
import { claudeDesktopModelStatusLabel } from "@/lib/claudeDesktopModelStatus";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ClaudeDesktopModelsSettings } from "./ClaudeDesktopModelsSettings";

const routes = [
  { routeId: "claude-fable-5", routeName: "Fable 5" },
  { routeId: "claude-opus-5", routeName: "Opus 5" },
  { routeId: "claude-sonnet-5", routeName: "Sonnet 5" },
  {
    routeId: "claude-haiku-4-5-20251001",
    routeName: "Haiku 4.5",
  },
  { routeId: "claude-sonnet-4-6", routeName: "Sonnet 4.6" },
];

function status(
  overrides: Partial<ClaudeDesktopStatus> = {},
): ClaudeDesktopStatus {
  return {
    supported: true,
    used: true,
    installed: true,
    configured: true,
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
        availability: "available",
      },
      {
        name: "qwen3:8b",
        displayName: "qwen3:8b",
        selected: true,
        availability: "available",
      },
    ],
    mappings: routes.map((route, index) => ({
      ...route,
      model: index === 0 ? "glm-5.2:cloud" : undefined,
    })),
    ...overrides,
  };
}

describe("ClaudeDesktopModelsSettings", () => {
  it("labels model plan and account requirements in the picker", () => {
    expect(
      claudeDesktopModelStatusLabel({
        name: "gemma4:31b-cloud",
        displayName: "gemma4:31b-cloud",
        cloud: true,
        selected: false,
        requiredPlan: "free",
      }),
    ).toBeNull();
    expect(
      claudeDesktopModelStatusLabel({
        name: "glm-5.2:cloud",
        displayName: "glm-5.2:cloud",
        cloud: true,
        selected: false,
        availability: "unavailable",
        reason: "upgrade_required",
        requiredPlan: "pro",
      }),
    ).toBe("Pro plan required");
    expect(
      claudeDesktopModelStatusLabel({
        name: "gemma4:31b-cloud",
        displayName: "gemma4:31b-cloud",
        cloud: true,
        selected: false,
        availability: "unavailable",
        reason: "sign_in_required",
        requiredPlan: "free",
      }),
    ).toBe("Sign in required");
  });

  it("renders the five explicit Claude routes and an Ollama model picker", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings initialStatus={status()} />,
    );

    expect(html).toContain(">Claude</h2>");
    for (const route of routes) {
      expect(html).toContain(route.routeName);
      expect(html).not.toContain(`>${route.routeId}<`);
    }
    expect((html.match(/aria-haspopup="listbox"/g) ?? []).length).toBe(5);
    expect(html).not.toContain('for="claude-route-');
    expect(html).toContain(
      "Choose which Ollama model Claude uses for each model option.",
    );
    expect(html).not.toContain("routing");
    expect(html).not.toContain("Built-in defaults");
    expect(html).not.toContain("Unassigned");
    expect(html).toContain("Select a model");
    expect(html).toContain("Start Claude");
  });

  it("allows the same Ollama model to be assigned to multiple routes", () => {
    const shared = routes.map((route) => ({
      ...route,
      model: "qwen3:8b",
    }));
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        initialStatus={status({ mappings: shared })}
      />,
    );

    expect((html.match(/>qwen3:8b<\/span>/g) ?? []).length).toBe(5);
  });

  it("keeps an unavailable default visible with its access status", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        initialStatus={status({
          models: [
            {
              name: "glm-5.2:cloud",
              displayName: "glm-5.2:cloud",
              cloud: true,
              selected: true,
              availability: "unavailable",
              reason: "upgrade_required",
              requiredPlan: "pro",
            },
            {
              name: "qwen3:8b",
              displayName: "qwen3:8b",
              selected: false,
              availability: "available",
            },
          ],
        })}
      />,
    );

    expect(html).toContain(">glm-5.2:cloud</span>");
  });

  it("presents Start or Restart based on whether Claude is running", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings
        initialStatus={status({ configured: false, connected: false })}
      />,
    );

    expect(html).toContain("Start Claude");
    expect(html).not.toContain("Apply changes");

    const runningHTML = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings initialStatus={status({ running: true })} />,
    );
    expect(runningHTML).toContain("Restart Claude");
    expect(runningHTML).toContain("disabled");
  });

  it("stays hidden until Claude has been enabled once", () => {
    const html = renderToStaticMarkup(
      <ClaudeDesktopModelsSettings initialStatus={status({ used: false })} />,
    );

    expect(html).toBe("");
  });
});
