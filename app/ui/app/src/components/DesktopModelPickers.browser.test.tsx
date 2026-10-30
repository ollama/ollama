import {
  getClaudeDesktopAvailableModels,
  getClaudeDesktopModelsSettings,
  getCodexDesktopModelsSettings,
} from "@/api";
import { queryClient } from "@/lib/queryClient";
import { page, userEvent } from "@vitest/browser/context";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ClaudeDesktopModelsSettings } from "./ClaudeDesktopModelsSettings";
import { CodexDesktopModelsSettings } from "./CodexDesktopModelsSettings";

vi.mock("@/api", () => ({
  getClaudeDesktopAvailableModels: vi.fn(),
  getClaudeDesktopModelsSettings: vi.fn(),
  getCodexDesktopModelsSettings: vi.fn(),
}));

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  const saved = {
    name: "saved-model",
    displayName: "saved-model",
    selected: true,
    availability: "unknown" as const,
  };
  const choices = [
    { ...saved, availability: "available" as const },
    {
      name: "another-model",
      displayName: "another-model",
      selected: false,
      availability: "available" as const,
    },
  ];
  vi.mocked(getCodexDesktopModelsSettings).mockImplementation(
    async (catalog) => ({
      settings: {
        supported: true,
        installed: true,
        connected: true,
        running: false,
        usesDefaults: false,
        maxModels: 5,
        selected: [saved.name],
        available: catalog ? choices.map((model) => model.name) : [],
        models: catalog ? choices : [saved],
      },
    }),
  );
  vi.mocked(getClaudeDesktopModelsSettings).mockImplementation(
    async (catalog) => ({
      supported: true,
      installed: true,
      used: true,
      connected: true,
      running: false,
      startFailed: false,
      portConflict: false,
      autoMode: false,
      mappings: [
        { routeId: "claude-fable-5", routeName: "Fable 5", model: saved.name },
      ],
      models: catalog ? choices : [saved],
    }),
  );
  vi.mocked(getClaudeDesktopAvailableModels).mockResolvedValue([]);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  queryClient.clear();
  vi.unstubAllGlobals();
});

it.each([
  ["ChatGPT", "{Enter}"],
  ["ChatGPT", "[Space]"],
  ["Claude", "{Enter}"],
  ["Claude", "[Space]"],
])("loads the real %s picker when opened with %s", async (integration, key) => {
  await act(async () =>
    root.render(
      integration === "ChatGPT" ? (
        <CodexDesktopModelsSettings />
      ) : (
        <ClaudeDesktopModelsSettings />
      ),
    ),
  );
  const read =
    integration === "ChatGPT"
      ? vi.mocked(getCodexDesktopModelsSettings)
      : vi.mocked(getClaudeDesktopModelsSettings);
  const button = page.getByRole("button", {
    name:
      integration === "ChatGPT"
        ? "Add ChatGPT model"
        : "Ollama model for Fable 5",
  });
  await expect.element(button).toBeEnabled();
  expect(read.mock.calls.some(([catalog]) => catalog)).toBe(false);
  await act(async () => {
    (button.element() as HTMLButtonElement).focus();
    await userEvent.keyboard(key);
  });
  await expect
    .element(page.getByRole("option", { name: "another-model", exact: true }))
    .toBeEnabled();
  expect(read.mock.calls.filter(([catalog]) => catalog)).toHaveLength(1);
});
