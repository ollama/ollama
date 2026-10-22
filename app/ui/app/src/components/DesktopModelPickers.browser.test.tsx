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
import type { ClaudeDesktopStatus } from "@/types/webview";
import { Model } from "@/gotypes";

vi.mock("@/api", () => ({
  getIntegrationStatuses: vi.fn().mockResolvedValue([]),
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

it.each([false, true])(
  "recovers first-use model setup with a visible cloud default: %s",
  async (hasCloudDefault) => {
    const previousURL = window.location.href;
    const previousApply = window.applyClaudeDesktopMappings;
    window.history.replaceState(null, "", "#claude-settings");
    const unavailableCloud = {
      name: "gemma4:31b-cloud",
      displayName: "gemma4:31b-cloud",
      cloud: true,
      selected: true,
      availability: "unavailable" as const,
      reason: "sign_in_required" as const,
    };
    const empty: ClaudeDesktopStatus = {
      supported: true,
      used: false,
      installed: true,
      configured: false,
      connected: false,
      running: false,
      startFailed: true,
      portConflict: false,
      modelSelectionRequired: true,
      mappings: [
        {
          routeId: "claude-sonnet-5",
          routeName: "Sonnet 5",
          model: hasCloudDefault ? unavailableCloud.name : undefined,
        },
      ],
      models: hasCloudDefault ? [unavailableCloud] : [],
    };
    let finishSummary!: (status: ClaudeDesktopStatus) => void;
    const summary = new Promise<ClaudeDesktopStatus>(
      (resolve) => (finishSummary = resolve),
    );
    vi.mocked(getClaudeDesktopModelsSettings).mockImplementation((catalog) =>
      catalog ? Promise.resolve(empty) : summary,
    );
    const open = vi.spyOn(window, "open").mockReturnValue(null);
    try {
      await act(async () => root.render(<ClaudeDesktopModelsSettings />));
      await act(async () => finishSummary(empty));
      await expect
        .element(
          page.getByRole("button", { name: "Browse models", exact: true }),
        )
        .toBeVisible();
      expect(document.activeElement?.id).toBe("claude-settings");
      expect(
        vi
          .mocked(getClaudeDesktopModelsSettings)
          .mock.calls.some(([catalog]) => catalog),
      ).toBe(true);
      await page
        .getByRole("button", { name: "Browse models", exact: true })
        .click();
      expect(open).toHaveBeenCalledExactlyOnceWith(
        "https://ollama.com/search?c=tools",
        "_blank",
      );
      await expect
        .element(
          page.getByRole("button", { name: "Start Claude", exact: true }),
        )
        .toBeDisabled();

      const available: ClaudeDesktopStatus = {
        ...empty,
        models: [
          ...(empty.models ?? []),
          {
            name: "qwen:8b",
            displayName: "qwen:8b",
            selected: false,
            availability: "available",
          },
        ],
      };
      vi.mocked(getClaudeDesktopAvailableModels).mockResolvedValue([
        new Model({ model: "qwen:8b" }),
      ]);
      vi.mocked(getClaudeDesktopModelsSettings).mockResolvedValue(available);
      await act(async () => window.dispatchEvent(new Event("focus")));
      await act(async () => {
        await page
          .getByRole("button", { name: "Ollama model for Sonnet 5" })
          .click();
      });
      await act(async () => {
        await page
          .getByRole("option", { name: "qwen:8b", exact: true })
          .click();
      });
      const connected: ClaudeDesktopStatus = {
        ...available,
        used: true,
        configured: true,
        connected: true,
        startFailed: false,
        modelSelectionRequired: false,
        mappings: [
          {
            routeId: "claude-sonnet-5",
            routeName: "Sonnet 5",
            model: "qwen:8b",
          },
        ],
      };
      const apply = vi
        .fn()
        .mockResolvedValueOnce({
          status: {
            ...empty,
            modelSelectionRequired: false,
            startFailed: false,
            models: [],
            mappings: [],
          },
          error: "Claude settings could not be saved.",
        })
        .mockImplementation(async () => {
          vi.mocked(getClaudeDesktopModelsSettings).mockResolvedValue(
            connected,
          );
          return { status: connected, mappingsApplied: true };
        });
      window.applyClaudeDesktopMappings = apply;
      await act(async () => {
        await page
          .getByRole("button", { name: "Start Claude", exact: true })
          .click();
      });
      await expect.element(page.getByRole("dialog")).toBeVisible();
      expect(apply).not.toHaveBeenCalled();
      await act(async () => {
        await page
          .getByRole("button", { name: "Continue", exact: true })
          .click();
      });
      expect(apply).toHaveBeenCalledExactlyOnceWith(
        { "claude-sonnet-5": "qwen:8b" },
        false,
      );
      await expect
        .element(page.getByRole("alert"))
        .toHaveTextContent("Claude settings could not be saved.");
      await expect
        .element(page.getByRole("heading", { name: "Claude", exact: true }))
        .toBeVisible();
      await act(async () => window.dispatchEvent(new Event("focus")));
      await act(async () => {
        await page
          .getByRole("button", { name: "Ollama model for Sonnet 5" })
          .click();
      });
      await act(async () => {
        await page
          .getByRole("option", { name: "qwen:8b", exact: true })
          .click();
      });
      await act(async () => {
        await page
          .getByRole("button", { name: "Start Claude", exact: true })
          .click();
      });
      await expect.element(page.getByRole("dialog")).toBeVisible();
      await act(async () => {
        await page
          .getByRole("button", { name: "Continue", exact: true })
          .click();
      });
      expect(apply).toHaveBeenCalledTimes(2);
      await expect
        .element(
          page.getByRole("button", { name: "Ollama model for Sonnet 5" }),
        )
        .toHaveTextContent("qwen:8b");
    } finally {
      finishSummary(empty);
      window.applyClaudeDesktopMappings = previousApply;
      open.mockRestore();
      window.history.replaceState(null, "", previousURL);
    }
  },
);
