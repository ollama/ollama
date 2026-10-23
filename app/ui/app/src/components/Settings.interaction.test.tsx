import { act, create, type ReactTestInstance } from "react-test-renderer";
import { forwardRef, useImperativeHandle } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Settings as SettingsType } from "@/gotypes";
import { Badge } from "./ui/badge";
import { Slider } from "./ui/slider";
import Settings from "./Settings";

const mocks = vi.hoisted(() => ({
  resetClaudeMappings: vi.fn(),
  resetChatGPTModels: vi.fn(),
  updateSettings: vi.fn(),
  updateCloudSetting: vi.fn(),
  setShowAppsInMenu: vi.fn(),
  refetchUser: vi.fn(),
  disconnectUser: vi.fn(),
  isWindows: false,
  queryClient: {
    cancelQueries: vi.fn().mockResolvedValue(undefined),
    getQueryData: vi.fn(),
    setQueryData: vi.fn(),
    invalidateQueries: vi.fn(),
  },
  settings: null as SettingsType | null,
  runningModels: [] as { name: string; context_length: number }[],
  modelContextInfo: {} as { numCtx?: number; trainCtx?: number },
  modelMTP: null as { supported: boolean; draftNumPredict?: number } | null,
  setModelDraftNumPredict: vi.fn(),
  unloadModel: vi.fn(),
  loadModel: vi.fn(),
}));

vi.mock("@/components/ClaudeDesktopModelsSettings", () => ({
  ClaudeDesktopModelsSettings: forwardRef(
    function MockClaudeDesktopSettings(_props, ref) {
      useImperativeHandle(ref, () => ({
        resetToDefaults: mocks.resetClaudeMappings,
      }));
      return <section aria-label="Claude settings" />;
    },
  ),
}));

vi.mock("@/components/CodexDesktopModelsSettings", () => ({
  CodexDesktopModelsSettings: forwardRef(
    function MockCodexDesktopSettings(_props, ref) {
      useImperativeHandle(ref, () => ({
        resetToDefaults: mocks.resetChatGPTModels,
      }));
      return <section aria-label="ChatGPT settings" />;
    },
  ),
}));

vi.mock("@/hooks/useUser", () => ({
  useUser: () => ({
    user: {
      id: "paid-user-id",
      name: "Paid user",
      email: "paid@example.com",
      plan: "pro",
    },
    isAuthenticated: true,
    refreshUser: vi.fn(),
    isRefreshing: false,
    refetchUser: mocks.refetchUser,
    fetchConnectUrl: vi.fn(),
    isLoading: false,
    disconnectUser: mocks.disconnectUser,
  }),
}));

vi.mock("@/hooks/useCloudStatus", () => ({
  useCloudStatus: () => ({
    cloudDisabled: false,
    cloudStatus: { disabled: false, source: "none" },
    isKnown: true,
  }),
}));

vi.mock("@/lib/platform", () => ({
  isWindowsPlatform: () => mocks.isWindows,
}));

vi.mock("@tanstack/react-router", () => ({
  useBlocker: vi.fn(),
}));

vi.mock("@tanstack/react-query", async (importOriginal) => ({
  QueryClient: (await importOriginal<typeof import("@tanstack/react-query")>())
    .QueryClient,
  useQueryClient: () => mocks.queryClient,
  useQuery: ({ queryKey }: { queryKey: string[] }) => {
    if (queryKey[0] === "settings") {
      return {
        data: { settings: mocks.settings },
        isLoading: false,
        error: null,
      };
    }
    if (queryKey[0] === "runningModels") {
      return { data: mocks.runningModels };
    }
    if (queryKey[0] === "modelContextInfo") {
      return { data: mocks.modelContextInfo };
    }
    if (queryKey[0] === "modelMTP") {
      return { data: mocks.modelMTP, isLoading: false };
    }
    return { data: { defaultContextLength: 65_536 } };
  },
  useMutation: ({
    mutationFn,
    onMutate,
    onSuccess,
    onError,
    onSettled,
  }: {
    mutationFn: (value: unknown) => Promise<unknown>;
    onMutate?: (value: unknown) => Promise<unknown>;
    onSuccess?: (result: unknown, value: unknown, context: unknown) => void;
    onError?: (error: unknown, value: unknown, context: unknown) => void;
    onSettled?: (
      result: unknown,
      error: unknown,
      value: unknown,
      context: unknown,
    ) => void;
  }) => {
    const run = async (
      value: unknown,
      callbacks?: { onSuccess?: () => void },
    ) => {
      const context = await onMutate?.(value);
      try {
        const result = await mutationFn(value);
        onSuccess?.(result, value, context);
        callbacks?.onSuccess?.();
        onSettled?.(result, null, value, context);
        return result;
      } catch (error) {
        onError?.(error, value, context);
        onSettled?.(undefined, error, value, context);
        throw error;
      }
    };

    return {
      mutate: (value: unknown, callbacks?: { onSuccess?: () => void }) => {
        // react-query reports failures through onError. mutate itself does
        // not reject, so a rejected mutation must not escape as an unhandled
        // rejection.
        void run(value, callbacks).catch(() => {});
      },
      mutateAsync: (value: unknown) => run(value),
    };
  },
}));

vi.mock("@/api", () => ({
  getSettings: vi.fn(),
  getInferenceCompute: vi.fn(),
  getModelContextInfo: vi.fn(),
  getModelMTPInfo: vi.fn(),
  listRunningModels: vi.fn(),
  loadModel: mocks.loadModel,
  setModelDraftNumPredict: mocks.setModelDraftNumPredict,
  unloadModel: mocks.unloadModel,
  updateSettings: mocks.updateSettings,
  updateCloudSetting: mocks.updateCloudSetting,
}));

function textContent(node: ReactTestInstance): string {
  return node.children
    .map((child) => (typeof child === "string" ? child : textContent(child)))
    .join("");
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe("Settings reset interactions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.isWindows = false;
    mocks.settings = new SettingsType({ ContextLength: 65_536 });
    mocks.runningModels = [];
    mocks.modelContextInfo = {};
    mocks.modelMTP = { supported: false };
    mocks.setModelDraftNumPredict.mockReset();
    mocks.unloadModel.mockReset();
    mocks.loadModel.mockReset();
    mocks.setModelDraftNumPredict.mockResolvedValue(undefined);
    mocks.unloadModel.mockResolvedValue(undefined);
    mocks.loadModel.mockResolvedValue(undefined);
    mocks.updateSettings.mockResolvedValue({ settings: mocks.settings });
    mocks.updateCloudSetting.mockResolvedValue({
      disabled: false,
      source: "none",
    });
    mocks.setShowAppsInMenu.mockResolvedValue(undefined);
    mocks.resetChatGPTModels.mockResolvedValue(true);
    mocks.disconnectUser.mockResolvedValue(undefined);

    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setTimeout: globalThis.setTimeout.bind(globalThis),
      clearTimeout: globalThis.clearTimeout.bind(globalThis),
      getShowAppsInMenu: vi.fn().mockResolvedValue(true),
      setShowAppsInMenu: mocks.setShowAppsInMenu,
      open: vi.fn(),
      confirm: vi.fn(() => true),
      location: { reload: vi.fn() },
      OLLAMA_TOOLS: false,
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  });

  it("locks every control and shows Saved after reset succeeds", async () => {
    const pendingClaudeReset = deferred<boolean>();
    mocks.resetClaudeMappings.mockImplementation(
      () => pendingClaudeReset.promise,
    );

    let renderer;
    try {
      await act(async () => {
        renderer = create(<Settings />);
        await Promise.resolve();
      });

      const resetButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button).includes("Reset to defaults"));
      if (!resetButton) throw new Error("Reset button not found");

      await act(async () => {
        resetButton.props.onClick();
        await Promise.resolve();
      });

      const settingsFieldset = renderer!.root.findByType("fieldset");
      expect(settingsFieldset.props.disabled).toBe(true);
      expect(settingsFieldset.props["aria-busy"]).toBe(true);
      expect(textContent(resetButton)).toContain("Resetting…");
      expect(renderer!.root.findAllByType(Badge)).toHaveLength(0);
      expect(mocks.resetChatGPTModels).toHaveBeenCalledOnce();

      await act(async () => {
        pendingClaudeReset.resolve(true);
        await pendingClaudeReset.promise;
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(renderer!.root.findByType("fieldset").props.disabled).toBe(false);
      expect(renderer!.root.findAllByType(Badge)).toHaveLength(1);
    } finally {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }
  });

  it("hides Claude and ChatGPT desktop settings on Windows", async () => {
    mocks.isWindows = true;

    let renderer;
    try {
      await act(async () => {
        renderer = create(<Settings />);
        await Promise.resolve();
      });

      expect(
        renderer!.root.findAllByProps({ "aria-label": "Claude settings" }),
      ).toHaveLength(0);
      expect(
        renderer!.root.findAllByProps({ "aria-label": "ChatGPT settings" }),
      ).toHaveLength(0);

      const resetButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button).includes("Reset to defaults"));
      if (!resetButton) throw new Error("Reset button not found");

      await act(async () => {
        resetButton.props.onClick();
        await vi.waitFor(() => expect(mocks.updateSettings).toHaveBeenCalled());
      });

      expect(mocks.resetClaudeMappings).not.toHaveBeenCalled();
      expect(mocks.resetChatGPTModels).not.toHaveBeenCalled();
    } finally {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }
  });

  it("reloads Settings after signing out", async () => {
    let renderer;
    try {
      await act(async () => {
        renderer = create(<Settings />);
        await Promise.resolve();
      });

      const signOutButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button) === "Sign out");
      if (!signOutButton) throw new Error("Sign out button not found");

      await act(async () => {
        signOutButton.props.onClick();
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mocks.disconnectUser).toHaveBeenCalledOnce();
      expect(window.location.reload).toHaveBeenCalledOnce();
    } finally {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }
  });

  describe("keep alive slider", () => {
    function findKeepAliveSlider(renderer: ReturnType<typeof create>) {
      const slider = renderer.root
        .findAllByType(Slider)
        .find((node) =>
          node.props.options?.some(
            (option: { label: string }) => option.label === "Never unload",
          ),
        );
      if (!slider) throw new Error("Keep alive slider not found");
      return slider;
    }

    it("renders directly under context length and defaults to 1h", async () => {
      let renderer;
      try {
        await act(async () => {
          renderer = create(<Settings />);
          await Promise.resolve();
        });

        const sliders = renderer!.root.findAllByType(Slider);
        const keepAlive = findKeepAliveSlider(renderer!);
        expect(sliders.indexOf(keepAlive)).toBe(1);
        expect(sliders[0].props.options.at(-1)).toEqual({
          value: 1_048_576,
          label: "1M",
        });
        expect(sliders[0].props.trackInset).toBe(keepAlive.props.trackInset);
        expect(keepAlive.props.value).toBe(3600);
        expect(
          keepAlive.props.options.map(
            (option: { label: string }) => option.label,
          ),
        ).toEqual(["5m", "15m", "1h", "4h", "24h", "Never unload"]);
        expect(
          keepAlive.props.options.find(
            (option: { label: string }) => option.label === "Never unload",
          ).value,
        ).toBe(-1);
      } finally {
        await act(async () => {
          renderer?.unmount();
          await Promise.resolve();
        });
        vi.unstubAllGlobals();
      }
    });

    it("saves the selected preset and skips no-op changes", async () => {
      let renderer;
      try {
        await act(async () => {
          renderer = create(<Settings />);
          await Promise.resolve();
        });

        // Selecting the effective default (1h) must not save or restart.
        await act(async () => {
          findKeepAliveSlider(renderer!).props.onChange(3600);
          await Promise.resolve();
        });
        expect(mocks.updateSettings).not.toHaveBeenCalled();

        await act(async () => {
          findKeepAliveSlider(renderer!).props.onChange(-1);
          await Promise.resolve();
        });
        expect(mocks.updateSettings).toHaveBeenCalledOnce();
        expect(mocks.updateSettings.mock.calls[0][0]).toMatchObject({
          ContextLength: 65_536,
          KeepAlive: -1,
        });
      } finally {
        await act(async () => {
          renderer?.unmount();
          await Promise.resolve();
        });
        vi.unstubAllGlobals();
      }
    });

    it("shows a stored value instead of the default", async () => {
      mocks.settings = new SettingsType({
        ContextLength: 65_536,
        KeepAlive: 86_400,
      });
      let renderer;
      try {
        await act(async () => {
          renderer = create(<Settings />);
          await Promise.resolve();
        });
        expect(findKeepAliveSlider(renderer!).props.value).toBe(86_400);
      } finally {
        await act(async () => {
          renderer?.unmount();
          await Promise.resolve();
        });
        vi.unstubAllGlobals();
      }
    });
  });

  describe("context length slider", () => {
    const glm = "glm5.3-flash:q6-vision";

    async function renderSettings() {
      let renderer!: ReturnType<typeof create>;
      await act(async () => {
        renderer = create(<Settings />);
        await Promise.resolve();
      });
      return renderer;
    }

    async function cleanup(renderer?: ReturnType<typeof create>) {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }

    function contextSlider(renderer: ReturnType<typeof create>) {
      return renderer.root.findAllByType(Slider)[0];
    }

    function lockedHint(renderer: ReturnType<typeof create>) {
      const hints = renderer.root.findAllByProps({
        "data-testid": "context-length-locked-hint",
      });
      return hints.length ? textContent(hints[0]) : null;
    }

    it("shows an explicit setting when the model has no own context length", async () => {
      mocks.runningModels = [{ name: glm, context_length: 1_048_576 }];
      const renderer = await renderSettings();
      try {
        expect(contextSlider(renderer).props.value).toBe(65_536);
        expect(contextSlider(renderer).props.disabled).toBe(false);
        expect(lockedHint(renderer)).toBeNull();
      } finally {
        await cleanup(renderer);
      }
    });

    it("stays adjustable and shows the loaded model's context when it has no num_ctx", async () => {
      mocks.settings = new SettingsType({
        ContextLength: 0,
        SelectedModel: glm,
      });
      mocks.runningModels = [{ name: glm, context_length: 1_048_576 }];
      const renderer = await renderSettings();
      try {
        expect(contextSlider(renderer).props.value).toBe(1_048_576);
        expect(contextSlider(renderer).props.disabled).toBe(false);
        expect(lockedHint(renderer)).toBeNull();
      } finally {
        await cleanup(renderer);
      }
    });

    it("locks when the loaded model sets its own num_ctx", async () => {
      mocks.settings = new SettingsType({
        ContextLength: 0,
        SelectedModel: glm,
      });
      mocks.runningModels = [{ name: glm, context_length: 1_048_576 }];
      mocks.modelContextInfo = { numCtx: 1_048_576, trainCtx: 1_048_576 };
      const renderer = await renderSettings();
      try {
        expect(contextSlider(renderer).props.value).toBe(1_048_576);
        expect(contextSlider(renderer).props.disabled).toBe(true);
        expect(lockedHint(renderer)).toBe(
          `Set by ${glm} (1M). Applies to models without their own context length.`,
        );
      } finally {
        await cleanup(renderer);
      }
    });

    it("locks even over an explicit setting, because num_ctx wins", async () => {
      mocks.runningModels = [{ name: glm, context_length: 1_048_576 }];
      mocks.modelContextInfo = { numCtx: 1_048_576 };
      const renderer = await renderSettings();
      try {
        expect(contextSlider(renderer).props.value).toBe(1_048_576);
        expect(contextSlider(renderer).props.disabled).toBe(true);
      } finally {
        await cleanup(renderer);
      }
    });

    it("locks for a selected model with num_ctx that is not loaded", async () => {
      mocks.settings = new SettingsType({
        ContextLength: 0,
        SelectedModel: glm,
      });
      mocks.modelContextInfo = { numCtx: 1_048_576 };
      const renderer = await renderSettings();
      try {
        expect(contextSlider(renderer).props.value).toBe(1_048_576);
        expect(contextSlider(renderer).props.disabled).toBe(true);
      } finally {
        await cleanup(renderer);
      }
    });

    it("does not save or restart on interaction while locked", async () => {
      mocks.settings = new SettingsType({
        ContextLength: 0,
        SelectedModel: glm,
      });
      mocks.runningModels = [{ name: glm, context_length: 1_048_576 }];
      mocks.modelContextInfo = { numCtx: 1_048_576 };
      const renderer = await renderSettings();
      try {
        const slider = contextSlider(renderer);
        // The rendered stop buttons are disabled...
        const stopButtons = slider.findAllByType("button");
        expect(stopButtons.length).toBe(9);
        expect(stopButtons.every((button) => button.props.disabled)).toBe(true);
        // ...and even a direct change is ignored.
        await act(async () => {
          slider.props.onChange(262_144);
          await Promise.resolve();
        });
        expect(mocks.updateSettings).not.toHaveBeenCalled();
      } finally {
        await cleanup(renderer);
      }
    });

    it("falls back to the server default last", async () => {
      mocks.settings = new SettingsType({
        ContextLength: 0,
        SelectedModel: glm,
      });
      const renderer = await renderSettings();
      try {
        expect(contextSlider(renderer).props.value).toBe(65_536);
        expect(contextSlider(renderer).props.disabled).toBe(false);
      } finally {
        await cleanup(renderer);
      }
    });

    it("does not save when the displayed stop is clicked again", async () => {
      mocks.settings = new SettingsType({ ContextLength: 0 });
      mocks.runningModels = [{ name: "glm", context_length: 1_048_576 }];
      const renderer = await renderSettings();
      try {
        await act(async () => {
          contextSlider(renderer).props.onChange(1_048_576);
          await Promise.resolve();
        });
        expect(mocks.updateSettings).not.toHaveBeenCalled();

        await act(async () => {
          contextSlider(renderer).props.onChange(262_144);
          await Promise.resolve();
        });
        expect(mocks.updateSettings.mock.calls[0][0]).toMatchObject({
          ContextLength: 262_144,
        });
      } finally {
        await cleanup(renderer);
      }
    });
  });

  describe("multi-token prediction", () => {
    const glm = "glm5.3-flash:q6-vision";

    async function renderSettings() {
      let renderer!: ReturnType<typeof create>;
      await act(async () => {
        renderer = create(<Settings />);
        await Promise.resolve();
      });
      return renderer;
    }

    async function cleanup(renderer?: ReturnType<typeof create>) {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }

    function toggle(renderer: ReturnType<typeof create>) {
      return renderer.root.findByProps({ "data-testid": "mtp-toggle" });
    }

    it("is hidden when the model has no MTP layers", async () => {
      mocks.settings = new SettingsType({ SelectedModel: glm });
      mocks.modelMTP = { supported: false };
      const renderer = await renderSettings();
      try {
        expect(
          renderer.root.findAllByProps({ "data-testid": "mtp-toggle" }),
        ).toHaveLength(0);
      } finally {
        await cleanup(renderer);
      }
    });

    it("shows an existing draft_num_predict parameter as on", async () => {
      mocks.settings = new SettingsType({ SelectedModel: glm });
      mocks.modelMTP = { supported: true, draftNumPredict: 3 };
      const renderer = await renderSettings();
      try {
        expect(toggle(renderer).props.checked).toBe(true);
        expect(textContent(renderer.root)).toContain(glm);
        expect(
          renderer.root.findByProps({ "data-testid": "mtp-draft-depth" }).props
            .value,
        ).toBe(3);
      } finally {
        await cleanup(renderer);
      }
    });

    it("turning off removes only draft_num_predict and does not reload an unloaded model", async () => {
      mocks.settings = new SettingsType({ SelectedModel: glm });
      mocks.modelMTP = { supported: true, draftNumPredict: 3 };
      const renderer = await renderSettings();
      try {
        await act(async () => {
          toggle(renderer).props.onChange(false);
          await Promise.resolve();
          await Promise.resolve();
        });
        expect(mocks.setModelDraftNumPredict).toHaveBeenCalledTimes(1);
        expect(mocks.setModelDraftNumPredict).toHaveBeenCalledWith(glm, 0);
        expect(mocks.unloadModel).not.toHaveBeenCalled();
        expect(mocks.loadModel).not.toHaveBeenCalled();
      } finally {
        await cleanup(renderer);
      }
    });

    it("turning on sets the default depth and reloads a loaded model", async () => {
      mocks.settings = new SettingsType({ SelectedModel: glm });
      mocks.modelMTP = { supported: true };
      mocks.runningModels = [{ name: glm, context_length: 1_048_576 }];
      const order: string[] = [];
      mocks.setModelDraftNumPredict.mockImplementation(async () => {
        order.push("set");
      });
      mocks.unloadModel.mockImplementation(async () => {
        order.push("unload");
      });
      mocks.loadModel.mockImplementation(async () => {
        order.push("load");
      });
      const renderer = await renderSettings();
      try {
        await act(async () => {
          toggle(renderer).props.onChange(true);
          await Promise.resolve();
          await Promise.resolve();
          await Promise.resolve();
        });
        expect(mocks.setModelDraftNumPredict).toHaveBeenCalledWith(glm, 3);
        expect(order).toEqual(["set", "unload", "load"]);
      } finally {
        await cleanup(renderer);
      }
    });

    it("does not unload the model when saving the parameter fails", async () => {
      mocks.settings = new SettingsType({ SelectedModel: glm });
      mocks.modelMTP = { supported: true, draftNumPredict: 3 };
      mocks.runningModels = [{ name: glm, context_length: 1_048_576 }];
      mocks.setModelDraftNumPredict.mockRejectedValue(new Error("disk full"));
      const renderer = await renderSettings();
      try {
        await act(async () => {
          toggle(renderer).props.onChange(false);
          await Promise.resolve();
          await Promise.resolve();
        });
        expect(mocks.unloadModel).not.toHaveBeenCalled();
        expect(mocks.loadModel).not.toHaveBeenCalled();
        expect(textContent(renderer.root)).toContain("disk full");
      } finally {
        await cleanup(renderer);
      }
    });
  });
});
