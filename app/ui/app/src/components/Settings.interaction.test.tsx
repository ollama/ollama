import { act, create, type ReactTestInstance } from "react-test-renderer";
import { forwardRef, useImperativeHandle } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Settings as SettingsType } from "@/gotypes";
import { Badge } from "./ui/badge";
import { Switch } from "./ui/switch";
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
  settingsLoading: false,
  settingsError: null as Error | null,
  cloudStatusKnown: true,
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
    isKnown: mocks.cloudStatusKnown,
  }),
}));

vi.mock("@/lib/platform", () => ({
  isWindowsPlatform: () => mocks.isWindows,
}));

vi.mock("@tanstack/react-router", () => ({
  useBlocker: vi.fn(),
}));

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => mocks.queryClient,
  useQuery: ({ queryKey }: { queryKey: string[] }) => {
    if (queryKey[0] === "settings") {
      return {
        data: { settings: mocks.settings },
        isLoading: mocks.settingsLoading,
        error: mocks.settingsError,
      };
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
        void run(value, callbacks);
      },
      mutateAsync: (value: unknown) => run(value),
    };
  },
}));

vi.mock("@/api", () => ({
  getSettings: vi.fn(),
  getInferenceCompute: vi.fn(),
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
    mocks.settingsLoading = false;
    mocks.settingsError = null;
    mocks.cloudStatusKnown = true;
    mocks.settings = new SettingsType({ ContextLength: 65_536 });
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

  it("renders the page before saved settings arrive without guessing values", async () => {
    mocks.settings = null;
    mocks.settingsLoading = true;
    mocks.cloudStatusKnown = false;
    const menuVisibility = deferred<boolean>();
    window.getShowAppsInMenu = () => menuVisibility.promise;
    let renderer;
    try {
      await act(async () => {
        renderer = create(<Settings />);
      });
      expect(renderer!.root.findAllByType("main")).toHaveLength(1);
      expect(textContent(renderer!.root)).toContain("Auto-download updates");
      expect(textContent(renderer!.root)).toContain("Model location");
      expect(renderer!.root.findAllByType(Switch)).toHaveLength(0);
      expect(renderer!.root.findAllByType(Slider)).toHaveLength(0);
      expect(
        renderer!.root.findAllByProps({ "aria-label": "ChatGPT settings" }),
      ).toHaveLength(1);
      const reset = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button).includes("Reset to defaults"))!;
      expect(reset.props.disabled).toBe(true);

      mocks.settings = new SettingsType({
        AutoUpdateEnabled: false,
        Expose: true,
        Models: "/saved/models",
        ContextLength: 32_768,
      });
      mocks.settingsLoading = false;
      await act(async () => {
        renderer!.update(<Settings />);
      });
      // These settings are usable even while the separate Cloud/menu reads wait.
      expect(
        renderer!.root
          .findAllByType(Switch)
          .map((control) => control.props.checked),
      ).toEqual([false, true]);
      expect(renderer!.root.findByType(Slider).props.value).toBe(32_768);
      expect(
        renderer!.root.findByProps({ value: "/saved/models", readOnly: true }),
      ).toBeTruthy();
      expect(reset.props.disabled).toBe(true);
      expect(mocks.updateSettings).not.toHaveBeenCalled();

      await act(async () => {
        menuVisibility.resolve(false);
        await menuVisibility.promise;
      });
      expect(
        renderer!.root
          .findAllByType(Switch)
          .map((control) => control.props.checked),
      ).toEqual([false, false, true]);
    } finally {
      await act(async () => renderer?.unmount());
      vi.unstubAllGlobals();
    }
  });

  it("keeps the page and independent controls available when saved settings fail", async () => {
    mocks.settings = null;
    mocks.settingsError = new Error("unavailable");
    let renderer;
    try {
      await act(async () => {
        renderer = create(<Settings />);
      });
      expect(renderer!.root.findAllByType("main")).toHaveLength(1);
      expect(textContent(renderer!.root.findByProps({ role: "alert" }))).toBe(
        "Failed to load settings",
      );
      expect(renderer!.root.findAllByType(Switch)).toHaveLength(2);
      expect(renderer!.root.findAllByType(Slider)).toHaveLength(0);
      expect(
        renderer!.root.findAllByProps({ "aria-label": "ChatGPT settings" }),
      ).toHaveLength(1);
    } finally {
      await act(async () => renderer?.unmount());
      vi.unstubAllGlobals();
    }
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
});
