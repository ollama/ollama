import { act, create, type ReactTestInstance } from "react-test-renderer";
import { forwardRef, useImperativeHandle } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Settings as SettingsType } from "@/gotypes";
import { Badge } from "./ui/badge";
import Settings from "./Settings";

const mocks = vi.hoisted(() => ({
  resetClaudeMappings: vi.fn(),
  updateSettings: vi.fn(),
  updateCloudSetting: vi.fn(),
  setShowAppsInMenu: vi.fn(),
  refetchUser: vi.fn(),
  isWindows: false,
  queryClient: {
    cancelQueries: vi.fn().mockResolvedValue(undefined),
    getQueryData: vi.fn(),
    setQueryData: vi.fn(),
    invalidateQueries: vi.fn(),
  },
  settings: null as SettingsType | null,
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

vi.mock("@/hooks/useUser", () => ({
  useUser: () => ({
    user: { name: "Paid user", email: "paid@example.com", plan: "pro" },
    isAuthenticated: true,
    refreshUser: vi.fn(),
    isRefreshing: false,
    refetchUser: mocks.refetchUser,
    fetchConnectUrl: vi.fn(),
    isLoading: false,
    disconnectUser: vi.fn(),
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

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => mocks.queryClient,
  useQuery: ({ queryKey }: { queryKey: string[] }) => {
    if (queryKey[0] === "settings") {
      return {
        data: { settings: mocks.settings },
        isLoading: false,
        error: null,
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
    mocks.settings = new SettingsType({ ContextLength: 65_536 });
    mocks.updateSettings.mockResolvedValue({ settings: mocks.settings });
    mocks.updateCloudSetting.mockResolvedValue({
      disabled: false,
      source: "none",
    });
    mocks.setShowAppsInMenu.mockResolvedValue(undefined);

    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setTimeout: globalThis.setTimeout.bind(globalThis),
      clearTimeout: globalThis.clearTimeout.bind(globalThis),
      getShowAppsInMenu: vi.fn().mockResolvedValue(true),
      setShowAppsInMenu: mocks.setShowAppsInMenu,
      open: vi.fn(),
      confirm: vi.fn(() => true),
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

  it("hides Claude Desktop settings and skips its reset on Windows", async () => {
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

      const resetButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button).includes("Reset to defaults"));
      if (!resetButton) throw new Error("Reset button not found");

      await act(async () => {
        resetButton.props.onClick();
        await vi.waitFor(() => expect(mocks.updateSettings).toHaveBeenCalled());
      });

      expect(mocks.resetClaudeMappings).not.toHaveBeenCalled();
    } finally {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }
  });
});
