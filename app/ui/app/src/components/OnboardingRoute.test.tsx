import { StrictMode, type ComponentType } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "@/api";
import { Settings } from "@/gotypes";
import { CURRENT_ONBOARDING_VERSION } from "@/lib/onboarding";
import { Route } from "@/routes/onboarding";
import { RunOllamaScreen, WelcomeScreen } from "./Onboarding";

const mocks = vi.hoisted(() => ({ navigate: vi.fn(), authenticated: true }));
vi.mock("@tanstack/react-router", async (importOriginal) =>
  Object.assign(
    {},
    await importOriginal<typeof import("@tanstack/react-router")>(),
    {
      useNavigate: () => mocks.navigate,
    },
  ),
);
vi.mock("@/hooks/useUser", () => ({
  useUser: () => ({
    isAuthenticated: mocks.authenticated,
    fetchConnectUrl: vi.fn(),
    refetchUser: vi.fn(),
  }),
}));

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  mocks.navigate.mockReset();
  mocks.authenticated = true;
});

// Use the real settings mutation: query notifications replace callbacks and
// must not start another completion or an unbounded retry after authentication.
async function renderOnboarding(authenticated: boolean, onboardingVersion = 0) {
  mocks.authenticated = authenticated;
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("navigator", { platform: "MacIntel" });
  vi.stubGlobal("window", {
    OLLAMA_PLATFORM: "darwin",
    location: { search: "" },
    setOnboardingWindow: vi.fn(),
  });
  let settingsResponse = {
    settings: new Settings({ OnboardingVersion: onboardingVersion }),
  };
  vi.spyOn(api, "getSettings").mockImplementation(async () => settingsResponse);
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: Infinity },
      mutations: { retry: false },
    },
  });
  client.setQueryData(["settings"], settingsResponse);
  const OnboardingRoute = Route.options.component as ComponentType;
  const element = () => (
    <StrictMode>
      <QueryClientProvider client={client}>
        <OnboardingRoute />
      </QueryClientProvider>
    </StrictMode>
  );
  let renderer: ReactTestRenderer;
  await act(async () => {
    renderer = create(element());
  });
  return {
    get root() {
      return renderer.root;
    },
    async continue() {
      await act(async () => {
        const button = renderer.root.find(
          (node) =>
            node.type === "button" && node.props.children === "Continue",
        );
        button.props.onClick();
        button.props.onClick();
      });
    },
    async authenticate() {
      mocks.authenticated = true;
      await act(async () => {
        renderer.update(element());
      });
    },
    async useLocal() {
      await act(async () => {
        const onLocal = renderer.root.findByType(WelcomeScreen).props.onLocal;
        onLocal();
        onLocal();
      });
    },
    async receiveCompletion() {
      settingsResponse = {
        settings: new Settings({
          OnboardingVersion: CURRENT_ONBOARDING_VERSION,
        }),
      };
      await act(async () => {
        client.setQueryData(["settings"], { ...settingsResponse });
        await vi.advanceTimersByTimeAsync(0);
      });
    },
    async unmount() {
      await act(async () => renderer.unmount());
      client.clear();
    },
  };
}

async function advanceTime(milliseconds = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds);
  });
}

const completionPaths = ["already signed in", "sign in", "local use"] as const;

async function finishOnboarding(path: (typeof completionPaths)[number]) {
  const onboarding = await renderOnboarding(path === "already signed in");
  await onboarding.continue();
  if (path === "sign in") await onboarding.authenticate();
  if (path === "local use") await onboarding.useLocal();
  return onboarding;
}

describe("Onboarding completion", () => {
  it.each([true, false])(
    "renders completed setup on Run Ollama without saving again (signed in: %s)",
    async (authenticated) => {
      const save = vi.spyOn(api, "updateSettings");
      const onboarding = await renderOnboarding(
        authenticated,
        CURRENT_ONBOARDING_VERSION,
      );
      try {
        expect(onboarding.root.findByType(RunOllamaScreen)).toBeTruthy();
        expect(save).not.toHaveBeenCalled();
        expect(mocks.navigate).not.toHaveBeenCalled();
      } finally {
        await onboarding.unmount();
      }
    },
  );

  it("shows Run Ollama without saving again when CLI completion arrives", async () => {
    const save = vi.spyOn(api, "updateSettings");
    const onboarding = await renderOnboarding(true);
    try {
      expect(mocks.navigate).not.toHaveBeenCalled();
      await onboarding.receiveCompletion();
      expect(mocks.navigate).not.toHaveBeenCalled();
      expect(onboarding.root.findByType(RunOllamaScreen)).toBeTruthy();
      expect(save).not.toHaveBeenCalled();
    } finally {
      await onboarding.unmount();
    }
  });

  it("keeps the app's own local completion screen visible", async () => {
    const save = vi
      .spyOn(api, "updateSettings")
      .mockImplementation(async (settings) => ({ settings }));
    const onboarding = await renderOnboarding(false);
    try {
      await onboarding.continue();
      await act(async () => {
        onboarding.root.findByType(WelcomeScreen).props.onLocal();
      });
      expect(save).toHaveBeenCalledOnce();
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          OnboardingVersion: CURRENT_ONBOARDING_VERSION,
        }),
      );
      await onboarding.receiveCompletion();
      expect(mocks.navigate).not.toHaveBeenCalled();
      expect(onboarding.root.findByType(RunOllamaScreen)).toBeTruthy();
    } finally {
      await onboarding.unmount();
    }
  });

  it.each(completionPaths)(
    "shows Run Ollama while saving in the background (%s)",
    async (path) => {
      let resolveSave!: (value: { settings: Settings }) => void;
      const save = vi.spyOn(api, "updateSettings").mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveSave = resolve;
          }),
      );
      const onboarding = await finishOnboarding(path);
      try {
        await advanceTime();
        expect(save).toHaveBeenCalledOnce();
        expect(save).toHaveBeenCalledWith(
          expect.objectContaining({
            OnboardingVersion: CURRENT_ONBOARDING_VERSION,
          }),
        );
        expect(mocks.navigate).not.toHaveBeenCalled();
        expect(onboarding.root.findByType(RunOllamaScreen)).toBeTruthy();
        await act(async () => {
          resolveSave({ settings: save.mock.calls[0][0] });
        });
        expect(onboarding.root.findByType(RunOllamaScreen)).toBeTruthy();
        await onboarding.receiveCompletion();
        expect(save).toHaveBeenCalledOnce();
        expect(mocks.navigate).not.toHaveBeenCalled();
      } finally {
        await onboarding.unmount();
      }
    },
  );

  it.each(completionPaths)(
    "quietly retries a failed save once while keeping Run Ollama visible (%s)",
    async (path) => {
      const save = vi
        .spyOn(api, "updateSettings")
        .mockRejectedValueOnce(new Error("connection interrupted"))
        .mockImplementation(async (settings) => ({ settings }));
      vi.spyOn(console, "error").mockImplementation(() => {});
      const onboarding = await finishOnboarding(path);
      try {
        await advanceTime(999);
        expect(save).toHaveBeenCalledOnce();
        expect(mocks.navigate).not.toHaveBeenCalled();
        expect(onboarding.root.findByType(RunOllamaScreen)).toBeTruthy();
        expect(onboarding.root.findAllByProps({ role: "alert" })).toHaveLength(
          0,
        );
        await advanceTime(1);
        expect(save).toHaveBeenCalledTimes(2);
        expect(save.mock.calls[1][0]).toEqual(save.mock.calls[0][0]);
        expect(onboarding.root.findByType(RunOllamaScreen)).toBeTruthy();
        await onboarding.receiveCompletion();
        await advanceTime(5000);
        expect(save).toHaveBeenCalledTimes(2);
        expect(mocks.navigate).not.toHaveBeenCalled();
      } finally {
        await onboarding.unmount();
      }
    },
  );

  it.each(completionPaths)(
    "logs a persistent save failure without blocking or repeatedly retrying (%s)",
    async (path) => {
      const error = new Error("disk full");
      const save = vi.spyOn(api, "updateSettings").mockRejectedValue(error);
      const log = vi.spyOn(console, "error").mockImplementation(() => {});
      const onboarding = await finishOnboarding(path);
      try {
        await advanceTime(1000);
        expect(save).toHaveBeenCalledTimes(2);
        expect(log).toHaveBeenCalledWith(
          "Failed to save onboarding state:",
          error,
        );
        await advanceTime(5000);
        expect(save).toHaveBeenCalledTimes(2);
        expect(onboarding.root.findByType(RunOllamaScreen)).toBeTruthy();
        expect(onboarding.root.findAllByProps({ role: "alert" })).toHaveLength(
          0,
        );
        expect(
          onboarding.root.findAll(
            (node) =>
              node.type === "button" && node.props.children === "Try again",
          ),
        ).toHaveLength(0);
        expect(mocks.navigate).not.toHaveBeenCalled();
      } finally {
        await onboarding.unmount();
      }
    },
  );
});
