import { StrictMode, type ComponentType } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "@/api";
import { Settings } from "@/gotypes";
import { CURRENT_ONBOARDING_VERSION } from "@/lib/onboarding";
import { Route } from "@/routes/onboarding";
import { WelcomeScreen } from "./Onboarding";

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

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  mocks.navigate.mockReset();
  mocks.authenticated = true;
});

// Use the real settings mutation: query notifications replace callbacks and
// must not automatically retry a failed handoff after authentication.
async function renderOnboarding(authenticated: boolean) {
  mocks.authenticated = authenticated;
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("navigator", { platform: "MacIntel" });
  vi.stubGlobal("window", {
    OLLAMA_PLATFORM: "darwin",
    setOnboardingWindow: vi.fn(),
  });
  const settingsResponse = { settings: new Settings({ OnboardingVersion: 0 }) };
  vi.spyOn(api, "getSettings").mockResolvedValue(settingsResponse);
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
  const primaryAction = () =>
    renderer.root.find(
      (node) => node.type === "button" && "aria-busy" in node.props,
    );
  return {
    get primaryAction() {
      return primaryAction();
    },
    get root() {
      return renderer.root;
    },
    async continue() {
      await act(async () => {
        const button = primaryAction();
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
    async unmount() {
      await act(async () => renderer.unmount());
      client.clear();
    },
  };
}

async function flushQueryNotifications() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

describe("Onboarding completion", () => {
  it.each([true, false])(
    "saves once before opening Apps directly (already signed in: %s)",
    async (authenticated) => {
      let resolveSave!: (value: { settings: Settings }) => void;
      const save = vi.spyOn(api, "updateSettings").mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveSave = resolve;
          }),
      );
      const onboarding = await renderOnboarding(authenticated);
      try {
        expect(save).not.toHaveBeenCalled();
        expect(mocks.navigate).not.toHaveBeenCalled();
        await onboarding.continue();
        if (!authenticated) {
          expect(save).not.toHaveBeenCalled();
          expect(onboarding.root.findByType(WelcomeScreen)).toBeTruthy();
          await onboarding.authenticate();
        }
        await flushQueryNotifications();
        expect(save).toHaveBeenCalledOnce();
        expect(save).toHaveBeenCalledWith(
          expect.objectContaining({
            OnboardingVersion: CURRENT_ONBOARDING_VERSION,
          }),
        );
        expect(mocks.navigate).not.toHaveBeenCalled();
        expect(onboarding.primaryAction.props.disabled).toBe(true);
        await act(async () => {
          resolveSave({ settings: save.mock.calls[0][0] });
        });
        expect(mocks.navigate).toHaveBeenCalledExactlyOnceWith({
          to: "/connect",
        });
      } finally {
        await onboarding.unmount();
      }
    },
  );

  it.each([true, false])(
    "keeps the current screen and waits for an explicit save retry (already signed in: %s)",
    async (authenticated) => {
      const save = vi
        .spyOn(api, "updateSettings")
        .mockRejectedValueOnce(new Error("disk full"))
        .mockImplementation(async (settings) => ({ settings }));
      vi.spyOn(console, "error").mockImplementation(() => {});
      const onboarding = await renderOnboarding(authenticated);
      try {
        await onboarding.continue();
        if (!authenticated) await onboarding.authenticate();
        await flushQueryNotifications();
        await flushQueryNotifications();
        expect(save).toHaveBeenCalledOnce();
        expect(mocks.navigate).not.toHaveBeenCalled();
        expect(onboarding.primaryAction.props.disabled).toBe(false);
        expect(onboarding.root.findByProps({ role: "alert" })).toBeTruthy();
        await act(async () => {
          onboarding.root
            .find(
              (node) =>
                node.type === "button" && node.props.children === "Try again",
            )
            .props.onClick();
        });
        expect(save).toHaveBeenCalledTimes(2);
        expect(save.mock.calls[1][0]).toEqual(save.mock.calls[0][0]);
        expect(mocks.navigate).toHaveBeenCalledExactlyOnceWith({
          to: "/connect",
        });
      } finally {
        await onboarding.unmount();
      }
    },
  );
});
