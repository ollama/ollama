import { createElement, type ComponentType } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Route } from "../routes/onboarding";

const state = vi.hoisted(() => ({
  settings: { OnboardingVersion: 0 },
  setSettings: vi.fn(),
  navigate: vi.fn(),
}));

vi.mock("@/hooks/useSettings", () => ({
  useSettings: () => ({
    settingsData: state.settings,
    setSettings: state.setSettings,
  }),
}));

vi.mock("@/hooks/useUser", () => ({
  useUser: () => ({ isAuthenticated: true }),
}));

vi.mock("@tanstack/react-router", () => ({
  createFileRoute: () => (options: unknown) => ({ options }),
  useNavigate: () => state.navigate,
  redirect: vi.fn(),
}));

vi.mock("@/components/Onboarding", () => ({
  default: ({ onUseLocal }: { onUseLocal: () => void }) => (
    <button onClick={onUseLocal}>Use local</button>
  ),
}));

const renderRoute = () =>
  createElement(Route.options.component as ComponentType);

let renderer: ReactTestRenderer | undefined;
beforeEach(() => {
  state.settings = { OnboardingVersion: 0 };
  state.navigate.mockReset();
  state.setSettings.mockReset().mockResolvedValue(undefined);
  vi.stubGlobal("window", { location: { search: "" } });
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
});

afterEach(async () => {
  await act(async () => renderer?.unmount());
  renderer = undefined;
  vi.unstubAllGlobals();
});

describe("shared onboarding completion", () => {
  it("leaves onboarding when CLI completion arrives, even when already signed in", async () => {
    await act(async () => {
      renderer = create(renderRoute());
    });
    expect(state.navigate).not.toHaveBeenCalled();

    state.settings = { OnboardingVersion: 1 };
    await act(async () => renderer!.update(renderRoute()));
    expect(state.navigate).toHaveBeenCalledWith({ to: "/" });
    expect(state.setSettings).not.toHaveBeenCalled();
  });

  it("keeps the app's own local completion flow visible", async () => {
    await act(async () => {
      renderer = create(renderRoute());
    });
    await act(async () => renderer!.root.findByType("button").props.onClick());
    expect(state.setSettings).toHaveBeenCalledWith({ OnboardingVersion: 1 });

    state.settings = { OnboardingVersion: 1 };
    await act(async () => renderer!.update(renderRoute()));
    expect(state.navigate).not.toHaveBeenCalled();
  });

});
