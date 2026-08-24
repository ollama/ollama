import { describe, expect, it, vi } from "vitest";
import { applySettingsDefaults } from "./Settings";

describe("Settings defaults", () => {
  it("restores product defaults across every settings backend", () => {
    const updateSettings = vi.fn();
    const updateCloud = vi.fn();
    const updateShowAppsInMenu = vi.fn();

    applySettingsDefaults({
      updateSettings,
      updateCloud,
      updateShowAppsInMenu,
      contextLength: 65_536,
      cloudDisabled: true,
      cloudOverriddenByEnv: false,
    });

    expect(updateSettings).toHaveBeenCalledOnce();
    expect(updateSettings.mock.calls[0][0]).toMatchObject({
      Expose: false,
      Models: "",
      ContextLength: 65_536,
      AutoUpdateEnabled: true,
    });
    expect(updateCloud).toHaveBeenCalledWith(true);
    expect(updateShowAppsInMenu).toHaveBeenCalledWith(true);
  });

  it("preserves an environment override that forces cloud off", () => {
    const updateCloud = vi.fn();
    const updateShowAppsInMenu = vi.fn();

    applySettingsDefaults({
      updateSettings: vi.fn(),
      updateCloud,
      updateShowAppsInMenu,
      contextLength: 32_768,
      cloudDisabled: true,
      cloudOverriddenByEnv: true,
    });

    expect(updateCloud).not.toHaveBeenCalled();
    expect(updateShowAppsInMenu).toHaveBeenCalledWith(true);
  });

  it("does not issue a redundant cloud update when cloud is already on", () => {
    const updateCloud = vi.fn();

    applySettingsDefaults({
      updateSettings: vi.fn(),
      updateCloud,
      updateShowAppsInMenu: vi.fn(),
      contextLength: 262_144,
      cloudDisabled: false,
      cloudOverriddenByEnv: false,
    });

    expect(updateCloud).not.toHaveBeenCalled();
  });
});
