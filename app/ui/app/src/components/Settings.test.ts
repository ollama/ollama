import { describe, expect, it, vi } from "vitest";
import { Settings as SettingsType } from "@/gotypes";
import { applySettingsDefaults } from "./Settings";

function currentSettings(overrides: Partial<SettingsType> = {}) {
  return new SettingsType({
    ContextLength: 65_536,
    ...overrides,
  });
}

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe("Settings defaults", () => {
  it("serializes a full reset before showing Saved", async () => {
    const settingsUpdate = deferred();
    const updateSettings = vi.fn(() => settingsUpdate.promise);
    const updateCloud = vi.fn().mockResolvedValue(undefined);
    const updateShowAppsInMenu = vi.fn().mockResolvedValue(undefined);
    const onSaved = vi.fn();

    const reset = applySettingsDefaults({
      updateSettings,
      updateCloud,
      updateShowAppsInMenu,
      currentSettings: currentSettings({
        Expose: true,
        Models: "/custom/models",
      }),
      cloudSource: "config",
      onSaved,
    });

    expect(updateSettings).toHaveBeenCalledOnce();
    expect(updateSettings.mock.calls[0][0]).toMatchObject({
      Expose: false,
      Models: "",
      ContextLength: 65_536,
      AutoUpdateEnabled: true,
    });
    expect(updateCloud).not.toHaveBeenCalled();
    expect(updateShowAppsInMenu).not.toHaveBeenCalled();
    expect(onSaved).not.toHaveBeenCalled();

    settingsUpdate.resolve();
    await reset;

    expect(updateCloud).toHaveBeenCalledWith(true);
    expect(updateShowAppsInMenu).toHaveBeenCalledWith(true);
    expect(onSaved).toHaveBeenCalledOnce();
    expect(updateSettings.mock.invocationCallOrder[0]).toBeLessThan(
      updateCloud.mock.invocationCallOrder[0],
    );
    expect(updateCloud.mock.invocationCallOrder[0]).toBeLessThan(
      updateShowAppsInMenu.mock.invocationCallOrder[0],
    );
    expect(updateShowAppsInMenu.mock.invocationCallOrder[0]).toBeLessThan(
      onSaved.mock.invocationCallOrder[0],
    );
  });

  it("preserves an environment-only Cloud override", async () => {
    const updateCloud = vi.fn().mockResolvedValue(undefined);
    const updateShowAppsInMenu = vi.fn().mockResolvedValue(undefined);

    await applySettingsDefaults({
      updateSettings: vi.fn().mockResolvedValue(undefined),
      updateCloud,
      updateShowAppsInMenu,
      currentSettings: currentSettings(),
      cloudSource: "env",
      onSaved: vi.fn(),
    });

    expect(updateCloud).not.toHaveBeenCalled();
    expect(updateShowAppsInMenu).toHaveBeenCalledWith(true);
  });

  it("clears the persisted Cloud override when the source is both", async () => {
    let environmentDisabled = true;
    let configDisabled = true;
    const updateCloud = vi.fn(async (enabled: boolean) => {
      configDisabled = !enabled;
    });

    await applySettingsDefaults({
      updateSettings: vi.fn().mockResolvedValue(undefined),
      updateCloud,
      updateShowAppsInMenu: vi.fn().mockResolvedValue(undefined),
      currentSettings: currentSettings(),
      cloudSource: "both",
      onSaved: vi.fn(),
    });

    expect(environmentDisabled || configDisabled).toBe(true);
    expect(configDisabled).toBe(false);

    environmentDisabled = false;
    expect(environmentDisabled || configDisabled).toBe(false);
  });

  it("does not issue a redundant Cloud update when Cloud is already on", async () => {
    const updateCloud = vi.fn().mockResolvedValue(undefined);

    await applySettingsDefaults({
      updateSettings: vi.fn().mockResolvedValue(undefined),
      updateCloud,
      updateShowAppsInMenu: vi.fn().mockResolvedValue(undefined),
      currentSettings: currentSettings(),
      cloudSource: "none",
      onSaved: vi.fn(),
    });

    expect(updateCloud).not.toHaveBeenCalled();
  });

  it("does not show Saved or continue after settings fail", async () => {
    const updateCloud = vi.fn().mockResolvedValue(undefined);
    const updateShowAppsInMenu = vi.fn().mockResolvedValue(undefined);
    const onSaved = vi.fn();

    await expect(
      applySettingsDefaults({
        updateSettings: vi.fn().mockRejectedValue(new Error("restart failed")),
        updateCloud,
        updateShowAppsInMenu,
        currentSettings: currentSettings({
          Expose: true,
          Models: "/custom/models",
        }),
        cloudSource: "config",
        onSaved,
      }),
    ).rejects.toThrow("restart failed");

    expect(updateCloud).not.toHaveBeenCalled();
    expect(updateShowAppsInMenu).not.toHaveBeenCalled();
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("does not show Saved when a later reset update fails", async () => {
    const updateShowAppsInMenu = vi.fn().mockResolvedValue(undefined);
    const onSaved = vi.fn();

    await expect(
      applySettingsDefaults({
        updateSettings: vi.fn().mockResolvedValue(undefined),
        updateCloud: vi.fn().mockRejectedValue(new Error("cloud failed")),
        updateShowAppsInMenu,
        currentSettings: currentSettings(),
        cloudSource: "config",
        onSaved,
      }),
    ).rejects.toThrow("cloud failed");

    expect(updateShowAppsInMenu).not.toHaveBeenCalled();
    expect(onSaved).not.toHaveBeenCalled();
  });
});
