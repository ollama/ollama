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
    let resolveClaudeReset!: (succeeded: boolean) => void;
    const claudeReset = new Promise<boolean>((resolve) => {
      resolveClaudeReset = resolve;
    });
    const resetClaudeMappings = vi.fn(() => claudeReset);
    const onSaved = vi.fn();

    const reset = applySettingsDefaults({
      updateSettings,
      updateCloud,
      updateShowAppsInMenu,
      resetClaudeMappings,
      currentSettings: currentSettings({
        Expose: true,
        Models: "/custom/models",
      }),
      currentShowAppsInMenu: false,
      cloudSource: "config",
      onSaved,
    });

    await vi.waitFor(() => expect(updateSettings).toHaveBeenCalledOnce());
    expect(updateCloud).toHaveBeenCalledWith(true);
    expect(updateShowAppsInMenu).not.toHaveBeenCalled();
    expect(resetClaudeMappings).not.toHaveBeenCalled();
    expect(onSaved).not.toHaveBeenCalled();

    expect(updateSettings.mock.calls[0][0]).toMatchObject({
      Expose: false,
      Models: "",
      ContextLength: 65_536,
      AutoUpdateEnabled: true,
    });
    expect(onSaved).not.toHaveBeenCalled();
    settingsUpdate.resolve();
    await vi.waitFor(() => expect(resetClaudeMappings).toHaveBeenCalledOnce());
    expect(updateShowAppsInMenu).toHaveBeenCalledWith(true);
    expect(onSaved).not.toHaveBeenCalled();

    resolveClaudeReset(true);
    await reset;

    expect(onSaved).toHaveBeenCalledOnce();
    expect(updateCloud.mock.invocationCallOrder[0]).toBeLessThan(
      updateSettings.mock.invocationCallOrder[0],
    );
    expect(updateSettings.mock.invocationCallOrder[0]).toBeLessThan(
      updateShowAppsInMenu.mock.invocationCallOrder[0],
    );
    expect(updateShowAppsInMenu.mock.invocationCallOrder[0]).toBeLessThan(
      resetClaudeMappings.mock.invocationCallOrder[0],
    );
    expect(resetClaudeMappings.mock.invocationCallOrder[0]).toBeLessThan(
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
      resetClaudeMappings: vi.fn().mockResolvedValue(true),
      currentSettings: currentSettings(),
      currentShowAppsInMenu: true,
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
      resetClaudeMappings: vi.fn().mockResolvedValue(true),
      currentSettings: currentSettings(),
      currentShowAppsInMenu: true,
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
      resetClaudeMappings: vi.fn().mockResolvedValue(true),
      currentSettings: currentSettings(),
      currentShowAppsInMenu: true,
      cloudSource: "none",
      onSaved: vi.fn(),
    });

    expect(updateCloud).not.toHaveBeenCalled();
  });

  it("restores Cloud and leaves Claude untouched when settings fail", async () => {
    const updateCloud = vi.fn().mockResolvedValue(undefined);
    const updateShowAppsInMenu = vi.fn().mockResolvedValue(undefined);
    const resetClaudeMappings = vi.fn().mockResolvedValue(true);
    const onSaved = vi.fn();

    await expect(
      applySettingsDefaults({
        updateSettings: vi.fn().mockRejectedValue(new Error("restart failed")),
        updateCloud,
        updateShowAppsInMenu,
        resetClaudeMappings,
        currentSettings: currentSettings({
          Expose: true,
          Models: "/custom/models",
        }),
        currentShowAppsInMenu: false,
        cloudSource: "config",
        onSaved,
      }),
    ).rejects.toThrow("restart failed");

    expect(updateCloud.mock.calls).toEqual([[true], [false]]);
    expect(resetClaudeMappings).not.toHaveBeenCalled();
    expect(updateShowAppsInMenu).not.toHaveBeenCalled();
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("does not continue when the Cloud reset fails", async () => {
    const updateShowAppsInMenu = vi.fn().mockResolvedValue(undefined);
    const onSaved = vi.fn();

    await expect(
      applySettingsDefaults({
        updateSettings: vi.fn().mockResolvedValue(undefined),
        updateCloud: vi.fn().mockRejectedValue(new Error("cloud failed")),
        updateShowAppsInMenu,
        resetClaudeMappings: vi.fn().mockResolvedValue(true),
        currentSettings: currentSettings(),
        currentShowAppsInMenu: true,
        cloudSource: "config",
        onSaved,
      }),
    ).rejects.toThrow("cloud failed");

    expect(updateShowAppsInMenu).not.toHaveBeenCalled();
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("rolls earlier changes back when Claude mappings cannot be reset", async () => {
    const onSaved = vi.fn();
    const updateSettings = vi.fn().mockResolvedValue(undefined);
    const updateCloud = vi.fn().mockResolvedValue(undefined);
    const updateShowAppsInMenu = vi.fn().mockResolvedValue(undefined);
    const previousSettings = currentSettings({
      Expose: true,
      Models: "/custom/models",
    });

    await expect(
      applySettingsDefaults({
        updateSettings,
        updateCloud,
        updateShowAppsInMenu,
        resetClaudeMappings: vi.fn().mockResolvedValue(false),
        currentSettings: previousSettings,
        currentShowAppsInMenu: false,
        cloudSource: "config",
        onSaved,
      }),
    ).rejects.toThrow("Claude model mappings could not be reset");

    expect(updateCloud.mock.calls).toEqual([[true], [false]]);
    expect(updateSettings).toHaveBeenCalledTimes(2);
    expect(updateSettings.mock.calls[1][0]).toBe(previousSettings);
    expect(updateShowAppsInMenu.mock.calls).toEqual([[true], [false]]);
    expect(onSaved).not.toHaveBeenCalled();
  });
});
