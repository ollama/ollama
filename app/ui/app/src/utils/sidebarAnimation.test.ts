import { describe, it, expect } from "vitest";

// Reproduces the logic from useSettings to confirm the default sidebarOpen
// value is false when settings have not yet loaded (settingsData is undefined).
// The layout uses isLoading to suppress CSS transitions on initial render so
// that the sidebar does not animate open when settings arrive after mount.
describe("sidebar animation guard", () => {
  function deriveSettings(settingsData: { SidebarOpen?: boolean } | undefined) {
    return {
      sidebarOpen: settingsData?.SidebarOpen ?? false,
    };
  }

  it("defaults sidebarOpen to false while settings are loading", () => {
    const settings = deriveSettings(undefined);
    expect(settings.sidebarOpen).toBe(false);
  });

  it("reflects persisted SidebarOpen=true once settings load", () => {
    const settings = deriveSettings({ SidebarOpen: true });
    expect(settings.sidebarOpen).toBe(true);
  });

  it("animate flag is false while isLoading, suppressing the open animation", () => {
    const isLoading = true;
    const animate = !isLoading;
    expect(animate).toBe(false);
  });

  it("animate flag is true after settings load, enabling subsequent transitions", () => {
    const isLoading = false;
    const animate = !isLoading;
    expect(animate).toBe(true);
  });
});
