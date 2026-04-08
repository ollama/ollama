import { describe, expect, it } from "vitest";
import {
  getLaunchRouteSettingsUpdates,
  requestLaunchSidebarOpen,
  resolveHomeChatId,
  shouldAutoOpenLaunchSidebarOnVisit,
} from "./homeView";

describe("home view routing helpers", () => {
  it("routes missing and launch preferences to launch", () => {
    expect(resolveHomeChatId(undefined)).toBe("launch");
    expect(resolveHomeChatId("launch")).toBe("launch");
    expect(resolveHomeChatId("openclaw")).toBe("launch");
  });

  it("routes chat preference to new chat", () => {
    expect(resolveHomeChatId("chat")).toBe("new");
  });

  it("opens the sidebar and preserves launch on launch route mount", () => {
    expect(
      getLaunchRouteSettingsUpdates({
        LastHomeView: "launch",
        SidebarOpen: false,
      }, true),
    ).toEqual({ SidebarOpen: true });

    expect(
      getLaunchRouteSettingsUpdates({
        LastHomeView: "openclaw",
        SidebarOpen: false,
      }, true),
    ).toEqual({ LastHomeView: "launch", SidebarOpen: true });
  });

  it("does not emit redundant updates when launch is already active and open", () => {
    expect(
      getLaunchRouteSettingsUpdates({
        LastHomeView: "launch",
        SidebarOpen: true,
      }),
    ).toEqual({});
  });

  it("only auto-opens launch once per session unless explicitly requested", () => {
    sessionStorage.clear();

    expect(shouldAutoOpenLaunchSidebarOnVisit()).toBe(true);
    expect(shouldAutoOpenLaunchSidebarOnVisit()).toBe(false);

    requestLaunchSidebarOpen();
    expect(shouldAutoOpenLaunchSidebarOnVisit()).toBe(true);
    expect(shouldAutoOpenLaunchSidebarOnVisit()).toBe(false);
  });
});
