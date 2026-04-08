export function resolveHomeChatId(lastHomeView?: string) {
  return lastHomeView === "chat" ? "new" : "launch";
}

// Session-scoped flags used to make launch-page sidebar opening a one-shot UX:
// open it the first time launch is shown in a session, and open it again only
// when navigation to launch was explicitly requested from the sidebar.
const launchSidebarRequestedKey = "ollama.launchSidebarRequested";
const launchSidebarSeenKey = "ollama.launchSidebarSeen";

export function getLaunchRouteSettingsUpdates(settings: {
  LastHomeView?: string;
  SidebarOpen?: boolean;
}, shouldOpenSidebar = false) {
  const updates: { LastHomeView?: string; SidebarOpen?: boolean } = {};

  // Keep launch as the persisted home preference while the user is on the
  // generic launch page. Explicit integration selection can still override this.
  if (settings.LastHomeView !== "launch") {
    updates.LastHomeView = "launch";
  }

  // Only force the sidebar open for the narrow entry cases we explicitly allow.
  if (shouldOpenSidebar && !settings.SidebarOpen) {
    updates.SidebarOpen = true;
  }

  return updates;
}

export function requestLaunchSidebarOpen() {
  sessionStorage.setItem(launchSidebarRequestedKey, "1");
}

export function shouldAutoOpenLaunchSidebarOnVisit() {
  // A sidebar click into Launch should reopen the sidebar once on arrival.
  if (sessionStorage.getItem(launchSidebarRequestedKey) === "1") {
    sessionStorage.removeItem(launchSidebarRequestedKey);
    sessionStorage.setItem(launchSidebarSeenKey, "1");
    return true;
  }

  // Otherwise only auto-open the first time Launch is shown in this session.
  if (sessionStorage.getItem(launchSidebarSeenKey) !== "1") {
    sessionStorage.setItem(launchSidebarSeenKey, "1");
    return true;
  }

  return false;
}
