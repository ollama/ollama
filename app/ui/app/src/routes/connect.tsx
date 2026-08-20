import { AppSidebar } from "@/components/AppSidebar";
import { ConnectAppsScreen } from "@/components/Onboarding";
import { SidebarLayout } from "@/components/layout/layout";
import { isOnboardingZoomShortcut } from "@/lib/onboarding";
import { createFileRoute } from "@tanstack/react-router";
import { useEffect } from "react";

export const Route = createFileRoute("/connect")({
  component: ConnectRoute,
});

function ConnectRoute() {
  useEffect(() => {
    const preventZoom = (event: KeyboardEvent) => {
      if (isOnboardingZoomShortcut(event)) event.preventDefault();
    };

    window.addEventListener("keydown", preventZoom, true);
    return () => window.removeEventListener("keydown", preventZoom, true);
  }, []);

  return (
    <SidebarLayout
      title="Connect your apps"
      sidebar={<AppSidebar current="apps" />}
    >
      <ConnectAppsScreen
        embedded
        completionError={null}
        onRetryCompletion={() => {}}
      />
    </SidebarLayout>
  );
}
