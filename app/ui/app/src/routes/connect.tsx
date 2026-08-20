import { AppSidebar } from "@/components/AppSidebar";
import { ConnectAppsScreen } from "@/components/Onboarding";
import { SidebarLayout } from "@/components/layout/layout";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/connect")({
  component: ConnectRoute,
});

function ConnectRoute() {
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
