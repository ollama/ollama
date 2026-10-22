import { AppSidebar } from "@/components/AppSidebar";
import { ConnectAppsScreen } from "@/components/Onboarding";
import { SidebarLayout } from "@/components/layout/layout";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/connect")({
  component: ConnectRoute,
});

function ConnectRoute() {
  const navigate = Route.useNavigate();
  return (
    <SidebarLayout title="Apps" sidebar={<AppSidebar current="apps" />}>
      <ConnectAppsScreen
        onChooseClaudeModel={() =>
          void navigate({ to: "/settings", hash: "claude-settings" })
        }
      />
    </SidebarLayout>
  );
}
