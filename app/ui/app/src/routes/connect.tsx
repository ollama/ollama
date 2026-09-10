import { AppSidebar } from "@/components/AppSidebar";
import { ConnectAppsScreen } from "@/components/Onboarding";
import { SidebarLayout } from "@/components/layout/layout";
import { parseConnectSearch } from "@/lib/connectSearch";
import { createFileRoute } from "@tanstack/react-router";
import { useCallback } from "react";

export const Route = createFileRoute("/connect")({
  validateSearch: parseConnectSearch,
  component: ConnectRoute,
});

function ConnectRoute() {
  const { connect, highlight } = Route.useSearch();
  const navigate = Route.useNavigate();
  const clearDeepLink = useCallback(
    () => void navigate({ to: "/connect", search: {}, replace: true }),
    [navigate],
  );

  return (
    <SidebarLayout title="Apps" sidebar={<AppSidebar current="apps" />}>
      <ConnectAppsScreen
        autoConnectClaude={connect === true}
        autoConnectChatGPT={connect === "chatgpt"}
        highlightIntegrationId={highlight}
        onDeepLinkHandled={clearDeepLink}
      />
    </SidebarLayout>
  );
}
