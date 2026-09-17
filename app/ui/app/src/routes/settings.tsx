import { AppSidebar } from "@/components/AppSidebar";
import { SidebarLayout } from "@/components/layout/layout";
import { createFileRoute } from "@tanstack/react-router";
import Settings from "@/components/Settings";

export const Route = createFileRoute("/settings")({
  component: SettingsRoute,
});

function SettingsRoute() {
  return (
    <SidebarLayout title="Settings" sidebar={<AppSidebar current="settings" />}>
      <Settings />
    </SidebarLayout>
  );
}
