import { AppSidebar } from "@/components/AppSidebar";
import { SidebarLayout } from "@/components/layout/layout";
import { createFileRoute } from "@tanstack/react-router";
import SavedModels from "@/components/SavedModels";

export const Route = createFileRoute("/saved")({
  component: SavedRoute,
});

function SavedRoute() {
  return (
    <SidebarLayout title="Saved" sidebar={<AppSidebar current="saved" />}>
      <SavedModels />
    </SidebarLayout>
  );
}
