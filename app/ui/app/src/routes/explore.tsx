import { AppSidebar } from "@/components/AppSidebar";
import { SidebarLayout } from "@/components/layout/layout";
import { createFileRoute } from "@tanstack/react-router";
import Explore from "@/components/Explore";

export const Route = createFileRoute("/explore")({
  component: ExploreRoute,
});

function ExploreRoute() {
  return (
    <SidebarLayout title="Explore" sidebar={<AppSidebar current="explore" />}>
      <Explore />
    </SidebarLayout>
  );
}
