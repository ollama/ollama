import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";
import { getSettings } from "@/api";
import { useQuery } from "@tanstack/react-query";
import { useCloudStatus } from "@/hooks/useCloudStatus";
import { FirstRunPrompt } from "@/components/FirstRunPrompt";

function RootComponent() {
  // This hook ensures settings are fetched on app startup
  const { data: settingsData, isLoading } = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
  });
  // Fetch cloud status on startup (best-effort)
  useCloudStatus();

  if (isLoading) {
    return <div className="min-h-screen bg-white dark:bg-neutral-900" />;
  }

  if (settingsData?.hasCompletedFirstRun === false) {
    return <FirstRunPrompt open />;
  }

  return <Outlet />;
}

export const Route = createRootRouteWithContext<{
  queryClient: QueryClient;
}>()({
  component: RootComponent,
});
