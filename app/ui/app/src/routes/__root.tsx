import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";
import { useInitializeSettings } from "@/hooks/useInitializeSettings";
// import { TanStackRouterDevtools } from "@tanstack/react-router-devtools";

function RootComponent() {
  useInitializeSettings();

  return (
    <div>
      <Outlet />
      {/* <TanStackRouterDevtools /> */}
    </div>
  );
}

export const Route = createRootRouteWithContext<{
  queryClient: QueryClient;
}>()({
  component: RootComponent,
});
