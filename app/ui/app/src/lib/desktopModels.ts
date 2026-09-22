import { queryClient } from "./queryClient";

// One inventory per integration/account/saved state, independent of picker search.
// Native actions still validate current model access before changing configuration.
export function desktopModels<T>(
  key: readonly unknown[],
  load: () => Promise<T>,
): Promise<T> {
  return queryClient.fetchQuery({
    queryKey: ["desktopModels", ...key],
    queryFn: load,
    staleTime: 30_000,
    gcTime: 60_000,
    retry: false,
  });
}

export async function invalidateDesktopModels(integration?: string) {
  const queryKey = integration
    ? ["desktopModels", integration]
    : ["desktopModels"];
  await queryClient.cancelQueries({ queryKey });
  queryClient.removeQueries({ queryKey });
}
