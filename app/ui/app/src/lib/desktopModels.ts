import { queryClient } from "./queryClient";

queryClient.setQueryDefaults(["desktopModels"], {
  staleTime: 30_000,
  gcTime: 60_000,
  retry: false,
});

// One inventory per integration/account/saved state, independent of picker search.
// Native actions still validate current model access before changing configuration.
export function desktopModels<T>(
  key: readonly unknown[],
  load: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  return queryClient.fetchQuery({
    queryKey: ["desktopModels", ...key],
    queryFn: ({ signal }) => load(signal),
  });
}

export function cachedDesktopModels<T>(key: readonly unknown[]): T | undefined {
  return queryClient.getQueryData<T>(["desktopModels", ...key]);
}

export function cacheDesktopModels<T>(key: readonly unknown[], models: T) {
  queryClient.setQueryData(["desktopModels", ...key], models);
}

export async function invalidateDesktopModels(integration?: string) {
  const queryKey = integration
    ? ["desktopModels", integration]
    : ["desktopModels"];
  await queryClient.cancelQueries({ queryKey });
  queryClient.removeQueries({ queryKey });
}
