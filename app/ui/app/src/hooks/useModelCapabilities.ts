import { useQuery } from "@tanstack/react-query";
import { getModelCapabilities } from "@/api";
import { ModelCapabilitiesResponse } from "@/gotypes";

export function useModelCapabilities(modelName: string | undefined) {
  return useQuery<ModelCapabilitiesResponse, Error>({
    queryKey: ["modelCapabilities", modelName],
    queryFn: () => {
      return getModelCapabilities(modelName!);
    },
    enabled: !!modelName, // Only run query if modelName is provided
    gcTime: 60 * 60 * 1000, // Keep in cache for 1 hour
    staleTime: 60 * 60 * 1000, // Consider data stale after 1 hour
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 5000),
  });
}

export function useHasVisionCapability(modelName: string | undefined) {
  const { data: capabilitiesResponse } = useModelCapabilities(modelName);
  return capabilitiesResponse?.capabilities?.includes("vision") ?? false;
}
