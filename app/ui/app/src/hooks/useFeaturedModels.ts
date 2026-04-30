import { useQuery } from "@tanstack/react-query";
import { getModelRecommendations } from "@/api";
import type { ModelRecommendation } from "@/api";

export function useFeaturedModels() {
  return useQuery<ModelRecommendation[], Error>({
    queryKey: ["modelRecommendations"],
    queryFn: getModelRecommendations,
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(100 * 2 ** attemptIndex, 5000),
    refetchOnWindowFocus: false,
  });
}
