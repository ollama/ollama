import { useQuery } from "@tanstack/react-query";
import { getModelRecommendations } from "@/api";
import type { ModelRecommendation } from "@/api";

export function useFeaturedModels() {
  return useQuery<ModelRecommendation[], Error>({
    queryKey: ["modelRecommendations"],
    queryFn: getModelRecommendations,
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}
