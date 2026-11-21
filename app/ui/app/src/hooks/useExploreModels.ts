import { useQuery } from "@tanstack/react-query";
import { getExploreModels, type ExploreModel } from "@/api";

/**
 * Loads the public model catalog. The server caches the catalog briefly, so
 * this is cheap to call from multiple components.
 */
export function useExploreModels() {
  return useQuery<ExploreModel[], Error>({
    queryKey: ["exploreModels"],
    queryFn: getExploreModels,
    staleTime: 10 * 60 * 1000,
    retry: 1,
  });
}
