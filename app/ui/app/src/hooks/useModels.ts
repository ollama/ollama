import { useQuery } from "@tanstack/react-query";
import { Model } from "@/gotypes";
import { getModels } from "@/api";
import { mergeModels } from "@/utils/mergeModels";
import { useSettings } from "./useSettings";
import { useMemo } from "react";

export function useModels(searchQuery = "") {
  const { settings } = useSettings();
  const localQuery = useQuery<Model[], Error>({
    queryKey: ["models", searchQuery],
    queryFn: () => getModels(searchQuery),
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
    retry: 10,
    // exponential backoff, starting at 100ms and capping at 5s
    retryDelay: (attemptIndex) => Math.min(100 * 2 ** attemptIndex, 5000),
    refetchOnWindowFocus: true,
    refetchInterval: 30 * 1000, // Refetch every 30 seconds to keep models updated
    refetchIntervalInBackground: true,
  });

  const allModels = useMemo(() => {
    const models = mergeModels(localQuery.data || [], settings.airplaneMode);

    if (searchQuery && searchQuery.trim()) {
      const query = searchQuery.toLowerCase().trim();
      const filteredModels = models.filter((model) =>
        model.model.toLowerCase().includes(query),
      );

      const seen = new Set<string>();
      return filteredModels.filter((model) => {
        const currentModel = model.model.toLowerCase();
        if (seen.has(currentModel)) {
          return false;
        }
        seen.add(currentModel);
        return true;
      });
    }

    return models;
  }, [localQuery.data, searchQuery, settings.airplaneMode]);

  return {
    ...localQuery,
    data: allModels,
    isLoading: localQuery.isLoading,
  };
}

export function useRefetchModels() {
  const { refetch } = useModels();
  return refetch;
}
