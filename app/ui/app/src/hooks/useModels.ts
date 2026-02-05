import { useQuery } from "@tanstack/react-query";
import { Model } from "@/gotypes";
import { getModels } from "@/api";
import { useMemo } from "react";

const DEFAULT_MODEL = "gemma3:4b";

export function useModels(searchQuery = "") {
  const query = useQuery<Model[], Error>({
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

  const models = useMemo(() => {
    const data = query.data || [];
    if (data.length === 0) {
      return [new Model({ model: DEFAULT_MODEL })];
    }
    return data;
  }, [query.data]);

  return {
    ...query,
    data: models,
    isLoading: query.isLoading,
  };
}

export function useRefetchModels() {
  const { refetch } = useModels();
  return refetch;
}
