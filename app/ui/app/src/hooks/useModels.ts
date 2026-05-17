import { useQuery } from "@tanstack/react-query";
import { Model } from "@/gotypes";
import { getModels } from "@/api";
import { useMemo } from "react";
import { useCloudStatus } from "./useCloudStatus";
import { useFeaturedModels } from "./useFeaturedModels";

export function useModels(searchQuery = "") {
  const { cloudDisabled } = useCloudStatus();
  const { data: recommendations, isLoading: recommendationsLoading } =
    useFeaturedModels();
  const localQuery = useQuery<Model[], Error>({
    queryKey: ["models", searchQuery],
    queryFn: () => getModels(searchQuery),
    gcTime: 10 * 60 * 1000,
    retry: 10,
    retryDelay: (attemptIndex) => Math.min(100 * 2 ** attemptIndex, 5000),
    refetchOnWindowFocus: true,
    refetchInterval: 30 * 1000,
    refetchIntervalInBackground: true,
  });

  const allModels = useMemo(() => {
    const local = localQuery.data || [];
    const featured = (recommendations || []).map((r) => r.model);
    const featuredSet = new Set(featured);

    // Recommended models first (using the local copy when downloaded),
    // then everything else from /api/tags in tags order.
    const recommended = featured.map(
      (name) =>
        local.find((m) => m.model === name) || new Model({ model: name }),
    );
    const rest = local.filter((m) => !featuredSet.has(m.model));
    const merged = [...recommended, ...rest];

    const visible = cloudDisabled
      ? merged.filter((m) => !m.isCloud())
      : merged;
    return filterBySearch(visible, searchQuery);
  }, [localQuery.data, searchQuery, cloudDisabled, recommendations]);

  return {
    ...localQuery,
    data: allModels,
    isLoading: localQuery.isLoading || recommendationsLoading,
  };
}

export function useRefetchModels() {
  const { refetch } = useModels();
  return refetch;
}

function filterBySearch(models: Model[], query: string): Model[] {
  const q = query.trim().toLowerCase();
  if (!q) return models;

  const seen = new Set<string>();
  return models.filter((m) => {
    const name = m.model.toLowerCase();
    if (!name.includes(q) || seen.has(name)) return false;
    seen.add(name);
    return true;
  });
}
