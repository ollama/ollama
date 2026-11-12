import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getSavedModels, saveModel, unsaveModel } from "@/api";
import { useCallback, useMemo } from "react";

const SAVED_MODELS_KEY = ["savedModels"];

export function useSavedModels() {
  const queryClient = useQueryClient();

  const {
    data: savedModels = [],
    isLoading,
    error,
  } = useQuery<string[], Error>({
    queryKey: SAVED_MODELS_KEY,
    queryFn: getSavedModels,
  });

  const savedSet = useMemo(() => new Set(savedModels), [savedModels]);

  const mutation = useMutation({
    mutationFn: ({ model, save }: { model: string; save: boolean }) =>
      save ? saveModel(model) : unsaveModel(model),
    onSuccess: (models) => {
      // The server returns the authoritative collection, so write it straight
      // into the cache instead of refetching.
      queryClient.setQueryData(SAVED_MODELS_KEY, models);
    },
  });

  const toggleSaved = useCallback(
    (model: string) => {
      const save = !savedSet.has(model);
      mutation.mutate({ model, save });
    },
    [mutation, savedSet],
  );

  const isSaved = useCallback(
    (model: string) => savedSet.has(model),
    [savedSet],
  );

  return {
    savedModels,
    savedSet,
    isSaved,
    toggleSaved,
    isLoading,
    error,
    isPending: mutation.isPending,
  };
}
