import { useCallback, useRef } from "react";
import { useQueryClient, QueryClient } from "@tanstack/react-query";

interface BatcherConfig {
  batchInterval?: number; // milliseconds, default 16ms (~60fps)
  immediateFirst?: boolean; // if true, first update is immediate
}

const DEFAULT_BATCH_INTERVAL = 16;

export const useQueryBatcher = <T>(
  queryKey: readonly unknown[],
  config: BatcherConfig = {},
) => {
  const queryClient = useQueryClient();
  const { batchInterval = DEFAULT_BATCH_INTERVAL, immediateFirst = false } =
    config;

  const batchRef = useRef<{
    updateBatch: T | undefined;
    batchTimeout: number | null;
    isFirstUpdate: boolean;
  }>({
    updateBatch: undefined,
    batchTimeout: null,
    isFirstUpdate: true,
  });

  const flushBatch = useCallback(() => {
    const { updateBatch, batchTimeout } = batchRef.current;
    if (updateBatch) {
      queryClient.setQueryData(queryKey, updateBatch);
      batchRef.current.updateBatch = undefined;
    }
    if (batchTimeout !== null) {
      clearTimeout(batchTimeout);
      batchRef.current.batchTimeout = null;
    }
  }, [queryClient, queryKey]);

  const scheduleBatch = useCallback(
    (updater: (old: T | undefined) => T | undefined) => {
      const currentData = queryClient.getQueryData<T>(queryKey);
      const newBatch = updater(batchRef.current.updateBatch || currentData);
      batchRef.current.updateBatch = newBatch;

      // If this is the first update and immediateFirst is enabled, apply immediately
      if (immediateFirst && batchRef.current.isFirstUpdate) {
        batchRef.current.isFirstUpdate = false;
        queryClient.setQueryData(queryKey, newBatch);
        batchRef.current.updateBatch = undefined;
        return;
      }

      // Keep the first scheduled flush in place so subsequent updates join
      // the same frame-sized batch instead of restarting a debounce timer.
      if (batchRef.current.batchTimeout === null) {
        batchRef.current.batchTimeout = setTimeout(flushBatch, batchInterval);
      }
    },
    [queryClient, queryKey, flushBatch, batchInterval, immediateFirst],
  );

  const cleanup = useCallback(() => {
    if (batchRef.current.batchTimeout !== null) {
      clearTimeout(batchRef.current.batchTimeout);
      batchRef.current.batchTimeout = null;
    }
    batchRef.current.updateBatch = undefined;
    batchRef.current.isFirstUpdate = true;
  }, []);

  return {
    scheduleBatch,
    flushBatch,
    cleanup,
  };
};

export const createQueryBatcher = <T>(
  queryClient: QueryClient,
  queryKey: readonly unknown[],
  config: BatcherConfig = {},
) => {
  const { batchInterval = DEFAULT_BATCH_INTERVAL, immediateFirst = false } =
    config;

  let updateBatch: T | undefined = undefined;
  let batchTimeout: number | null = null;
  let isFirstUpdate = true;

  const flushBatch = () => {
    if (updateBatch) {
      queryClient.setQueryData(queryKey, updateBatch);
      updateBatch = undefined;
    }
    if (batchTimeout !== null) {
      clearTimeout(batchTimeout);
      batchTimeout = null;
    }
  };

  const scheduleBatch = (updater: (old: T | undefined) => T | undefined) => {
    const currentData = queryClient.getQueryData<T>(queryKey);
    updateBatch = updater(updateBatch || currentData);

    // If this is the first update and immediateFirst is enabled, apply immediately
    if (immediateFirst && isFirstUpdate) {
      isFirstUpdate = false;
      queryClient.setQueryData(queryKey, updateBatch);
      updateBatch = undefined;
      return;
    }

    // Keep the first scheduled flush in place so subsequent updates join
    // the same frame-sized batch instead of restarting a debounce timer.
    if (batchTimeout === null) {
      batchTimeout = setTimeout(flushBatch, batchInterval);
    }
  };

  const cleanup = () => {
    if (batchTimeout !== null) {
      clearTimeout(batchTimeout);
      batchTimeout = null;
    }
    updateBatch = undefined;
    isFirstUpdate = true;
  };

  return {
    scheduleBatch,
    flushBatch,
    cleanup,
  };
};
