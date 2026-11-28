import { useCallback, useEffect, useRef, useState } from "react";
import { pullModel } from "@/api";
import {
  applyPullEvent,
  type DownloadState,
  type PullEvent,
  IDLE_DOWNLOAD,
} from "@/lib/modelDownload";

export type PullModelFn = (
  modelName: string,
  signal?: AbortSignal,
) => AsyncGenerator<PullEvent>;

export interface ModelDownloadControls {
  states: Record<string, DownloadState>;
  start: (modelName: string) => void;
  cancel: (modelName: string) => void;
}

export function useModelDownload(
  onCompleted?: (modelName: string) => void,
  pull: PullModelFn = pullModel,
): ModelDownloadControls {
  const [states, setStates] = useState<Record<string, DownloadState>>({});
  const controllers = useRef(new Map<string, AbortController>());
  const onCompletedRef = useRef(onCompleted);
  onCompletedRef.current = onCompleted;

  useEffect(() => {
    const active = controllers.current;
    return () => {
      for (const controller of active.values()) {
        controller.abort();
      }
      active.clear();
    };
  }, []);

  const start = useCallback(
    (modelName: string) => {
      if (controllers.current.has(modelName)) {
        return;
      }

      const controller = new AbortController();
      controllers.current.set(modelName, controller);
      setStates((previous) => ({
        ...previous,
        [modelName]: { status: "starting" },
      }));

      void (async () => {
        try {
          for await (const event of pull(modelName, controller.signal)) {
            setStates((previous) => ({
              ...previous,
              [modelName]: applyPullEvent(
                previous[modelName] ?? IDLE_DOWNLOAD,
                event,
              ),
            }));
          }
          setStates((previous) => ({
            ...previous,
            [modelName]: { status: "completed" },
          }));
          onCompletedRef.current?.(modelName);
        } catch (error) {
          if (controller.signal.aborted) {
            // Cancelling returns the row to its available state.
            setStates((previous) => ({
              ...previous,
              [modelName]: IDLE_DOWNLOAD,
            }));
            return;
          }

          setStates((previous) => ({
            ...previous,
            [modelName]: {
              status: "error",
              error: error instanceof Error ? error.message : String(error),
            },
          }));
        } finally {
          controllers.current.delete(modelName);
        }
      })();
    },
    [pull],
  );

  const cancel = useCallback((modelName: string) => {
    const controller = controllers.current.get(modelName);
    if (controller) {
      controller.abort();
      return;
    }

    setStates((previous) => ({ ...previous, [modelName]: IDLE_DOWNLOAD }));
  }, []);

  return { states, start, cancel };
}
