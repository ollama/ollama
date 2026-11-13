import { useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { pullModel } from "@/api";

export type PullState =
  | { status: "idle" }
  | { status: "downloading"; completed: number; total: number }
  | { status: "done" }
  | { status: "error"; message: string };

/**
 * Downloads a model using the existing pull endpoint and reports progress.
 * Progress is tracked per model so several downloads can run independently.
 */
export function usePullModel() {
  const queryClient = useQueryClient();
  const [states, setStates] = useState<Record<string, PullState>>({});

  const setState = useCallback((model: string, state: PullState) => {
    setStates((prev) => ({ ...prev, [model]: state }));
  }, []);

  const pull = useCallback(
    async (model: string) => {
      setState(model, { status: "downloading", completed: 0, total: 0 });

      try {
        for await (const event of pullModel(model)) {
          if (event.done) {
            setState(model, { status: "done" });
          } else {
            setState(model, {
              status: "downloading",
              completed: event.completed ?? 0,
              total: event.total ?? 0,
            });
          }
        }
        setState(model, { status: "done" });
        // The installed model list changed, so refresh it.
        queryClient.invalidateQueries({ queryKey: ["models"] });
      } catch (err) {
        setState(model, {
          status: "error",
          message: err instanceof Error ? err.message : "Download failed",
        });
      }
    },
    [queryClient, setState],
  );

  const stateFor = useCallback(
    (model: string): PullState => states[model] ?? { status: "idle" },
    [states],
  );

  return { pull, stateFor };
}
