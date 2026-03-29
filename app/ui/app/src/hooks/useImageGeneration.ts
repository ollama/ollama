import { useState, useCallback, useRef } from "react";
import { generateImage } from "@/api";

interface ImageGenerationState {
  isGenerating: boolean;
  image: string | null;
  progress: { completed: number; total: number } | null;
  error: string | null;
}

export function useImageGeneration() {
  const [state, setState] = useState<ImageGenerationState>({
    isGenerating: false,
    image: null,
    progress: null,
    error: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const generate = useCallback(
    async (
      model: string,
      prompt: string,
      options?: {
        width?: number;
        height?: number;
        steps?: number;
        seed?: number;
      },
    ) => {
      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setState({
        isGenerating: true,
        image: null,
        progress: null,
        error: null,
      });

      try {
        for await (const event of generateImage(
          model,
          prompt,
          controller.signal,
          options,
        )) {
          if (event.image) {
            setState((prev) => ({ ...prev, image: event.image! }));
          }
          if (
            event.completed !== undefined &&
            event.total !== undefined &&
            event.total > 0
          ) {
            setState((prev) => ({
              ...prev,
              progress: {
                completed: event.completed!,
                total: event.total!,
              },
            }));
          }
          if (event.done) {
            setState((prev) => ({
              ...prev,
              isGenerating: false,
              progress: null,
            }));
          }
        }
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          setState((prev) => ({
            ...prev,
            isGenerating: false,
            progress: null,
          }));
          return;
        }
        setState((prev) => ({
          ...prev,
          isGenerating: false,
          progress: null,
          error: err instanceof Error ? err.message : String(err),
        }));
      }
    },
    [],
  );

  const abort = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  }, []);

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
      image: null,
      progress: null,
      error: null,
    });
  }, []);

  return { ...state, generate, abort, reset };
}
