import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { pullModel } from "@/api";
import { useSelectedModel } from "./useSelectedModel";
import { useSettings } from "./useSettings";

interface DownloadProgress {
  status: string;
  digest?: string;
  total?: number;
  completed?: number;
  done?: boolean;
}

export function useDownloadModel(chatId?: string) {
  const queryClient = useQueryClient();
  const { selectedModel } = useSelectedModel(chatId);
  const { setSettings } = useSettings();
  const [downloadProgress, setDownloadProgress] =
    useState<DownloadProgress | null>(null);
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);
  const [downloadingChatIds, setDownloadingChatIds] = useState<Set<string>>(
    new Set(),
  );

  const mutation = useMutation({
    mutationFn: async (modelName: string) => {
      const controller = new AbortController();
      setAbortController(controller);
      setDownloadProgress({ status: "Starting download..." });
      if (chatId) {
        setDownloadingChatIds((prev) => new Set(prev).add(chatId));
      }

      try {
        for await (const progress of pullModel(modelName, controller.signal)) {
          setDownloadProgress(progress);

          if (progress.status === "success") {
            // Update selected model to indicate it's now available locally
            if (selectedModel && selectedModel.model === modelName) {
              setSettings({ SelectedModel: modelName });
            }
            // Invalidate models query to refresh the list
            await queryClient.invalidateQueries({ queryKey: ["models"] });
            break;
          }
        }
      } finally {
        setAbortController(null);
        if (chatId) {
          setDownloadingChatIds((prev) => {
            const newSet = new Set(prev);
            newSet.delete(chatId);
            return newSet;
          });
        }
      }
    },
    onSuccess: () => {
      setDownloadProgress(null);
      if (chatId) {
        setDownloadingChatIds((prev) => {
          const newSet = new Set(prev);
          newSet.delete(chatId);
          return newSet;
        });
      }
    },
    onError: (error: Error) => {
      const status =
        error.name === "AbortError" ? "Download cancelled" : "Download failed";
      setDownloadProgress({ status, done: true });

      // Clear error message after delay
      const delay = error.name === "AbortError" ? 1500 : 3000;
      setTimeout(() => {
        setDownloadProgress(null);
        if (chatId) {
          setDownloadingChatIds((prev) => {
            const newSet = new Set(prev);
            newSet.delete(chatId);
            return newSet;
          });
        }
      }, delay);
    },
  });

  const cancelDownload = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      if (chatId) {
        setDownloadingChatIds((prev) => {
          const newSet = new Set(prev);
          newSet.delete(chatId);
          return newSet;
        });
      }
    }
  };

  return {
    downloadModel: mutation.mutate,
    isDownloading:
      mutation.isPending && chatId ? downloadingChatIds.has(chatId) : false,
    downloadProgress:
      chatId && downloadingChatIds.has(chatId) ? downloadProgress : null,
    error: mutation.error,
    cancelDownload,
  };
}
