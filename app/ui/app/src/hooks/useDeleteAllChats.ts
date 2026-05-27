import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteAllChats } from "@/api";
import { useStreamingContext } from "@/contexts/StreamingContext";

export function useDeleteAllChats() {
  const queryClient = useQueryClient();
  const {
    abortControllers,
    setStreamingChatIds,
    setAbortControllers,
    setDownloadProgress,
  } = useStreamingContext();

  return useMutation({
    mutationFn: () => {
      // Abort every active stream before issuing the delete so that each
      // server-side streaming goroutine sees a cancelled context and skips
      // its final SetChat upsert. Without this, a goroutine that completes
      // *after* DELETE FROM chats would re-insert the deleted chat row.
      for (const controller of abortControllers.values()) {
        controller.abort();
      }
      return deleteAllChats();
    },
    onSuccess: () => {
      // Clear all streaming state now that every stream has been aborted.
      setStreamingChatIds(new Set());
      setAbortControllers(new Map());
      setDownloadProgress(new Map());
      // Remove the chat list and all per-chat entries from the cache
      // immediately so the sidebar is empty when the user navigates back
      // and no stale deleted-chat data can be served from cache.
      queryClient.removeQueries({ queryKey: ["chats"] });
      queryClient.removeQueries({ queryKey: ["chat"] });
    },
    onError: (error) => {
      console.error("Failed to delete all chats:", error);
    },
  });
}
