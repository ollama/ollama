import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteAllChats } from "@/api";
import { useNavigate } from "@tanstack/react-router";
import { useStreamingContext } from "@/contexts/StreamingContext";

export function useDeleteAllChats() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
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
      // immediately so the sidebar is empty on navigation and no stale
      // deleted-chat data can be served from cache on a direct URL visit.
      queryClient.removeQueries({ queryKey: ["chats"] });
      queryClient.removeQueries({ queryKey: ["chat"] });
      navigate({ to: "/c/$chatId", params: { chatId: "new" } });
    },
    onError: (error) => {
      console.error("Failed to delete all chats:", error);
    },
  });
}
