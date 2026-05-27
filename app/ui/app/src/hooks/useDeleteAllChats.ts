import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteAllChats } from "@/api";
import { useNavigate } from "@tanstack/react-router";

export function useDeleteAllChats() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: () => deleteAllChats(),
    onSuccess: () => {
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
