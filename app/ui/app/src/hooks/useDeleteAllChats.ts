import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteAllChats } from "@/api";
import { useNavigate } from "@tanstack/react-router";

export function useDeleteAllChats() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: () => deleteAllChats(),
    onSuccess: () => {
      // Invalidate first so the sidebar is already cleared when we navigate
      queryClient.invalidateQueries({ queryKey: ["chats"] });
      navigate({ to: "/c/$chatId", params: { chatId: "new" } });
    },
    onError: (error) => {
      console.error("Failed to delete all chats:", error);
    },
  });
}
