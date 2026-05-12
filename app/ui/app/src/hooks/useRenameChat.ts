import { useMutation, useQueryClient } from "@tanstack/react-query";
import { renameChat } from "@/api";

export function useRenameChat() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ chatId, title }: { chatId: string; title: string }) =>
      renameChat(chatId, title),
    onSuccess: (_, { chatId }) => {
      // Invalidate and refetch chats list
      queryClient.invalidateQueries({ queryKey: ["chats"] });
      // Invalidate the specific chat to update its title
      queryClient.invalidateQueries({ queryKey: ["chat", chatId] });
    },
    onError: (error) => {
      console.error("Failed to rename chat:", error);
    },
  });
}
