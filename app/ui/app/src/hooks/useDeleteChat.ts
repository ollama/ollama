import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteChat } from "@/api";
import type { ChatInfo } from "@/gotypes";

export function useDeleteChat() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (chatId: string) => deleteChat(chatId),
    retry: false,
    networkMode: "always",
    onSuccess: async (_, chatId) => {
      await queryClient.cancelQueries({ queryKey: ["history-chats"] });
      queryClient.setQueryData<ChatInfo[]>(["history-chats"], (chats) =>
        chats?.filter((chat) => chat.id !== chatId),
      );
      queryClient.removeQueries({
        queryKey: ["history-chat", chatId],
        exact: true,
      });
    },
  });
}
