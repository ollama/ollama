import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteChat, getChats } from "@/api";
import { useNavigate } from "@tanstack/react-router";

export function useDeleteAllChats() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: async () => {
      const chatsResponse = await getChats();
      const chatIds = chatsResponse.chatInfos?.map((chat) => chat.id) || [];

      if (chatIds.length === 0) {
        return { deletedCount: 0 };
      }

      const results = await Promise.allSettled(
        chatIds.map((chatId) => deleteChat(chatId))
      );

      const deletedCount = results.filter(
        (result) => result.status === "fulfilled"
      ).length;

      const failures = results.filter((result) => result.status === "rejected");
      if (failures.length > 0) {
        console.error(
          `Failed to delete ${failures.length} chat(s):`,
          failures.map((f) => (f as PromiseRejectedResult).reason)
        );
      }

      return { deletedCount, totalCount: chatIds.length };
    },
    onSuccess: () => {
      navigate({ to: "/c/$chatId", params: { chatId: "new" } });
      queryClient.invalidateQueries({ queryKey: ["chats"] });
    },
    onError: (error) => {
      console.error("Failed to delete all chats:", error);
    },
  });
}
