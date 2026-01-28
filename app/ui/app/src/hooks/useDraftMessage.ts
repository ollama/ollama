import { useCallback } from "react";
import { updateChatDraft } from "@/api";

export function useDraftMessage(chatId: string) {
  const saveDraft = useCallback(async (content: string) => {
    try {
      if (chatId === "new") {
        return;
      }

      await updateChatDraft(chatId, content);
    } catch (error) {
      console.error("Error saving draft message:", error);
    }
  }, [chatId]);

  const clearDraft = useCallback(async () => {
    try {
      if (chatId === "new") {
        return;
      }

      await updateChatDraft(chatId, "");
    } catch (error) {
      console.error("Error clearing draft message:", error);
    }
  }, [chatId]);

  return {
    saveDraft,
    clearDraft,
  };
}

