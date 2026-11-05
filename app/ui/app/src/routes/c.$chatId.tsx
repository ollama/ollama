import { createFileRoute, redirect } from "@tanstack/react-router";
import { useChat } from "@/hooks/useChats";
import Chat from "@/components/Chat";
import { getChat } from "@/api";
import { SidebarLayout } from "@/components/layout/layout";
import { ChatSidebar } from "@/components/ChatSidebar";
import { useEffect } from "react";
import { useSettings } from "@/hooks/useSettings";

export const Route = createFileRoute("/c/$chatId")({
  component: RouteComponent,
  beforeLoad: ({ params }) => {
    if (params.chatId === "launch") {
      throw redirect({
        to: "/c/$chatId",
        params: { chatId: "new" },
        mask: { to: "/" },
      });
    }
  },
  loader: async ({ context, params }) => {
    if (params.chatId !== "new") {
      context.queryClient.ensureQueryData({
        queryKey: ["chat", params.chatId],
        queryFn: () => getChat(params.chatId),
        staleTime: 1500,
      });
    }
  },
});

function RouteComponent() {
  const { chatId } = Route.useParams();
  const { settingsData, setSettings } = useSettings();

  // Always call hooks at the top level - use a flag to skip data when chatId is a special view
  const {
    data: chatData,
    isLoading: chatLoading,
    error: chatError,
  } = useChat(chatId === "new" ? "" : chatId);

  useEffect(() => {
    if (!settingsData) {
      return;
    }

    if (settingsData.LastHomeView === "chat") {
      return;
    }

    setSettings({ LastHomeView: "chat" }).catch(() => {
      // Best effort persistence for home view preference.
    });
  }, [chatId, settingsData, setSettings]);

  // Handle "new" chat case - just use Chat component which handles everything
  if (chatId === "new") {
    return (
      <SidebarLayout sidebar={<ChatSidebar currentChatId={chatId} />}>
        <Chat chatId={chatId} />
      </SidebarLayout>
    );
  }

  // Handle existing chat case
  if (chatLoading) {
    return (
      <SidebarLayout sidebar={<ChatSidebar currentChatId={chatId} />}>
        <div className="p-4">Loading chat...</div>
      </SidebarLayout>
    );
  }

  if (chatError) {
    return (
      <SidebarLayout sidebar={<ChatSidebar currentChatId={chatId} />}>
        <div className="p-4 text-red-500">Error loading chat</div>
      </SidebarLayout>
    );
  }

  if (!chatData) {
    return (
      <SidebarLayout sidebar={<ChatSidebar currentChatId={chatId} />}>
        <div className="p-4">Chat not found</div>
      </SidebarLayout>
    );
  }

  return (
    <SidebarLayout sidebar={<ChatSidebar currentChatId={chatId} />}>
      <Chat chatId={chatId} />
    </SidebarLayout>
  );
}
