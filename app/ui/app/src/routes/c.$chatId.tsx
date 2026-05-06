import { createFileRoute } from "@tanstack/react-router";
import { useChat } from "@/hooks/useChats";
import Chat from "@/components/Chat";
import { getChat } from "@/api";
import { SidebarLayout } from "@/components/layout/layout";
import { ChatSidebar } from "@/components/ChatSidebar";
import LaunchCommands from "@/components/LaunchCommands";
import { useEffect, useRef } from "react";
import { useSettings } from "@/hooks/useSettings";

const launchSidebarRequestedKey = "ollama.launchSidebarRequested";
const launchSidebarSeenKey = "ollama.launchSidebarSeen";
const fallbackSessionState = new Map<string, string>();

function getSessionState() {
  if (typeof sessionStorage !== "undefined") {
    return sessionStorage;
  }

  return {
    getItem(key: string) {
      return fallbackSessionState.get(key) ?? null;
    },
    setItem(key: string, value: string) {
      fallbackSessionState.set(key, value);
    },
    removeItem(key: string) {
      fallbackSessionState.delete(key);
    },
  };
}

export const Route = createFileRoute("/c/$chatId")({
  component: RouteComponent,
  loader: async ({ context, params }) => {
    // Skip loading for special non-chat views
    if (params.chatId !== "new" && params.chatId !== "launch") {
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
  const previousChatIdRef = useRef<string | null>(null);

  // Always call hooks at the top level - use a flag to skip data when chatId is a special view
  const {
    data: chatData,
    isLoading: chatLoading,
    error: chatError,
  } = useChat(chatId === "new" || chatId === "launch" ? "" : chatId);

  useEffect(() => {
    if (!settingsData) {
      return;
    }

    const previousChatId = previousChatIdRef.current;
    previousChatIdRef.current = chatId;

    if (chatId === "launch") {
      const sessionState = getSessionState();
      const shouldOpenSidebar =
        previousChatId !== "launch" &&
        (() => {
          if (sessionState.getItem(launchSidebarRequestedKey) === "1") {
            sessionState.removeItem(launchSidebarRequestedKey);
            sessionState.setItem(launchSidebarSeenKey, "1");
            return true;
          }

          if (sessionState.getItem(launchSidebarSeenKey) !== "1") {
            sessionState.setItem(launchSidebarSeenKey, "1");
            return true;
          }

          return false;
        })();
      const updates: { LastHomeView?: string; SidebarOpen?: boolean } = {};

      if (settingsData.LastHomeView !== "launch") {
        updates.LastHomeView = "launch";
      }

      if (shouldOpenSidebar && !settingsData.SidebarOpen) {
        updates.SidebarOpen = true;
      }

      if (Object.keys(updates).length === 0) {
        return;
      }

      setSettings(updates).catch(() => {
        // Best effort persistence for home view preference.
      });
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

  if (chatId === "launch") {
    return (
      <SidebarLayout sidebar={<ChatSidebar currentChatId={chatId} />}>
        <LaunchCommands />
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
