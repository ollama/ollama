import { useMemo, type ReactNode, type MouseEvent } from "react";
import type { ChatInfo } from "@/gotypes";

export function ChatSidebarList({
  chatInfos,
  currentChatId,
  isLoading,
  error,
  navigation,
  renderChat,
  onChatContextMenu,
}: {
  chatInfos?: ChatInfo[];
  currentChatId?: string;
  isLoading?: boolean;
  error?: unknown;
  navigation: ReactNode;
  renderChat: (chat: ChatInfo) => ReactNode;
  onChatContextMenu?: (event: MouseEvent, chat: ChatInfo) => void;
}) {
  const chatGroups = useMemo(() => {
    const sorted = [...(chatInfos ?? [])].sort((a, b) => {
      const comparison = b.updatedAt.getTime() - a.updatedAt.getTime();
      return comparison || b.id.localeCompare(a.id);
    });
    const today = new Date();
    const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
    const groups = [
      { name: "Today", chats: [] as ChatInfo[] },
      { name: "This week", chats: [] as ChatInfo[] },
      { name: "Older", chats: [] as ChatInfo[] },
    ];
    for (const chat of sorted) {
      const isToday = chat.updatedAt.toDateString() === today.toDateString();
      groups[isToday ? 0 : chat.updatedAt > weekAgo ? 1 : 2].chats.push(chat);
    }
    return groups.filter((group) => group.chats.length > 0);
  }, [chatInfos]);

  return (
    <nav
      aria-label="Chats"
      aria-busy={isLoading || undefined}
      className="flex flex-1 flex-col min-h-0 select-none"
    >
      <header className="flex flex-col gap-0.5 px-4 pb-2">{navigation}</header>
      <div className="flex flex-1 flex-col px-4 py-1 overflow-y-auto overscroll-auto scrollbar-gutter">
        {error ? (
          <div className="px-2 pt-4 text-sm text-red-500">
            Error loading chats
          </div>
        ) : (
          <div className="flex flex-col gap-3 pt-4">
            {chatGroups.map((group) => (
              <div key={group.name} className="flex flex-col gap-0.5">
                <h3 className="text-xs font-medium text-neutral-400 dark:text-neutral-500 px-2 py-1 select-none">
                  {group.name}
                </h3>
                {group.chats.map((chat) => (
                  <div
                    key={chat.id}
                    className={`allow-context-menu flex items-center relative text-sm text-neutral-800 dark:text-neutral-400 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 ${
                      chat.id === currentChatId
                        ? "bg-neutral-100 text-black dark:bg-neutral-800"
                        : ""
                    }`}
                    onContextMenu={
                      onChatContextMenu
                        ? (event) => onChatContextMenu(event, chat)
                        : undefined
                    }
                  >
                    {renderChat(chat)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
}
