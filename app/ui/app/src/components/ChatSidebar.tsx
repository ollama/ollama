import { useChats } from "@/hooks/useChats";
import { useRenameChat } from "@/hooks/useRenameChat";
import { useDeleteChat } from "@/hooks/useDeleteChat";
import { useQueryClient } from "@tanstack/react-query";
import { getChat } from "@/api";
import { Link } from "@/components/ui/link";
import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { ChatsResponse } from "@/gotypes";
import { CogIcon } from "@heroicons/react/24/outline";

// there's a hidden debug feature to copy a chat's data to the clipboard by
// holding shift and clicking this many times within this many seconds
const DEBUG_SHIFT_CLICKS_REQUIRED = 5;
const DEBUG_SHIFT_CLICK_WINDOW_MS = 7000; // 7 seconds

interface ChatSidebarProps {
  currentChatId?: string;
}

export function ChatSidebar({ currentChatId }: ChatSidebarProps) {
  const { data, isLoading, error } = useChats();
  const queryClient = useQueryClient();
  const renameMutation = useRenameChat();
  const deleteMutation = useDeleteChat();
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const [shiftClicks, setShiftClicks] = useState<Record<string, number[]>>({});
  const [copiedChatId, setCopiedChatId] = useState<string | null>(null);

  const handleMouseEnter = useCallback(
    (chatId: string) => {
      queryClient.prefetchQuery({
        queryKey: ["chat", chatId],
        queryFn: () => getChat(chatId),
        staleTime: 1500,
      });
    },
    [queryClient],
  );

  const startEditing = useCallback((chatId: string, currentTitle: string) => {
    setEditingChatId(chatId);
    setEditValue(currentTitle);
  }, []);

  const saveRename = useCallback(async () => {
    if (!editingChatId || !editValue.trim()) {
      setEditingChatId(null);
      return;
    }

    const newTitle = editValue.trim();
    const chatId = editingChatId;

    // Exit edit mode immediately to prevent flash
    setEditingChatId(null);
    setEditValue("");

    // Optimistically update the cache
    queryClient.setQueryData(
      ["chats"],
      (oldData: ChatsResponse | undefined) => {
        if (!oldData?.chatInfos) return oldData;
        return {
          ...oldData,
          chatInfos: oldData.chatInfos.map((chat) =>
            chat.id === chatId ? { ...chat, title: newTitle } : chat,
          ),
        };
      },
    );

    try {
      await renameMutation.mutateAsync({
        chatId: chatId,
        title: newTitle,
      });
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (error: unknown) {
      // Revert optimistic update on error
      queryClient.invalidateQueries({ queryKey: ["chats"] });
    }
  }, [editingChatId, editValue, renameMutation, queryClient]);

  useEffect(() => {
    if (editingChatId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingChatId]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        saveRename();
      }
    };

    if (editingChatId) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => {
        document.removeEventListener("mousedown", handleClickOutside);
      };
    }
  }, [editingChatId, editValue, saveRename]);

  const sortedChats = useMemo(() => {
    if (!data?.chatInfos) return [];
    return [...data.chatInfos].sort((a, b) => {
      const comparison = b.updatedAt.getTime() - a.updatedAt.getTime();
      if (comparison === 0) {
        return b.id.localeCompare(a.id);
      }
      return comparison;
    });
  }, [data?.chatInfos]);

  const isToday = (date: Date) => {
    const today = new Date();
    return (
      date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear()
    );
  };

  const isThisWeek = (date: Date) => {
    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    return date > weekAgo && !isToday(date);
  };

  // Group chats by time period
  const groupedChats = useMemo(() => {
    const groups = {
      today: [] as typeof sortedChats,
      thisWeek: [] as typeof sortedChats,
      older: [] as typeof sortedChats,
    };

    sortedChats.forEach((chat) => {
      if (isToday(chat.updatedAt)) {
        groups.today.push(chat);
      } else if (isThisWeek(chat.updatedAt)) {
        groups.thisWeek.push(chat);
      } else {
        groups.older.push(chat);
      }
    });

    return groups;
  }, [sortedChats]);

  const chatGroups = useMemo(() => {
    return [
      { name: "Today", chats: groupedChats.today },
      { name: "This week", chats: groupedChats.thisWeek },
      { name: "Older", chats: groupedChats.older },
    ].filter((group) => group.chats.length > 0);
  }, [groupedChats]);

  const handleDeleteChat = useCallback(
    async (chatId: string) => {
      const confirmed = window.confirm(
        `Are you sure you want to remove this chat?`,
      );

      if (!confirmed) return;

      try {
        await deleteMutation.mutateAsync(chatId);
      } catch (error) {
        console.error("Failed to delete chat:", error);
      }
    },
    [deleteMutation],
  );

  // implementation of the hidden debug feature to copy a chat's data to the clipboard
  const handleShiftClick = useCallback(
    async (e: React.MouseEvent, chatId: string) => {
      if (!e.shiftKey) return false;

      e.preventDefault();
      const now = Date.now();

      const clicks = shiftClicks[chatId] || [];
      const recentClicks = clicks.filter(
        (timestamp) => now - timestamp < DEBUG_SHIFT_CLICK_WINDOW_MS,
      );
      recentClicks.push(now);

      setShiftClicks((prev) => ({
        ...prev,
        [chatId]: recentClicks,
      }));

      if (recentClicks.length >= DEBUG_SHIFT_CLICKS_REQUIRED) {
        try {
          const chatData = await getChat(chatId);
          const jsonString = JSON.stringify(chatData, null, 2);
          await navigator.clipboard.writeText(jsonString);

          // visual feedback
          setCopiedChatId(chatId);
          setTimeout(() => setCopiedChatId(null), 2000);

          setShiftClicks((prev) => ({
            ...prev,
            [chatId]: [],
          }));
        } catch (error) {
          console.error("Failed to copy chat data:", error);
        }
      }

      return true;
    },
    [shiftClicks],
  );

  const handleContextMenu = useCallback(
    async (_: React.MouseEvent, chatId: string, chatTitle: string) => {
      const selectedAction = await window.menu([
        { label: "Rename", enabled: true },
        { label: "Delete", enabled: true },
      ]);

      if (selectedAction === "Rename") {
        startEditing(chatId, chatTitle);
      } else if (selectedAction === "Delete") {
        handleDeleteChat(chatId);
      }
    },
    [startEditing, handleDeleteChat],
  );

  if (isLoading) {
    return (
      <nav className="flex min-h-0 flex-col">
        <div className="flex flex-1 flex-col p-4">
          <div className="p-4">Loading...</div>
        </div>
      </nav>
    );
  }

  if (error) {
    return (
      <nav className="flex min-h-0 flex-col">
        <div className="flex flex-1 flex-col p-4">
          <div className="p-4 text-red-500">Error loading chats</div>
        </div>
      </nav>
    );
  }

  const isWindows = navigator.platform.toLowerCase().includes("win");

  return (
    <nav className="flex flex-1 flex-col min-h-0 select-none">
      <header className="flex flex-col gap-0.5 px-4 pb-2">
        <Link
          href="/c/new"
          mask={{ to: "/" }}
          className={`flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-sm text-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800 dark:text-neutral-100 ${
            currentChatId === "new" ? "bg-neutral-100 dark:bg-neutral-800" : ""
          }`}
          draggable={false}
        >
          <svg
            className="h-5 w-5 fill-current"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M17.0859 3.39949L15.2135 5.27196H7.27028C5.78649 5.27196 4.94684 6.11336 4.94684 7.59716V16.664C4.94684 18.1558 5.78649 18.9892 7.27028 18.9892H16.3406C17.8324 18.9892 18.6623 18.1558 18.6623 16.664V8.79514L20.5428 6.9115C20.567 7.11532 20.5773 7.33066 20.5773 7.55419V16.7149C20.5773 19.4069 19.0818 20.9024 16.3898 20.9024H7.22107C4.53708 20.9024 3.03357 19.4069 3.03357 16.7149V7.55419C3.03357 4.8622 4.53708 3.35869 7.22107 3.35869H16.3898C16.6329 3.35869 16.8662 3.37094 17.0859 3.39949Z" />
            <path d="M9.92714 14.381L11.914 13.5403L20.8312 4.63114L19.3404 3.1581L10.433 12.0655L9.55234 13.9964C9.45664 14.2169 9.70293 14.4714 9.92714 14.381ZM21.5767 3.89364L22.2588 3.19384C22.6347 2.80184 22.6435 2.2663 22.2711 1.90536L22.0148 1.64287C21.6822 1.31377 21.1334 1.36513 20.7689 1.72158L20.0859 2.39833L21.5767 3.89364Z" />
          </svg>
          <span className="truncate">New Chat</span>
        </Link>
        {isWindows && (
          <Link
            href="/settings"
            className={`flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-sm text-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800 dark:text-neutral-300`}
            draggable={false}
          >
            <CogIcon className="h-5 w-5 stroke-current" />
            <span className="truncate">Settings</span>
          </Link>
        )}
      </header>
      <div className="flex flex-1 flex-col px-4 py-1 overflow-y-auto overscroll-auto scrollbar-gutter">
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
                  onMouseEnter={() => handleMouseEnter(chat.id)}
                  onContextMenu={(e) =>
                    handleContextMenu(
                      e,
                      chat.id,
                      chat.title ||
                        chat.userExcerpt ||
                        chat.createdAt.toLocaleString(),
                    )
                  }
                >
                  {editingChatId === chat.id ? (
                    <div className="flex-1 flex items-center min-w-0 px-2 py-2 bg-neutral-100 text-black dark:bg-neutral-800 rounded-lg">
                      <span className="truncate font-sans text-sm w-full">
                        <input
                          ref={inputRef}
                          type="text"
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              e.preventDefault();
                              saveRename();
                            } else if (e.key === "Escape") {
                              setEditingChatId(null);
                              setEditValue("");
                            }
                          }}
                          className="bg-transparent border-0 focus:outline-none w-full dark:text-white"
                          style={{
                            font: "inherit",
                            lineHeight: "inherit",
                            padding: 0,
                            margin: 0,
                          }}
                        />
                      </span>
                    </div>
                  ) : (
                    <Link
                      to="/c/$chatId"
                      params={{ chatId: chat.id }}
                      className="flex-1 flex items-center min-w-0 px-2 py-2 select-none"
                      onClick={(e) => {
                        handleShiftClick(e, chat.id);
                      }}
                      draggable={false}
                    >
                      <span className="truncate font-sans text-sm">
                        {chat.title ||
                          chat.userExcerpt ||
                          chat.createdAt.toLocaleString()}
                      </span>
                      {copiedChatId === chat.id && (
                        <span className="ml-2 text-xs text-green-600 dark:text-green-400">
                          Copied!
                        </span>
                      )}
                    </Link>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </nav>
  );
}
