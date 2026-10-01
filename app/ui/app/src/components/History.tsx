import { useEffect, useState } from "react";
import { flushSync } from "react-dom";
import { ArrowDownTrayIcon, ArrowPathIcon } from "@heroicons/react/24/outline";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Navigate } from "@tanstack/react-router";
import { exportChat, getChat, getChats } from "@/api";
import { SidebarLayout } from "./layout/layout";
import { ChatSidebarList } from "./ChatSidebarList";
import { AppNavigation } from "./AppSidebar";
import MessageList from "./MessageList";
import { isWindowsPlatform } from "@/lib/platform";
import { useDeleteChat } from "@/hooks/useDeleteChat";

// Retain the selected conversation when visiting Apps or Settings.
let lastSelectedID = "";

export function History({
  chatId,
  onSelect,
}: {
  chatId?: string;
  onSelect?: (id: string) => void;
}) {
  const [requestedID, setSelectedID] = useState(
    chatId && chatId !== "new" ? chatId : lastSelectedID,
  );
  const deletion = useDeleteChat();
  const list = useQuery({
    queryKey: ["history-chats"],
    retry: false,
    networkMode: "always",
    queryFn: getChats,
  });
  const chats = list.data ?? [];
  const selectedID =
    chats.find((chat) => chat.id === requestedID)?.id ?? chats[0]?.id ?? "";
  const selected = useQuery({
    queryKey: ["history-chat", selectedID],
    enabled: !!selectedID,
    retry: false,
    networkMode: "always",
    queryFn: () => getChat(selectedID),
  });
  const selectedChat = selected.data;
  const lastModel = selectedChat?.messages
    .slice()
    .reverse()
    .find((message) => message.role === "assistant" && message.model?.trim())
    ?.model?.trim();
  const [isExporting, setIsExporting] = useState(false);
  const exporting = useMutation({
    mutationFn: exportChat,
    // Allow a painted frame before the native save dialog takes over.
    onMutate: () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      }),
    onSettled: () => setIsExporting(false),
    retry: false,
    networkMode: "always",
  });
  const loading = list.isPending;
  const messageLoading = !!selectedID && selected.isPending;
  const exportError =
    exporting.variables === selectedID ? exporting.error : null;
  const result = exporting.variables === selectedID ? exporting.data : null;
  const error = list.error ?? selected.error ?? deletion.error ?? exportError;
  const busy = isExporting || deletion.isPending;

  useEffect(() => {
    if (chatId && chatId !== "new") setSelectedID(chatId);
  }, [chatId]);

  useEffect(() => {
    lastSelectedID = selectedID;
    deletion.reset();
    exporting.reset();
  }, [selectedID, deletion.reset, exporting.reset]);

  if (list.isSuccess && chats.length === 0) {
    return <Navigate to="/connect" replace />;
  }

  return (
    <SidebarLayout
      sidebar={
        <ChatSidebarList
          chatInfos={chats}
          currentChatId={selectedID}
          isLoading={loading}
          navigation={<AppNavigation current="chat" />}
          onChatContextMenu={async (event, chat) => {
            event.preventDefault();
            if (busy) return;
            const action = await window.menu([
              { label: "Delete", enabled: true },
            ]);
            if (
              action !== "Delete" ||
              !window.confirm("Are you sure you want to remove this chat?")
            )
              return;
            deletion.mutate(chat.id, {
              onSuccess: () => {
                if (chat.id !== selectedID) return;
                const nextID =
                  chats.find((saved) => saved.id !== chat.id)?.id ?? "";
                setSelectedID(nextID);
                onSelect?.(nextID || "new");
              },
            });
          }}
          renderChat={(chat) => (
            <button
              aria-current={chat.id === selectedID ? "page" : undefined}
              disabled={busy}
              onClick={() => {
                setSelectedID(chat.id);
                onSelect?.(chat.id);
              }}
              className="flex-1 flex items-center min-w-0 px-2 py-2 select-none text-left"
              title={chat.title || chat.userExcerpt}
            >
              <span className="truncate font-sans text-sm">
                {chat.title ||
                  chat.userExcerpt ||
                  chat.createdAt.toLocaleString()}
              </span>
            </button>
          )}
        />
      }
    >
      <main className="flex min-h-0 flex-1 w-full flex-col relative allow-context-menu select-none">
        <section
          key={selectedID}
          className={`flex-1 overflow-y-auto overscroll-contain relative min-h-0 select-none ${isWindowsPlatform() ? "xl:pt-4" : "xl:pt-8"}`}
        >
          {loading || messageLoading ? (
            <p className="mx-auto max-w-[768px] px-6 py-4 text-sm text-neutral-500">
              Loading chats…
            </p>
          ) : selectedChat ? (
            <MessageList
              messages={selectedChat.messages}
              browserToolResult={selectedChat.browser_state}
            />
          ) : (
            <p className="mx-auto max-w-[768px] px-6 py-4 text-sm text-neutral-500">
              No conversations found.
            </p>
          )}
        </section>
        <div className="flex-shrink-0">
          <div className="mx-auto max-w-[768px] px-6 pb-4">
            {error && (
              <p
                role="alert"
                className="mb-3 break-words text-sm text-red-600 dark:text-red-400"
              >
                {error.message}
              </p>
            )}
            {(lastModel || (result && !isExporting)) && (
              <div className="mb-2 flex items-baseline justify-between gap-3 pt-3 text-xs text-neutral-500 dark:text-neutral-400">
                {result && !isExporting && (
                  <p role="status" className="shrink-0">
                    Export complete.
                  </p>
                )}
                {lastModel && (
                  <p className="ml-auto min-w-0 break-all text-right">
                    Last used: {lastModel}
                  </p>
                )}
              </div>
            )}
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-3xl border border-neutral-200 bg-white px-4 py-3 dark:border-neutral-700 dark:bg-neutral-900">
              <p className="w-3/4 flex-none text-sm text-neutral-500 dark:text-neutral-400">
                The Ollama app no longer supports chat—export this conversation
                to continue with an{" "}
                <a
                  href="https://docs.ollama.com/integrations"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline"
                >
                  integration
                </a>
                .
              </p>
              <button
                onClick={() => {
                  if (busy) return;
                  flushSync(() => setIsExporting(true));
                  exporting.mutate(selectedID);
                }}
                disabled={!selectedChat || busy || messageLoading}
                aria-busy={isExporting || undefined}
                className="inline-flex items-center justify-center gap-2 rounded-full bg-neutral-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 disabled:cursor-wait dark:bg-white dark:text-neutral-900"
              >
                {isExporting ? (
                  <ArrowPathIcon
                    aria-hidden="true"
                    className="h-4 w-4 animate-spin motion-reduce:animate-none"
                  />
                ) : (
                  <ArrowDownTrayIcon aria-hidden="true" className="h-4 w-4" />
                )}
                {isExporting ? "Exporting…" : "Export"}
              </button>
            </div>
          </div>
        </div>
      </main>
    </SidebarLayout>
  );
}
