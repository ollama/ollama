import MessageList from "./MessageList";
import ChatForm from "./ChatForm";
import { FileUpload } from "./FileUpload";
import { DisplayUpgrade } from "./DisplayUpgrade";
import { DisplayStale } from "./DisplayStale";
import { DisplayLogin } from "./DisplayLogin";
import {
  useChat,
  useSendMessage,
  useIsStreaming,
  useIsWaitingForLoad,
  useDownloadProgress,
  useChatError,
  useShouldShowStaleDisplay,
  useDismissStaleModel,
} from "@/hooks/useChats";
import { useHealth } from "@/hooks/useHealth";
import { useMessageAutoscroll } from "@/hooks/useMessageAutoscroll";
import {
  useState,
  useEffect,
  useLayoutEffect,
  useRef,
  useCallback,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useSelectedModel } from "@/hooks/useSelectedModel";
import { useUser } from "@/hooks/useUser";
import { useHasVisionCapability } from "@/hooks/useModelCapabilities";
import { Message } from "@/gotypes";

export default function Chat({ chatId }: { chatId: string }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const chatQuery = useChat(chatId === "new" ? "" : chatId);
  const chatErrorQuery = useChatError(chatId === "new" ? "" : chatId);
  const { selectedModel } = useSelectedModel(chatId);
  const { user } = useUser();
  const hasVisionCapability = useHasVisionCapability(selectedModel?.model);
  const shouldShowStaleDisplay = useShouldShowStaleDisplay(selectedModel);
  const dismissStaleModel = useDismissStaleModel();
  const { isHealthy } = useHealth();

  const [editingMessage, setEditingMessage] = useState<{
    content: string;
    index: number;
    originalMessage: Message;
  } | null>(null);
  const prevChatIdRef = useRef<string>(chatId);

  const chatFormCallbackRef = useRef<
    | ((
        files: Array<{ filename: string; data: Uint8Array; type?: string }>,
        errors: Array<{ filename: string; error: string }>,
      ) => void)
    | null
  >(null);

  const handleFilesReceived = useCallback(
    (
      callback: (
        files: Array<{
          filename: string;
          data: Uint8Array;
          type?: string;
        }>,
        errors: Array<{ filename: string; error: string }>,
      ) => void,
    ) => {
      chatFormCallbackRef.current = callback;
    },
    [],
  );

  const handleFilesProcessed = useCallback(
    (
      files: Array<{ filename: string; data: Uint8Array; type?: string }>,
      errors: Array<{ filename: string; error: string }> = [],
    ) => {
      chatFormCallbackRef.current?.(files, errors);
    },
    [],
  );

  const allMessages = chatQuery?.data?.chat?.messages ?? [];
  // TODO(parthsareen): will need to consolidate when used with more tools with state
  const browserToolResult = chatQuery?.data?.chat?.browser_state;
  const chatError = chatErrorQuery.data;

  const messages = allMessages;
  const isStreaming = useIsStreaming(chatId);
  const isWaitingForLoad = useIsWaitingForLoad(chatId);
  const downloadProgress = useDownloadProgress(chatId);
  const isDownloadingModel = downloadProgress && !downloadProgress.done;
  const isDisabled = !isHealthy;

  // Clear editing state when navigating to a different chat
  useEffect(() => {
    setEditingMessage(null);
  }, [chatId]);

  const sendMessageMutation = useSendMessage(chatId);

  const { containerRef, handleNewUserMessage, spacerHeight } =
    useMessageAutoscroll({
      messages,
      isStreaming,
      chatId,
    });

  // Scroll to bottom only when switching to a different existing chat
  useLayoutEffect(() => {
    // Only scroll if the chatId actually changed (not just messages updating)
    if (
      prevChatIdRef.current !== chatId &&
      containerRef.current &&
      messages.length > 0 &&
      chatId !== "new"
    ) {
      // Always scroll to the bottom when opening a chat
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
    prevChatIdRef.current = chatId;
  }, [chatId, messages.length]);

  // Simplified submit handler - ChatForm handles all the attachment logic
  const handleChatFormSubmit = (
    message: string,
    options: {
      attachments?: Array<{ filename: string; data: Uint8Array }>;
      index?: number;
      webSearch?: boolean;
      fileTools?: boolean;
      think?: boolean | string;
    },
  ) => {
    // Clear any existing errors when sending a new message
    sendMessageMutation.reset();
    if (chatError) {
      clearChatError();
    }

    // Prepare attachments for backend
    const allAttachments = (options.attachments || []).map((att) => ({
      filename: att.filename,
      data: att.data.length === 0 ? new Uint8Array(0) : att.data,
    }));

    sendMessageMutation.mutate({
      message,
      attachments: allAttachments,
      index: editingMessage ? editingMessage.index : options.index,
      webSearch: options.webSearch,
      fileTools: options.fileTools,
      think: options.think,
      onChatEvent: (event) => {
        if (event.eventName === "chat_created" && event.chatId) {
          navigate({
            to: "/c/$chatId",
            params: {
              chatId: event.chatId,
            },
          });
        }
      },
    });

    // Clear edit mode after submission
    setEditingMessage(null);
    handleNewUserMessage();
  };

  const handleEditMessage = (content: string, index: number) => {
    setEditingMessage({
      content,
      index,
      originalMessage: messages[index],
    });
  };

  const handleCancelEdit = () => {
    setEditingMessage(null);
    if (chatError) {
      clearChatError();
    }
  };

  const clearChatError = () => {
    queryClient.setQueryData(
      ["chatError", chatId === "new" ? "" : chatId],
      null,
    );
  };

  const isWindows = navigator.platform.toLowerCase().includes("win");

  return chatId === "new" || chatQuery ? (
    <FileUpload
      onFilesAdded={handleFilesProcessed}
      selectedModel={selectedModel}
      hasVisionCapability={hasVisionCapability}
    >
      {chatId === "new" ? (
        <div className="flex flex-col h-screen justify-center relative">
          <div className="px-6">
            <ChatForm
              hasMessages={false}
              onSubmit={handleChatFormSubmit}
              chatId={chatId}
              autoFocus={true}
              editingMessage={editingMessage}
              onCancelEdit={handleCancelEdit}
              isDownloadingModel={isDownloadingModel}
              isDisabled={isDisabled}
              onFilesReceived={handleFilesReceived}
            />
          </div>
        </div>
      ) : (
        <main className="flex h-screen w-full flex-col relative allow-context-menu select-none">
          <section
            key={chatId} // This key forces React to recreate the element when chatId changes
            ref={containerRef}
            className={`flex-1 overflow-y-auto overscroll-contain relative min-h-0 select-none ${isWindows ? "xl:pt-4" : "xl:pt-8"}`}
          >
            <MessageList
              messages={messages}
              spacerHeight={spacerHeight}
              isWaitingForLoad={isWaitingForLoad}
              isStreaming={isStreaming}
              downloadProgress={downloadProgress}
              onEditMessage={(content: string, index: number) => {
                handleEditMessage(content, index);
              }}
              editingMessageIndex={editingMessage?.index}
              error={chatError}
              browserToolResult={browserToolResult}
            />
          </section>

          <div className="flex-shrink-0 sticky bottom-0 z-20">
            {selectedModel && shouldShowStaleDisplay && (
              <div className="pb-2">
                <DisplayStale
                  model={selectedModel}
                  onDismiss={() =>
                    dismissStaleModel(selectedModel?.model || "")
                  }
                  chatId={chatId}
                  onScrollToBottom={() => {
                    if (containerRef.current) {
                      containerRef.current.scrollTo({
                        top: containerRef.current.scrollHeight,
                        behavior: "smooth",
                      });
                    }
                  }}
                />
              </div>
            )}
            {chatError && chatError.code === "usage_limit_upgrade" && (
              <div className="pb-2">
                <DisplayUpgrade
                  error={chatError}
                  onDismiss={clearChatError}
                  href={
                    user?.plan === "pro"
                      ? "https://ollama.com/settings/billing"
                      : "https://ollama.com/upgrade"
                  }
                />
              </div>
            )}
            {chatError && chatError.code === "cloud_unauthorized" && (
              <div className="pb-2">
                <DisplayLogin error={chatError} />
              </div>
            )}
            <ChatForm
              hasMessages={messages.length > 0}
              onSubmit={handleChatFormSubmit}
              chatId={chatId}
              autoFocus={true}
              editingMessage={editingMessage}
              onCancelEdit={handleCancelEdit}
              isDisabled={isDisabled}
              isDownloadingModel={isDownloadingModel}
              onFilesReceived={handleFilesReceived}
            />
          </div>
        </main>
      )}
    </FileUpload>
  ) : (
    <div>Loading...</div>
  );
}
