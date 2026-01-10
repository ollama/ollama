import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getChats, getChat, sendMessage, type ChatEventUnion } from "../api";
import { Chat, ErrorEvent, Model } from "@/gotypes";
import { Message } from "@/gotypes";
import { useSelectedModel } from "./useSelectedModel";
import { createQueryBatcher } from "./useQueryBatcher";
import { useRefetchModels } from "./useModels";
import { useStreamingContext } from "@/contexts/StreamingContext";
import { useSettings } from "./useSettings";
import { getModelCapabilities } from "@/api";

export const useChats = () => {
  return useQuery({
    queryKey: ["chats"],
    queryFn: getChats,
  });
};

export const useChat = (chatId: string) => {
  const queryClient = useQueryClient();
  const { streamingChatIds } = useStreamingContext();

  return useQuery({
    queryKey: ["chat", chatId],
    queryFn: async () => {
      // Check if we have optimistic data and this chat is currently streaming
      const existingData = queryClient.getQueryData<{ chat: Chat }>([
        "chat",
        chatId,
      ]);
      const isStreaming = streamingChatIds.has(chatId);

      const response = await getChat(chatId);

      // If we have existing optimistic data with more messages than the server
      // and this chat is currently streaming, preserve the optimistic data
      if (existingData && isStreaming) {
        const existingCount = existingData.chat.messages?.length || 0;
        const serverCount = response.chat?.messages?.length || 0;

        if (existingCount > serverCount) {
          return existingData;
        }
      }
      // Process messages to ensure tool calls are properly structured
      if (response.chat && response.chat.messages) {
        response.chat.messages = response.chat.messages.map((msg) => {
          // If this is a tool message without tool_calls but has content about a tool call
          if (
            msg.role === "tool" &&
            (!msg.tool_calls || msg.tool_calls.length === 0)
          ) {
            // Check if content indicates this is a tool call (not a tool result)
            const toolCallMatch = msg.content.match(/Tool (\w+) called/);
            if (toolCallMatch) {
              // This is likely a tool call message that lost its structure
              // For now, we'll leave it as-is but could enhance this later
              // to parse the content and reconstruct the tool_calls array
            }
          }
          return msg;
        });
      }
      return response;
    },
    enabled: !!chatId && chatId !== "new",
    staleTime: 1500,
  });
};

export const useStaleModels = () => {
  return useQuery({
    queryKey: ["staleModels"],
    queryFn: () => new Map<string, boolean>(),
    initialData: new Map<string, boolean>(),
    staleTime: Infinity,
    gcTime: Infinity,
  });
};

export const useDismissedStaleModels = () => {
  return useQuery({
    queryKey: ["dismissedStaleModels"],
    queryFn: () => new Set<string>(),
    initialData: new Set<string>(),
    staleTime: Infinity,
    gcTime: Infinity,
  });
};

export const useChatError = (chatId: string) => {
  return useQuery({
    queryKey: ["chatError", chatId],
    queryFn: () => null as ErrorEvent | null,
    initialData: null,
    staleTime: Infinity,
    gcTime: 1000 * 60 * 5, // Keep in cache for 5 minutes
  });
};

export const useIsStreaming = (chatId: string) => {
  const { streamingChatIds } = useStreamingContext();
  return streamingChatIds.has(chatId);
};

export const useDownloadProgress = (chatId: string) => {
  const { downloadProgress } = useStreamingContext();
  return downloadProgress.get(chatId);
};

export const useIsModelStale = (modelName: string) => {
  const { data: staleModels } = useStaleModels();
  return staleModels?.get(modelName) || false;
};

export const useShouldShowStaleDisplay = (model: Model | null) => {
  const isStale = useIsModelStale(model?.model || "");
  const { data: dismissedModels } = useDismissedStaleModels();
  const {
    settings: { airplaneMode },
  } = useSettings();

  if (model?.isCloud() && !airplaneMode) {
    return false;
  }

  return isStale && !dismissedModels?.has(model?.model || "");
};

export const useDismissStaleModel = () => {
  const queryClient = useQueryClient();
  const refetchModels = useRefetchModels();

  return (modelName: string) => {
    const currentDismissedModels =
      queryClient.getQueryData<Set<string>>(["dismissedStaleModels"]) ||
      new Set();
    const newSet = new Set(currentDismissedModels);
    newSet.add(modelName);
    queryClient.setQueryData(["dismissedStaleModels"], newSet);

    refetchModels();
  };
};

// Helper hook to check if we should show loading bar (streaming but no first token yet)
export const useIsWaitingForLoad = (chatId: string) => {
  const { streamingChatIds, loadingChats } = useStreamingContext();
  const { selectedModel } = useSelectedModel();
  const { data: chatsData } = useChats();
  const queryClient = useQueryClient();

  // Basic check: is this chat streaming but hasn't loaded yet?
  const isWaitingForLoad =
    streamingChatIds.has(chatId) && !loadingChats.has(chatId);

  if (!isWaitingForLoad || !selectedModel || !chatsData?.chatInfos) {
    return isWaitingForLoad;
  }

  // Find the most recent chat that isn't the current one
  const sortedChats = [...chatsData.chatInfos].sort(
    (a, b) => b.updatedAt.getTime() - a.updatedAt.getTime(),
  );
  const mostRecentOtherChat = sortedChats.find((chat) => chat.id !== chatId);
  if (!mostRecentOtherChat) {
    return isWaitingForLoad;
  }

  // Check if the most recent chat used the same model within 5 minutes
  const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
  const wasRecentlyActive =
    mostRecentOtherChat.updatedAt.getTime() > fiveMinutesAgo;
  if (!wasRecentlyActive) {
    return isWaitingForLoad;
  }
  const recentChatData = queryClient.getQueryData<{ chat: Chat }>([
    "chat",
    mostRecentOtherChat.id,
  ]);
  if (
    !recentChatData?.chat?.messages ||
    recentChatData.chat.messages.length === 0
  ) {
    return isWaitingForLoad;
  }
  const lastAssistantMessage = [...recentChatData.chat.messages]
    .reverse()
    .find((msg) => msg.role === "assistant" && msg.model);
  if (!lastAssistantMessage?.model) {
    return isWaitingForLoad;
  }

  // If the same model was used recently, skip the loading bar
  const isSameModel = lastAssistantMessage.model === selectedModel.model;
  return isWaitingForLoad && !isSameModel;
};

export const useSendMessage = (chatId: string) => {
  let updatableChatId = chatId;
  const queryClient = useQueryClient();
  const { selectedModel } = useSelectedModel();
  const {
    setStreamingChatIds,
    loadingChats,
    setLoadingChats,
    setAbortControllers,
    setDownloadProgress,
  } = useStreamingContext();

  const cleanupStreaming = (id: string) => {
    setStreamingChatIds((prev: Set<string>) => {
      const newSet = new Set(prev);
      newSet.delete(id);
      return newSet;
    });
    setAbortControllers((prev) => {
      const newMap = new Map(prev);
      newMap.delete(id);
      return newMap;
    });
    setDownloadProgress((prev) => {
      const newMap = new Map(prev);
      newMap.delete(id);
      return newMap;
    });
  };

  return useMutation({
    mutationKey: ["sendMessage", chatId],
    onSuccess: () => {
      cleanupStreaming(updatableChatId);
    },
    onError: (error) => {
      console.error("error mutating sendMessage", error);
      cleanupStreaming(updatableChatId);
    },
    mutationFn: async ({
      message,
      attachments,
      index,
      webSearch,
      fileTools,
      forceUpdate,
      think,
      onChatEvent,
    }: {
      message: string;
      attachments?: Array<{ filename: string; data: Uint8Array }>;
      index?: number;
      webSearch?: boolean;
      fileTools?: boolean;
      forceUpdate?: boolean;
      think?: boolean | string;
      onChatEvent?: (event: ChatEventUnion) => void;
    }) => {
      // For existing chats, set streaming state and add optimistic user message
      if (chatId !== "new") {
        setStreamingChatIds((prev: Set<string>) => {
          const newSet = new Set(prev);
          newSet.add(chatId);
          return newSet;
        });
        queryClient.cancelQueries({ queryKey: ["chat", chatId] });

        // Only add optimistic message for non-empty messages
        if (message.trim() !== "") {
          // Optimistically add the user message
          queryClient.setQueryData(
            ["chat", chatId],
            (old: { chat: Chat } | undefined) => {
              if (!old) return old;

              const newMessage = new Message({
                role: "user",
                content: message,
                attachments: attachments,
              });

              let messages = old.chat.messages || [];

              // If editing a message (index provided), truncate messages array
              if (
                index !== undefined &&
                index >= 0 &&
                index < messages.length
              ) {
                messages = messages.slice(0, index);
              }

              return {
                ...old,
                chat: new Chat({
                  ...old.chat,
                  messages: [...messages, newMessage],
                }),
              };
            },
          );
        }
      }

      if (!selectedModel) {
        throw new Error("No model selected");
      }

      const effectiveModel = new Model({
        model: selectedModel.model,
        digest: selectedModel.digest,
        modified_at: selectedModel.modified_at,
      });

      const abortController = new AbortController();
      setAbortControllers((prev) => {
        const newMap = new Map(prev);
        newMap.set(updatableChatId, abortController);
        return newMap;
      });

      const events = sendMessage(
        chatId,
        message,
        effectiveModel,
        attachments,
        abortController.signal,
        index,
        webSearch,
        fileTools,
        forceUpdate,
        think,
      );
      let currentChatId = chatId;
      let isCancelled = false;

      // Listen for abort signal to set cancelled flag
      abortController.signal.addEventListener("abort", () => {
        isCancelled = true;
      });

      // Create batcher for streaming updates with smoother intervals, prevents state update depth being exceeded
      // and allows for smoother updates at high frame rates
      let batcher = createQueryBatcher<{ chat: Chat }>(
        queryClient,
        ["chat", currentChatId],
        { batchInterval: 4, immediateFirst: true }, // ~250fps for smoother updates
      );

      for await (const event of events) {
        // If cancelled, continue draining the stream but don't update UI
        if (isCancelled) {
          continue;
        }

        // download events don't count as loaded
        // TODO(jmorganca): loading should potentially be an event instead of
        // reducing it this way
        if (
          event.eventName !== "download" &&
          !loadingChats.has(currentChatId)
        ) {
          // If this is the first time loading this chat, mark it as loaded
          setLoadingChats((prev: Set<string>) => {
            const newSet = new Set(prev);
            newSet.add(currentChatId);
            return newSet;
          });
        }

        switch (event.eventName) {
          case "chat": {
            // Update the current chat data with streaming content
            batcher.scheduleBatch((old: { chat: Chat } | undefined) => {
              if (!old) return old;

              const existingMessages = old.chat.messages || [];
              const newMessages = [...existingMessages];

              // Find or create the assistant message
              let lastMessage = newMessages[newMessages.length - 1];
              if (!lastMessage || lastMessage.role !== "assistant") {
                newMessages.push(
                  new Message({
                    role: "assistant",
                    content: "",
                    thinking: "",
                    model: effectiveModel,
                  }),
                );
                lastMessage = newMessages[newMessages.length - 1];
              }

              // Update the last message with new content
              if (lastMessage) {
                const updatedContent =
                  (lastMessage.content || "") + (event.content || "");
                const updatedThinking =
                  (lastMessage.thinking || "") + (event.thinking || "");
                const updatedMessage = new Message({
                  ...lastMessage,
                  content: updatedContent,
                  thinking: updatedThinking,
                });
                if (event.thinkingTimeStart) {
                  updatedMessage.thinkingTimeStart = event.thinkingTimeStart;
                }
                if (event.thinkingTimeEnd) {
                  updatedMessage.thinkingTimeEnd = event.thinkingTimeEnd;
                }
                newMessages[newMessages.length - 1] = updatedMessage;
              }

              return {
                ...old,
                chat: new Chat({
                  ...old.chat,
                  messages: newMessages,
                }),
              };
            });
            break;
          }
          case "thinking": {
            // Handle thinking content
            batcher.scheduleBatch((old: { chat: Chat } | undefined) => {
              if (!old) return old;

              const existingMessages = old.chat.messages || [];
              const newMessages = [...existingMessages];

              // Find or create the assistant message
              let lastMessage = newMessages[newMessages.length - 1];
              if (!lastMessage || lastMessage.role !== "assistant") {
                newMessages.push(
                  new Message({
                    role: "assistant",
                    content: "",
                    thinking: "",
                    model: effectiveModel,
                  }),
                );
                lastMessage = newMessages[newMessages.length - 1];
              }

              // Update the last message with new thinking content
              if (lastMessage) {
                const updatedThinking =
                  (lastMessage.thinking || "") + (event.thinking || "");
                const updatedMessage = new Message({
                  ...lastMessage,
                  thinking: updatedThinking,
                });
                if (event.thinkingTimeStart) {
                  updatedMessage.thinkingTimeStart = event.thinkingTimeStart;
                }
                newMessages[newMessages.length - 1] = updatedMessage;
              }

              return {
                ...old,
                chat: new Chat({
                  ...old.chat,
                  messages: newMessages,
                }),
              };
            });
            break;
          }
          case "tool_call": {
            // Handle tool call events - these are now mostly handled by assistant_with_tools
            // but kept for backward compatibility, potentially still good for normal tool calling models
            queryClient.setQueryData(
              ["chat", currentChatId],
              (old: { chat: Chat } | undefined) => {
                if (!old) return old;

                const existingMessages = old.chat.messages || [];
                const newMessages = [...existingMessages];

                // Add tool call message
                if (event.toolCall) {
                  newMessages.push(
                    new Message({
                      role: "tool",
                      content: `Tool ${event.toolCall.function.name} called`,
                      tool_calls: [event.toolCall],
                      thinkingTimeStart: event.thinkingTimeStart,
                      thinkingTimeEnd: event.thinkingTimeEnd,
                    }),
                  );
                }

                return {
                  ...old,
                  chat: new Chat({
                    ...old.chat,
                    messages: newMessages,
                  }),
                };
              },
            );
            break;
          }
          case "assistant_with_tools": {
            // Handle assistant messages that include tool calls
            queryClient.setQueryData(
              ["chat", currentChatId],
              (old: { chat: Chat } | undefined) => {
                if (!old) return old;

                const existingMessages = old.chat.messages || [];
                const newMessages = [...existingMessages];

                // Find the last assistant message and update it with tool calls
                const lastMessage = newMessages[newMessages.length - 1];
                if (lastMessage && lastMessage.role === "assistant") {
                  // Update existing assistant message with tool calls
                  const updatedMessage = new Message({
                    ...lastMessage,
                    content: lastMessage.content + (event.content || ""),
                    thinking: lastMessage.thinking + (event.thinking || ""),
                    tool_calls: event.toolCalls,
                    thinkingTimeStart:
                      lastMessage.thinkingTimeStart || event.thinkingTimeStart,
                    thinkingTimeEnd: event.thinkingTimeEnd,
                    model: selectedModel,
                  });
                  newMessages[newMessages.length - 1] = updatedMessage;
                } else {
                  // No existing assistant message, create new one
                  newMessages.push(
                    new Message({
                      role: "assistant",
                      content: event.content,
                      thinking: event.thinking,
                      tool_calls: event.toolCalls,
                      thinkingTimeStart: event.thinkingTimeStart,
                      thinkingTimeEnd: event.thinkingTimeEnd,
                      model: selectedModel,
                    }),
                  );
                }

                return {
                  ...old,
                  chat: new Chat({
                    ...old.chat,
                    messages: newMessages,
                  }),
                };
              },
            );
            break;
          }
          case "tool_result": {
            // Handle tool result events
            queryClient.setQueryData(
              ["chat", currentChatId],
              (old: { chat: Chat } | undefined) => {
                if (!old) return old;

                const existingMessages = old.chat.messages || [];
                const newMessages = [...existingMessages];

                newMessages.push(
                  Object.assign(
                    new Message({
                      role: "tool",
                      content: event.content,
                      thinkingTimeStart: event.thinkingTimeStart,
                      thinkingTimeEnd: event.thinkingTimeEnd,
                    }),
                    {
                      tool_result: (event as any).toolResultData,
                      ...((event as any).toolName
                        ? { tool_name: (event as any).toolName }
                        : {}),
                    },
                  ),
                );

                return {
                  ...old,
                  chat: new Chat({
                    ...old.chat,
                    messages: newMessages,
                    browser_state: event.toolState ?? old.chat.browser_state,
                  }),
                };
              },
            );
            break;
          }
          case "download": {
            setDownloadProgress((prev) => {
              const newMap = new Map(prev);
              newMap.set(currentChatId, event);
              return newMap;
            });

            if (event.done && selectedModel) {
              const currentStaleModels =
                queryClient.getQueryData<Map<string, boolean>>([
                  "staleModels",
                ]) || new Map();
              const newStaleMap = new Map(currentStaleModels);
              newStaleMap.delete(selectedModel.model);
              queryClient.setQueryData(["staleModels"], newStaleMap);

              queryClient.invalidateQueries({ queryKey: ["models"] });

              // Fetch fresh capabilities for the downloaded model
              getModelCapabilities(selectedModel.model)
                .then((capabilities) => {
                  queryClient.setQueryData(
                    ["modelCapabilities", selectedModel.model],
                    capabilities,
                  );
                })
                .catch((error) => {
                  console.error(
                    "Failed to fetch capabilities after download:",
                    error,
                  );
                  queryClient.invalidateQueries({
                    queryKey: ["modelCapabilities", selectedModel.model],
                  });
                });
            }
            break;
          }
          case "error": {
            // Clean up streaming state
            setStreamingChatIds((prev: Set<string>) => {
              const newSet = new Set(prev);
              newSet.delete(currentChatId);
              return newSet;
            });
            setDownloadProgress((prev) => {
              const newMap = new Map(prev);
              newMap.delete(currentChatId);
              return newMap;
            });

            // Set error using separate React Query cache
            queryClient.setQueryData(
              ["chatError", currentChatId],
              event as ErrorEvent,
            );
            break;
          }
          case "done":
            // TODO(drifkin): update the chat with the thinking time for cases
            // where there is thinking content, but no other content (which
            // should be very rare)
            setStreamingChatIds((prev: Set<string>) => {
              const newSet = new Set(prev);
              newSet.delete(currentChatId);
              return newSet;
            });
            // Clear download progress when streaming is done
            setDownloadProgress((prev) => {
              const newMap = new Map(prev);
              newMap.delete(currentChatId);
              return newMap;
            });
            // Ensure chat is fresh for next fetch
            queryClient.invalidateQueries({
              queryKey: ["chat", currentChatId],
            });
            break;
          case "chat_created": {
            if (!event.chatId) break;
            const newId = event.chatId;
            updatableChatId = newId;
            setStreamingChatIds((prev: Set<string>) => {
              const newSet = new Set(prev);
              newSet.add(newId);
              return newSet;
            });
            setAbortControllers((prev) => {
              const newMap = new Map(prev);
              const controller = newMap.get(chatId);
              if (controller) {
                newMap.delete(chatId);
                newMap.set(newId, controller);
              }
              return newMap;
            });

            // Flush current batcher and create new one for the new chat ID
            batcher.flushBatch();
            batcher.cleanup();
            currentChatId = newId;
            batcher = createQueryBatcher<{ chat: Chat }>(
              queryClient,
              ["chat", currentChatId],
              { batchInterval: 4, immediateFirst: true },
            );

            // Create initial chat data for the new chat
            queryClient.setQueryData(["chat", newId], {
              chat: new Chat({
                id: newId,
                model: effectiveModel,
                messages: [
                  new Message({
                    role: "user",
                    content: message,
                    attachments: attachments,
                  }),
                ],
              }),
            });

            // Cancel the old "new" chat query if it exists
            if (chatId === "new") {
              queryClient.cancelQueries({ queryKey: ["chat", "new"] });
            }

            // Invalidate chats list to include the new chat
            queryClient.invalidateQueries({ queryKey: ["chats"] });
            break;
          }
        }
        onChatEvent?.(event);
      }

      // Flush any remaining batched updates and cleanup
      batcher.flushBatch();
      batcher.cleanup();
    },
  });
};

export const useCancelMessage = () => {
  const {
    abortControllers,
    setStreamingChatIds,
    setAbortControllers,
    setDownloadProgress,
  } = useStreamingContext();

  return (chatId: string) => {
    const controller = abortControllers.get(chatId);
    if (controller) {
      controller.abort();
      setStreamingChatIds(
        (prev) => new Set([...prev].filter((id) => id !== chatId)),
      );
      setAbortControllers((prev) => {
        const newMap = new Map(prev);
        newMap.delete(chatId);
        return newMap;
      });
      setDownloadProgress((prev) => {
        const newMap = new Map(prev);
        newMap.delete(chatId);
        return newMap;
      });
    }
  };
};
