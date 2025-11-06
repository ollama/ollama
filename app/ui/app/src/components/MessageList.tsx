import { Message as MessageType, DownloadEvent, ErrorEvent } from "@/gotypes";
import React from "react";
import Message from "./Message";
import Downloading from "./Downloading";
import { ErrorMessage } from "./ErrorMessage";

export default function MessageList({
  messages,
  spacerHeight,
  isWaitingForLoad,
  isStreaming,
  downloadProgress,
  onEditMessage,
  editingMessageIndex,
  error,
  browserToolResult,
}: {
  messages: MessageType[];
  spacerHeight: number;
  isWaitingForLoad?: boolean;
  isStreaming: boolean;
  downloadProgress?: DownloadEvent;
  onEditMessage?: (content: string, index: number) => void | Promise<void>;
  editingMessageIndex?: number;
  error?: ErrorEvent | null;
  browserToolResult?: any;
}) {
  const [showDots, setShowDots] = React.useState(false);
  const isDownloadingModel = downloadProgress && !downloadProgress.done;
  const shouldShowDownload = messages.length > 0;

  React.useEffect(() => {
    let timer: number;
    if (
      (isStreaming || isWaitingForLoad) &&
      !isDownloadingModel &&
      messages.length > 0 &&
      messages[messages.length - 1]?.role === "user"
    ) {
      timer = window.setTimeout(() => {
        setShowDots(true);
      }, 750); // Wait 750ms before showing dots
    } else {
      setShowDots(false);
    }

    return () => window.clearTimeout(timer);
  }, [isStreaming, isWaitingForLoad, isDownloadingModel, messages]);

  const lastIdx = messages.length - 1;

  // Memoize the last tool query (web_search query or web_fetch url) at each message index
  const lastToolQueries = React.useMemo(() => {
    const queries: (string | undefined)[] = [];
    let lastQuery: string | undefined = undefined;
    for (let i = 0; i < messages.length; i++) {
      const m: any = messages[i] as any;
      const toolCalls: any[] | undefined = Array.isArray(m?.tool_calls)
        ? (m.tool_calls as any[])
        : m?.tool_call
          ? [m.tool_call]
          : undefined;
      if (toolCalls && toolCalls.length > 0) {
        for (const tc of toolCalls) {
          const name = tc?.function?.name;
          if (name === "web_search" || name === "web_fetch") {
            try {
              const args = JSON.parse(tc.function.arguments || "{}");
              const candidate =
                typeof args.query === "string" && args.query.trim()
                  ? String(args.query).trim()
                  : typeof args.url === "string" && args.url.trim()
                    ? String(args.url).trim()
                    : "";
              if (candidate) lastQuery = candidate;
            } catch {}
          }
        }
      }
      queries.push(lastQuery);
    }
    return queries;
  }, [messages]);

  return (
    <div
      className="mx-auto flex max-w-[768px] flex-1 flex-col px-6 pb-12 select-text"
      data-role="message-list"
    >
      {messages.map((message, idx) => {
        const lastToolQuery = lastToolQueries[idx];
        return (
          <div key={`${message.created_at}-${idx}`} data-message-index={idx}>
            <Message
              message={message}
              onEditMessage={onEditMessage}
              messageIndex={idx}
              isStreaming={isStreaming && idx === lastIdx}
              isFaded={
                editingMessageIndex !== undefined && idx >= editingMessageIndex
              }
              browserToolResult={browserToolResult}
              lastToolQuery={lastToolQuery}
            />
          </div>
        );
      })}

      {/* Inline error message */}
      {error &&
        error.code !== "usage_limit_upgrade" &&
        error.code !== "cloud_unauthorized" && <ErrorMessage error={error} />}

      {/* Indeterminate loading indicator */}
      {showDots && (
        <div className="flex items-center space-x-1.5 py-3 mt-2 self-start rounded-full px-4 min-w-0 bg-neutral-100 dark:bg-neutral-800 border-neutral-200 dark:border-neutral-700">
          <div
            className="w-1.5 h-1.5 bg-neutral-400 dark:bg-neutral-500 rounded-full opacity-0"
            style={{
              animation: "typing 1.4s infinite",
              animationDelay: "0s",
            }}
          />
          <div
            className="w-1.5 h-1.5 bg-neutral-400 dark:bg-neutral-500 rounded-full opacity-0"
            style={{
              animation: "typing 1.4s infinite",
              animationDelay: "0.15s",
            }}
          />
          <div
            className="w-1.5 h-1.5 bg-neutral-400 dark:bg-neutral-500 rounded-full opacity-0"
            style={{
              animation: "typing 1.4s infinite",
              animationDelay: "0.3s",
            }}
          />
        </div>
      )}

      {/* Downloading model */}
      {/* Only show for models larger than 1KiB */}
      {downloadProgress?.total && downloadProgress.total > 1024 && (
        <section
          className={`
          transition-all ease-out
          ${shouldShowDownload ? "duration-300" : "duration-0"}
          ${
            downloadProgress
              ? downloadProgress.done
                ? "opacity-0 -translate-y-8"
                : "opacity-100 translate-y-0"
              : "opacity-0 translate-y-4 pointer-events-none"
          }
        `}
        >
          <Downloading
            completed={downloadProgress?.completed || 0}
            total={downloadProgress?.total || 0}
          />
        </section>
      )}

      {/* Dynamic spacer to allow scrolling the last message to the top of the container */}
      <div style={{ height: `${spacerHeight}px` }} aria-hidden="true" />
    </div>
  );
}
