import type { Message as MessageType } from "@/gotypes";
import { useMemo } from "react";
import Message from "./Message";

export default function MessageList({
  messages,
  browserToolResult,
}: {
  messages: MessageType[];
  browserToolResult?: any;
}) {
  // Memoize the last tool query (web_search query or web_fetch url) at each message index
  const lastToolQueries = useMemo(() => {
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
            } catch {
              /* ignored */
            }
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
      {messages.map((message, index) => (
        <div key={`${message.created_at}-${index}`} data-message-index={index}>
          <Message
            message={message}
            browserToolResult={browserToolResult}
            lastToolQuery={lastToolQueries[index]}
          />
        </div>
      ))}
    </div>
  );
}
