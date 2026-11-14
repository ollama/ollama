import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, create } from "react-test-renderer";
import { afterEach, expect, it, vi } from "vitest";
import { ChatEvent, Chat, ErrorEvent, Message } from "@/gotypes";
import { StreamingProvider } from "@/contexts/StreamingContext";
import { sendMessage } from "@/api";
import { useSendMessage } from "./useChats";
import { useSelectedModel } from "./useSelectedModel";

vi.mock("@/api", () => ({ sendMessage: vi.fn() }));
vi.mock("./useSelectedModel", () => ({ useSelectedModel: vi.fn() }));

const chatId = "chat-1";

// A tool call pushes a role:"tool" message, so the next tool call is the
// "no existing assistant message" branch and a further one takes the update
// branch. That is where the missing content/thinking guards used to append to
// undefined and write the string "undefined" into the message.
function events(...payloads: Record<string, unknown>[]) {
  vi.mocked(sendMessage).mockImplementation(async function* () {
    for (const payload of payloads) {
      yield new ChatEvent(payload);
    }
  });
}

async function renderSend(messages: Message[]) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  client.setQueryData(["chat", chatId], {
    chat: new Chat({ id: chatId, messages }),
  });

  let mutate!: ReturnType<typeof useSendMessage>["mutate"];
  function Harness() {
    mutate = useSendMessage(chatId).mutate;
    return null;
  }

  const element = (
    <QueryClientProvider client={client}>
      <StreamingProvider>
        <Harness />
      </StreamingProvider>
    </QueryClientProvider>
  );

  let renderer: ReturnType<typeof create> | undefined;
  await act(async () => {
    renderer = create(element);
  });

  return {
    client,
    send: (message: string) =>
      act(async () => {
        mutate({ message });
      }),
    unmount: async () => {
      await act(async () => renderer?.unmount());
      client.clear();
    },
  };
}

const toolCall = (content?: string) => ({
  eventName: "assistant_with_tools",
  ...(content === undefined ? {} : { content }),
  toolCalls: [
    { type: "function", function: { name: "web_search", arguments: "{}" } },
  ],
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

it('does not write "undefined" into content or thinking across tool calls', async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.mocked(useSelectedModel).mockReturnValue({
    selectedModel: { model: "qwen3:8b", digest: "abc" },
  } as unknown as ReturnType<typeof useSelectedModel>);

  // Both tool-call events omit content/thinking, which the Go server does
  // whenever they are empty (omitempty).
  events(toolCall(), toolCall(), toolCall());

  const harness = await renderSend([
    new Message({ role: "user", content: "search the web" }),
  ]);
  try {
    await harness.send("search the web");

    const { chat } = harness.client.getQueryData(["chat", chatId]) as {
      chat: Chat;
    };
    const assistant = chat.messages.filter((m) => m.role === "assistant");

    expect(assistant.length).toBe(1);
    expect(assistant[0].content).toBe("");
    expect(assistant[0].thinking).toBe("");
  } finally {
    await harness.unmount();
  }
});

it("still accumulates content and thinking that the server does send", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.mocked(useSelectedModel).mockReturnValue({
    selectedModel: { model: "qwen3:8b", digest: "abc" },
  } as unknown as ReturnType<typeof useSelectedModel>);

  events({ ...toolCall(), content: "Looking", thinking: "hmm" }, toolCall(), {
    ...toolCall(),
    content: " up",
  });

  const harness = await renderSend([
    new Message({ role: "user", content: "search the web" }),
  ]);
  try {
    await harness.send("search the web");

    const { chat } = harness.client.getQueryData(["chat", chatId]) as {
      chat: Chat;
    };
    const assistant = chat.messages.filter((m) => m.role === "assistant");

    expect(assistant[0].content).toBe("Looking up");
    expect(assistant[0].thinking).toBe("hmm");
  } finally {
    await harness.unmount();
  }
});

it("surfaces a request that fails before the stream starts", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.mocked(useSelectedModel).mockReturnValue({
    selectedModel: { model: "qwen3:8b", digest: "abc" },
  } as unknown as ReturnType<typeof useSelectedModel>);

  vi.mocked(sendMessage).mockImplementation(() => {
    throw new Error('{"error":"Token is required"}');
  });

  const harness = await renderSend([
    new Message({ role: "user", content: "hi" }),
  ]);
  try {
    await harness.send("hi");

    const error = harness.client.getQueryData(["chatError", chatId]) as
      | ErrorEvent
      | undefined;
    expect(error?.error).toContain("Token is required");
  } finally {
    await harness.unmount();
  }
});
