import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Chat, ErrorEvent, Model } from "@/gotypes";
import {
  StreamingProvider,
  useStreamingContext,
} from "@/contexts/StreamingContext";
import { useCancelMessage, useChatError, useSendMessage } from "./useChats";

vi.mock("./useSelectedModel", () => ({
  useSelectedModel: () => ({ selectedModel: new Model({ model: "test" }) }),
}));

const cleanups: Array<() => Promise<void>> = [];

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(async () => {
  for (const cleanup of cleanups.splice(0)) await cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function renderChat(chatId: string) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: Infinity },
      mutations: { retry: false, gcTime: 0 },
    },
  });
  client.setQueryData(["chat", chatId], {
    chat: new Chat({ id: chatId, messages: [] }),
  });
  let mutation: ReturnType<typeof useSendMessage>;
  let streaming: ReturnType<typeof useStreamingContext>;
  let cancel: ReturnType<typeof useCancelMessage>;
  let renderer: ReactTestRenderer;
  function Probe() {
    mutation = useSendMessage(chatId);
    streaming = useStreamingContext();
    cancel = useCancelMessage();
    const { data: error } = useChatError(chatId === "new" ? "" : chatId);
    return <span>{error?.error}</span>;
  }
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={client}>
        <StreamingProvider>
          <Probe />
        </StreamingProvider>
      </QueryClientProvider>,
    );
  });
  cleanups.push(async () => {
    await act(async () => renderer.unmount());
    client.clear();
  });
  return {
    client,
    get streaming() {
      return streaming!;
    },
    cancel: (id: string) => cancel!(id),
    start: () => mutation!.mutateAsync({ message: "hello" }),
  };
}

function controlledResponse() {
  let controller: ReadableStreamDefaultController<Uint8Array>;
  const response = new Response(
    new ReadableStream<Uint8Array>({
      start(value) {
        controller = value;
      },
    }),
  );
  return {
    response,
    send: (...events: object[]) =>
      controller!.enqueue(
        new TextEncoder().encode(
          events.map((event) => JSON.stringify(event)).join("\n") + "\n",
        ),
      ),
    fail: (error: Error) => controller!.error(error),
    close: () => controller!.close(),
  };
}

describe("chat stream failures", () => {
  it("shows an unexpected transport abort when the user did not cancel", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockRejectedValue(
          new DOMException("The operation was aborted", "AbortError"),
        ),
    );
    const chat = await renderChat("existing");
    await act(async () => {
      await expect(chat.start()).rejects.toThrow("The operation was aborted");
    });
    expect(
      chat.client.getQueryData<ErrorEvent>(["chatError", "existing"])?.error,
    ).toBe("The operation was aborted");
    expect(chat.streaming.streamingChatIds.size).toBe(0);
  });

  it.each(["existing", "new"])(
    "preserves server error codes for a %s chat",
    async (chatId) => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(
            JSON.stringify({
              eventName: "error",
              error: "Sign in required",
              code: "cloud_unauthorized",
            }),
          ),
        ),
      );
      const chat = await renderChat(chatId);
      await act(async () => {
        await chat.start();
      });
      expect(
        chat.client.getQueryData(["chatError", chatId === "new" ? "" : chatId]),
      ).toMatchObject({
        error: "Sign in required",
        code: "cloud_unauthorized",
      });
      expect(chat.streaming.streamingChatIds.size).toBe(0);
      expect(chat.streaming.abortControllers.size).toBe(0);
    },
  );

  it("finishes a normal response without an error or leftover streaming state", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            '{"eventName":"chat","content":"complete"}\n{"eventName":"done"}\n',
          ),
        ),
    );
    const chat = await renderChat("existing");
    await act(async () => {
      await chat.start();
    });
    expect(chat.client.getQueryData(["chatError", "existing"])).toBeNull();
    expect(
      chat.client
        .getQueryData<{ chat: Chat }>(["chat", "existing"])
        ?.chat.messages?.at(-1)?.content,
    ).toBe("complete");
    expect(chat.streaming.streamingChatIds.size).toBe(0);
    expect(chat.streaming.abortControllers.size).toBe(0);
    expect(chat.streaming.loadingChats.size).toBe(0);
  });

  it.each(["existing", "new"])(
    "shows a request error for a %s chat and clears streaming state",
    async (chatId) => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockRejectedValue(new TypeError("Load failed")),
      );
      const chat = await renderChat(chatId);
      await act(async () => {
        await expect(chat.start()).rejects.toThrow("Load failed");
      });
      expect(
        chat.client.getQueryData(["chatError", chatId === "new" ? "" : chatId]),
      ).toMatchObject({
        eventName: "error",
        error: "Load failed",
      });
      expect(chat.streaming.streamingChatIds.size).toBe(0);
      expect(chat.streaming.abortControllers.size).toBe(0);
    },
  );

  it("reports a failure after chat creation against the new ID and preserves buffered content", async () => {
    const stream = controlledResponse();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(stream.response));
    const chat = await renderChat("new");
    let request: Promise<unknown>;
    await act(async () => {
      request = chat.start();
      request.catch(() => {});
    });
    await act(async () =>
      stream.send(
        { eventName: "chat_created", chatId: "created" },
        { eventName: "download", total: 100, completed: 50 },
        { eventName: "chat", content: "first" },
        { eventName: "chat", content: " second" },
      ),
    );
    expect(chat.streaming.streamingChatIds.has("created")).toBe(true);
    expect(chat.streaming.abortControllers.has("created")).toBe(true);
    await act(async () => {
      stream.fail(new TypeError("Connection lost"));
      await expect(request!).rejects.toThrow("Connection lost");
    });
    expect(
      chat.client.getQueryData<ErrorEvent>(["chatError", "created"])?.error,
    ).toBe("Connection lost");
    expect(
      chat.client
        .getQueryData<{ chat: Chat }>(["chat", "created"])
        ?.chat.messages?.at(-1)?.content,
    ).toBe("first second");
    expect(chat.streaming.streamingChatIds.size).toBe(0);
    expect(chat.streaming.abortControllers.size).toBe(0);
    expect(chat.streaming.downloadProgress.size).toBe(0);
    expect(chat.streaming.loadingChats.size).toBe(0);
  });

  it("keeps waiting for inference after chat creation and cleans up a failure before the first token", async () => {
    const stream = controlledResponse();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(stream.response));
    const chat = await renderChat("new");
    let request: Promise<unknown>;
    await act(async () => {
      request = chat.start();
      request.catch(() => {});
    });
    await act(async () =>
      stream.send({ eventName: "chat_created", chatId: "created" }),
    );
    expect(chat.streaming.streamingChatIds.has("created")).toBe(true);
    expect(chat.streaming.loadingChats.size).toBe(0);
    await act(async () => {
      stream.fail(new TypeError("Connection lost"));
      await expect(request!).rejects.toThrow("Connection lost");
    });
    expect(chat.streaming.streamingChatIds.size).toBe(0);
    expect(chat.streaming.loadingChats.size).toBe(0);
    expect(chat.streaming.abortControllers.size).toBe(0);
  });

  it("shows a premature EOF instead of leaving the chat waiting", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response('{"eventName":"chat","content":"partial"}\n'),
        ),
    );
    const chat = await renderChat("existing");
    await act(async () => {
      await expect(chat.start()).rejects.toThrow("response was complete");
    });
    expect(
      chat.client.getQueryData<ErrorEvent>(["chatError", "existing"])?.error,
    ).toContain("response was complete");
    expect(chat.streaming.streamingChatIds.size).toBe(0);
  });

  it("keeps an intentional cancellation quiet", async () => {
    const stream = controlledResponse();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((_url, init: RequestInit) => {
        init.signal!.addEventListener("abort", () =>
          stream.fail(new DOMException("Aborted", "AbortError")),
        );
        return Promise.resolve(stream.response);
      }),
    );
    const chat = await renderChat("existing");
    let request: Promise<unknown>;
    await act(async () => {
      request = chat.start();
      request.catch(() => {});
    });
    await act(async () => {
      chat.cancel("existing");
      await expect(request!).rejects.toThrow("Aborted");
    });
    expect(chat.client.getQueryData(["chatError", "existing"])).toBeNull();
    expect(chat.streaming.streamingChatIds.size).toBe(0);
    expect(chat.streaming.abortControllers.size).toBe(0);
  });
});
