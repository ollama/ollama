import { act, create } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Message as MessageType } from "@/gotypes";

vi.mock("./Message", () => ({
  default: ({ autoStart }: { autoStart?: boolean }) => (
    <div data-testid="message" data-auto-start={autoStart ? "true" : "false"} />
  ),
}));

import MessageList from "./MessageList";

beforeEach(() => {
  vi.stubGlobal("window", { clearTimeout: vi.fn(), setTimeout: vi.fn() });
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
});

afterEach(() => vi.unstubAllGlobals());

function assistantMessage() {
  return new MessageType({
    role: "assistant",
    content: "Completed response.",
    stream: false,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  });
}

describe("MessageList auto read", () => {
  it("starts only the response that just finished streaming", async () => {
    let renderer!: ReturnType<typeof create>;

    await act(async () => {
      renderer = create(
        <MessageList
          chatId="chat-1"
          messages={[assistantMessage()]}
          spacerHeight={0}
          isStreaming
          didCompleteStreaming={false}
          autoReadEnabled
        />,
      );
    });

    expect(
      renderer.root.findByProps({ "data-testid": "message" }).props[
        "data-auto-start"
      ],
    ).toBe("false");

    await act(async () => {
      renderer.update(
        <MessageList
          chatId="chat-1"
          messages={[assistantMessage()]}
          spacerHeight={0}
          isStreaming={false}
          didCompleteStreaming
          autoReadEnabled
        />,
      );
    });

    expect(
      renderer.root.findByProps({ "data-testid": "message" }).props[
        "data-auto-start"
      ],
    ).toBe("true");
  });

  it("does not start a cancelled response", async () => {
    let renderer!: ReturnType<typeof create>;

    await act(async () => {
      renderer = create(
        <MessageList
          chatId="chat-1"
          messages={[assistantMessage()]}
          spacerHeight={0}
          isStreaming
          didCompleteStreaming={false}
          autoReadEnabled
        />,
      );
    });

    await act(async () => {
      renderer.update(
        <MessageList
          chatId="chat-1"
          messages={[assistantMessage()]}
          spacerHeight={0}
          isStreaming={false}
          didCompleteStreaming={false}
          autoReadEnabled
        />,
      );
    });

    expect(
      renderer.root.findByProps({ "data-testid": "message" }).props[
        "data-auto-start"
      ],
    ).toBe("false");
  });

  it("does not read a response that completed before auto read was enabled", async () => {
    let renderer!: ReturnType<typeof create>;
    const onCompletionConsumed = vi.fn();

    await act(async () => {
      renderer = create(
        <MessageList
          chatId="chat-1"
          messages={[assistantMessage()]}
          spacerHeight={0}
          isStreaming
          didCompleteStreaming={false}
          autoReadEnabled={false}
          onCompletionConsumed={onCompletionConsumed}
        />,
      );
    });

    await act(async () => {
      renderer.update(
        <MessageList
          chatId="chat-1"
          messages={[assistantMessage()]}
          spacerHeight={0}
          isStreaming={false}
          didCompleteStreaming
          autoReadEnabled={false}
          onCompletionConsumed={onCompletionConsumed}
        />,
      );
    });

    expect(
      renderer.root.findByProps({ "data-testid": "message" }).props[
        "data-auto-start"
      ],
    ).toBe("false");
    expect(onCompletionConsumed).toHaveBeenCalledOnce();

    await act(async () => {
      renderer.update(
        <MessageList
          chatId="chat-1"
          messages={[assistantMessage()]}
          spacerHeight={0}
          isStreaming={false}
          didCompleteStreaming={false}
          autoReadEnabled
          onCompletionConsumed={onCompletionConsumed}
        />,
      );
    });

    expect(
      renderer.root.findByProps({ "data-testid": "message" }).props[
        "data-auto-start"
      ],
    ).toBe("false");
  });
});
