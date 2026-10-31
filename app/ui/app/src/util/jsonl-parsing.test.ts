import { describe, expect, it, vi } from "vitest";
import { parseJsonlFromResponse, parseJsonlFromStream } from "./jsonl-parsing";

async function collect<T>(events: AsyncIterable<T>) {
  const result: T[] = [];
  for await (const event of events) result.push(event);
  return result;
}

describe("JSONL streams", () => {
  it("cancels unread data after a malformed event", async () => {
    const cancel = vi.fn();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("invalid\n"));
      },
      cancel,
    });
    await expect(collect(parseJsonlFromStream(stream))).rejects.toThrow();
    expect(cancel).toHaveBeenCalledOnce();
    expect(stream.locked).toBe(false);
  });

  it("handles split UTF-8 characters, blank lines, and a final line without a newline", async () => {
    const bytes = new TextEncoder().encode(
      '\n{"content":"你好"}\r\n{"done":true}',
    );
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
        controller.close();
      },
    });

    await expect(collect(parseJsonlFromStream(stream))).resolves.toEqual([
      { content: "你好" },
      { done: true },
    ]);
    expect(stream.locked).toBe(false);
  });

  it.each(['{"content":\n', '{"content":'])(
    "rejects malformed or truncated JSON: %s",
    async (body) => {
      const response = new Response(body);
      await expect(collect(parseJsonlFromResponse(response))).rejects.toThrow();
      expect(response.body!.locked).toBe(false);
    },
  );

  it("reports an HTTP error instead of yielding it as stream data", async () => {
    const response = new Response('{"error":"model failed to load"}', {
      status: 500,
    });
    await expect(collect(parseJsonlFromResponse(response))).rejects.toThrow(
      "model failed to load",
    );
  });

  it("includes the HTTP status when the error response is not JSON", async () => {
    const response = new Response("Bad Gateway", { status: 502 });
    await expect(collect(parseJsonlFromResponse(response))).rejects.toThrow(
      "502",
    );
  });
});
