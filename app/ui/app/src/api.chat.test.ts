import { afterEach, describe, expect, it, vi } from "vitest";
import { sendMessage, type ChatEventUnion } from "@/api";
import { Model } from "@/gotypes";

async function collectEvents() {
  const events: ChatEventUnion[] = [];
  for await (const event of sendMessage(
    "chat-1",
    "hello",
    new Model({ model: "llama3" }),
  )) {
    events.push(event);
  }
  return events;
}

describe("sendMessage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("yields an error event for non-OK JSON responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: "server exploded" }), {
          status: 500,
          statusText: "Internal Server Error",
          headers: {
            "Content-Type": "application/json",
            "X-Request-ID": "req-123",
          },
        }),
      ),
    );

    const events = await collectEvents();

    expect(events).toHaveLength(1);
    expect(events[0].eventName).toBe("error");
    expect((events[0] as { error: string }).error).toBe("server exploded");
    expect((events[0] as { details?: string }).details).toContain("req-123");
  });

  it("turns malformed stream frames into error events", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: "broken stream frame" }), {
          status: 200,
          headers: {
            "Content-Type": "text/jsonl",
          },
        }),
      ),
    );

    const events = await collectEvents();

    expect(events).toHaveLength(1);
    expect(events[0].eventName).toBe("error");
    expect((events[0] as { error: string }).error).toBe("broken stream frame");
  });

  it("preserves valid chat stream events", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          [
            JSON.stringify({ eventName: "chat", content: "Hello" }),
            JSON.stringify({ eventName: "done" }),
          ].join("\n"),
          {
            status: 200,
            headers: {
              "Content-Type": "text/jsonl",
            },
          },
        ),
      ),
    );

    const events = await collectEvents();

    expect(events).toHaveLength(2);
    expect(events[0].eventName).toBe("chat");
    expect((events[0] as { content?: string }).content).toBe("Hello");
    expect(events[1].eventName).toBe("done");
  });

  it("accepts loading and download stream events", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          [
            JSON.stringify({ eventName: "loading" }),
            JSON.stringify({
              eventName: "download",
              total: 10,
              completed: 4,
              done: false,
            }),
            JSON.stringify({ eventName: "done" }),
          ].join("\n"),
          {
            status: 200,
            headers: {
              "Content-Type": "text/jsonl",
            },
          },
        ),
      ),
    );

    const events = await collectEvents();

    expect(events).toHaveLength(3);
    expect(events[0].eventName).toBe("loading");
    expect(events[1].eventName).toBe("download");
    expect((events[1] as { total?: number }).total).toBe(10);
    expect(events[2].eventName).toBe("done");
  });
});
