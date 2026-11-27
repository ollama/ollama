import { describe, expect, it } from "vitest";
import { Message } from "@/gotypes";
import { readAloudOwnerKey } from "./readAloudOwnerKey";

function assistant(content: string) {
  return new Message({
    role: "assistant",
    content,
    stream: false,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  });
}

describe("readAloudOwnerKey", () => {
  it("changes when assistant content changes at the same index", () => {
    const first = readAloudOwnerKey("chat-1", 2, assistant("First answer."));
    const second = readAloudOwnerKey("chat-1", 2, assistant("Second answer."));
    expect(second).not.toBe(first);
  });

  it("stays stable for the same message content", () => {
    const message = assistant("Stable text.");
    const a = readAloudOwnerKey("chat-1", 0, message);
    const b = readAloudOwnerKey("chat-1", 0, assistant("Stable text."));
    expect(b).toBe(a);
  });
});
