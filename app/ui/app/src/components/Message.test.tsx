import { renderToStaticMarkup } from "react-dom/server";
import type React from "react";
import { describe, expect, it, vi } from "vitest";
import { Message as MessageType } from "@/gotypes";

vi.mock("streamdown", () => ({
  Streamdown: ({ children }: { children?: React.ReactNode }) => children,
  defaultRehypePlugins: { katex: "katex", raw: "raw" },
  defaultRemarkPlugins: { gfm: "gfm", math: "math" },
}));

vi.mock("@/hooks/useSettings", () => ({
  useSettings: () => ({
    settings: { speechVoice: "", speechRate: 1, speechVolume: 1 },
  }),
}));

import Message from "./Message";

function assistantMessage(content = "The answer is ready.") {
  return new MessageType({
    role: "assistant",
    content,
    thinking: "",
    stream: false,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  });
}

describe("Message read aloud action", () => {
  it("renders Read aloud for a completed assistant response", () => {
    const html = renderToStaticMarkup(
      <Message message={assistantMessage()} isStreaming={false} />,
    );

    expect(html).toContain('title="Read aloud"');
  });

  it("does not render Read aloud while an assistant response is streaming", () => {
    const html = renderToStaticMarkup(
      <Message message={assistantMessage()} isStreaming />,
    );

    expect(html).not.toContain('title="Read aloud"');
  });
});
