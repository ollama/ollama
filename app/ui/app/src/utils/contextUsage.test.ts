import { describe, expect, it } from "vitest";
import {
  buildContextUsageData,
  getLatestAssistantMessage,
} from "./contextUsage";

function normalizeNumberSeparators(value: string) {
  return value.replace(/[.,]/g, "");
}

describe("contextUsage", () => {
  it("uses the latest assistant message instead of older reported metrics", () => {
    const latestAssistant = getLatestAssistantMessage([
      { role: "assistant", promptEvalCount: 1200, evalCount: 250 },
      { role: "user", promptEvalCount: undefined, evalCount: undefined },
      { role: "assistant", promptEvalCount: undefined, evalCount: undefined },
    ]);

    expect(latestAssistant).toEqual({
      role: "assistant",
      promptEvalCount: undefined,
      evalCount: undefined,
    });
  });

  it("shows an unavailable state when the latest assistant message has no token metrics", () => {
    const usage = buildContextUsageData({
      messages: [
        { role: "assistant", promptEvalCount: 1200, evalCount: 250 },
        { role: "assistant", promptEvalCount: undefined, evalCount: undefined },
      ],
      contextLength: 4096,
    });

    expect(usage.state).toBe("unavailable");
    expect(usage.contextUsagePercent).toBeNull();
    expect(normalizeNumberSeparators(usage.summary)).toBe(
      "Context limit 4096 tokens",
    );
    expect(usage.breakdown).toBe("Latest response did not report token counts");
    expect(usage.tone.label).toBe("Unavailable");
  });

  it("keeps the previous metrics visible while the latest response is still streaming", () => {
    const usage = buildContextUsageData({
      messages: [
        { role: "assistant", promptEvalCount: 1200, evalCount: 250 },
        { role: "assistant", promptEvalCount: undefined, evalCount: undefined },
      ],
      contextLength: 4096,
      isStreaming: true,
    });

    expect(usage.state).toBe("available");
    expect(usage.isPendingRefresh).toBe(true);
    expect(usage.contextUsagePercent).toBe(35);
    expect(normalizeNumberSeparators(usage.summary)).toBe(
      "~1450 / 4096 tokens",
    );
    expect(normalizeNumberSeparators(usage.breakdown)).toContain(
      "Latest prompt 1200 + reply 250",
    );
    expect(usage.breakdown).toContain("updating after the current response finishes");
  });

  it("shows an empty state before the first completed assistant response", () => {
    const usage = buildContextUsageData({
      messages: [{ role: "user", promptEvalCount: undefined, evalCount: undefined }],
      contextLength: 4096,
    });

    expect(usage.state).toBe("empty");
    expect(usage.contextUsagePercent).toBe(0);
    expect(normalizeNumberSeparators(usage.summary)).toBe("~0 / 4096 tokens");
    expect(usage.breakdown).toBe("No completed response yet");
    expect(usage.tone.label).toBe("Comfortable");
  });

  it("flags over-limit usage from the latest assistant message", () => {
    const usage = buildContextUsageData({
      messages: [{ role: "assistant", promptEvalCount: 3000, evalCount: 1500 }],
      contextLength: 4096,
    });

    expect(usage.state).toBe("available");
    expect(usage.isContextOverLimit).toBe(true);
    expect(usage.contextUsagePercent).toBe(110);
    expect(normalizeNumberSeparators(usage.summary)).toBe(
      "~4500 / 4096 tokens (+404 over)",
    );
    expect(usage.hint).toBe("Older messages may already be dropped from the prompt.");
    expect(usage.tone.label).toBe("Over limit");
  });
});
