import { describe, expect, it, vi } from "vitest";
import type { QueryClient } from "@tanstack/react-query";
import { preloadChatData } from "./chatPreload";

describe("preloadChatData", () => {
  it("warms Chat data once without installing polling options", async () => {
    const prefetchQuery = vi.fn().mockResolvedValue(undefined);

    await preloadChatData({ prefetchQuery } as unknown as Pick<
      QueryClient,
      "prefetchQuery"
    >);

    expect(prefetchQuery).toHaveBeenCalledTimes(4);
    expect(
      prefetchQuery.mock.calls.map(([options]) => options.queryKey),
    ).toEqual([
      ["chats"],
      ["health"],
      ["models", ""],
      ["modelRecommendations"],
    ]);
    for (const [options] of prefetchQuery.mock.calls) {
      expect(options).not.toHaveProperty("refetchInterval");
      expect(options).not.toHaveProperty("refetchIntervalInBackground");
    }
  });
});
