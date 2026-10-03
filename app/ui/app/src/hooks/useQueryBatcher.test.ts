import { QueryClient } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createQueryBatcher } from "./useQueryBatcher";

describe("createQueryBatcher", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("coalesces a high-rate stream", () => {
    vi.useFakeTimers();

    const queryClient = new QueryClient();
    const queryKey = ["chat", "benchmark"] as const;
    queryClient.setQueryData(queryKey, 0);

    const setQueryData = vi.spyOn(queryClient, "setQueryData");
    const batcher = createQueryBatcher<number>(queryClient, queryKey, {
      immediateFirst: true,
    });

    batcher.scheduleBatch((value = 0) => value + 1);
    expect(queryClient.getQueryData(queryKey)).toBe(1);

    // Simulate the remaining chunks arriving over one second. A 5 ms cadence
    // is fast enough to expose excessive cache commits without using wall time.
    for (let i = 1; i < 200; i += 1) {
      batcher.scheduleBatch((value = 0) => value + 1);
      vi.advanceTimersByTime(5);
    }
    batcher.flushBatch();

    const streamCommits = setQueryData.mock.calls.length;
    expect(queryClient.getQueryData(queryKey)).toBe(200);
    expect(streamCommits).toBeLessThanOrEqual(64);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("cancels a pending batch during cleanup", () => {
    vi.useFakeTimers();

    const queryClient = new QueryClient();
    const queryKey = ["chat", "cancelled"] as const;
    queryClient.setQueryData(queryKey, 0);

    const batcher = createQueryBatcher<number>(queryClient, queryKey, {
      immediateFirst: true,
    });

    batcher.scheduleBatch((value = 0) => value + 1);
    batcher.scheduleBatch((value = 0) => value + 1);
    batcher.cleanup();
    vi.runAllTimers();

    expect(queryClient.getQueryData(queryKey)).toBe(1);
    expect(vi.getTimerCount()).toBe(0);
  });
});
