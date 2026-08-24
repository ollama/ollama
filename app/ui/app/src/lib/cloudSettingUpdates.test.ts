import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { cloudSettingMutationScope } from "./cloudSettingUpdates";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe("cloudSettingMutationScope", () => {
  it("serializes rapid Cloud updates in request order", async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    const updateCloud = vi
      .fn<(enabled: boolean) => Promise<string>>()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const queryClient = new QueryClient();
    const mutationCache = queryClient.getMutationCache();
    const createMutation = () =>
      mutationCache.build(queryClient, {
        mutationFn: updateCloud,
        scope: cloudSettingMutationScope,
      });

    const firstResult = createMutation().execute(true);
    const secondResult = createMutation().execute(false);
    await vi.waitFor(() => expect(updateCloud).toHaveBeenCalledTimes(1));
    expect(updateCloud).toHaveBeenNthCalledWith(1, true);

    first.resolve("enabled");
    await firstResult;
    await vi.waitFor(() => expect(updateCloud).toHaveBeenCalledTimes(2));
    expect(updateCloud).toHaveBeenNthCalledWith(2, false);

    second.resolve("disabled");
    await expect(secondResult).resolves.toBe("disabled");
  });
});
