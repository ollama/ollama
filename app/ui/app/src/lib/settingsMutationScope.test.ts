import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { settingsMutationScope } from "./settingsMutationScope";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe("settingsMutationScope", () => {
  it("serializes settings and Cloud updates in request order", async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    const updateSettings = vi
      .fn<() => Promise<string>>()
      .mockReturnValue(first.promise);
    const updateCloud = vi
      .fn<() => Promise<string>>()
      .mockReturnValue(second.promise);
    const queryClient = new QueryClient();
    const mutationCache = queryClient.getMutationCache();
    const settingsMutation = mutationCache.build(queryClient, {
      mutationFn: updateSettings,
      scope: settingsMutationScope,
    });
    const cloudMutation = mutationCache.build(queryClient, {
      mutationFn: updateCloud,
      scope: settingsMutationScope,
    });

    const firstResult = settingsMutation.execute(undefined);
    const secondResult = cloudMutation.execute(undefined);
    await vi.waitFor(() => expect(updateSettings).toHaveBeenCalledOnce());
    expect(updateCloud).not.toHaveBeenCalled();

    first.resolve("settings updated");
    await firstResult;
    await vi.waitFor(() => expect(updateCloud).toHaveBeenCalledOnce());

    second.resolve("cloud updated");
    await expect(secondResult).resolves.toBe("cloud updated");
  });
});
