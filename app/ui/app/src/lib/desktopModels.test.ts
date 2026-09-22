import { afterEach, expect, it, vi } from "vitest";
import { desktopModels, invalidateDesktopModels } from "./desktopModels";
import { queryClient } from "./queryClient";

afterEach(() => queryClient.clear());

it("shares pending and recent inventory but separates accounts", async () => {
  let resolve!: (models: string[]) => void;
  const load = vi.fn(
    () =>
      new Promise<string[]>((done) => {
        resolve = done;
      }),
  );
  const first = desktopModels(["chatgpt", "account-a"], load);
  const second = desktopModels(["chatgpt", "account-a"], load);
  resolve(["local-model"]);
  expect(await first).toEqual(["local-model"]);
  expect(await second).toEqual(["local-model"]);
  expect(await desktopModels(["chatgpt", "account-a"], load)).toEqual([
    "local-model",
  ]);
  expect(load).toHaveBeenCalledOnce();
  const otherAccount = vi.fn().mockResolvedValue(["other-model"]);
  expect(await desktopModels(["chatgpt", "account-b"], otherAccount)).toEqual([
    "other-model",
  ]);
});

it("does not restore stale inventory after an action invalidates it", async () => {
  let resolve!: (models: string[]) => void;
  const pending = desktopModels(
    ["claude"],
    () =>
      new Promise<string[]>((done) => {
        resolve = done;
      }),
  );
  const canceled = expect(pending).rejects.toThrow();
  await invalidateDesktopModels();
  resolve(["old-model"]);
  await canceled;
  expect(await desktopModels(["claude"], async () => ["new-model"])).toEqual([
    "new-model",
  ]);
});
