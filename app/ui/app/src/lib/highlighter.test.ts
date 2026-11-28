import { beforeEach, describe, expect, it, vi } from "vitest";

const createHighlighter = vi.hoisted(() => vi.fn());
vi.mock("shiki", () => ({ createHighlighter }));

describe("loadHighlighter", () => {
  beforeEach(() => {
    vi.resetModules();
    createHighlighter.mockReset();
  });

  it("waits for first use and shares initialization across code blocks", async () => {
    const { loadHighlighter } = await import("./highlighter");
    expect(createHighlighter).not.toHaveBeenCalled();

    const highlighter = { codeToTokensBase: vi.fn() };
    let resolve!: (value: typeof highlighter) => void;
    createHighlighter.mockReturnValue(
      new Promise((done) => {
        resolve = done;
      }),
    );
    const first = loadHighlighter();
    expect(loadHighlighter()).toBe(first);
    expect(createHighlighter).toHaveBeenCalledOnce();

    resolve(highlighter);
    await expect(first).resolves.toBe(highlighter);
    expect(loadHighlighter()).toBe(first);
    expect(createHighlighter).toHaveBeenCalledOnce();
  });

  it("allows a later code block to retry failed initialization", async () => {
    const { loadHighlighter } = await import("./highlighter");
    const error = new Error("grammar unavailable");
    const highlighter = { codeToTokensBase: vi.fn() };
    createHighlighter
      .mockRejectedValueOnce(error)
      .mockResolvedValueOnce(highlighter);

    await expect(loadHighlighter()).rejects.toBe(error);
    await expect(loadHighlighter()).resolves.toBe(highlighter);
    expect(createHighlighter).toHaveBeenCalledTimes(2);
  });
});
