import { describe, expect, it, vi, beforeEach } from "vitest";
import { copyTextToClipboard } from "./clipboard";

describe("copyTextToClipboard", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("copies via Clipboard API when available", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", {
      clipboard: {
        writeText,
      },
    });

    const copied = await copyTextToClipboard("ollama launch claude");

    expect(copied).toBe(true);
    expect(writeText).toHaveBeenCalledWith("ollama launch claude");
  });

  it("falls back to execCommand when Clipboard API fails", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("not allowed"));
    vi.stubGlobal("navigator", {
      clipboard: {
        writeText,
      },
    });

    const textarea = {
      value: "",
      setAttribute: vi.fn(),
      style: {} as Record<string, string>,
      focus: vi.fn(),
      select: vi.fn(),
    };
    const appendChild = vi.fn();
    const removeChild = vi.fn();
    const execCommand = vi.fn().mockReturnValue(true);
    vi.stubGlobal("document", {
      createElement: vi.fn().mockReturnValue(textarea),
      body: {
        appendChild,
        removeChild,
      },
      execCommand,
    });

    const copied = await copyTextToClipboard("ollama launch openclaw");

    expect(copied).toBe(true);
    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(appendChild).toHaveBeenCalled();
    expect(removeChild).toHaveBeenCalled();
  });
});
