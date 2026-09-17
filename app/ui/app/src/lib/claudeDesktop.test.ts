import { describe, expect, it, vi } from "vitest";
import {
  addClaudeModelSelection,
  ClaudeConnectionTimeoutError,
  claudeDesktopRecoveryMessage,
  claudeDesktopMaxModels,
  claudeDesktopMaxModelsMessage,
  claudeDesktopRequestCountLabel,
  claudeDesktopUsableSelection,
  defaultClaudeDesktopMaxModels,
  isClaudeConfigured,
  optimisticClaudeConnectionState,
  withClaudeConnectionTimeout,
} from "./claudeDesktop";

describe("isClaudeConfigured", () => {
  it("keeps a failed configured profile switchable off", () => {
    expect(
      isClaudeConfigured({
        supported: true,
        used: true,
        installed: true,
        configured: true,
        connected: false,
        running: false,
        startFailed: true,
        portConflict: true,
      }),
    ).toBe(true);
  });
});

describe("optimisticClaudeConnectionState", () => {
  it("shows the requested state while a connection change is pending", () => {
    expect(optimisticClaudeConnectionState(false, true)).toBe(true);
    expect(optimisticClaudeConnectionState(true, false)).toBe(false);
  });

  it("uses the confirmed state when no change is pending", () => {
    expect(optimisticClaudeConnectionState(true, null)).toBe(true);
    expect(optimisticClaudeConnectionState(false, null)).toBe(false);
  });
});

describe("claudeDesktopRequestCountLabel", () => {
  it("formats zero, singular, and plural session counts", () => {
    expect(claudeDesktopRequestCountLabel(0)).toBe("0 requests this session");
    expect(claudeDesktopRequestCountLabel(1)).toBe("1 request this session");
    expect(claudeDesktopRequestCountLabel(2)).toBe("2 requests this session");
  });
});

describe("withClaudeConnectionTimeout", () => {
  it("returns a native result that finishes before the watchdog", async () => {
    await expect(
      withClaudeConnectionTimeout(Promise.resolve("connected"), 100),
    ).resolves.toBe("connected");
  });

  it("reports when native work settles after the watchdog", async () => {
    vi.useFakeTimers();
    try {
      let resolveNative!: (value: string) => void;
      const nativeAction = new Promise<string>((resolve) => {
        resolveNative = resolve;
      });
      const onLateSettled = vi.fn();
      const result = withClaudeConnectionTimeout(
        nativeAction,
        100,
        onLateSettled,
      );
      const timedOut = expect(result).rejects.toBeInstanceOf(
        ClaudeConnectionTimeoutError,
      );

      await vi.advanceTimersByTimeAsync(100);
      await timedOut;
      expect(onLateSettled).not.toHaveBeenCalled();
      resolveNative("late connection");
      await Promise.resolve();

      expect(onLateSettled).toHaveBeenCalledOnce();
      expect(onLateSettled).toHaveBeenCalledWith({
        status: "fulfilled",
        value: "late connection",
      });
      await expect(result).rejects.toBeInstanceOf(ClaudeConnectionTimeoutError);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("claudeDesktopMaxModels", () => {
  it("falls back to the five literal Claude slots without a status", () => {
    expect(claudeDesktopMaxModels(undefined)).toBe(5);
    expect(
      claudeDesktopMaxModels({
        supported: true,
        used: true,
        installed: true,
        connected: false,
        running: false,
        startFailed: false,
        portConflict: false,
      }),
    ).toBe(defaultClaudeDesktopMaxModels);
  });

  it("uses the server-provided limit when present", () => {
    expect(
      claudeDesktopMaxModels({
        supported: true,
        used: true,
        installed: true,
        connected: false,
        running: false,
        startFailed: false,
        portConflict: false,
        maxModels: 3,
      }),
    ).toBe(3);
  });
});

describe("claudeDesktopRecoveryMessage", () => {
  it("prefers current native guidance over stale action errors", () => {
    expect(
      claudeDesktopRecoveryMessage(
        "Cloud models are off. Select an installed model in Settings.",
        "Ollama could not open Claude.",
      ),
    ).toBe("Cloud models are off. Select an installed model in Settings.");
  });

  it("clears after native recovery when no action error remains", () => {
    expect(claudeDesktopRecoveryMessage(undefined, null)).toBeNull();
  });
});

describe("addClaudeModelSelection", () => {
  it("appends models below the limit", () => {
    expect(addClaudeModelSelection(["kimi-k3:cloud"], "qwen3:8b", 5)).toEqual({
      selection: ["kimi-k3:cloud", "qwen3:8b"],
    });
  });

  it("rejects a sixth selection with a clear message", () => {
    const selection = [
      "glm-5.2:cloud",
      "kimi-k3:cloud",
      "deepseek-v4-pro",
      "deepseek-v4-flash",
      "gemma4:26b:cloud",
    ];
    const result = addClaudeModelSelection(selection, "qwen3:8b", 5);
    expect(result.selection).toBe(selection);
    expect(result.error).toBe(
      "Claude supports up to 5 models. Deselect one to add another.",
    );
  });

  it("honors a smaller server-provided limit", () => {
    const result = addClaudeModelSelection(["qwen3:8b"], "llama3.2", 1);
    expect(result.selection).toEqual(["qwen3:8b"]);
    expect(result.error).toBe(claudeDesktopMaxModelsMessage(1));
  });

  it("is a no-op for an already selected model", () => {
    expect(addClaudeModelSelection(["qwen3:8b"], "qwen3:8b", 5)).toEqual({
      selection: ["qwen3:8b"],
    });
  });
});

describe("claudeDesktopUsableSelection", () => {
  it("replaces unavailable paid selections with an available free model", () => {
    expect(
      claudeDesktopUsableSelection([
        {
          name: "glm-5.2:cloud",
          displayName: "glm-5.2:cloud",
          selected: true,
          availability: "unavailable",
          reason: "upgrade_required",
          requiredPlan: "pro",
        },
        {
          name: "gemma4:31b-cloud",
          displayName: "gemma4:31b-cloud",
          selected: false,
          availability: "available",
          requiredPlan: "free",
        },
      ]),
    ).toEqual(["gemma4:31b-cloud"]);
  });

  it("preserves selected models that remain available", () => {
    expect(
      claudeDesktopUsableSelection([
        {
          name: "gemma4:31b-cloud",
          displayName: "gemma4:31b-cloud",
          selected: true,
          availability: "available",
        },
        {
          name: "qwen3:8b",
          displayName: "qwen3:8b",
          selected: false,
          availability: "available",
        },
      ]),
    ).toEqual(["gemma4:31b-cloud"]);
  });

  it("selects all available recommendations for a default catalog", () => {
    expect(
      claudeDesktopUsableSelection(
        [
          {
            name: "glm-5.2:cloud",
            displayName: "glm-5.2:cloud",
            selected: false,
            availability: "available",
          },
          {
            name: "kimi-k3:cloud",
            displayName: "kimi-k3:cloud",
            selected: false,
            availability: "available",
          },
        ],
        true,
        5,
      ),
    ).toEqual(["glm-5.2:cloud", "kimi-k3:cloud"]);
  });

  it("returns no selection when no model is available", () => {
    expect(
      claudeDesktopUsableSelection([
        {
          name: "glm-5.2:cloud",
          displayName: "glm-5.2:cloud",
          selected: true,
          availability: "unavailable",
        },
      ]),
    ).toEqual([]);
  });
});
