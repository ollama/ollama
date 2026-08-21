import { describe, expect, it } from "vitest";
import {
  addClaudeModelSelection,
  claudeDesktopRecoveryMessage,
  claudeDesktopMaxModels,
  claudeDesktopMaxModelsMessage,
  defaultClaudeDesktopMaxModels,
  isClaudeConfigured,
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
