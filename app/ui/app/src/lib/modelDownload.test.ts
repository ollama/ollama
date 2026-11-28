import { describe, expect, it } from "vitest";
import { Model } from "@/gotypes";
import {
  applyPullEvent,
  formatBytes,
  IDLE_DOWNLOAD,
  isInstalled,
  layerProgressText,
} from "./modelDownload";

describe("isInstalled", () => {
  it("treats a model without a digest as not installed", () => {
    expect(isInstalled(new Model({ model: "qwen3:9b" }))).toBe(false);
  });

  it("treats an empty digest as not installed", () => {
    expect(isInstalled(new Model({ model: "qwen3:9b", digest: "" }))).toBe(
      false,
    );
  });

  it("treats a model with a digest as installed", () => {
    expect(
      isInstalled(new Model({ model: "qwen3:9b", digest: "abc123" })),
    ).toBe(true);
  });
});

describe("applyPullEvent", () => {
  it("moves from starting to downloading on the first event", () => {
    expect(
      applyPullEvent({ status: "starting" }, { status: "pulling manifest" }),
    ).toEqual({
      status: "downloading",
      statusText: "pulling manifest",
      completed: undefined,
      total: undefined,
    });
  });

  it("tracks per-layer byte progress", () => {
    const state = applyPullEvent(
      { status: "starting" },
      { status: "downloading", completed: 50, total: 100 },
    );

    expect(state.status).toBe("downloading");
    expect(state.completed).toBe(50);
    expect(state.total).toBe(100);
  });

  it("keeps the last known totals when an event omits them", () => {
    const state = applyPullEvent(
      { status: "downloading", completed: 10, total: 40 },
      { status: "downloading", completed: 20 },
    );

    expect(state.completed).toBe(20);
    expect(state.total).toBe(40);
  });

  it("completes on a success event", () => {
    expect(applyPullEvent(IDLE_DOWNLOAD, { status: "success" })).toEqual({
      status: "completed",
    });
  });

  it("completes on a done event even without a success status", () => {
    expect(
      applyPullEvent(
        { status: "downloading" },
        { status: "verifying", done: true },
      ),
    ).toEqual({ status: "completed" });
  });
});

describe("formatBytes", () => {
  it("formats zero as 0 B", () => {
    expect(formatBytes(0)).toBe("0 B");
  });

  it("formats bytes without decimals", () => {
    expect(formatBytes(2048)).toBe("2 KB");
  });

  it("uses the supplied unit", () => {
    expect(formatBytes(1536, "KB")).toBe("2 KB");
  });

  it("keeps one decimal place for large units", () => {
    expect(formatBytes(1.5 * 1024 ** 3, "GB")).toBe("1.5 GB");
  });
});

describe("layerProgressText", () => {
  it("is undefined while byte progress is unknown", () => {
    expect(layerProgressText({ status: "downloading" })).toBeUndefined();
    expect(
      layerProgressText({ status: "downloading", completed: 10, total: 0 }),
    ).toBeUndefined();
  });

  it("describes the current layer rather than claiming overall progress", () => {
    expect(
      layerProgressText({
        status: "downloading",
        completed: 50,
        total: 100,
      }),
    ).toBe("50 B / 100 B (50%)");
  });
});
