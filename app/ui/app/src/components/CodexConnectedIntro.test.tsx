import type { ReactNode } from "react";
import { act, create } from "react-test-renderer";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CodexConnectedIntro } from "./CodexConnectedIntro";

// Exercise acknowledgment independently of Headless UI's browser-only portal.
vi.mock("@headlessui/react", () => {
  const Container = ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  );
  return {
    Dialog: Container,
    DialogPanel: Container,
    DialogTitle: Container,
    Description: Container,
  };
});
afterEach(() => vi.unstubAllGlobals());

describe("CodexConnectedIntro", () => {
  it("keeps save failures retryable and closes only after a successful save", async () => {
    const save = vi
      .fn()
      .mockResolvedValueOnce("disk error")
      .mockResolvedValueOnce("");
    const done = vi.fn();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", { markCodexDesktopIntegrationUsed: save });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexConnectedIntro onConnect={async () => true} onDone={done} />,
        );
      });
      await act(async () => {
        renderer!.root.findByType("button").props.onClick();
      });
      expect(done).not.toHaveBeenCalled();
      expect(
        renderer!.root.findByProps({ role: "alert" }).children.join(""),
      ).toContain("couldn’t save");
      await act(async () => {
        renderer!.root.findByType("button").props.onClick();
      });
      expect(done).toHaveBeenCalledOnce();
      expect(save).toHaveBeenCalledTimes(2);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("does not duplicate pending acknowledgment requests", async () => {
    let resolve!: (error: string) => void;
    const save = vi.fn(
      () =>
        new Promise<string>((done) => {
          resolve = done;
        }),
    );
    const done = vi.fn();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", { markCodexDesktopIntegrationUsed: save });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexConnectedIntro onConnect={async () => true} onDone={done} />,
        );
      });
      await act(async () => {
        const button = renderer!.root.findByType("button");
        button.props.onClick();
        button.props.onClick();
      });
      expect(save).toHaveBeenCalledOnce();
      expect(renderer!.root.findByType("button").props.disabled).toBe(true);
      expect(done).not.toHaveBeenCalled();
      await act(async () => {
        resolve("");
      });
      expect(done).toHaveBeenCalledOnce();
    } finally {
      await act(async () => renderer?.unmount());
    }
  });
});

describe("launch before acknowledgment", () => {
  it.each(["cancelled", "failed"])(
    "keeps the intro unacknowledged when launch is %s",
    async (outcome) => {
      const save = vi.fn();
      const done = vi.fn();
      const connect = vi.fn().mockImplementation(async () => {
        if (outcome === "failed") throw new Error("ChatGPT could not open");
        return false;
      });
      vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
      vi.stubGlobal("window", { markCodexDesktopIntegrationUsed: save });
      let renderer;
      try {
        await act(async () => {
          renderer = create(
            <CodexConnectedIntro onConnect={connect} onDone={done} />,
          );
        });
        expect(connect).not.toHaveBeenCalled();
        await act(async () => {
          renderer!.root.findByType("button").props.onClick();
        });
        expect(save).not.toHaveBeenCalled();
        expect(done).not.toHaveBeenCalled();
        expect(renderer!.root.findByType("button").props.disabled).toBe(false);
        if (outcome === "failed")
          expect(
            renderer!.root.findByProps({ role: "alert" }).children,
          ).toContain("ChatGPT could not open");
      } finally {
        await act(async () => renderer?.unmount());
      }
    },
  );

  it("does not relaunch when retrying a failed acknowledgment save", async () => {
    const connect = vi.fn().mockResolvedValue(true);
    const save = vi
      .fn()
      .mockResolvedValueOnce("disk error")
      .mockResolvedValueOnce("");
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", { markCodexDesktopIntegrationUsed: save });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexConnectedIntro onConnect={connect} onDone={vi.fn()} />,
        );
      });
      await act(async () => {
        renderer!.root.findByType("button").props.onClick();
      });
      await act(async () => {
        renderer!.root.findByType("button").props.onClick();
      });
      expect(connect).toHaveBeenCalledOnce();
      expect(save).toHaveBeenCalledTimes(2);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });
});
