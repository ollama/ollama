import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ClaudeDesktopIcon } from "./ClaudeDesktopIcon";

describe("ClaudeDesktopIcon", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses the placeholder without probing for an uninstalled app", () => {
    const getClaudeDesktopIcon = vi.fn();
    vi.stubGlobal("window", { getClaudeDesktopIcon });

    const html = renderToStaticMarkup(<ClaudeDesktopIcon installed={false} />);

    expect(html).toContain("<svg");
    expect(html).not.toContain("<img");
    expect(getClaudeDesktopIcon).not.toHaveBeenCalled();
  });

  it("shows the installed application icon", async () => {
    const dataURL = "data:image/png;base64,aWNvbg==";
    const getClaudeDesktopIcon = vi.fn().mockResolvedValue(dataURL);
    vi.stubGlobal("window", { getClaudeDesktopIcon });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

    let renderer: ReactTestRenderer | undefined;
    await act(async () => {
      renderer = create(<ClaudeDesktopIcon installed />);
      await Promise.resolve();
    });

    expect(getClaudeDesktopIcon).toHaveBeenCalledOnce();
    expect(renderer!.root.findByType("img").props.src).toBe(dataURL);
    await act(async () => {
      renderer!.unmount();
    });
  });

  it("keeps the placeholder when icon loading fails", async () => {
    vi.stubGlobal("window", {
      getClaudeDesktopIcon: vi.fn().mockRejectedValue(new Error("unavailable")),
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

    let renderer: ReactTestRenderer | undefined;
    await act(async () => {
      renderer = create(<ClaudeDesktopIcon installed />);
      await Promise.resolve();
    });

    expect(renderer!.root.findAllByType("img")).toHaveLength(0);
    expect(renderer!.root.findAllByType("svg")).toHaveLength(1);
    await act(async () => {
      renderer!.unmount();
    });
  });
});
