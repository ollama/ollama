import { renderToStaticMarkup } from "react-dom/server";
import { act, create } from "react-test-renderer";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SidebarLayout } from "./layout";

describe("SidebarLayout", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders sidebar open without initial transition classes to avoid animation on load", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    const html = renderToStaticMarkup(
      <SidebarLayout title="Connect your apps" sidebar={<nav />}>
        <div />
      </SidebarLayout>,
    );

    // Should render open initially
    expect(html).toContain("w-48");
    expect(html).toContain("pl-6");
    expect(html).toContain("left-[140px]");

    // Initial render should not have transition classes active
    expect(html).not.toContain("transition-[padding-left]");
    expect(html).not.toContain("transition-[width]");
    expect(html).not.toContain("transition-[left]");
  });

  it("enables transitions once mounted for smooth user interactions", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    let renderer: ReturnType<typeof create>;
    act(() => {
      renderer = create(
        <SidebarLayout title="Connect your apps" sidebar={<nav />}>
          <div />
        </SidebarLayout>,
      );
    });

    const root = renderer!.root;
    const title = root.findByType("h1");
    expect(title.props.className).toContain("transition-[padding-left]");
    expect(title.props.className).toContain("duration-300");
    expect(title.props.className).toContain("pl-6");
  });

  it("toggles sidebar between open and closed when the toggle button is clicked", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    let renderer: ReturnType<typeof create>;
    act(() => {
      renderer = create(
        <SidebarLayout title="Connect your apps" sidebar={<nav />}>
          <div />
        </SidebarLayout>,
      );
    });

    const root = renderer!.root;
    const button = root.findByType("button");

    // Initially open
    expect(button.props["aria-label"]).toBe("Hide sidebar");
    const title = root.findByType("h1");
    expect(title.props.className).toContain("pl-6");

    // Click to close
    act(() => {
      button.props.onClick();
    });

    expect(button.props["aria-label"]).toBe("Show sidebar");
    expect(title.props.className).toContain("pl-36");

    // Click to open again
    act(() => {
      button.props.onClick();
    });

    expect(button.props["aria-label"]).toBe("Hide sidebar");
    expect(title.props.className).toContain("pl-6");
  });

  it("adjusts offsets for Windows platform", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "windows" });

    let renderer: ReturnType<typeof create>;
    act(() => {
      renderer = create(
        <SidebarLayout title="Settings" sidebar={<nav />}>
          <div />
        </SidebarLayout>,
      );
    });

    const root = renderer!.root;
    const button = root.findByType("button");
    const title = root.findByType("h1");

    // On Windows, toggle button is at left-2 when open
    expect(button.parent?.props.className).toContain("left-2");
    expect(title.props.className).toContain("pl-6");

    // Toggle closed on Windows: title gets pl-16
    act(() => {
      button.props.onClick();
    });

    expect(title.props.className).toContain("pl-16");
    expect(button.parent?.props.className).toContain("left-2");
  });
});
