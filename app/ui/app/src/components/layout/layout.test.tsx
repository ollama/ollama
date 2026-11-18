import { renderToStaticMarkup } from "react-dom/server";
import { act } from "react";
import { create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("SidebarLayout", () => {
  let SidebarLayout: (typeof import("./layout"))["SidebarLayout"];
  let renderer: ReactTestRenderer | undefined;

  beforeEach(async () => {
    vi.resetModules();
    ({ SidebarLayout } = await import("./layout"));
  });

  afterEach(async () => {
    if (renderer) {
      await act(async () => renderer?.unmount());
      renderer = undefined;
    }
    vi.unstubAllGlobals();
  });

  it("keeps the macOS title offset in step with the sidebar transition", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    const html = renderToStaticMarkup(
      <SidebarLayout title="Connect your apps" sidebar={<nav />}>
        <div />
      </SidebarLayout>,
    );

    expect(html).toContain("pl-36");
    expect(html).toContain("transition-[padding-left]");
    expect(html).toContain("duration-300");
  });

  it("resizes the open sidebar with pointer and keyboard input", async () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

    await act(async () => {
      renderer = create(
        <SidebarLayout title="Chat" sidebar={<nav />}>
          <div />
        </SidebarLayout>,
      );
    });
    if (!renderer) throw new Error("failed to render sidebar layout");
    const rendered = renderer;

    await act(async () => {
      rendered.root
        .findByProps({ "aria-label": "Show sidebar" })
        .props.onClick();
    });

    let separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    const pointerTarget = {
      setPointerCapture: vi.fn(),
      hasPointerCapture: vi.fn(() => true),
      releasePointerCapture: vi.fn(),
    };

    await act(async () => {
      separator.props.onPointerDown({
        button: 0,
        pointerId: 1,
        currentTarget: pointerTarget,
        preventDefault: vi.fn(),
        stopPropagation: vi.fn(),
      });
    });
    separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    await act(async () => {
      separator.props.onPointerMove({ clientX: 50 });
    });

    separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    expect(separator.props["aria-valuenow"]).toBe(160);
    expect(separator.props["aria-valuetext"]).toBe("160 pixels");

    await act(async () => {
      separator.props.onPointerMove({ clientX: 320 });
    });

    separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    expect(separator.props["aria-valuenow"]).toBe(320);
    expect(separator.parent?.props.style).toEqual({ width: 320 });
    const sidebarControls = rendered.root.findAll(
      (node) => node.type === "div" && node.props.style?.left !== undefined,
    );
    expect(sidebarControls).toHaveLength(1);
    expect(sidebarControls[0].props.style.left).toBe(268);

    await act(async () => {
      separator.props.onPointerUp({
        pointerId: 1,
        currentTarget: pointerTarget,
      });
    });
    expect(pointerTarget.releasePointerCapture).toHaveBeenCalledWith(1);

    separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    await act(async () => {
      separator.props.onPointerMove({ clientX: 400 });
    });
    separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    expect(separator.props["aria-valuenow"]).toBe(320);

    await act(async () => {
      separator.props.onPointerDown({
        button: 0,
        pointerId: 2,
        currentTarget: pointerTarget,
        preventDefault: vi.fn(),
        stopPropagation: vi.fn(),
      });
    });
    separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    await act(async () => separator.props.onLostPointerCapture());
    separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    await act(async () => {
      separator.props.onPointerMove({ clientX: 400 });
    });
    separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    expect(separator.props["aria-valuenow"]).toBe(320);

    await act(async () => {
      separator.props.onKeyDown({
        key: "ArrowRight",
        preventDefault: vi.fn(),
      });
    });
    separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    expect(separator.props["aria-valuenow"]).toBe(336);

    await act(async () => {
      separator.props.onKeyDown({
        key: "End",
        preventDefault: vi.fn(),
      });
    });
    separator = rendered.root.findByProps({
      "aria-label": "Resize sidebar",
    });
    expect(separator.props["aria-valuenow"]).toBe(480);
  });
});
