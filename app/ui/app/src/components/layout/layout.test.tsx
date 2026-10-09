import { renderToStaticMarkup } from "react-dom/server";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { SidebarLayout } from "./layout";

vi.mock("@tanstack/react-router", async (importOriginal) =>
  Object.assign(
    {},
    await importOriginal<typeof import("@tanstack/react-router")>(),
    {
      Link: ({
        children,
        className,
        title,
      }: {
        children: ReactNode;
        className?: string;
        title?: string;
      }) => (
        <a className={className} title={title}>
          {children}
        </a>
      ),
    },
  ),
);

describe("SidebarLayout", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("keeps the macOS title offset in step with the sidebar state", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    const html = renderToStaticMarkup(
      <SidebarLayout title="Connect your apps" sidebar={<nav />}>
        <div />
      </SidebarLayout>,
    );

    expect(html).toContain("pl-36");
  });

  it("renders without sidebar transition classes on first paint so the sidebar appears instead of animating open on load", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    const html = renderToStaticMarkup(
      <SidebarLayout title="Connect your apps" sidebar={<nav />}>
        <div />
      </SidebarLayout>,
    );

    // The sidebar starts closed and must be painted in its initial state
    // with no transitions; transitions are only enabled after mount so
    // user toggles still animate.
    expect(html).toContain("w-0");
    expect(html).not.toContain("transition-[width]");
    expect(html).not.toContain("transition-[left]");
    expect(html).not.toContain("transition-all");
    expect(html).not.toContain("transition-[padding-left]");
  });

  it("renders the new chat button without transition classes on first paint", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    const html = renderToStaticMarkup(
      <SidebarLayout sidebar={<nav />}>
        <div />
      </SidebarLayout>,
    );

    expect(html).toContain("New chat");
    expect(html).not.toContain("transition-opacity");
  });

  it("enables sidebar transitions after mount so toggling still animates", async () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    // Fire requestAnimationFrame callbacks immediately so the mount effect runs.
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
      cb(0);
      return 0;
    });
    vi.stubGlobal("cancelAnimationFrame", () => {});

    let renderer: ReactTestRenderer | undefined;
    await act(async () => {
      renderer = create(
        <SidebarLayout title="Connect your apps" sidebar={<nav />}>
          <div />
        </SidebarLayout>,
      );
    });
    expect(renderer).toBeDefined();

    const html = JSON.stringify(renderer!.toJSON());
    expect(html).toContain("transition-[width]");
    expect(html).toContain("transition-[padding-left]");

    await act(async () => {
      renderer!.unmount();
    });
  });
});
