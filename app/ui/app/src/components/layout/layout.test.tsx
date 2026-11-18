import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SidebarLayout } from "./layout";

describe("SidebarLayout", () => {
  afterEach(() => {
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
    // Transition classes are deferred until after mount to prevent the sidebar
    // from animating open on initial load (see issue #12954).
    expect(html).not.toContain("transition-[padding-left]");
    expect(html).not.toContain("transition-[width]");
    expect(html).not.toContain("transition-all");
  });
});
