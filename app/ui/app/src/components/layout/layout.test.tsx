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
    expect(html).toContain("translate-y-px");
    expect(html).toContain("transition-[padding-left]");
    expect(html).toContain("duration-300");
  });

  it("keeps the titlebar control adjustment specific to macOS", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "windows" });

    const html = renderToStaticMarkup(
      <SidebarLayout title="Settings" sidebar={<nav />}>
        <div />
      </SidebarLayout>,
    );

    expect(html).not.toContain("translate-y-px");
  });
});
