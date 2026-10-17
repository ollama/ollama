import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SidebarLayout } from "./layout";

describe("SidebarLayout", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders sidebar open by default on macOS", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    const html = renderToStaticMarkup(
      <SidebarLayout title="Connect your apps" sidebar={<nav />}>
        <div />
      </SidebarLayout>,
    );

    expect(html).toContain("pl-6");
    expect(html).toContain("transition-[padding-left]");
    expect(html).toContain("duration-300");
  });

  it("does not animate the sidebar on the initial render", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    const html = renderToStaticMarkup(
      <SidebarLayout title="Apps" sidebar={<nav />}>
        <div />
      </SidebarLayout>,
    );

    expect(html).not.toContain("transition-[width]");
    expect(html).not.toContain("transition-[left]");
  });
});
