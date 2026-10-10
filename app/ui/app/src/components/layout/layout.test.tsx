import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SidebarLayout } from "./layout";

describe("SidebarLayout", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps the macOS title offset in step with the sidebar state", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    const html = renderToStaticMarkup(
      <SidebarLayout title="Connect your apps" sidebar={<nav />}>
        <div />
      </SidebarLayout>,
    );

    // The macOS title offset is applied so the title clears the sidebar.
    expect(html).toContain("pl-36");
  });

  it("does not apply open/close transitions on the initial mount (#12954)", () => {
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });

    const html = renderToStaticMarkup(
      <SidebarLayout title="Connect your apps" sidebar={<nav />}>
        <div />
      </SidebarLayout>,
    );

    // On the first render the open/close transitions must be absent, otherwise
    // the sidebar animates from closed to open on load instead of appearing
    // in its initial state. Transitions are enabled only after mount.
    expect(html).not.toContain("transition-[width]");
    expect(html).not.toContain("transition-[left]");
    expect(html).not.toContain("transition-all");
    expect(html).not.toContain("transition-[padding-left]");
  });
});
