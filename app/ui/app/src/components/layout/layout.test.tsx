import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SidebarLayout } from "./layout";

describe("SidebarLayout", () => {
  it("does not animate its initial layout", () => {
    const html = renderToStaticMarkup(
      <SidebarLayout title="Settings" sidebar={<div />}>
        <div />
      </SidebarLayout>,
    );

    expect(html).not.toContain("transition-[left]");
    expect(html).not.toContain("transition-[width]");
    expect(html).not.toContain("transition-all");
  });
});
