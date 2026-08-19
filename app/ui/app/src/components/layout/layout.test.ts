import { describe, expect, it } from "vitest";
import { getSidebarLayoutClassNames } from "./layout";

describe("getSidebarLayoutClassNames", () => {
  it("keeps the sidebar open on first paint without transition classes", () => {
    const classes = getSidebarLayoutClassNames({
      animate: false,
      isWindows: false,
      sidebarOpen: true,
    });

    expect(classes.toggleContainer).toContain("left-[204px]");
    expect(classes.toggleContainer).not.toContain("transition-[left]");
    expect(classes.newChatLink).toContain("opacity-0");
    expect(classes.newChatLink).not.toContain("transition-opacity");
    expect(classes.sidebar).toContain("w-64");
    expect(classes.sidebar).not.toContain("transition-[width]");
    expect(classes.main).not.toContain("transition-all");
  });

  it("restores sidebar transition classes after the initial paint", () => {
    const classes = getSidebarLayoutClassNames({
      animate: true,
      isWindows: false,
      sidebarOpen: true,
    });

    expect(classes.root).toContain("transition-[width]");
    expect(classes.toggleContainer).toContain("transition-[left]");
    expect(classes.newChatLink).toContain("transition-opacity");
    expect(classes.sidebar).toContain("transition-[width]");
    expect(classes.main).toContain("transition-all");
  });
});
