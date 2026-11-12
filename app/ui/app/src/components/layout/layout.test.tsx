import { act, create } from "react-test-renderer";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", () => ({
  Link: () => <a title="New chat" />,
}));

describe("SidebarLayout", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it.each(["darwin", "windows"])(
    "opens immediately and animates toggles on %s, preserving state across routes",
    async (platform) => {
      vi.stubGlobal("window", { OLLAMA_PLATFORM: platform });
      vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
      const { SidebarLayout } = await import("./layout");
      const layout = (
        <SidebarLayout title="Connect your apps" sidebar={<nav />}>
          <div />
        </SidebarLayout>
      );
      let renderer: ReturnType<typeof create>;
      await act(async () => {
        renderer = create(layout);
      });
      const html = () => JSON.stringify(renderer.toJSON());
      const toggle = async () => {
        await act(async () => {
          renderer.root.findByType("button").props.onClick();
        });
      };
      try {
        expect(renderer!.root.findByType("nav")).toBeDefined();
        expect(html()).toContain("w-48");
        expect(html()).toContain("pl-6");
        expect(html()).not.toContain("transition-");
        expect(renderer!.root.findByType("button").props["aria-label"]).toBe(
          "Hide sidebar",
        );

        await toggle();
        expect(renderer!.root.findAllByType("nav")).toHaveLength(0);
        expect(html()).toContain("w-0");
        expect(html()).toContain(platform === "windows" ? "pl-16" : "pl-36");
        expect(html()).toContain("transition-[width]");
        expect(html()).toContain("transition-[left]");
        expect(html()).toContain("transition-[padding-left]");

        await act(async () => renderer!.unmount());
        await act(async () => {
          renderer = create(layout);
        });
        expect(renderer!.root.findAllByType("nav")).toHaveLength(0);
        expect(html()).not.toContain("transition-");

        await toggle();
        expect(renderer!.root.findByType("nav")).toBeDefined();
        expect(html()).toContain("w-48");
        expect(html()).toContain("transition-[width]");
      } finally {
        await act(async () => renderer!.unmount());
      }
    },
  );
});
