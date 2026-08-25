import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { describe, expect, it, vi } from "vitest";
import { ClaudeDesktopModelsSettings } from "./ClaudeDesktopModelsSettings";

describe("ClaudeDesktopModelsSettings interactions", () => {
  it("disables auto mode while model changes are not applied", async () => {
    class TestHTMLElement {
      focus() {}
    }
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      HTMLElement: TestHTMLElement,
    });
    vi.stubGlobal("document", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(
          <ClaudeDesktopModelsSettings
            initialLocalModels={[]}
            initialStatus={{
              supported: true,
              used: true,
              installed: true,
              connected: true,
              running: false,
              startFailed: false,
              portConflict: false,
              autoMode: true,
              modelSource: "user",
              mappings: [
                {
                  routeId: "claude-fable-5",
                  routeName: "Fable 5",
                  model: "glm-5.2:cloud",
                },
              ],
              models: [
                {
                  name: "glm-5.2:cloud",
                  displayName: "glm-5.2:cloud",
                  cloud: true,
                  selected: true,
                  autoMode: true,
                },
                {
                  name: "kimi-k3:cloud",
                  displayName: "kimi-k3:cloud",
                  cloud: true,
                  selected: false,
                  autoMode: true,
                },
              ],
            }}
          />,
        );
        await Promise.resolve();
      });

      const autoModeSwitch = () =>
        renderer!.root.findByProps({ role: "switch" });
      expect(autoModeSwitch().props.disabled).not.toBe(true);
      expect(autoModeSwitch().props["aria-checked"]).toBe(true);

      await act(async () => {
        renderer!.root
          .findByProps({ "aria-label": "Ollama model for Fable 5" })
          .props.onClick();
        await Promise.resolve();
      });
      await act(async () => {
        const options = renderer!.root.findAllByProps({ role: "option" });
        options[1].props.onClick();
        await Promise.resolve();
      });

      expect(autoModeSwitch().props.disabled).toBe(true);
      expect(autoModeSwitch().props["aria-checked"]).toBe(true);
      expect(
        renderer!.root
          .findAllByType("p")
          .some((node) =>
            node.children
              .join("")
              .includes(
                "Start or restart Claude to apply model changes before changing auto mode.",
              ),
          ),
      ).toBe(true);
    } finally {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }
  });
});
