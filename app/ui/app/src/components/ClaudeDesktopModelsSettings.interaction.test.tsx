import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { describe, expect, it, vi } from "vitest";
import { ClaudeDesktopModelsSettings } from "./ClaudeDesktopModelsSettings";

const fableRoute = {
  routeId: "claude-fable-5",
  routeName: "Fable 5",
};

function testStatus(model = "glm-5.2:cloud", running = false) {
  return {
    supported: true,
    used: true,
    installed: true,
    connected: true,
    running,
    startFailed: false,
    portConflict: false,
    autoMode: false,
    modelSource: "user" as const,
    mappings: [{ ...fableRoute, model }],
    defaultMappings: [{ ...fableRoute, model: "glm-5.2:cloud" }],
    models: [
      {
        name: "glm-5.2:cloud",
        displayName: "glm-5.2:cloud",
        cloud: true,
        selected: model === "glm-5.2:cloud",
        availability: "available" as const,
      },
      {
        name: "kimi-k3:cloud",
        displayName: "kimi-k3:cloud",
        cloud: true,
        selected: model === "kimi-k3:cloud",
        availability: "available" as const,
      },
    ],
  };
}

async function selectKimi(renderer: ReactTestRenderer) {
  await act(async () => {
    renderer.root
      .findByProps({ "aria-label": "Ollama model for Fable 5" })
      .props.onClick();
    await Promise.resolve();
  });
  await act(async () => {
    renderer.root.findAllByProps({ role: "option" })[1].props.onClick();
    await Promise.resolve();
  });
}

function actionButton(renderer: ReactTestRenderer) {
  const button = renderer.root
    .findAllByType("button")
    .find(
      (candidate) =>
        !candidate.props["aria-label"] &&
        candidate.props.className?.includes("flex-shrink-0"),
    );
  if (!button) throw new Error("Claude action button not found");
  return button;
}

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

  it("asks for confirmation from live native state before restarting", async () => {
    class TestHTMLElement {
      focus() {}
    }
    const apply = vi
      .fn()
      .mockResolvedValueOnce({
        status: testStatus("glm-5.2:cloud", true),
        error:
          "Claude Desktop restart confirmation is required before changing its profile",
        restartConfirmationRequired: true,
      })
      .mockResolvedValueOnce({
        status: testStatus("kimi-k3:cloud", true),
        mappingsApplied: true,
      });
    const confirm = vi.fn(() => true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      HTMLElement: TestHTMLElement,
      applyClaudeDesktopMappings: apply,
      confirm,
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
            initialStatus={testStatus()}
          />,
        );
        await Promise.resolve();
      });
      await selectKimi(renderer!);
      await act(async () => {
        actionButton(renderer!).props.onClick();
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(confirm).toHaveBeenCalledWith(
        "Restart Claude Desktop? Any running task will stop.",
      );
      expect(apply).toHaveBeenNthCalledWith(
        1,
        { "claude-fable-5": "kimi-k3:cloud" },
        false,
      );
      expect(apply).toHaveBeenNthCalledWith(
        2,
        { "claude-fable-5": "kimi-k3:cloud" },
        true,
      );
    } finally {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }
  });

  it("ignores a stale focus refresh that finishes after apply", async () => {
    class TestHTMLElement {
      focus() {}
    }
    let focusHandler: (() => void) | undefined;
    let resolveRefresh:
      | ((status: ReturnType<typeof testStatus>) => void)
      | undefined;
    const staleRefresh = new Promise<ReturnType<typeof testStatus>>(
      (resolve) => {
        resolveRefresh = resolve;
      },
    );
    vi.stubGlobal("window", {
      addEventListener: vi.fn((event: string, handler: () => void) => {
        if (event === "focus") focusHandler = handler;
      }),
      removeEventListener: vi.fn(),
      HTMLElement: TestHTMLElement,
      getClaudeDesktopStatus: vi.fn(() => staleRefresh),
      applyClaudeDesktopMappings: vi.fn().mockResolvedValue({
        status: testStatus("kimi-k3:cloud"),
        mappingsApplied: true,
      }),
      confirm: vi.fn(() => true),
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
            initialStatus={testStatus()}
          />,
        );
        await Promise.resolve();
      });
      await selectKimi(renderer!);
      await act(async () => {
        focusHandler?.();
        actionButton(renderer!).props.onClick();
        await Promise.resolve();
        await Promise.resolve();
      });
      await act(async () => {
        resolveRefresh?.(testStatus("glm-5.2:cloud"));
        await staleRefresh;
        await Promise.resolve();
      });

      const picker = renderer!.root.findByProps({
        "aria-label": "Ollama model for Fable 5",
      });
      expect(picker.findAllByType("span")[0].children.join("")).toBe(
        "kimi-k3:cloud",
      );
    } finally {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }
  });

  it("accepts committed mappings when launching Claude fails", async () => {
    class TestHTMLElement {
      focus() {}
    }
    const onDraftChange = vi.fn();
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      HTMLElement: TestHTMLElement,
      applyClaudeDesktopMappings: vi.fn().mockResolvedValue({
        status: testStatus("kimi-k3:cloud"),
        error:
          "Claude model mappings were saved, but Claude Desktop could not open",
        mappingsApplied: true,
      }),
      confirm: vi.fn(() => true),
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
            initialStatus={testStatus()}
            onDraftChange={onDraftChange}
          />,
        );
        await Promise.resolve();
      });
      await selectKimi(renderer!);
      await act(async () => {
        actionButton(renderer!).props.onClick();
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(onDraftChange).toHaveBeenLastCalledWith(false);
      const picker = renderer!.root.findByProps({
        "aria-label": "Ollama model for Fable 5",
      });
      expect(picker.findAllByType("span")[0].children.join("")).toBe(
        "kimi-k3:cloud",
      );
    } finally {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }
  });

  it("resets sparse mappings to the defaults for the current account", async () => {
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
    const initialStatus = {
      ...testStatus("glm-5.2:cloud"),
      mappings: [
        { ...fableRoute, model: "glm-5.2:cloud" },
        {
          routeId: "claude-sonnet-5",
          routeName: "Sonnet 5",
          model: "kimi-k3:cloud",
        },
      ],
      defaultMappings: [
        { ...fableRoute },
        {
          routeId: "claude-sonnet-5",
          routeName: "Sonnet 5",
          model: "glm-5.2:cloud",
        },
      ],
    };

    let renderer: ReactTestRenderer | undefined;
    try {
      await act(async () => {
        renderer = create(
          <ClaudeDesktopModelsSettings
            initialLocalModels={[]}
            initialStatus={initialStatus}
            resetVersion={0}
          />,
        );
        await Promise.resolve();
      });
      await act(async () => {
        renderer!.update(
          <ClaudeDesktopModelsSettings
            initialLocalModels={[]}
            initialStatus={initialStatus}
            resetVersion={1}
          />,
        );
        await Promise.resolve();
      });

      const fable = renderer!.root.findByProps({
        "aria-label": "Ollama model for Fable 5",
      });
      const sonnet = renderer!.root.findByProps({
        "aria-label": "Ollama model for Sonnet 5",
      });
      expect(fable.findAllByType("span")[0].children.join("")).toBe(
        "Select a model",
      );
      expect(sonnet.findAllByType("span")[0].children.join("")).toBe(
        "glm-5.2:cloud",
      );
    } finally {
      await act(async () => {
        renderer?.unmount();
        await Promise.resolve();
      });
      vi.unstubAllGlobals();
    }
  });
});
