import {
  act,
  create,
  type ReactTestInstance,
  type ReactTestRenderer,
} from "react-test-renderer";
import {
  createRef,
  type ButtonHTMLAttributes,
  type HTMLAttributes,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
} from "react";
import { describe, expect, it, vi } from "vitest";
import { Switch } from "./ui/switch";
import {
  ClaudeDesktopModelsSettings,
  type ClaudeDesktopModelsSettingsHandle,
} from "./ClaudeDesktopModelsSettings";

vi.mock("@headlessui/react", async (importOriginal) => {
  const React = await import("react");
  const original = await importOriginal<typeof import("@headlessui/react")>();
  type PopoverContextValue = {
    open: boolean;
    close: () => void;
    toggle: () => void;
  };
  const PopoverContext = React.createContext<PopoverContextValue | null>(null);
  const usePopover = () => {
    const context = React.useContext(PopoverContext);
    if (!context) throw new Error("Popover components must be nested");
    return context;
  };

  function TestPopover({
    children,
    className,
  }: {
    children: ReactNode;
    className?: string;
  }) {
    const [open, setOpen] = React.useState(false);
    const context = {
      open,
      close: () => setOpen(false),
      toggle: () => setOpen((current) => !current),
    };
    return (
      <PopoverContext.Provider value={context}>
        <div className={className}>{children}</div>
      </PopoverContext.Provider>
    );
  }

  function TestPopoverButton({
    onClick,
    ...props
  }: ButtonHTMLAttributes<HTMLButtonElement>) {
    const { open, toggle } = usePopover();
    return (
      <button
        {...props}
        aria-expanded={open}
        onClick={(event: ReactMouseEvent<HTMLButtonElement>) => {
          onClick?.(event);
          toggle();
        }}
      />
    );
  }

  function TestPopoverPanel({
    anchor,
    children,
    ...props
  }: HTMLAttributes<HTMLDivElement> & {
    anchor?: unknown;
    children: ReactNode | ((props: { close: () => void }) => ReactNode);
  }) {
    const { open, close } = usePopover();
    if (!open) return null;
    return (
      <div {...props} data-anchor={JSON.stringify(anchor)}>
        {typeof children === "function" ? children({ close }) : children}
      </div>
    );
  }

  return Object.assign({}, original, {
    Popover: TestPopover,
    PopoverButton: TestPopoverButton,
    PopoverPanel: TestPopoverPanel,
  });
});

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
    pickerButton(renderer).props.onClick();
    await Promise.resolve();
  });
  await act(async () => {
    renderer.root.findAllByProps({ role: "option" })[1].props.onClick();
    await Promise.resolve();
  });
}

function pickerButton(renderer: ReactTestRenderer) {
  const button = renderer.root
    .findAllByType("button")
    .find(
      (candidate) =>
        candidate.props["aria-label"] === "Ollama model for Fable 5",
    );
  if (!button) throw new Error("Claude model picker button not found");
  return button;
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

function textContent(node: ReactTestInstance): string {
  return node.children
    .map((child) => (typeof child === "string" ? child : textContent(child)))
    .join("");
}

describe("ClaudeDesktopModelsSettings interactions", () => {
  it("opens below without scrolling and disables auto mode for draft changes", async () => {
    class TestHTMLElement {
      focus() {}
    }
    const focus = vi.fn();
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
          {
            createNodeMock: (element) =>
              element.type === "input" ? { focus } : null,
          },
        );
        await Promise.resolve();
      });

      const autoModeSwitch = () =>
        renderer!.root.findByProps({ role: "switch" });
      expect(autoModeSwitch().props.disabled).not.toBe(true);
      expect(autoModeSwitch().props["aria-checked"]).toBe(true);

      await act(async () => {
        pickerButton(renderer!).props.onClick();
        await Promise.resolve();
      });
      expect(
        renderer!.root.findByProps({
          "data-anchor": JSON.stringify({
            to: "bottom end",
            gap: 8,
            padding: 8,
          }),
        }),
      ).toBeDefined();
      expect(focus).toHaveBeenCalledWith({ preventScroll: true });
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

  it("restores Auto mode when restart confirmation is canceled", async () => {
    class TestHTMLElement {
      focus() {}
    }
    const runningStatus = {
      ...testStatus("glm-5.2:cloud", true),
      autoMode: true,
      models: testStatus().models.map((model) => ({
        ...model,
        autoMode: true,
      })),
    };
    const setAutoMode = vi.fn().mockResolvedValue({
      status: runningStatus,
      error:
        "Claude Desktop restart confirmation is required before changing its profile",
      restartConfirmationRequired: true,
    });
    const confirm = vi.fn(() => false);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      HTMLElement: TestHTMLElement,
      setClaudeDesktopAutoMode: setAutoMode,
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
            initialStatus={runningStatus}
          />,
        );
        await Promise.resolve();
      });
      await act(async () => {
        renderer!.root.findByType(Switch).props.onChange(false);
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(setAutoMode).toHaveBeenCalledTimes(1);
      expect(setAutoMode).toHaveBeenCalledWith(false, false);
      expect(confirm).toHaveBeenCalledWith(
        "Restart Claude to change auto mode? Any running task will stop.",
      );
      expect(
        renderer!.root.findByProps({ role: "switch" }).props["aria-checked"],
      ).toBe(true);
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

  it("keeps the previous mappings when reset restart is canceled", async () => {
    class TestHTMLElement {
      focus() {}
    }
    const currentStatus = testStatus("kimi-k3:cloud", true);
    const resetMappings = vi.fn().mockResolvedValue({
      status: currentStatus,
      restartConfirmationRequired: true,
    });
    const confirm = vi.fn(() => false);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      HTMLElement: TestHTMLElement,
      resetClaudeDesktopMappings: resetMappings,
      confirm,
    });
    vi.stubGlobal("document", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

    let renderer: ReactTestRenderer | undefined;
    const settingsRef = createRef<ClaudeDesktopModelsSettingsHandle>();
    try {
      await act(async () => {
        renderer = create(
          <ClaudeDesktopModelsSettings
            ref={settingsRef}
            initialLocalModels={[]}
            initialStatus={currentStatus}
          />,
        );
        await Promise.resolve();
      });

      let resetSucceeded = true;
      await act(async () => {
        resetSucceeded =
          (await settingsRef.current?.resetToDefaults()) ?? false;
      });

      expect(resetSucceeded).toBe(false);
      expect(confirm).toHaveBeenCalledOnce();
      expect(resetMappings).toHaveBeenCalledWith(false);
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

  it("applies the native reset result and shows progress", async () => {
    class TestHTMLElement {
      focus() {}
    }
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
    };
    const resetStatus = {
      ...initialStatus,
      mappings: [
        { ...fableRoute },
        {
          routeId: "claude-sonnet-5",
          routeName: "Sonnet 5",
          model: "glm-5.2:cloud",
        },
      ],
    };
    const resetResult = {
      status: resetStatus,
      mappingsApplied: true,
    };
    let resolveReset!: (result: typeof resetResult) => void;
    const resetRequestResult = new Promise<typeof resetResult>((resolve) => {
      resolveReset = resolve;
    });
    const resetMappings = vi.fn(() => resetRequestResult);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      HTMLElement: TestHTMLElement,
      resetClaudeDesktopMappings: resetMappings,
    });
    vi.stubGlobal("document", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

    let renderer: ReactTestRenderer | undefined;
    const settingsRef = createRef<ClaudeDesktopModelsSettingsHandle>();
    try {
      await act(async () => {
        renderer = create(
          <ClaudeDesktopModelsSettings
            ref={settingsRef}
            initialLocalModels={[]}
            initialStatus={initialStatus}
          />,
        );
        await Promise.resolve();
        await Promise.resolve();
      });

      let resetRequest: Promise<boolean> | undefined;
      await act(async () => {
        resetRequest = settingsRef.current?.resetToDefaults();
        await Promise.resolve();
      });
      expect(actionButton(renderer!).props.disabled).toBe(true);
      expect(textContent(actionButton(renderer!))).toContain("Resetting…");

      resolveReset(resetResult);
      let resetSucceeded = false;
      await act(async () => {
        resetSucceeded = (await resetRequest) ?? false;
      });

      expect(resetSucceeded).toBe(true);
      expect(resetMappings).toHaveBeenCalledWith(false);

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
