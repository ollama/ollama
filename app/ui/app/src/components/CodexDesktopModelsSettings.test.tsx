import type { CodexDesktopModelsSettings as ModelsSettings } from "@/types/webview";
import type {
  ButtonHTMLAttributes,
  HTMLAttributes,
  MouseEvent as ReactMouseEvent,
  ReactNode,
} from "react";
import { act, create } from "react-test-renderer";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CodexDesktopModelsSettings } from "./CodexDesktopModelsSettings";

vi.mock("@headlessui/react", async (importOriginal) => {
  const React = await import("react");
  const original = await importOriginal<typeof import("@headlessui/react")>();
  type PopoverContextValue = {
    open: boolean;
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
    return (
      <PopoverContext.Provider
        value={{
          open,
          toggle: () => setOpen((current) => !current),
        }}
      >
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
    anchor: _anchor,
    children,
    ...props
  }: HTMLAttributes<HTMLDivElement> & {
    anchor?: unknown;
    children: ReactNode;
  }) {
    void _anchor;
    const { open } = usePopover();
    return open ? <div {...props}>{children}</div> : null;
  }

  return Object.assign({}, original, {
    Popover: TestPopover,
    PopoverButton: TestPopoverButton,
    PopoverPanel: TestPopoverPanel,
  });
});

const available = [
  "glm-5.2:cloud",
  "kimi-k3:cloud",
  "gemma4:31b-cloud",
  "llama3.1",
  "llama3",
  "qwen3:8b",
];

function textContent(node: {
  children: Array<string | { children: unknown[] }>;
}): string {
  return node.children
    .map((child) =>
      typeof child === "string"
        ? child
        : textContent(
            child as { children: Array<string | { children: unknown[] }> },
          ),
    )
    .join("");
}

function settings(overrides: Partial<ModelsSettings> = {}): ModelsSettings {
  return {
    supported: true,
    installed: true,
    running: false,
    selected: available.slice(0, 5),
    available,
    maxModels: 5,
    ...overrides,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CodexDesktopModelsSettings", () => {
  it("treats null native model arrays as empty", async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings
            initialSettings={settings({
              selected: null as unknown as string[],
              available: null as unknown as string[],
            })}
          />,
        );
      });

      expect(textContent(renderer!.root)).not.toContain("0 of 5 selected");
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("renders five selected model chips without a selection counter", async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings initialSettings={settings()} />,
        );
      });

      expect(textContent(renderer!.root)).toContain("ChatGPT");
      expect(textContent(renderer!.root)).toContain(
        "Replace ChatGPT's OpenAI model list with up to 5 Ollama models. Your existing profile stays signed in. Codex CLI and IDE share this configuration while Ollama is enabled.",
      );
      expect(textContent(renderer!.root)).not.toContain("5 of 5 selected");
      expect(
        renderer!.root.findByProps({
          src: "/launch-icons/codex-color.svg",
        }),
      ).toBeTruthy();
      expect(
        renderer!.root.findByProps({
          id: "chatgpt-model-settings-heading",
        }).props.className,
      ).toContain("text-sm");
      for (const model of available.slice(0, 5)) {
        expect(
          renderer!.root.findAllByProps({
            "aria-label": `Remove ${model}`,
          }),
        ).toHaveLength(1);
      }
      expect(
        renderer!.root
          .findAllByProps({
            "aria-label": "Add ChatGPT model",
          })
          .filter((node) => node.type === "button"),
      ).toHaveLength(1);
      expect(textContent(renderer!.root)).toContain("Save & start ChatGPT");
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("anchors the dropdown to the full-width model field", async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings
            initialSettings={settings({ selected: [] })}
          />,
        );
      });
      await act(async () => {
        const addButton = renderer!.root
          .findAllByProps({ "aria-label": "Add ChatGPT model" })
          .find((node) => node.type === "button");
        if (!addButton) throw new Error("Add model button not found");
        addButton.props.onClick();
      });

      const dropdown = renderer!.root.findByProps({
        "data-testid": "chatgpt-model-options",
      });
      const picker = renderer!.root.findByProps({
        "data-testid": "chatgpt-model-picker",
      });
      expect(picker.props.className).toContain("w-full");
      expect(dropdown.props.className).toContain("w-[var(--button-width)]");
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("navigates and selects dropdown models with the keyboard", async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings
            initialSettings={settings({ selected: [] })}
          />,
        );
      });
      await act(async () => {
        const addButton = renderer!.root
          .findAllByProps({ "aria-label": "Add ChatGPT model" })
          .find((node) => node.type === "button");
        if (!addButton) throw new Error("Add model button not found");
        addButton.props.onClick();
      });

      const search = renderer!.root.findByProps({
        "aria-label": "Find ChatGPT model",
      });
      const preventDefault = vi.fn();
      await act(async () => {
        search.props.onKeyDown({ key: "ArrowDown", preventDefault });
      });
      let options = renderer!.root.findAllByProps({ role: "option" });
      expect(options[0].props.className).toContain("bg-neutral-100");

      await act(async () => {
        search.props.onKeyDown({ key: "ArrowDown", preventDefault });
      });
      options = renderer!.root.findAllByProps({ role: "option" });
      expect(options[1].props.className).toContain("bg-neutral-100");

      await act(async () => {
        search.props.onKeyDown({ key: "ArrowUp", preventDefault });
      });
      await act(async () => {
        renderer!.root
          .findByProps({ "aria-label": "Find ChatGPT model" })
          .props.onKeyDown({ key: "Enter", preventDefault });
      });
      expect(
        renderer!.root.findAllByProps({
          "aria-label": `Remove ${available[0]}`,
        }),
      ).toHaveLength(1);
      expect(preventDefault).toHaveBeenCalledTimes(4);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("removes a model and applies the remaining selection", async () => {
    const next = available.slice(0, 4);
    const apply = vi.fn().mockResolvedValue({
      settings: settings({ selected: next, running: true }),
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      applyCodexDesktopModels: apply,
      confirm: vi.fn(() => true),
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings initialSettings={settings()} />,
        );
      });
      await act(async () => {
        renderer!.root
          .findByProps({
            "aria-label": `Remove ${available[4]}`,
          })
          .props.onClick();
      });
      expect(textContent(renderer!.root)).not.toContain("4 of 5 selected");

      const applyButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button).includes("Save & start ChatGPT"));
      if (!applyButton) throw new Error("Apply button not found");
      await act(async () => {
        await applyButton.props.onClick();
      });

      expect(apply).toHaveBeenCalledWith(next);
      expect(applyButton.props.disabled).toBe(true);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("does not restart a running profile when confirmation is canceled", async () => {
    const apply = vi.fn();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      applyCodexDesktopModels: apply,
      confirm: vi.fn(() => false),
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings
            initialSettings={settings({ running: true })}
          />,
        );
      });
      await act(async () => {
        renderer!.root
          .findByProps({
            "aria-label": `Remove ${available[4]}`,
          })
          .props.onClick();
      });
      const applyButton = renderer!.root
        .findAllByType("button")
        .find((button) =>
          textContent(button).includes("Save & restart ChatGPT"),
        );
      if (!applyButton) throw new Error("Apply button not found");
      await act(async () => {
        await applyButton.props.onClick();
      });

      expect(window.confirm).toHaveBeenCalledOnce();
      expect(apply).not.toHaveBeenCalled();
    } finally {
      await act(async () => renderer?.unmount());
    }
  });
});
