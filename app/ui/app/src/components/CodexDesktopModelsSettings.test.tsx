import type {
  CodexDesktopModelsSettings as ModelsSettings,
  CodexDesktopStatus,
} from "@/types/webview";
import type {
  ButtonHTMLAttributes,
  HTMLAttributes,
  MouseEvent as ReactMouseEvent,
  ReactNode,
} from "react";
import { createRef } from "react";
import { act, create } from "react-test-renderer";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CodexDesktopModelsSettings,
  type CodexDesktopModelsSettingsHandle,
} from "./CodexDesktopModelsSettings";

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
  "glm-5.3-flash:cloud",
  "glm-5.3:cloud",
  "kimi-k3:cloud",
  "deepseek-v4-flash:cloud",
  "gemma4:31b-cloud",
  "llama3.1",
];

const recommendationDefaults = available.slice(0, 5);

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
    connected: false,
    running: false,
    usesDefaults: false,
    selected: available.slice(0, 5),
    available,
    maxModels: 5,
    ...overrides,
  };
}

function codexStatus(
  overrides: Partial<CodexDesktopStatus> = {},
): CodexDesktopStatus {
  return {
    supported: true,
    installed: true,
    connected: true,
    running: false,
    ...overrides,
  };
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("CodexDesktopModelsSettings", () => {
  it("resets to native recommendation defaults through its settings handle", async () => {
    const resetModels = vi.fn().mockResolvedValue({
      settings: settings({
        usesDefaults: true,
        selected: recommendationDefaults,
      }),
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      resetCodexDesktopModels: resetModels,
    });
    const ref = createRef<CodexDesktopModelsSettingsHandle>();
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings
            ref={ref}
            initialSettings={settings({ selected: ["qwen3:8b"] })}
          />,
        );
      });

      let succeeded = false;
      await act(async () => {
        succeeded = (await ref.current?.resetToDefaults()) ?? false;
      });

      expect(succeeded).toBe(true);
      expect(resetModels).toHaveBeenCalledWith();
      expect(
        renderer!.root.findAllByProps({
          "aria-label": "Remove glm-5.3-flash:cloud",
        }),
      ).toHaveLength(1);
      for (const model of recommendationDefaults) {
        expect(
          renderer!.root.findAllByProps({ "aria-label": `Remove ${model}` }),
        ).toHaveLength(1);
      }
      expect(
        renderer!.root.findAllByProps({ "aria-label": "Remove qwen3:8b" }),
      ).toHaveLength(0);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("resets models without restarting running ChatGPT", async () => {
    const resetModels = vi.fn().mockResolvedValue({
      settings: settings({
        connected: true,
        running: true,
        usesDefaults: true,
        selected: recommendationDefaults,
      }),
    });
    const confirm = vi.fn();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      resetCodexDesktopModels: resetModels,
      confirm,
    });
    const ref = createRef<CodexDesktopModelsSettingsHandle>();
    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings
            ref={ref}
            initialSettings={settings({ connected: true, running: true })}
          />,
        );
      });

      await act(async () => {
        await ref.current?.resetToDefaults();
      });

      expect(resetModels).toHaveBeenCalledTimes(1);
      expect(resetModels).toHaveBeenCalledWith();
      expect(confirm).not.toHaveBeenCalled();
      expect(
        renderer!.root
          .findAllByType("button")
          .some((button) => textContent(button) === "Restart ChatGPT"),
      ).toBe(true);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("stops loading and explains when the native settings bridge is unavailable", async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
    let renderer;
    try {
      await act(async () => {
        renderer = create(<CodexDesktopModelsSettings />);
      });

      expect(textContent(renderer!.root)).toContain(
        "ChatGPT model settings are unavailable in this Ollama build.",
      );
      expect(textContent(renderer!.root)).not.toContain("Loading models…");
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("keeps restart available when live model inventory cannot refresh", async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getCodexDesktopModelsSettings: vi.fn().mockResolvedValue({
        settings: settings({
          connected: true,
          running: true,
          selected: ["qwen3:8b"],
          available: [],
        }),
        warning:
          "Couldn’t refresh available models. Your saved models are unchanged.",
      }),
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(<CodexDesktopModelsSettings />);
      });

      expect(textContent(renderer!.root)).toContain(
        "Couldn’t refresh available models. Your saved models are unchanged.",
      );
      const restartButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button) === "Restart ChatGPT");
      if (!restartButton) throw new Error("Restart button not found");
      expect(Boolean(restartButton.props.disabled)).toBe(false);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("keeps one stable restart operation until ChatGPT is running", async () => {
    vi.useFakeTimers();
    const apply = vi
      .fn()
      .mockResolvedValueOnce({
        settings: settings({ connected: true, running: true }),
        restartConfirmationRequired: true,
      })
      .mockResolvedValueOnce({
        settings: settings({ connected: true, running: false }),
      });
    const getStatus = vi
      .fn()
      .mockResolvedValueOnce(codexStatus())
      .mockResolvedValueOnce(codexStatus({ running: true }));
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      applyCodexDesktopModels: apply,
      getCodexDesktopStatus: getStatus,
      confirm: vi.fn(() => true),
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings
            initialSettings={settings({ connected: true, running: true })}
          />,
        );
      });
      const restartButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button) === "Restart ChatGPT");
      if (!restartButton) throw new Error("Restart button not found");

      await act(async () => {
        restartButton.props.onClick();
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(apply).toHaveBeenCalledTimes(2);
      expect(getStatus).toHaveBeenCalledTimes(1);
      const busyButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button) === "Restarting…");
      if (!busyButton) throw new Error("Busy restart button not found");
      expect(busyButton.props.disabled).toBe(true);
      expect(textContent(renderer!.root)).not.toContain("Starting…");

      await act(async () => {
        restartButton.props.onClick();
        await Promise.resolve();
      });
      expect(apply).toHaveBeenCalledTimes(2);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(250);
      });

      const readyButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button) === "Restart ChatGPT");
      if (!readyButton) throw new Error("Ready restart button not found");
      expect(Boolean(readyButton.props.disabled)).toBe(false);
      expect(getStatus).toHaveBeenCalledTimes(2);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("keeps untouched recommendation defaults implicit when starting ChatGPT", async () => {
    const apply = vi.fn().mockResolvedValue({
      settings: settings({
        connected: true,
        running: true,
        usesDefaults: true,
      }),
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      applyCodexDesktopModels: apply,
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings
            initialSettings={settings({ usesDefaults: true })}
          />,
        );
      });
      const startButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button) === "Start ChatGPT");
      if (!startButton) throw new Error("Start button not found");

      await act(async () => {
        await startButton.props.onClick();
      });

      expect(apply).toHaveBeenCalledWith([], false);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("keeps the same automatic list after the Ollama account changes", async () => {
    const refresh = vi.fn().mockResolvedValue({
      settings: settings({
        usesDefaults: true,
        selected: recommendationDefaults,
      }),
    });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getCodexDesktopModelsSettings: refresh,
    });

    let renderer;
    try {
      await act(async () => {
        renderer = create(
          <CodexDesktopModelsSettings
            accountKey="paid-user:team:cloud-on"
            initialSettings={settings({ usesDefaults: true })}
          />,
        );
      });
      expect(refresh).not.toHaveBeenCalled();

      await act(async () => {
        renderer!.update(
          <CodexDesktopModelsSettings
            accountKey="free-user:free:cloud-on"
            initialSettings={settings({ usesDefaults: true })}
          />,
        );
        await Promise.resolve();
      });

      expect(refresh).toHaveBeenCalledOnce();
      for (const model of recommendationDefaults) {
        expect(
          renderer!.root.findAllByProps({ "aria-label": `Remove ${model}` }),
        ).toHaveLength(1);
      }
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

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
        "Choose up to 5 Ollama models to use in ChatGPT.",
      );
      expect(textContent(renderer!.root)).not.toContain("5 of 5 selected");
      const lightIcon = renderer!.root.findByProps({
        src: "/launch-icons/codex.svg",
      });
      const darkIcon = renderer!.root.findByProps({
        src: "/launch-icons/codex-dark.svg",
      });
      expect(lightIcon.props.className).toContain("h-5 w-5");
      expect(darkIcon.props.className).toContain("h-5 w-5");
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
      expect(textContent(renderer!.root)).toContain("Start ChatGPT");
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

  it("keeps paid recommendations visible and editable for a Free account", async () => {
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
              selected: ["gemma4:31b-cloud"],
              available: ["glm-5.3-flash:cloud", "gemma4:31b-cloud"],
              models: [
                {
                  name: "glm-5.3-flash:cloud",
                  displayName: "glm-5.3-flash:cloud",
                  recommended: true,
                  selected: false,
                  availability: "unavailable",
                  reason: "upgrade_required",
                  requiredPlan: "pro",
                },
                {
                  name: "gemma4:31b-cloud",
                  displayName: "gemma4:31b-cloud",
                  recommended: true,
                  selected: true,
                  availability: "available",
                  requiredPlan: "free",
                },
              ],
            })}
          />,
        );
      });
      await act(async () => {
        renderer!.root
          .findAllByProps({ "aria-label": "Add ChatGPT model" })
          .find((node) => node.type === "button")!
          .props.onClick({});
      });

      const options = renderer!.root.findAllByProps({ role: "option" });
      expect(textContent(options[0])).toContain("glm-5.3-flash:cloud");
      expect(textContent(options[0])).toContain("Pro plan required");
      expect(options[0].props.disabled).toBe(false);
      await act(async () => options[0].props.onClick());
      expect(
        renderer!.root.findAllByProps({
          "aria-label": "Remove glm-5.3-flash:cloud",
        }),
      ).toHaveLength(1);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("shows cloud recommendations to signed-out users", async () => {
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
              selected: ["qwen3:8b"],
              available: ["qwen3:8b"],
              models: [
                {
                  name: "qwen3:8b",
                  displayName: "qwen3:8b",
                  selected: true,
                  availability: "available",
                },
                {
                  name: "glm-5.3-flash:cloud",
                  displayName: "glm-5.3-flash:cloud",
                  recommended: true,
                  selected: false,
                  availability: "unavailable",
                  reason: "sign_in_required",
                  requiredPlan: "pro",
                },
              ],
            })}
          />,
        );
      });
      await act(async () => {
        renderer!.root
          .findAllByProps({ "aria-label": "Add ChatGPT model" })
          .find((node) => node.type === "button")!
          .props.onClick({});
      });

      const cloud = renderer!.root
        .findAllByProps({ role: "option" })
        .find((option) => textContent(option).includes("glm-5.3-flash:cloud"));
      if (!cloud) throw new Error("Cloud recommendation not found");
      expect(textContent(cloud)).toContain("Sign in required");
      expect(cloud.props.disabled).toBe(false);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("removes a model and applies the remaining selection", async () => {
    const next = available.slice(0, 4);
    const apply = vi.fn().mockResolvedValue({
      settings: settings({ connected: true, selected: next, running: true }),
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
        const stopPropagation = vi.fn();
        renderer!.root
          .findByProps({
            "aria-label": `Remove ${available[4]}`,
          })
          .props.onClick({ stopPropagation });
        expect(stopPropagation).toHaveBeenCalledOnce();
      });
      expect(renderer!.root.findAllByProps({ role: "option" })).toHaveLength(0);
      expect(textContent(renderer!.root)).not.toContain("4 of 5 selected");

      const applyButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button).includes("Save & start ChatGPT"));
      if (!applyButton) throw new Error("Apply button not found");
      await act(async () => {
        await applyButton.props.onClick();
      });

      expect(apply).toHaveBeenCalledWith(next, false);
      const restartButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button) === "Restart ChatGPT");
      if (!restartButton) throw new Error("Restart button not found");
      expect(Boolean(restartButton.props.disabled)).toBe(false);
      expect(textContent(renderer!.root)).not.toContain(
        "Your selected Ollama models are ready in ChatGPT.",
      );
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it.each([
    {
      connected: false,
      confirmation:
        "Restart ChatGPT to add Ollama models? Any running task will stop.",
    },
    {
      connected: true,
      confirmation:
        "Restart ChatGPT to update Ollama models? Any running task will stop.",
    },
  ])(
    "uses the native restart copy when connected is $connected",
    async ({ connected, confirmation }) => {
      const next = available.slice(0, 4);
      const apply = vi.fn().mockResolvedValue({
        settings: settings({ connected, running: true }),
        restartConfirmationRequired: true,
      });
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
              initialSettings={settings({ connected, running: true })}
            />,
          );
        });
        await act(async () => {
          renderer!.root
            .findByProps({
              "aria-label": `Remove ${available[4]}`,
            })
            .props.onClick({ stopPropagation: vi.fn() });
        });
        const applyButton = renderer!.root
          .findAllByType("button")
          .find((button) =>
            textContent(button).includes("Save & restart ChatGPT"),
          );
        if (!applyButton) throw new Error("Apply button not found");
        expect(Boolean(applyButton.props.disabled)).toBe(false);
        await act(async () => {
          await applyButton.props.onClick();
        });

        expect(window.confirm).toHaveBeenCalledWith(confirmation);
        expect(apply).toHaveBeenCalledOnce();
        expect(apply).toHaveBeenCalledWith(next, false);
      } finally {
        await act(async () => renderer?.unmount());
      }
    },
  );

  it("retries with live restart consent and applies the saved selection", async () => {
    const next = available.slice(0, 4);
    const apply = vi
      .fn()
      .mockResolvedValueOnce({
        settings: settings({ connected: true, running: true }),
        restartConfirmationRequired: true,
      })
      .mockResolvedValueOnce({
        settings: settings({ connected: true, running: true, selected: next }),
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
          <CodexDesktopModelsSettings
            initialSettings={settings({ connected: true, running: false })}
          />,
        );
      });
      await act(async () => {
        renderer!.root
          .findByProps({ "aria-label": `Remove ${available[4]}` })
          .props.onClick({ stopPropagation: vi.fn() });
      });
      const applyButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button).includes("Save & start ChatGPT"));
      if (!applyButton) throw new Error("Apply button not found");
      await act(async () => {
        await applyButton.props.onClick();
      });

      expect(window.confirm).toHaveBeenCalledWith(
        "Restart ChatGPT to update Ollama models? Any running task will stop.",
      );
      expect(apply.mock.calls).toEqual([
        [next, false],
        [next, true],
      ]);
      expect(
        renderer!.root.findAllByProps({
          "aria-label": `Remove ${available[4]}`,
        }),
      ).toHaveLength(0);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("ignores a stale focus refresh that finishes after an apply", async () => {
    const next = available.slice(0, 4);
    let resolveRefresh!: (value: { settings: ModelsSettings }) => void;
    const refresh = vi.fn(
      () =>
        new Promise<{ settings: ModelsSettings }>((resolve) => {
          resolveRefresh = resolve;
        }),
    );
    const apply = vi.fn().mockResolvedValue({
      settings: settings({ connected: true, running: true, selected: next }),
    });
    let focusHandler: (() => void) | undefined;
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn((event: string, handler: () => void) => {
        if (event === "focus") focusHandler = handler;
      }),
      removeEventListener: vi.fn(),
      getCodexDesktopModelsSettings: refresh,
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
          .findByProps({ "aria-label": `Remove ${available[4]}` })
          .props.onClick({ stopPropagation: vi.fn() });
        focusHandler?.();
      });
      const applyButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button).includes("Save & start ChatGPT"));
      if (!applyButton) throw new Error("Apply button not found");
      await act(async () => {
        await applyButton.props.onClick();
      });
      await act(async () => {
        resolveRefresh({ settings: settings() });
        await Promise.resolve();
      });

      expect(apply).toHaveBeenCalledWith(next, false);
      expect(
        renderer!.root.findAllByProps({
          "aria-label": `Remove ${available[4]}`,
        }),
      ).toHaveLength(0);
      expect(textContent(renderer!.root)).not.toContain("Save & start ChatGPT");
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("keeps restart available without a separate removal button", async () => {
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
            initialSettings={settings({ connected: true, running: true })}
          />,
        );
      });
      const content = textContent(renderer!.root);
      expect(content).toContain("Restart ChatGPT");
      expect(content).not.toContain("Remove Ollama models");
      const restartButton = renderer!.root
        .findAllByType("button")
        .find((button) => textContent(button) === "Restart ChatGPT");
      if (!restartButton) throw new Error("Restart button not found");
      expect(Boolean(restartButton.props.disabled)).toBe(false);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("keeps pill bodies on the field trigger and reserves removal for the x", async () => {
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

      const pill = renderer!.root
        .findAllByType("span")
        .find(
          (node) =>
            textContent(node) === available[0] &&
            String(node.props.className).includes(
              "pointer-events-none inline-flex",
            ),
        );
      if (!pill) throw new Error("Selected model pill not found");
      expect(
        renderer!.root
          .findAllByType("button")
          .filter((button) => textContent(button) === available[0]),
      ).toHaveLength(0);
      expect(pill.findAllByType("button")).toHaveLength(1);
      expect(pill.findByType("button").props["aria-label"]).toBe(
        `Remove ${available[0]}`,
      );

      const fieldTrigger = renderer!.root
        .findAllByProps({ "aria-label": "Add ChatGPT model" })
        .find((node) => node.type === "button");
      if (!fieldTrigger) throw new Error("Model field trigger not found");
      await act(async () => {
        fieldTrigger.props.onClick({});
      });

      expect(renderer!.root.findAllByProps({ role: "option" })).toHaveLength(
        available.length,
      );
      expect(
        renderer!.root.findAllByProps({
          "aria-label": `Remove ${available[0]}`,
        }),
      ).toHaveLength(1);
      expect(
        renderer!.root
          .findAllByProps({ role: "option" })
          .filter((option) => option.props["aria-selected"] === true),
      ).toHaveLength(5);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });

  it("does not show an Ollama approval control", async () => {
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

      expect(textContent(renderer!.root)).not.toContain("Approve for me");
      expect(
        renderer!.root.findAllByProps({ "aria-label": "Approve for me" }),
      ).toHaveLength(0);
    } finally {
      await act(async () => renderer?.unmount());
    }
  });
});
