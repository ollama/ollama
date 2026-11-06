import React from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import type { Highlighter } from "shiki";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  loadHighlighter: vi.fn(),
  language: "javascript" as string | null,
}));

vi.mock("@/lib/highlighter", () => ({
  loadHighlighter: mocks.loadHighlighter,
}));
vi.mock("streamdown", () => ({
  Streamdown: ({
    children,
    components,
  }: {
    children: React.ReactNode;
    components: {
      pre: React.ComponentType<React.HTMLAttributes<HTMLPreElement>>;
    };
  }) => {
    if (children === "Plain prose") return <p>{children}</p>;
    const Pre = components.pre;
    return (
      <Pre>
        <code
          className={mocks.language ? `language-${mocks.language}` : undefined}
        >
          {children}
        </code>
      </Pre>
    );
  },
  defaultRehypePlugins: { katex: "katex" },
  defaultRemarkPlugins: { gfm: "gfm", math: "math" },
}));

import StreamingMarkdownContent from "./StreamingMarkdownContent";

describe("lazy code highlighting", () => {
  let renderer: ReactTestRenderer | undefined;

  beforeEach(() => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    mocks.loadHighlighter.mockReset();
    mocks.language = "javascript";
  });

  afterEach(async () => {
    await act(async () => renderer?.unmount());
    renderer = undefined;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("does not initialize for prose or unlabelled code", async () => {
    await act(async () => {
      renderer = create(<StreamingMarkdownContent content="Plain prose" />);
    });
    mocks.language = null;
    await act(async () => {
      renderer!.update(<StreamingMarkdownContent content="unlabelled code" />);
    });
    expect(mocks.loadHighlighter).not.toHaveBeenCalled();
    expect(renderer!.root.findAllByType("code")[0].children).toEqual([
      "unlabelled code",
    ]);
  });

  it("renders plain code immediately, then highlights both themes and streamed updates", async () => {
    let resolve!: (value: Highlighter) => void;
    mocks.loadHighlighter.mockReturnValue(
      new Promise<Highlighter>((done) => {
        resolve = done;
      }),
    );
    await act(async () => {
      renderer = create(
        <StreamingMarkdownContent content="const answer = 42;" />,
      );
    });
    expect(renderer!.root.findAllByType("code")[0].children).toEqual([
      "const answer = 42;",
    ]);

    const codeToTokensBase = vi.fn(
      (text: string, { theme }: { theme: string }) => [
        [
          {
            content: text,
            color: theme === "one-light" ? "#a626a4" : "#c678dd",
          },
        ],
      ],
    );
    await act(async () => {
      resolve({ codeToTokensBase } as unknown as Highlighter);
    });
    const blocks = renderer!.root.findAllByType("code");
    expect(blocks[0].findByType("span").props.style.color).toBe("#a626a4");
    expect(blocks[1].findByType("span").props.style.color).toBe("#c678dd");

    await act(async () => {
      renderer!.update(
        <StreamingMarkdownContent content="const answer = 43;" />,
      );
    });
    expect(
      renderer!.root.findAllByType("code")[0].findByType("span").children,
    ).toEqual(["const answer = 43;"]);
    expect(mocks.loadHighlighter).toHaveBeenCalledOnce();
  });

  it("keeps code readable when initialization fails", async () => {
    const error = new Error("grammar unavailable");
    mocks.loadHighlighter.mockRejectedValue(error);
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    await act(async () => {
      renderer = create(
        <StreamingMarkdownContent content="const answer = 42;" />,
      );
    });
    expect(renderer!.root.findAllByType("code")[0].children).toEqual([
      "const answer = 42;",
    ]);
    expect(log).toHaveBeenCalledWith(
      "Failed to initialize syntax highlighting:",
      error,
    );
  });
});
