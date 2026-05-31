import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import StreamingMarkdownContent from "./StreamingMarkdownContent";

const renderMarkdown = (content: string) =>
  renderToStaticMarkup(<StreamingMarkdownContent content={content} />);

describe("StreamingMarkdownContent", () => {
  it("drops raw HTML elements", () => {
    const html = renderMarkdown(`
Regular markdown

<div class="html-snippet">HTML block</div>
<iframe src="https://example.test/embed"></iframe>
<img src="https://example.test/image.png" />

More markdown
`);

    expect(html).toContain("Regular markdown");
    expect(html).toContain("More markdown");
    expect(html).not.toContain("html-snippet");
    expect(html).not.toContain("<iframe");
    expect(html).not.toContain("<img");
  });

  it("does not auto-load markdown images", () => {
    const html = renderMarkdown("![diagram](https://example.test/diagram.png)");

    expect(html).not.toContain("<img");
    expect(html).not.toContain("https://example.test/diagram.png");
  });

  it("blocks unsupported markdown link protocols", () => {
    const html = renderMarkdown("[example](data:text/plain,hello)");

    expect(html).toContain("example");
    expect(html).not.toContain("data:text");
  });
});
