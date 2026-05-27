import { describe, it, expect } from "vitest";
import { sanitize } from "hast-util-sanitize";
import { defaultSchema } from "rehype-sanitize";

// Mirror the sanitizeSchema from StreamingMarkdownContent.tsx
const sanitizeSchema = {
  ...defaultSchema,
  tagNames: [...(defaultSchema.tagNames || []), "ol-citation"],
  attributes: {
    ...defaultSchema.attributes,
    div: [
      ...(defaultSchema.attributes?.div || []),
      ["className", "math", "math-display"],
    ],
    span: [
      ...(defaultSchema.attributes?.span || []),
      ["className", "math", "math-inline"],
    ],
    "ol-citation": ["cursor", "start", "end"],
  },
  strip: ["script", "style"],
};

// Helper to create a hast element node
function h(
  tagName: string,
  properties: Record<string, unknown>,
  children: any[] = [],
): any {
  return { type: "element", tagName, properties, children };
}

function text(value: string): any {
  return { type: "text", value };
}

function root(...children: any[]): any {
  return { type: "root", children };
}

describe("sanitizeSchema", () => {
  it("should strip <style> tags and their content", () => {
    const tree = root(
      h("style", {}, [
        text("body { background: red; } button { background: linear-gradient(blue, green); }"),
      ]),
      h("p", {}, [text("Hello world")]),
    );

    const result = sanitize(tree, sanitizeSchema);

    // <style> should be completely stripped (including content)
    const hasStyle = JSON.stringify(result).includes("background");
    expect(hasStyle).toBe(false);

    // <p> should survive
    expect(result.children).toHaveLength(1);
    expect(result.children[0].tagName).toBe("p");
  });

  it("should strip <script> tags and their content", () => {
    const tree = root(
      h("script", {}, [text("alert('xss')")]),
      h("p", {}, [text("Safe content")]),
    );

    const result = sanitize(tree, sanitizeSchema);

    const hasScript = JSON.stringify(result).includes("alert");
    expect(hasScript).toBe(false);
    expect(result.children).toHaveLength(1);
    expect(result.children[0].tagName).toBe("p");
  });

  it("should strip <iframe> tags", () => {
    const tree = root(
      h("iframe", { src: "https://evil.com" }, []),
      h("p", {}, [text("Safe content")]),
    );

    const result = sanitize(tree, sanitizeSchema);

    const hasIframe = result.children.some(
      (c: any) => c.tagName === "iframe",
    );
    expect(hasIframe).toBe(false);
  });

  it("should preserve math block elements (div.math.math-display)", () => {
    const tree = root(
      h("div", { className: ["math", "math-display"] }, [
        text("E = mc^2"),
      ]),
    );

    const result = sanitize(tree, sanitizeSchema);

    expect(result.children).toHaveLength(1);
    expect(result.children[0].tagName).toBe("div");
    expect(result.children[0].properties.className).toEqual([
      "math",
      "math-display",
    ]);
  });

  it("should preserve math inline elements (span.math.math-inline)", () => {
    const tree = root(
      h("span", { className: ["math", "math-inline"] }, [text("x^2")]),
    );

    const result = sanitize(tree, sanitizeSchema);

    expect(result.children).toHaveLength(1);
    expect(result.children[0].tagName).toBe("span");
    expect(result.children[0].properties.className).toEqual([
      "math",
      "math-inline",
    ]);
  });

  it("should preserve ol-citation elements with attributes", () => {
    const tree = root(
      h("ol-citation", { cursor: "1", start: "25", end: "30" }, []),
    );

    const result = sanitize(tree, sanitizeSchema);

    expect(result.children).toHaveLength(1);
    expect(result.children[0].tagName).toBe("ol-citation");
    expect(result.children[0].properties.cursor).toBe("1");
    expect(result.children[0].properties.start).toBe("25");
    expect(result.children[0].properties.end).toBe("30");
  });

  it("should preserve code elements with language classes", () => {
    const tree = root(
      h("pre", {}, [
        h("code", { className: ["language-python"] }, [
          text("print('hello')"),
        ]),
      ]),
    );

    const result = sanitize(tree, sanitizeSchema);

    expect(result.children).toHaveLength(1);
    const code = result.children[0].children[0];
    expect(code.tagName).toBe("code");
    expect(code.properties.className).toEqual(["language-python"]);
  });

  it("should preserve standard markdown elements", () => {
    const tree = root(
      h("h1", {}, [text("Title")]),
      h("p", {}, [
        text("Some "),
        h("strong", {}, [text("bold")]),
        text(" and "),
        h("em", {}, [text("italic")]),
        text(" text."),
      ]),
      h("ul", {}, [
        h("li", {}, [text("Item 1")]),
        h("li", {}, [text("Item 2")]),
      ]),
      h("a", { href: "https://example.com" }, [text("A link")]),
    );

    const result = sanitize(tree, sanitizeSchema);

    const tagNames = result.children.map((c: any) => c.tagName);
    expect(tagNames).toEqual(["h1", "p", "ul", "a"]);
  });

  it("should strip model-generated HTML page that would corrupt the UI", () => {
    // Simulate a model generating a full HTML page
    const tree = root(
      h("style", {}, [
        text(`
          * { margin: 0; padding: 0; }
          button { background: linear-gradient(to right, #ff0000, #0000ff); }
          .some-class { font-size: 72px; }
        `),
      ]),
      h("div", {}, [
        h("h1", {}, [text("My Generated Page")]),
        h("p", {}, [text("This is model-generated content")]),
      ]),
    );

    const result = sanitize(tree, sanitizeSchema);

    // Style tag and its content should be gone
    const serialized = JSON.stringify(result);
    expect(serialized).not.toContain("linear-gradient");
    expect(serialized).not.toContain("margin: 0");

    // The safe content should remain
    expect(serialized).toContain("My Generated Page");
    expect(serialized).toContain("model-generated content");
  });
});
