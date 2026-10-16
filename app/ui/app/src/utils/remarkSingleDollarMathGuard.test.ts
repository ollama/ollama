import { describe, it, expect } from "vitest";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { VFile } from "vfile";
import { visit } from "unist-util-visit";
import type { Root } from "mdast";
import remarkSingleDollarMathGuard from "./remarkSingleDollarMathGuard";

// Runs the same mdast pipeline StreamingMarkdownContent configures, minus the
// citation parser (not relevant to this guard), and returns every node type
// plus its rendered text so tests can assert both what got parsed as math and
// what the reader would actually see.
function parse(markdown: string) {
  const processor = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath, { singleDollarTextMath: true })
    .use(remarkSingleDollarMathGuard);
  const file = new VFile({ value: markdown });
  const tree = processor.runSync(processor.parse(file), file) as Root;

  const nodes: { type: string; value?: string }[] = [];
  visit(tree, (node) => {
    const value = (node as { value?: string }).value;
    nodes.push({ type: node.type, value });
  });

  const text = nodes
    .filter((n) => typeof n.value === "string")
    .map((n) => n.value)
    .join("");

  return { nodes, text };
}

function inlineMathValues(markdown: string): string[] {
  return parse(markdown)
    .nodes.filter((n) => n.type === "inlineMath")
    .map((n) => n.value ?? "");
}

describe("remarkSingleDollarMathGuard", () => {
  it("keeps a single-dollar TeX span (issue #15310)", () => {
    const values = inlineMathValues(
      "$C \\rightarrow S \\rightarrow L \\rightarrow P \\rightarrow A$",
    );
    expect(values).toEqual([
      "C \\rightarrow S \\rightarrow L \\rightarrow P \\rightarrow A",
    ]);
  });

  it("keeps a short single-dollar TeX span (issue #17074)", () => {
    expect(inlineMathValues("$\\rightarrow$")).toEqual(["\\rightarrow"]);
  });

  it("keeps two separate single-dollar spans on one line", () => {
    expect(inlineMathValues("Both $a$ and $b$ are vars.")).toEqual(["a", "b"]);
  });

  it("never touches double-dollar math", () => {
    // A single-line "$$...$$" isn't a math fence (that needs its own line),
    // so remark-math still parses it as inlineMath. The guard must recognize
    // the doubled delimiter and leave it alone rather than demoting it.
    expect(inlineMathValues("$$e^{i\\pi} + 1 = 0$$")).toEqual([
      "e^{i\\pi} + 1 = 0",
    ]);
  });

  it("demotes currency mentioned twice on one line", () => {
    const { nodes, text } = parse("It costs $5 and $10 in total.");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("It costs $5 and $10 in total.");
  });

  it("demotes currency repeated across sentences", () => {
    const { nodes, text } = parse("He paid $50. She paid $30. Total was $80.");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("He paid $50. She paid $30. Total was $80.");
  });

  it("demotes comma-grouped currency ranges", () => {
    const { nodes, text } = parse("between $1,500 and $2,000 per month");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("between $1,500 and $2,000 per month");
  });

  it("demotes postfix currency notation", () => {
    const { nodes, text } = parse("He paid 5 $ and 10 $ in total");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("5 $ and 10 $");
  });

  it("demotes parenthesized currency", () => {
    const { nodes, text } = parse("Prices are ($5) and ($10) respectively.");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("($5) and ($10)");
  });

  it("demotes word characters butting the delimiters", () => {
    const { nodes, text } = parse("BEWARE OF $HA$TA BEA$T$");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("$HA$TA BEA$T$");
  });

  it("demotes a span padded with whitespace inside the delimiters", () => {
    const { nodes, text } = parse("padded $ x^2 $ here");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("$ x^2 $");
  });

  it("leaves escaped dollars as literal text", () => {
    const { nodes, text } = parse("Escaped: \\$5 and \\$10 stay text");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("$5 and $10 stay text");
  });

  it("demotes a span flanked by CJK characters", () => {
    const { nodes, text } = parse("预算是$充足$的");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("预算是$充足$的");
  });

  it("demotes a span whose closing delimiter butts an accented letter", () => {
    const { nodes, text } = parse("This is $x$é");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("$x$é");
  });

  // Known limitation: when a currency "$" precedes real math on the SAME line,
  // remark-math pairs the currency dollar with the math opener, so the span is
  // demoted wholesale and the trailing math is not rendered. This must at least
  // degrade safely - visible text is preserved and no currency leaks as math.
  it("degrades safely when currency precedes real math on one line", () => {
    const { nodes, text } = parse("Cost $5. Formula $x$ works.");
    expect(nodes.some((n) => n.type === "inlineMath")).toBe(false);
    expect(text).toContain("Cost $5. Formula $x$ works.");
  });

  it("keeps real math that appears before a currency amount", () => {
    const { nodes } = parse("Formula $x$ then it costs $5 total.");
    const math = nodes
      .filter((n) => n.type === "inlineMath")
      .map((n) => n.value);
    expect(math).toEqual(["x"]);
  });
});
