import { visit } from "unist-util-visit";
import type { Root, Text } from "mdast";

// remark-math's `singleDollarTextMath` treats any `$...$` span as math, which
// turns currency like "$5 and $10" into a formula. Demote a single-dollar
// span back to literal text unless it plausibly is TeX: the content must not
// start or end with whitespace, and the delimiters must not butt against
// word characters outside the span.
export default function remarkSingleDollarMathGuard() {
  return (tree: Root, file: { value?: unknown }) => {
    const source = String(file.value ?? "");
    visit(tree, "inlineMath", (node, index, parent) => {
      if (!parent || index === undefined) return;
      const start = node.position?.start.offset;
      const end = node.position?.end.offset;
      if (start == null || end == null) return;
      const raw = source.slice(start, end);
      // Only guard single-dollar spans; `$$...$$` is always intentional math.
      if (!raw.startsWith("$") || raw.startsWith("$$")) return;
      // Pandoc's tex_math_dollars rules: the opening $ must be immediately
      // followed by a non-space, the closing $ immediately preceded by one,
      // and the delimiters must not butt against word characters outside.
      // The outer-boundary check is Unicode-aware (letters/numbers in any
      // script, not just ASCII), so spans abutting CJK, accented Latin,
      // Cyrillic, Greek, or other non-ASCII word characters are also demoted.
      const looksLikeMath =
        raw.length >= 3 &&
        !/\s/.test(raw[1]) &&
        !/\s/.test(raw[raw.length - 2]) &&
        !/[\p{L}\p{N}_$]/u.test(source[start - 1] ?? "") &&
        !/[\p{L}\p{N}_$]/u.test(source[end] ?? "");
      if (looksLikeMath) return;
      const replacement: Text = { type: "text", value: raw };
      parent.children[index] = replacement;
    });
  };
}
