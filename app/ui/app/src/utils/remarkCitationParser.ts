import { visit } from "unist-util-visit";
import type { Root, RootContent } from "mdast";

export default function remarkMyDelimiter() {
  return (tree: Root) => {
    // First pass: convert citations to nodes
    visit(tree, "text", (node, index, parent) => {
      // Example: 【1†L25-L30】
      // should be parsed into:
      // cursor: 1, start: 25, end: 30
      const regex = /【(\d+)†L(\d+)-L(\d+)】/g;
      let match;
      let last = 0;
      const pieces: RootContent[] = [];

      while ((match = regex.exec(node.value))) {
        // text before the delimiter
        if (match.index > last) {
          pieces.push({
            type: "text",
            value: node.value.slice(last, match.index),
          });
        }
        // the delimited content → new custom node
        pieces.push({
          // @ts-expect-error: custom type
          type: "custom-citation" as const,
          data: {
            // tell rehype/rehype-react to render <Citation>
            hName: "ol-citation",
            hProperties: {
              cursor: match[1],
              start: match[2],
              end: match[3],
            },
          },
        });
        last = match.index + match[0].length;
      }

      // After handling range-style citations, handle generic ones in the remaining text
      // Generic style citations like [1†...] should also be parsed as citations
      const remaining = node.value.slice(last);
      const generic = /【(\d+)†[^】]*】/g;
      let gLast = 0;
      while ((match = generic.exec(remaining))) {
        if (match.index > gLast) {
          pieces.push({
            type: "text",
            value: remaining.slice(gLast, match.index),
          });
        }
        pieces.push({
          // @ts-expect-error: custom type
          type: "custom-citation" as const,
          data: {
            hName: "ol-citation",
            hProperties: {
              cursor: match[1],
            },
          },
        });
        gLast = match.index + match[0].length;
      }

      // trailing text after generic
      if (gLast < remaining.length) {
        pieces.push({ type: "text", value: remaining.slice(gLast) });
      }

      if (pieces.length) {
        parent?.children?.splice(index ?? 0, 1, ...pieces);
        return (index ?? 0) + pieces.length;
      }
    });

    // Second pass: remove adjacent duplicate citations
    visit(tree, (node, index, parent) => {
      if (
        parent &&
        parent.children &&
        index !== null &&
        index !== undefined &&
        index > 0
      ) {
        const currentNode = node as any;
        const prevNode = parent.children[index - 1] as any;

        // Check if both nodes are citations with the same cursor
        if (
          currentNode.type === "custom-citation" &&
          prevNode.type === "custom-citation" &&
          currentNode.data?.hProperties?.cursor ===
            prevNode.data?.hProperties?.cursor
        ) {
          // Remove the current duplicate citation
          parent.children.splice(index, 1);
          return index;
        }
      }
    });
  };
}
