import { parents, type Proxy } from "unist-util-parents";
import type { Plugin } from "unified";
import type {
  Emphasis,
  Node,
  Parent,
  Root,
  RootContent,
  Text,
  Strong,
  PhrasingContent,
  Paragraph,
} from "mdast";
import { u } from "unist-builder";

declare module "unist" {
  interface Node {
    /** Added by `unist-util-parents` (or your own walk). */
    parent?: Proxy & Parent;
  }
}

// interface SimpleTextRule {
//   pattern: RegExp;
//   transform: (matches: RegExpExecArray[], lastNode: Proxy) => void;
// }

// const simpleTextRules: SimpleTextRule[] = [
//   // TODO(drifkin): generalize this for `__`/`_`/`~~`/`~` etc.
//   {
//     pattern: /(\*\*)(?=\S|$)/g,
//     transform: (matchesIterator, lastNode) => {
//       const textNode = lastNode.node as Text;

//       const matches = [...matchesIterator];
//       const lastMatch = matches[matches.length - 1];
//       const origValue = textNode.value;
//       const start = lastMatch.index;
//       const sep = lastMatch[1];

//       const before = origValue.slice(0, start);
//       const after = origValue.slice(start + sep.length);

//       if (lastNode.parent) {
//         const index = (lastNode.parent.node as Parent).children.indexOf(
//           lastNode.node as RootContent,
//         );
//         const shouldRemove = before.length === 0;
//         if (!shouldRemove) {
//           textNode.value = before;
//         }

//         const newNode = u("strong", {
//           children: [u("text", { value: after })],
//         });
//         (lastNode.parent.node as Parent).children.splice(
//           index + (shouldRemove ? 0 : 1),
//           shouldRemove ? 1 : 0,
//           newNode,
//         );
//       }
//     },
//   },
// ];

interface Options {
  debug?: boolean;
  onLastNode?: (info: LastNodeInfo) => void;
}

export interface LastNodeInfo {
  path: string[];
  type: string;
  value?: string;
  lastChars?: string;
  fullNode: Node;
}

/**
 * Removes `child` from `parent` in-place.
 * @returns `true` if the child was found and removed; `false` otherwise.
 */
export function removeChildFromParent(
  child: RootContent,
  parent: Node,
): boolean {
  if (!isParent(parent)) return false; // parent isn’t a Parent → nothing to do

  const idx = parent.children.indexOf(child);
  if (idx < 0) return false; // not a child → nothing to remove

  parent.children.splice(idx, 1);
  return true; // removal successful
}

/** Narrow a generic `Node` to a `Parent` (i.e. one that really has children). */
function isParent(node: Node): node is Parent {
  // A `Parent` always has a `children` array; make sure it's an array first.
  return Array.isArray((node as Partial<Parent>).children);
}

/**
 * Follow “last-child” pointers until you reach a leaf.
 * Returns the right-most, deepest node in source order.
 */
export function findRightmostDeepestNode(root: Node): Node {
  let current: Node = root;

  // While the current node *is* a Parent and has at least one child…
  while (isParent(current) && current.children.length > 0) {
    const lastIndex = current.children.length - 1;
    current = current.children[lastIndex];
  }

  return current; // Leaf: no further children
}

const remarkStreamingMarkdown: Plugin<[Options?], Root> = () => {
  return (tree) => {
    const treeWithParents = parents(tree);
    const lastNode = findRightmostDeepestNode(treeWithParents) as Proxy;

    const parentNode = lastNode.parent;
    const grandparentNode = parentNode?.parent;

    let ruleMatched = false;

    // handling `* *` -> ``
    //
    // if the last node is part of a <list item (otherwise empty)> ->
    // <list (otherwise empty)> -> <list item (last node, empty)>, then we need to
    // remove everything up to and including the first list item. This happens
    // when we have `* *`, which can become a bolded list item OR a horizontal
    // line
    if (
      lastNode.type === "listItem" &&
      parentNode &&
      grandparentNode &&
      parentNode.type === "list" &&
      grandparentNode.type === "listItem" &&
      parentNode.children.length === 1 &&
      grandparentNode.children.length === 1
    ) {
      ruleMatched = true;
      if (grandparentNode.parent) {
        removeChildFromParent(
          grandparentNode.node as RootContent,
          grandparentNode.parent.node,
        );
      }
      // Handle `*` -> ``:
      //
      // if the last node is just an empty list item, we need to remove it
      // because it could become something else (e.g., a horizontal line)
    } else if (
      lastNode.type === "listItem" &&
      parentNode &&
      parentNode.type === "list"
    ) {
      ruleMatched = true;
      removeChildFromParent(lastNode.node as RootContent, parentNode.node);
    } else if (lastNode.type === "thematicBreak") {
      ruleMatched = true;
      const parent = lastNode.parent;
      if (parent) {
        removeChildFromParent(lastNode.node as RootContent, parent.node);
      }
    } else if (lastNode.type === "text") {
      const textNode = lastNode.node as Text;
      if (textNode.value.endsWith("**")) {
        ruleMatched = true;
        textNode.value = textNode.value.slice(0, -2);
        // if there's a newline then a number, this is very very likely a
        // numbered list item. Let's just hide it until the period comes (or
        // other text disambiguates it)
      } else {
        const match = textNode.value.match(/^([0-9]+)$/m);
        if (match) {
          const number = match[1];
          textNode.value = textNode.value.slice(0, -number.length - 1);
          ruleMatched = true;
          // if the text node is now empty, then we might want to remove other
          // elements, like a now-empty containing paragraph, or a break that
          // might disappear once more tokens come in
          if (textNode.value.length === 0) {
            if (
              lastNode.parent?.type === "paragraph" &&
              lastNode.parent.children.length === 1
            ) {
              // remove the whole paragraph if it's now empty (otherwise it'll
              // cause an extra newline that might not last)
              removeChildFromParent(
                lastNode.parent.node as Paragraph,
                lastNode.parent.parent?.node as Node,
              );
            } else {
              const prev = prevSibling(lastNode);
              if (prev?.type === "break") {
                removeChildFromParent(
                  prev.node as RootContent,
                  lastNode.parent?.node as Node,
                );
                removeChildFromParent(
                  lastNode.node as RootContent,
                  lastNode.parent?.node as Node,
                );
              }
            }
          }
        }
      }
    }

    if (ruleMatched) {
      return tree;
    }

    // we need to
    // a case like
    //     - *def `abc` [abc **def**](abc)*
    // is pretty tricky, because if we land just after def, then we actually
    // have two separate tags to process at two different parents. Maybe we
    // need to keep iterating up until we find a paragraph, but process each
    // parent on the way up. Hmm, well actually after `def` we won't even be a proper link yet
    // TODO(drifkin): it's really if the last node's parent is a paragraph, for which the following is a sub-cas where the lastNode is a text node.
    // And instead of just processing simple text rules, they need to operate on the whole paragraph
    // like `**[abc](def)` needs to become `**[abc](def)**`

    // if we're just text at the end, then we should remove some ambiguous characters

    if (lastNode.parent) {
      const didChange = processParent(lastNode.parent as Parent & Proxy);
      if (didChange) {
        // TODO(drifkin): need to fix up the tree, but not sure lastNode will still exist? Check all the transforms to see if it's safe to find the last node again
        //
        // need to regen the tree w/ parents since reparenting could've happened
        // treeWithParents = parents(tree);
      }
    }

    const grandparent = lastNode.parent?.parent;
    // TODO(drifkin): let's go arbitrarily high up the tree, but limiting it
    // to 2 levels for now until I think more about the stop condition
    if (grandparent) {
      processParent(grandparent as Parent & Proxy);
    }

    // console.log("ruleMatched", ruleMatched);

    // } else if (lastNode.parent?.type === "paragraph") {
    //   console.log("!!! paragraph");
    //   console.log("lastNode.parent", lastNode.parent);

    //   // Handle `**abc*` -> `**abc**`:
    //   // We detect this when the last child is an emphasis node, and it's preceded by a text node that ends with `*`
    //   const paragraph = lastNode.parent as Proxy & Paragraph;
    //   if (paragraph.children.length >= 2) {
    //     const lastChild = paragraph.children[paragraph.children.length - 1];
    //     if (lastChild.type === "emphasis") {
    //       const sibling = paragraph.children[paragraph.children.length - 2];
    //       if (sibling.type === "text") {
    //         const siblingText = sibling as Text & Proxy;
    //         if (siblingText.value.endsWith("*")) {
    //           ruleMatched = true;
    //           const textNode = (lastNode as Proxy).node as Text;
    //           textNode.value = textNode.value.slice(0, -1);
    //           paragraph.node.type = "strong";
    //         }
    //       }
    //     }
    //   }
    // } else if (lastNode.type === "text") {
    //   // Handle `**abc*` -> `**abc**`:
    //   //
    //   // this gets parsed as a text node ending in `*` followed by an emphasis
    //   // node. So if we're in text, we need to check if our parent is emphasis,
    //   // and then get our parent's sibling before it and check if it ends with
    //   // `*`
    //   const parent = lastNode.parent;
    //   if (parent && parent.type === "emphasis") {
    //     const grandparent = parent.parent;
    //     if (grandparent) {
    //       const index = (grandparent.node as Parent).children.indexOf(
    //         parent.node as RootContent,
    //       );
    //       if (index > 0) {
    //         const prevNode = grandparent.children[index - 1];
    //         if (
    //           prevNode.type === "text" &&
    //           (prevNode as Text).value.endsWith("*")
    //         ) {
    //           ruleMatched = true;
    //           const textNode = (prevNode as Proxy).node as Text;
    //           textNode.value = textNode.value.slice(0, -1);
    //           parent.node.type = "strong";
    //         }
    //       }
    //     }
    // }

    //   if (!ruleMatched) {
    //     // if the last node is just text, then we process it in order to fix up certain unclosed items
    //     // e.g., `**abc` -> `**abc**`
    //     const textNode = lastNode.node as Text;
    //     for (const rule of simpleTextRules) {
    //       const matchesIterator = textNode.value.matchAll(rule.pattern);
    //       const matches = [...matchesIterator];
    //       if (matches.length > 0) {
    //         rule.transform(matches, lastNode);
    //         ruleMatched = true;
    //         break;
    //       }
    //     }
    //   }
    // } else if (!ruleMatched) {
    //   // console.log("no rule matched", lastNode);
    // }

    return tree;
  };
};

function processParent(parent: Parent & Proxy): boolean {
  if (parent.type === "emphasis") {
    // Handle `**abc*` -> `**abc**`:
    // We detect this when we end with an emphasis node, and it's preceded by
    // a text node that ends with `*`
    // TODO(drifkin): the last node can be more deeply nested (e.g., a code
    // literal in a link), so we probably need to walk up the tree until we
    // find an emphasis node or a block? For now we'll just go up one layer to
    // catch the most common cases
    const emphasisNode = parent as Emphasis & Proxy;
    const grandparent = emphasisNode.parent;
    if (grandparent) {
      const indexOfEmphasisNode = (grandparent.node as Parent).children.indexOf(
        emphasisNode.node as RootContent,
      );
      if (indexOfEmphasisNode >= 0) {
        const nodeBefore = grandparent.children[indexOfEmphasisNode - 1] as
          | (Node & Proxy)
          | undefined;
        if (nodeBefore?.type === "text") {
          const textNode = nodeBefore.node as Text;
          if (textNode.value.endsWith("*")) {
            const strBefore = textNode.value.slice(0, -1);
            textNode.value = strBefore;
            const strongNode = u("strong", {
              children: emphasisNode.children,
            });
            (grandparent.node as Parent).children.splice(
              indexOfEmphasisNode,
              1,
              strongNode,
            );
            return true;
          }
        }
      }
    }
  }

  // Let's check if we have any bold items to close
  for (let i = parent.children.length - 1; i >= 0; i--) {
    const child = parent.children[i];
    if (child.type === "text") {
      const textNode = child as Text & Proxy;
      const sep = "**";
      const index = textNode.value.lastIndexOf(sep);
      if (index >= 0) {
        let isValidOpening = false;
        if (index + sep.length < textNode.value.length) {
          const charAfter = textNode.value[index + sep.length];
          if (!isWhitespace(charAfter)) {
            isValidOpening = true;
          }
        } else {
          if (i < parent.children.length - 1) {
            // TODO(drifkin): I'm not sure that this check is strict enough.
            // We're trying to detect cases like `**[abc]()` where the char
            // after the opening ** is indeed a non-whitespace character. We're
            // using the heuristic that there's another item after the current
            // one, but I'm not sure if that is good enough. In a well
            // constructed tree, there aren't two text nodes in a row, so this
            // _seems_ good, but I should think through it more
            isValidOpening = true;
          }
        }

        if (isValidOpening) {
          // TODO(drifkin): close the bold
          const strBefore = textNode.value.slice(0, index);
          const strAfter = textNode.value.slice(index + sep.length);
          (textNode.node as Text).value = strBefore;
          // TODO(drifkin): the node above could be empty in which case we probably want to delete it
          const children: PhrasingContent[] = [
            ...(strAfter.length > 0 ? [u("text", { value: strAfter })] : []),
          ];
          const strongNode: Strong = u("strong", {
            children,
          });
          const nodesAfter = (parent.node as Parent).children.splice(
            i + 1,
            parent.children.length - i - 1,
            strongNode,
          );
          // TODO(drifkin): this cast seems iffy, should see if we can cast the
          // parent instead, which would also help us check some of our
          // assumptions
          strongNode.children.push(...(nodesAfter as PhrasingContent[]));
          return true;
        }
      }
    }
  }

  return false;
}

function prevSibling(node: Node & Proxy): (Node & Proxy) | null {
  const parent = node.parent;
  if (parent) {
    const index = parent.children.indexOf(node);
    return parent.children[index - 1] as Node & Proxy;
  }
  return null;
}

function isWhitespace(str: string) {
  return str.trim() === "";
}

// function debugPrintTreeNoPos(tree: Node) {
//   console.log(
//     JSON.stringify(
//       tree,
//       (key, value) => {
//         if (key === "position") {
//           return undefined;
//         }
//         return value;
//       },
//       2,
//     ),
//   );
// }

export default remarkStreamingMarkdown;
