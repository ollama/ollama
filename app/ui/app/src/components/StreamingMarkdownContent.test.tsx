import { expect, test, suite } from "vitest";
import { processStreamingMarkdown } from "@/utils/processStreamingMarkdown";

suite("common llm outputs that cause issues", () => {
  test("prefix of bolded list item shouldn't make a horizontal line", () => {
    // we're going to go in order of incrementally adding characters. This
    // happens really commonly with LLMs that like to make lists like so:
    //
    // * **point 1**: explanatory text
    // * **point 2**: more explanatory text
    //
    // Partial rendering of `*` (A), followed by `* *` (B), followed by `* **`
    // (C) is a total mess.  (A) renders as a single bullet point in an
    // otherwise empty list, (B) renders as two nested lists (and therefore
    // two bullet points, styled differently by default in html), and (C)
    // renders as a horizontal line because in markdown apparently `***` or `*
    // * *` horizontal rules don't have as strict whitespace rules as I
    // expected them to

    // these are alone (i.e., they would be the first list item)
    expect(processStreamingMarkdown("*")).toBe("");
    expect(processStreamingMarkdown("* *")).toBe("");
    expect(processStreamingMarkdown("* **")).toBe("");
    // expect(processStreamingMarkdown("* **b")).toBe("* **b**");

    // with a list item before them
    expect(
      processStreamingMarkdown(
        // prettier-ignore
        [
          "* abc", 
          "*"
        ].join("\n"),
      ),
    ).toBe("* abc");

    expect(
      processStreamingMarkdown(
        // prettier-ignore
        [
          "* abc", 
          "* *"
        ].join("\n"),
      ),
    ).toBe("* abc");

    expect(
      processStreamingMarkdown(
        // prettier-ignore
        [
          "* abc", 
          "* **"
        ].join("\n"),
      ),
    ).toBe("* abc");
  });

  test("bolded list items with text should be rendered properly", () => {
    expect(processStreamingMarkdown("* **abc**")).toBe("* **abc**");
  });

  test("partially bolded list items should be autoclosed", () => {
    expect(processStreamingMarkdown("* **abc")).toBe("* **abc**");
  });

  suite(
    "partially bolded list items should be autoclosed, even if the last node isn't a text node",
    () => {
      test("inline code", () => {
        expect(
          processStreamingMarkdown("* **Asynchronous Function `async`*"),
        ).toBe("* **Asynchronous Function `async`**");
      });
    },
  );
});

suite("autoclosing bold", () => {
  suite("endings with no asterisks", () => {
    test("should autoclose bold", () => {
      expect(processStreamingMarkdown("**abc")).toBe("**abc**");
      expect(processStreamingMarkdown("abc **abc")).toBe("abc **abc**");
    });

    suite("should autoclose, even if the last node isn't a text node", () => {
      test("inline code", () => {
        expect(
          processStreamingMarkdown("* **Asynchronous Function `async`"),
        ).toBe("* **Asynchronous Function `async`**");
      });

      test("opening ** is at the end of the text", () => {
        expect(processStreamingMarkdown("abc **`def` jhk [lmn](opq)")).toBe(
          "abc **`def` jhk [lmn](opq)**",
        );
      });

      test("if there's a space after the **, it should NOT be autoclosed", () => {
        expect(processStreamingMarkdown("abc ** `def` jhk [lmn](opq)")).toBe(
          "abc \\*\\* `def` jhk [lmn](opq)",
        );
      });
    });

    test("should autoclose bold, even if the last node isn't a text node", () => {
      expect(
        processStreamingMarkdown("* **Asynchronous Function ( `async`"),
      ).toBe("* **Asynchronous Function ( `async`**");
    });

    test("whitespace fakeouts should not be modified", () => {
      expect(processStreamingMarkdown("** abc")).toBe("\\*\\* abc");
    });

    // TODO(drifkin): arguably this should just be removed entirely, but empty
    // isn't so bad
    test("should handle empty bolded items", () => {
      expect(processStreamingMarkdown("**")).toBe("");
    });
  });

  suite("partially closed bolded items", () => {
    test("simple partial", () => {
      expect(processStreamingMarkdown("**abc*")).toBe("**abc**");
    });

    test("partial with non-text node at end", () => {
      expect(processStreamingMarkdown("**abc`def`*")).toBe("**abc`def`**");
    });

    test("partial with multiply nested ending nodes", () => {
      expect(processStreamingMarkdown("**abc[abc](`def`)*")).toBe(
        "**abc[abc](`def`)**",
      );
    });

    test("normal emphasis should not be affected", () => {
      expect(processStreamingMarkdown("*abc*")).toBe("*abc*");
    });

    test("normal emphasis with nested code should not be affected", () => {
      expect(processStreamingMarkdown("*`abc`*")).toBe("*`abc`*");
    });
  });

  test.skip("shouldn't autoclose immediately if there's a space before the closing *", () => {
    expect(processStreamingMarkdown("**abc *")).toBe("**abc**");
  });

  // skipping for now because this requires partial link completion as well
  suite.skip("nested blocks that each need autoclosing", () => {
    test("emph nested in link nested in strong nested in list item", () => {
      expect(processStreamingMarkdown("* **[abc **def")).toBe(
        "* **[abc **def**]()**",
      );
    });

    test("* **[ab *`def`", () => {
      expect(processStreamingMarkdown("* **[ab *`def`")).toBe(
        "* **[ab *`def`*]()**",
      );
    });
  });
});

suite("numbered list items", () => {
  test("should remove trailing numbers", () => {
    expect(processStreamingMarkdown("1. First\n2")).toBe("1. First");
  });

  test("should remove trailing numbers with breaks before", () => {
    expect(processStreamingMarkdown("1. First    \n2")).toBe("1. First");
  });

  test("should remove trailing numbers that form a new paragraph", () => {
    expect(processStreamingMarkdown("1. First\n\n2")).toBe("1. First");
  });

  test("but should leave list items separated by two newlines", () => {
    expect(processStreamingMarkdown("1. First\n\n2. S")).toBe(
      "1. First\n\n2. S",
    );
  });
});

// TODO(drifkin):slop tests ahead, some are decent, but need to manually go
// through them as I implement
/*
describe("StreamingMarkdownContent - processStreamingMarkdown", () => {
  describe("Ambiguous endings removal", () => {
    it("should remove list markers at the end", () => {
      expect(processStreamingMarkdown("Some text\n* ")).toBe("Some text");
      expect(processStreamingMarkdown("Some text\n*")).toBe("Some text");
      expect(processStreamingMarkdown("* Item 1\n- ")).toBe("* Item 1");
      expect(processStreamingMarkdown("* Item 1\n-")).toBe("* Item 1");
      expect(processStreamingMarkdown("Text\n+ ")).toBe("Text");
      expect(processStreamingMarkdown("Text\n+")).toBe("Text");
      expect(processStreamingMarkdown("1. First\n2. ")).toBe("1. First");
    });

    it("should remove heading markers at the end", () => {
      expect(processStreamingMarkdown("Some text\n# ")).toBe("Some text");
      expect(processStreamingMarkdown("Some text\n#")).toBe("Some text\n#"); // # without space is not removed
      expect(processStreamingMarkdown("# Title\n## ")).toBe("# Title");
      expect(processStreamingMarkdown("# Title\n##")).toBe("# Title\n##"); // ## without space is not removed
    });

    it("should remove ambiguous bold markers at the end", () => {
      expect(processStreamingMarkdown("Text **")).toBe("Text ");
      expect(processStreamingMarkdown("Some text\n**")).toBe("Some text");
    });

    it("should remove code block markers at the end", () => {
      expect(processStreamingMarkdown("Text\n```")).toBe("Text");
      expect(processStreamingMarkdown("```")).toBe("");
    });

    it("should remove single backtick at the end", () => {
      expect(processStreamingMarkdown("Text `")).toBe("Text ");
      expect(processStreamingMarkdown("`")).toBe("");
    });

    it("should remove single asterisk at the end", () => {
      expect(processStreamingMarkdown("Text *")).toBe("Text ");
      expect(processStreamingMarkdown("*")).toBe("");
    });

    it("should handle empty content", () => {
      expect(processStreamingMarkdown("")).toBe("");
    });

    it("should handle single line removals correctly", () => {
      expect(processStreamingMarkdown("* ")).toBe("");
      expect(processStreamingMarkdown("# ")).toBe("");
      expect(processStreamingMarkdown("**")).toBe("");
      expect(processStreamingMarkdown("`")).toBe("");
    });

    it("shouldn't have this regexp capture group bug", () => {
      expect(
        processStreamingMarkdown("Here's a shopping list:\n*"),
      ).not.toContain("0*");
      expect(processStreamingMarkdown("Here's a shopping list:\n*")).toBe(
        "Here's a shopping list:",
      );
    });
  });

  describe("List markers", () => {
    it("should preserve complete list items", () => {
      expect(processStreamingMarkdown("* Complete item")).toBe(
        "* Complete item",
      );
      expect(processStreamingMarkdown("- Another item")).toBe("- Another item");
      expect(processStreamingMarkdown("+ Plus item")).toBe("+ Plus item");
      expect(processStreamingMarkdown("1. Numbered item")).toBe(
        "1. Numbered item",
      );
    });

    it("should handle indented list markers", () => {
      expect(processStreamingMarkdown("  * ")).toBe("  ");
      expect(processStreamingMarkdown("    - ")).toBe("    ");
      expect(processStreamingMarkdown("\t+ ")).toBe("\t");
    });
  });

  describe("Heading markers", () => {
    it("should preserve complete headings", () => {
      expect(processStreamingMarkdown("# Complete Heading")).toBe(
        "# Complete Heading",
      );
      expect(processStreamingMarkdown("## Subheading")).toBe("## Subheading");
      expect(processStreamingMarkdown("### H3 Title")).toBe("### H3 Title");
    });

    it("should not affect # in other contexts", () => {
      expect(processStreamingMarkdown("C# programming")).toBe("C# programming");
      expect(processStreamingMarkdown("Issue #123")).toBe("Issue #123");
    });
  });

  describe("Bold text", () => {
    it("should close incomplete bold text", () => {
      expect(processStreamingMarkdown("This is **bold text")).toBe(
        "This is **bold text**",
      );
      expect(processStreamingMarkdown("Start **bold and more")).toBe(
        "Start **bold and more**",
      );
      expect(processStreamingMarkdown("**just bold")).toBe("**just bold**");
    });

    it("should not affect complete bold text", () => {
      expect(processStreamingMarkdown("**complete bold**")).toBe(
        "**complete bold**",
      );
      expect(processStreamingMarkdown("Text **bold** more")).toBe(
        "Text **bold** more",
      );
    });

    it("should handle nested bold correctly", () => {
      expect(processStreamingMarkdown("**bold** and **another")).toBe(
        "**bold** and **another**",
      );
    });
  });

  describe("Italic text", () => {
    it("should close incomplete italic text", () => {
      expect(processStreamingMarkdown("This is *italic text")).toBe(
        "This is *italic text*",
      );
      expect(processStreamingMarkdown("Start *italic and more")).toBe(
        "Start *italic and more*",
      );
    });

    it("should differentiate between list markers and italic", () => {
      expect(processStreamingMarkdown("* Item\n* ")).toBe("* Item");
      expect(processStreamingMarkdown("Some *italic text")).toBe(
        "Some *italic text*",
      );
      expect(processStreamingMarkdown("*just italic")).toBe("*just italic*");
    });

    it("should not affect complete italic text", () => {
      expect(processStreamingMarkdown("*complete italic*")).toBe(
        "*complete italic*",
      );
      expect(processStreamingMarkdown("Text *italic* more")).toBe(
        "Text *italic* more",
      );
    });
  });

  describe("Code blocks", () => {
    it("should close incomplete code blocks", () => {
      expect(processStreamingMarkdown("```javascript\nconst x = 42;")).toBe(
        "```javascript\nconst x = 42;\n```",
      );
      expect(processStreamingMarkdown("```\ncode here")).toBe(
        "```\ncode here\n```",
      );
    });

    it("should not affect complete code blocks", () => {
      expect(processStreamingMarkdown("```\ncode\n```")).toBe("```\ncode\n```");
      expect(processStreamingMarkdown("```js\nconst x = 1;\n```")).toBe(
        "```js\nconst x = 1;\n```",
      );
    });

    it("should handle nested code blocks correctly", () => {
      expect(processStreamingMarkdown("```\ncode\n```\n```python")).toBe(
        "```\ncode\n```\n```python\n```",
      );
    });

    it("should not process markdown inside code blocks", () => {
      expect(processStreamingMarkdown("```\n* not a list\n**not bold**")).toBe(
        "```\n* not a list\n**not bold**\n```",
      );
    });
  });

  describe("Inline code", () => {
    it("should close incomplete inline code", () => {
      expect(processStreamingMarkdown("This is `inline code")).toBe(
        "This is `inline code`",
      );
      expect(processStreamingMarkdown("Use `console.log")).toBe(
        "Use `console.log`",
      );
    });

    it("should not affect complete inline code", () => {
      expect(processStreamingMarkdown("`complete code`")).toBe(
        "`complete code`",
      );
      expect(processStreamingMarkdown("Use `code` here")).toBe(
        "Use `code` here",
      );
    });

    it("should handle multiple inline codes correctly", () => {
      expect(processStreamingMarkdown("`code` and `more")).toBe(
        "`code` and `more`",
      );
    });

    it("should not confuse inline code with code blocks", () => {
      expect(processStreamingMarkdown("```\nblock\n```\n`inline")).toBe(
        "```\nblock\n```\n`inline`",
      );
    });
  });

  describe("Complex streaming scenarios", () => {
    it("should handle progressive streaming of a heading", () => {
      const steps = [
        { input: "#", expected: "#" }, // # alone is not removed (needs space)
        { input: "# ", expected: "" },
        { input: "# H", expected: "# H" },
        { input: "# Hello", expected: "# Hello" },
      ];
      steps.forEach(({ input, expected }) => {
        expect(processStreamingMarkdown(input)).toBe(expected);
      });
    });

    it("should handle progressive streaming of bold text", () => {
      const steps = [
        { input: "*", expected: "" },
        { input: "**", expected: "" },
        { input: "**b", expected: "**b**" },
        { input: "**bold", expected: "**bold**" },
        { input: "**bold**", expected: "**bold**" },
      ];
      steps.forEach(({ input, expected }) => {
        expect(processStreamingMarkdown(input)).toBe(expected);
      });
    });

    it("should handle multiline content with various patterns", () => {
      const multiline = `# Title
      
This is a paragraph with **bold text** and *italic text*.

* Item 1
* Item 2
* `;

      const expected = `# Title
      
This is a paragraph with **bold text** and *italic text*.

* Item 1
* Item 2`;

      expect(processStreamingMarkdown(multiline)).toBe(expected);
    });

    it("should only fix the last line", () => {
      expect(processStreamingMarkdown("# Complete\n# Another\n# ")).toBe(
        "# Complete\n# Another",
      );
      expect(processStreamingMarkdown("* Item 1\n* Item 2\n* ")).toBe(
        "* Item 1\n* Item 2",
      );
    });

    it("should handle mixed content correctly", () => {
      const input = `# Header

This has **bold** text and *italic* text.

\`\`\`js
const x = 42;
\`\`\`

Now some \`inline code\` and **unclosed bold`;

      const expected = `# Header

This has **bold** text and *italic* text.

\`\`\`js
const x = 42;
\`\`\`

Now some \`inline code\` and **unclosed bold**`;

      expect(processStreamingMarkdown(input)).toBe(expected);
    });
  });

  describe("Edge cases with escaping", () => {
    it("should handle escaped asterisks (future enhancement)", () => {
      // Note: Current implementation doesn't handle escaping
      // This is a known limitation - escaped characters still trigger closing
      expect(processStreamingMarkdown("Text \\*not italic")).toBe(
        "Text \\*not italic*",
      );
    });

    it("should handle escaped backticks (future enhancement)", () => {
      // Note: Current implementation doesn't handle escaping
      // This is a known limitation - escaped characters still trigger closing
      expect(processStreamingMarkdown("Text \\`not code")).toBe(
        "Text \\`not code`",
      );
    });
  });

  describe("Code block edge cases", () => {
    it("should handle triple backticks in the middle of lines", () => {
      expect(processStreamingMarkdown("Text ``` in middle")).toBe(
        "Text ``` in middle\n```",
      );
      expect(processStreamingMarkdown("```\nText ``` in code\nmore")).toBe(
        "```\nText ``` in code\nmore\n```",
      );
    });

    it("should properly close code blocks with language specifiers", () => {
      expect(processStreamingMarkdown("```typescript")).toBe(
        "```typescript\n```",
      );
      expect(processStreamingMarkdown("```typescript\nconst x = 1")).toBe(
        "```typescript\nconst x = 1\n```",
      );
    });

    it("should remove a completely empty partial code block", () => {
      expect(processStreamingMarkdown("```\n")).toBe("");
    });
  });
});

*/
