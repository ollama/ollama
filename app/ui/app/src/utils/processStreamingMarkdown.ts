import { remark } from "remark";
import remarkStringify from "remark-stringify";
import remarkStreamingMarkdown from "./remarkStreamingMarkdown";

/**
 * Process markdown content for streaming display using the remark plugin.
 * This is primarily used for testing the remark plugin with string inputs/outputs.
 */
export function processStreamingMarkdown(content: string): string {
  if (!content) return content;

  const result = remark()
    .use(remarkStreamingMarkdown, { debug: false })
    .use(remarkStringify)
    .processSync(content);

  // remove trailing newline to keep tests cleaner
  let output = result.toString();
  if (output.endsWith("\n")) {
    output = output.slice(0, -1);
  }

  return output;
}
