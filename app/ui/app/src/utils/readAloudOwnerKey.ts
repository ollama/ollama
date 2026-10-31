import { Message } from "@/gotypes";

/** Revision must change when assistant content changes; do not use generated `Time` (empty class). */
function messageContentRevision(content: string): string {
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    hash = (hash * 31 + content.charCodeAt(i)) | 0;
  }
  return `${content.length}:${hash}`;
}

export function readAloudOwnerKey(
  chatId: string,
  messageIndex: number,
  message: Message,
) {
  const content = message.content?.trim() ?? "";
  return `${chatId}:${messageIndex}:${messageContentRevision(content)}`;
}
