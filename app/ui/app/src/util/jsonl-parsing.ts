export async function* parseJsonlFromStream<T>(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<T, void, unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        // Process any remaining data in buffer
        buffer += decoder.decode();
        if (buffer.trim()) {
          yield JSON.parse(buffer.trim());
        }
        break;
      }

      // Decode the chunk and add to buffer
      buffer += decoder.decode(value, { stream: true });

      // Process complete lines
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) {
          yield JSON.parse(trimmed);
        }
      }
    }
  } finally {
    // Release unread data if parsing fails or the consumer stops early.
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

/**
 * Helper function to parse JSONL from a Response object
 */
export async function* parseJsonlFromResponse<T>(
  response: Response,
): AsyncGenerator<T, void, unknown> {
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body?.error === "string" && body.error) {
        message = body.error;
      }
    } catch {
      // Non-JSON error responses still need to report the HTTP status.
    }
    throw new Error(message);
  }
  if (!response.body) {
    throw new Error("Response body is null");
  }
  yield* parseJsonlFromStream<T>(response.body);
}
