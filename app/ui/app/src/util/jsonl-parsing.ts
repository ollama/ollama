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
        if (buffer.trim()) {
          try {
            yield JSON.parse(buffer.trim());
          } catch (error) {
            console.error(`Failed to parse final buffer: ${buffer}`, error);
          }
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
          try {
            yield JSON.parse(trimmed);
          } catch (error) {
            console.error(`Failed to parse line: ${trimmed}`, error);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Helper function to parse JSONL from a Response object
 */
export async function* parseJsonlFromResponse<T>(
  response: Response,
): AsyncGenerator<T, void, unknown> {
  if (!response.body) {
    throw new Error("Response body is null");
  }
  yield* parseJsonlFromStream<T>(response.body);
}
