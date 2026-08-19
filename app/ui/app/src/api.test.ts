import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchConnectUrl } from "./api";

describe("fetchConnectUrl", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("requests a desktop handoff after account creation", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            signin_url:
              "https://ollama.com/connect?name=MacBook&key=public-key",
          }),
          { status: 401 },
        ),
      ),
    );

    await expect(fetchConnectUrl()).resolves.toBe(
      "https://ollama.com/connect?name=MacBook&key=public-key&launch=true",
    );
  });
});
