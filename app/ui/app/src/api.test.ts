import { afterEach, describe, expect, it, vi } from "vitest";
const { listModels } = vi.hoisted(() => ({ listModels: vi.fn() }));
vi.mock("./lib/ollama-client", () => ({
  ollamaClient: { list: listModels },
}));

import {
  fetchConnectUrl,
  getClaudeDesktopAvailableModels,
  getIntegrationStatuses,
} from "./api";

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

describe("getIntegrationStatuses", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns desktop and launcher integration metadata", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            id: "claude-desktop",
            name: "Claude",
            description: "Use Ollama models in Claude Desktop",
            installed: true,
          },
          {
            id: "opencode",
            name: "OpenCode",
            description: "Open-source coding agent",
            command: "ollama launch opencode",
          },
        ]),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetch);

    await expect(getIntegrationStatuses()).resolves.toEqual([
      {
        id: "claude-desktop",
        name: "Claude",
        description: "Use Ollama models in Claude Desktop",
        installed: true,
      },
      {
        id: "opencode",
        name: "OpenCode",
        description: "Open-source coding agent",
        command: "ollama launch opencode",
      },
    ]);
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:3001/api/v1/integrations",
    );
  });
});

describe("getClaudeDesktopAvailableModels", () => {
  afterEach(() => {
    listModels.mockReset();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("merges local and cloud lists while pruning local remote entries", async () => {
    listModels.mockResolvedValue({
      models: [
        { name: "llama3.2:latest", digest: "local" },
        {
          name: "remote-placeholder",
          digest: "remote",
          remote_host: "https://ollama.com",
        },
      ],
    });
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          models: [
            { name: "glm-5.2", digest: "cloud" },
            { name: "llama3.2:latest", digest: "duplicate" },
          ],
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetch);

    const models = await getClaudeDesktopAvailableModels(true);

    expect(models.map((model) => model.model)).toEqual([
      "llama3.2",
      "glm-5.2:cloud",
    ]);
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:3001/api/v1/models/cloud",
    );
  });

  it("does not request cloud models when they are unavailable to the user", async () => {
    listModels.mockResolvedValue({
      models: [{ name: "qwen3:8b", digest: "local" }],
    });
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    const models = await getClaudeDesktopAvailableModels(false);

    expect(models.map((model) => model.model)).toEqual(["qwen3:8b"]);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("keeps local models when the cloud list fails", async () => {
    listModels.mockResolvedValue({
      models: [{ name: "qwen3:8b", digest: "local" }],
    });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    vi.spyOn(console, "warn").mockImplementation(() => {});

    const models = await getClaudeDesktopAvailableModels(true);

    expect(models.map((model) => model.model)).toEqual(["qwen3:8b"]);
  });
});
