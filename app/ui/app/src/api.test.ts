import { afterEach, describe, expect, it, vi } from "vitest";
const { listModels } = vi.hoisted(() => ({ listModels: vi.fn() }));
vi.mock("./lib/ollama-client", () => ({
  ollamaClient: { list: listModels },
}));

import {
  fetchConnectUrl,
  getFeatureFlag,
  getClaudeDesktopAvailableModels,
  getIntegrationStatuses,
} from "./api";

describe("getFeatureFlag", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns boolean and string values from the local app service", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ value: true })))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ value: "compact" })),
      );
    vi.stubGlobal("fetch", fetch);

    await expect(getFeatureFlag("new-chat", false)).resolves.toBe(true);
    await expect(getFeatureFlag("chat-layout", "standard")).resolves.toBe(
      "compact",
    );
    expect(fetch).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:3001/api/v1/feature-flags/new-chat?type=boolean&default=false",
    );
    expect(fetch).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:3001/api/v1/feature-flags/chat-layout?type=string&default=standard",
    );
  });

  it("returns the compiled fallback for unavailable or invalid responses", async () => {
    const fetch = vi
      .fn()
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(new Response(JSON.stringify({ value: "wrong" })));
    vi.stubGlobal("fetch", fetch);

    await expect(getFeatureFlag("new-chat", false)).resolves.toBe(false);
    await expect(getFeatureFlag("new-chat", true)).resolves.toBe(true);
  });
});

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

  it("returns installed local models while pruning remote entries", async () => {
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
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    const models = await getClaudeDesktopAvailableModels();

    expect(models.map((model) => model.model)).toEqual(["llama3.2"]);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("does not request cloud models when they are unavailable to the user", async () => {
    listModels.mockResolvedValue({
      models: [
        { name: "qwen3:8b", digest: "local" },
        { name: "deepseek-v4-flash:cloud", digest: "cached-cloud" },
        { name: "gemma4:31b-cloud", digest: "legacy-cached-cloud" },
      ],
    });
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    const models = await getClaudeDesktopAvailableModels();

    expect(models.map((model) => model.model)).toEqual(["qwen3:8b"]);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("loads the account cloud list in parallel when Cloud is available", async () => {
    listModels.mockResolvedValue({
      models: [{ name: "qwen3:8b", digest: "local" }],
    });
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          models: [
            { name: "glm-5.2", digest: "cloud" },
            { name: "gemma4:31b-cloud", digest: "legacy-cloud" },
            { name: "qwen3:8b", digest: "cloud-duplicate" },
          ],
        }),
      ),
    );
    vi.stubGlobal("fetch", fetch);

    const models = await getClaudeDesktopAvailableModels(true);

    expect(models.map((model) => model.model)).toEqual([
      "qwen3:8b",
      "glm-5.2:cloud",
      "gemma4:31b-cloud",
    ]);
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:3001/api/v1/models/cloud",
    );
  });

  it("keeps local models when the account cloud list fails", async () => {
    listModels.mockResolvedValue({
      models: [{ name: "qwen3:8b", digest: "local" }],
    });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    const models = await getClaudeDesktopAvailableModels(true);

    expect(models.map((model) => model.model)).toEqual(["qwen3:8b"]);
  });
});
