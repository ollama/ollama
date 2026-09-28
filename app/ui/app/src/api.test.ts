import { afterEach, describe, expect, it, vi } from "vitest";
const { listModels } = vi.hoisted(() => ({ listModels: vi.fn() }));
vi.mock("./lib/ollama-client", () => ({
  ollamaClient: { list: listModels },
}));

import {
  fetchConnectUrl,
  getClaudeDesktopAvailableModels,
  getClaudeDesktopModelsSettings,
  getCodexDesktopModelsSettings,
  getIntegrationStatuses,
  exportAllChats,
} from "./api";

describe("exportAllChats", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("reads progress and the saved path across stream chunks", async () => {
    const data = new TextEncoder().encode(
      '{"completed":0,"total":2}\n{"completed":1,"total":2}\n{"path":"/exports/café.zip"}\n',
    );
    const stream = new ReadableStream({
      start(controller) {
        for (const byte of data) controller.enqueue(Uint8Array.of(byte));
        controller.close();
      },
    });
    const fetch = vi.fn().mockResolvedValue(new Response(stream));
    vi.stubGlobal("fetch", fetch);
    const controller = new AbortController();
    const progress = vi.fn();
    await expect(exportAllChats(controller.signal, progress)).resolves.toEqual({
      path: "/exports/café.zip",
    });
    expect(progress.mock.calls).toEqual([
      [{ completed: 0, total: 2 }],
      [{ completed: 1, total: 2 }],
    ]);
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:3001/api/v1/chats/export",
      { method: "POST", signal: controller.signal },
    );
  });

  it.each([
    ["null\n", null],
    ['{"error":"Disk is full"}\n', "Disk is full"],
    ['{"completed":1,"total":2}\n', "Export stopped before it finished."],
  ])(
    "handles cancellation, errors, and incomplete exports: %s",
    async (body, error) => {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body)));
      const result = exportAllChats(new AbortController().signal, vi.fn());
      if (error) await expect(result).rejects.toThrow(error);
      else await expect(result).resolves.toBeNull();
    },
  );
});

describe("desktop model settings", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("requests summaries and catalogs through the configured API server", async () => {
    const response = { settings: { selected: ["saved-model"] } };
    const fetch = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(response)));
    vi.stubGlobal("fetch", fetch);
    await expect(getCodexDesktopModelsSettings(false)).resolves.toEqual(
      response,
    );
    expect(fetch).toHaveBeenLastCalledWith(
      "http://127.0.0.1:3001/api/v1/integrations/chatgpt/models?catalog=false",
      { signal: undefined },
    );
    fetch.mockResolvedValue(new Response(JSON.stringify({ installed: true })));
    const controller = new AbortController();
    await expect(
      getClaudeDesktopModelsSettings(true, controller.signal),
    ).resolves.toEqual({ installed: true });
    expect(fetch).toHaveBeenLastCalledWith(
      "http://127.0.0.1:3001/api/v1/integrations/claude-desktop/models?catalog=true",
      { signal: controller.signal },
    );
  });

  it("rejects failed discovery responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("timed out", { status: 504 })),
    );
    await expect(getCodexDesktopModelsSettings(true)).rejects.toThrow("504");
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
