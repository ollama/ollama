import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { getSavedModels, saveModel, unsaveModel } from "@/api";

function mockFetch(response: Partial<Response> & { json?: () => unknown }) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    text: async () => "",
    json: async () => ({}),
    ...response,
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("saved models api", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns the saved model list", async () => {
    mockFetch({ json: async () => ({ models: ["qwen3:8b", "gemma3:4b"] }) });

    await expect(getSavedModels()).resolves.toEqual(["qwen3:8b", "gemma3:4b"]);
  });

  it("treats a missing models field as an empty collection", async () => {
    mockFetch({ json: async () => ({}) });

    await expect(getSavedModels()).resolves.toEqual([]);
  });

  it("throws when the saved models request fails", async () => {
    mockFetch({ ok: false, status: 500 });

    await expect(getSavedModels()).rejects.toThrow(
      "Failed to fetch saved models",
    );
  });

  it("posts the model name when saving", async () => {
    const fetchMock = mockFetch({
      json: async () => ({ models: ["qwen3:8b"] }),
    });

    await expect(saveModel("qwen3:8b")).resolves.toEqual(["qwen3:8b"]);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain("/api/v1/saved-models");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ model: "qwen3:8b" });
  });

  it("deletes the model name when unsaving", async () => {
    const fetchMock = mockFetch({ json: async () => ({ models: [] }) });

    await expect(unsaveModel("qwen3:8b")).resolves.toEqual([]);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain("/api/v1/saved-models");
    expect(init.method).toBe("DELETE");
    expect(JSON.parse(init.body)).toEqual({ model: "qwen3:8b" });
  });

  it("surfaces the server error message when a mutation fails", async () => {
    mockFetch({
      ok: false,
      status: 500,
      text: async () => "model name is required",
    });

    await expect(saveModel("")).rejects.toThrow("model name is required");
  });
});
