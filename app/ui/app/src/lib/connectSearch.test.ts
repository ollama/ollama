import { describe, expect, it } from "vitest";
import { parseConnectSearch } from "./connectSearch";

describe("parseConnectSearch", () => {
  it("keeps only recognized Apps page intents", () => {
    expect(parseConnectSearch({})).toEqual({});
    expect(parseConnectSearch({ connect: true })).toEqual({ connect: true });
    expect(parseConnectSearch({ connect: "true" })).toEqual({ connect: true });
    expect(parseConnectSearch({ connect: "chatgpt" })).toEqual({
      connect: "chatgpt",
    });
    expect(parseConnectSearch({ connect: "yes", highlight: "" })).toEqual({});
    expect(parseConnectSearch({ highlight: "codex" })).toEqual({
      highlight: "codex",
    });
    expect(parseConnectSearch({ highlight: 7, other: "x" })).toEqual({});
  });
});
