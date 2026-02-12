import { describe, it, expect } from "vitest";
import { parseParamsB, isWithinParams } from "./modelSize";

describe("parseParamsB", () => {
  it("extracts size in billions from simple tags", () => {
    expect(parseParamsB("gemma3:12b")).toBe(12);
    expect(parseParamsB("qwen3:4b")).toBe(4);
    expect(parseParamsB("gpt-oss:20B")).toBe(20);
  });

  it("extracts size when cloud suffix present", () => {
    expect(parseParamsB("qwen3-coder:480b-cloud")).toBe(480);
    expect(parseParamsB("deepseek-v3.1:671b-cloud")).toBe(671);
  });

  it("returns null when no size encoded", () => {
    expect(parseParamsB("minimax-m2:cloud")).toBeNull();
    expect(parseParamsB("glm-4.6:cloud")).toBeNull();
    expect(parseParamsB("plainmodel")).toBeNull();
  });
});

describe("isWithinParams", () => {
  it("checks within inclusive range", () => {
    expect(isWithinParams("gemma3:12b", 0, 14)).toBe(true);
    expect(isWithinParams("gemma3:12b", 13, 20)).toBe(false);
    expect(isWithinParams("gpt-oss:120b-cloud", 30, 120)).toBe(true);
    expect(isWithinParams("qwen3:4b", 5, 10)).toBe(false);
  });
});
