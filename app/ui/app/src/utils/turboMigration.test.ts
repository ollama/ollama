import { describe, expect, it } from "vitest";
import { getTurboMigrationCloudModel } from "./turboMigration";

describe("getTurboMigrationCloudModel", () => {
  it("should append explicit cloud source if model needs Turbo migration", () => {
    expect(getTurboMigrationCloudModel("gpt-oss:20b")).toBe(
      "gpt-oss:20b:cloud",
    );
    expect(getTurboMigrationCloudModel("gpt-oss:120b")).toBe(
      "gpt-oss:120b:cloud",
    );
    expect(getTurboMigrationCloudModel("deepseek-v3.1:671b")).toBe(
      "deepseek-v3.1:671b:cloud",
    );
    expect(getTurboMigrationCloudModel("qwen3-coder:480b")).toBe(
      "qwen3-coder:480b:cloud",
    );
  });

  it("should return null if model does not need Turbo migration", () => {
    expect(getTurboMigrationCloudModel("gemma3:4b")).toBeNull();
  });
});
