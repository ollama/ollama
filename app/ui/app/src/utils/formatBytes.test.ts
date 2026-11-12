import { describe, it, expect } from "vitest";
import { formatBytes } from "@/utils/formatBytes";

describe("formatBytes", () => {
  it("formats zero and invalid values", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(-5)).toBe("0 B");
    expect(formatBytes(Number.NaN)).toBe("0 B");
  });

  it("formats bytes without decimals", () => {
    expect(formatBytes(512)).toBe("512 B");
  });

  it("formats kilobytes and megabytes", () => {
    expect(formatBytes(1024)).toBe("1 KB");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5 MB");
  });

  it("formats gigabytes the way model sizes are shown", () => {
    expect(formatBytes(4 * 1024 ** 3)).toBe("4 GB");
    expect(formatBytes(13_780_162_412)).toBe("12.8 GB");
  });

  it("formats terabytes", () => {
    expect(formatBytes(2 * 1024 ** 4)).toBe("2 TB");
  });
});
