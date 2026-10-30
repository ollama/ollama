import { describe, it, expect } from "vitest";
import { parseVRAM, getTotalVRAM } from "./vram";

describe("VRAM Utilities", () => {
  describe("parseVRAM", () => {
    it("should parse GB (decimal) values correctly", () => {
      expect(parseVRAM("1 GB")).toBeCloseTo(1000 / 1024); // ≈0.9765625 GiB
      expect(parseVRAM("16.5 GB")).toBeCloseTo(16.5 * (1000 / 1024));
      expect(parseVRAM("32GB")).toBeCloseTo(32 * (1000 / 1024));
    });

    it("should parse GiB (binary) values correctly", () => {
      expect(parseVRAM("8 GiB")).toBe(8);
      expect(parseVRAM("12.8 GiB")).toBe(12.8);
      expect(parseVRAM("24GiB")).toBe(24);
    });

    it("should convert MB (decimal) to GiB correctly", () => {
      expect(parseVRAM("1000 MB")).toBeCloseTo(1000 / (1024 * 1024));
      expect(parseVRAM("8192 MB")).toBeCloseTo(8192 / (1024 * 1024));
      expect(parseVRAM("512.5 MB")).toBeCloseTo(512.5 / (1024 * 1024));
    });

    it("should convert MiB (binary) to GiB correctly", () => {
      expect(parseVRAM("1024 MiB")).toBe(1);
      expect(parseVRAM("2048MiB")).toBe(2);
      expect(parseVRAM("512.5 MiB")).toBe(512.5 / 1024);
      expect(parseVRAM("8192 MiB")).toBe(8);
    });

    it("should handle case insensitive units", () => {
      expect(parseVRAM("8 gb")).toBeCloseTo(8 * (1000 / 1024));
      expect(parseVRAM("8 Gb")).toBeCloseTo(8 * (1000 / 1024));
      expect(parseVRAM("8 GiB")).toBe(8);
      expect(parseVRAM("1024 mib")).toBe(1);
      expect(parseVRAM("1000 mb")).toBeCloseTo(1000 / (1024 * 1024));
    });

    it("should return null for invalid inputs", () => {
      expect(parseVRAM("")).toBeNull();
      expect(parseVRAM("invalid")).toBeNull();
      expect(parseVRAM("8 TB")).toBeNull();
      expect(parseVRAM("GB 8")).toBeNull();
      expect(parseVRAM("8")).toBeNull();
    });

    it("should handle edge cases", () => {
      expect(parseVRAM("0 GB")).toBe(0);
      expect(parseVRAM("0.1 GiB")).toBe(0.1);
      expect(parseVRAM("999999 GiB")).toBe(999999);
    });
  });

  describe("Integration tests", () => {
    it("should parse various VRAM formats consistently", () => {
      const testCases = ["8 GB", "16 GiB", "1024 MB", "512 MiB"];

      testCases.forEach((testCase) => {
        const parsed = parseVRAM(testCase);
        expect(parsed).not.toBeNull();
        expect(typeof parsed).toBe("number");
        expect(parsed).toBeGreaterThan(0);
      });
    });
  });

  describe("getTotalVRAM", () => {
    it("should sum VRAM values from multiple computes", () => {
      const computes = [
        { vram: "8 GiB" },
        { vram: "16 GiB" },
        { vram: "4 GiB" },
      ];
      expect(getTotalVRAM(computes)).toBe(28);
    });

    it("should handle MB and MiB conversions correctly", () => {
      const computes = [
        { vram: "1024 MiB" }, // 1 GiB
        { vram: "2048 MiB" }, // 2 GiB
        { vram: "1000 MB" }, // ~0.000953 GiB
      ];
      expect(getTotalVRAM(computes)).toBeCloseTo(3.000953);
    });

    it("should handle mixed units", () => {
      const computes = [
        { vram: "8 GiB" },
        { vram: "1000 MB" }, // ~0.000953 GiB
        { vram: "16 GB" }, // 16 * 0.9765625 GiB
      ];

      expect(getTotalVRAM(computes)).toBeCloseTo(8 + 0.000953 + 15.625);
    });

    it("should skip invalid VRAM strings", () => {
      const computes = [
        { vram: "8 GiB" },
        { vram: "invalid" },
        { vram: "16 GiB" },
        { vram: "" },
      ];

      expect(getTotalVRAM(computes)).toBe(24);
    });

    it("should handle empty array", () => {
      expect(getTotalVRAM([])).toBe(0);
    });

    it("should handle single compute", () => {
      const computes = [{ vram: "12 GiB" }];
      expect(getTotalVRAM(computes)).toBe(12);
    });

    it("should handle decimal values", () => {
      const computes = [{ vram: "8.5 GiB" }, { vram: "4.25 GiB" }];
      expect(getTotalVRAM(computes)).toBe(12.75);
    });
  });
});
