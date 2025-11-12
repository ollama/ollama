import { describe, it, expect } from "vitest";
import { Model } from "@/gotypes";
import { mergeModels, FEATURED_MODELS } from "@/utils/mergeModels";
import "@/api";

describe("Model merging logic", () => {
  it("should handle cloud models with -cloud suffix", () => {
    const localModels: Model[] = [
      new Model({ model: "gpt-oss:120b-cloud" }),
      new Model({ model: "llama3:latest" }),
      new Model({ model: "mistral:latest" }),
    ];

    const merged = mergeModels(localModels);

    // First verify cloud models are first and in FEATURED_MODELS order
    const cloudModels = FEATURED_MODELS.filter((m: string) =>
      m.endsWith("cloud"),
    );
    for (let i = 0; i < cloudModels.length; i++) {
      expect(merged[i].model).toBe(cloudModels[i]);
      expect(merged[i].isCloud()).toBe(true);
    }

    // Then verify non-cloud featured models are next and in FEATURED_MODELS order
    const nonCloudFeatured = FEATURED_MODELS.filter(
      (m: string) => !m.endsWith("cloud"),
    );
    for (let i = 0; i < nonCloudFeatured.length; i++) {
      const model = merged[i + cloudModels.length];
      expect(model.model).toBe(nonCloudFeatured[i]);
      expect(model.isCloud()).toBe(false);
    }

    // Verify local models are preserved and come after featured models
    const featuredCount = FEATURED_MODELS.length;
    expect(merged[featuredCount].model).toBe("llama3:latest");
    expect(merged[featuredCount + 1].model).toBe("mistral:latest");

    // Length should be exactly featured models plus our local models
    expect(merged.length).toBe(FEATURED_MODELS.length + 2);
  });

  it("should hide cloud models in airplane mode", () => {
    const localModels: Model[] = [
      new Model({ model: "gpt-oss:120b-cloud" }),
      new Model({ model: "llama3:latest" }),
      new Model({ model: "mistral:latest" }),
    ];

    const merged = mergeModels(localModels, true); // airplane mode = true

    // No cloud models should be present
    const cloudModels = merged.filter((m) => m.isCloud());
    expect(cloudModels.length).toBe(0);

    // Should have non-cloud featured models
    const nonCloudFeatured = FEATURED_MODELS.filter(
      (m) => !m.endsWith("cloud"),
    );
    for (let i = 0; i < nonCloudFeatured.length; i++) {
      const model = merged[i];
      expect(model.model).toBe(nonCloudFeatured[i]);
      expect(model.isCloud()).toBe(false);
    }

    // Local models should be preserved
    const featuredCount = nonCloudFeatured.length;
    expect(merged[featuredCount].model).toBe("llama3:latest");
    expect(merged[featuredCount + 1].model).toBe("mistral:latest");
  });

  it("should handle empty input", () => {
    const merged = mergeModels([]);

    // First verify cloud models are first and in FEATURED_MODELS order
    const cloudModels = FEATURED_MODELS.filter((m) => m.endsWith("cloud"));
    for (let i = 0; i < cloudModels.length; i++) {
      expect(merged[i].model).toBe(cloudModels[i]);
      expect(merged[i].isCloud()).toBe(true);
    }

    // Then verify non-cloud featured models are next and in FEATURED_MODELS order
    const nonCloudFeatured = FEATURED_MODELS.filter(
      (m) => !m.endsWith("cloud"),
    );
    for (let i = 0; i < nonCloudFeatured.length; i++) {
      const model = merged[i + cloudModels.length];
      expect(model.model).toBe(nonCloudFeatured[i]);
      expect(model.isCloud()).toBe(false);
    }

    // Length should be exactly FEATURED_MODELS length
    expect(merged.length).toBe(FEATURED_MODELS.length);
  });

  it("should sort models correctly", () => {
    const localModels: Model[] = [
      new Model({ model: "zephyr:latest" }),
      new Model({ model: "alpha:latest" }),
      new Model({ model: "gpt-oss:120b-cloud" }),
    ];

    const merged = mergeModels(localModels);

    // First verify cloud models are first and in FEATURED_MODELS order
    const cloudModels = FEATURED_MODELS.filter((m) => m.endsWith("cloud"));
    for (let i = 0; i < cloudModels.length; i++) {
      expect(merged[i].model).toBe(cloudModels[i]);
      expect(merged[i].isCloud()).toBe(true);
    }

    // Then verify non-cloud featured models are next and in FEATURED_MODELS order
    const nonCloudFeatured = FEATURED_MODELS.filter(
      (m) => !m.endsWith("cloud"),
    );
    for (let i = 0; i < nonCloudFeatured.length; i++) {
      const model = merged[i + cloudModels.length];
      expect(model.model).toBe(nonCloudFeatured[i]);
      expect(model.isCloud()).toBe(false);
    }

    // Non-featured local models should be at the end in alphabetical order
    const featuredCount = FEATURED_MODELS.length;
    expect(merged[featuredCount].model).toBe("alpha:latest");
    expect(merged[featuredCount + 1].model).toBe("zephyr:latest");
  });
});
