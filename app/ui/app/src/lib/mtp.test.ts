import { describe, expect, it, vi } from "vitest";
import {
  ModelMTPReloadError,
  applyModelDraftNumPredict,
  defaultDraftNumPredict,
  inspectModelMTP,
  modelHasMTPLayers,
  modelIsLoaded,
  parseDraftNumPredict,
} from "./mtp";

// /api/show prints parameters with the key padded to 30 columns, then a space,
// then Go's %#v. Large floats use exponent notation; small integers do not.
function showParameter(name: string, value: string): string {
  return `${name.padEnd(30)} ${value}`;
}

const productionParameters = [
  showParameter("num_ctx", "1.048576e+06"),
  showParameter("draft_num_predict", "3"),
].join("\n");

describe("modelHasMTPLayers", () => {
  it("detects nextn predict layers from GGUF model_info", () => {
    expect(
      modelHasMTPLayers({
        "general.architecture": "glm4moe",
        "glm4moe.nextn_predict_layers": 1,
        "glm4moe.block_count": 40,
      }),
    ).toBe(true);
    expect(modelHasMTPLayers({ "qwen35.nextn_predict_layers": 3 })).toBe(true);
  });

  it("does not offer MTP when nextn layers are missing or zero", () => {
    expect(modelHasMTPLayers(undefined)).toBe(false);
    expect(
      modelHasMTPLayers({
        "general.architecture": "llama",
        "llama.block_count": 32,
      }),
    ).toBe(false);
    expect(modelHasMTPLayers({ "glm4moe.nextn_predict_layers": 0 })).toBe(
      false,
    );
    expect(modelHasMTPLayers({ "glm4moe.nextn_predict_layers_extra": 4 })).toBe(
      false,
    );
  });
});

describe("parseDraftNumPredict", () => {
  it("reads an existing draft_num_predict parameter as on", () => {
    expect(parseDraftNumPredict(productionParameters)).toBe(3);
    expect(
      inspectModelMTP(productionParameters, {
        "glm4moe.nextn_predict_layers": 1,
      }),
    ).toEqual({ supported: true, draftNumPredict: 3 });
    // num_ctx in the same block is not consumed by the MTP parse.
    expect(productionParameters).toContain("num_ctx");
    expect(productionParameters).toContain("1.048576e+06");
  });

  it("reads a missing or zero parameter as off", () => {
    expect(parseDraftNumPredict(showParameter("num_ctx", "32768"))).toBe(
      undefined,
    );
    expect(parseDraftNumPredict(showParameter("draft_num_predict", "0"))).toBe(
      0,
    );
    expect(
      inspectModelMTP(showParameter("num_ctx", "1048576"), {
        "glm4moe.nextn_predict_layers": 1,
      }),
    ).toEqual({ supported: true });
  });

  it("hides the control for models without MTP even if the parameter is set", () => {
    expect(
      inspectModelMTP(showParameter("draft_num_predict", "3"), {
        "llama.block_count": 32,
      }).supported,
    ).toBe(false);
  });
});

describe("applyModelDraftNumPredict", () => {
  it("turns on by setting draft_num_predict and does not reload an unloaded model", async () => {
    const setDraft = vi.fn().mockResolvedValue(undefined);
    const unload = vi.fn().mockResolvedValue(undefined);
    const load = vi.fn().mockResolvedValue(undefined);

    await applyModelDraftNumPredict(
      "glm5.3-flash:q6-vision",
      defaultDraftNumPredict,
      false,
      {
        setDraft,
        unload,
        load,
      },
    );

    expect(setDraft).toHaveBeenCalledTimes(1);
    expect(setDraft).toHaveBeenCalledWith("glm5.3-flash:q6-vision", 3);
    expect(unload).not.toHaveBeenCalled();
    expect(load).not.toHaveBeenCalled();
  });

  it("turns off by removing the parameter and reloads a loaded model", async () => {
    const order: string[] = [];
    const setDraft = vi.fn(async () => {
      order.push("set");
    });
    const unload = vi.fn(async () => {
      order.push("unload");
    });
    const load = vi.fn(async () => {
      order.push("load");
    });

    await applyModelDraftNumPredict("glm5.3-flash:q6-vision", 0, true, {
      setDraft,
      unload,
      load,
    });

    expect(setDraft).toHaveBeenCalledWith("glm5.3-flash:q6-vision", 0);
    expect(order).toEqual(["set", "unload", "load"]);
  });

  it("does not unload when saving the parameter fails", async () => {
    const setDraft = vi.fn().mockRejectedValue(new Error("disk full"));
    const unload = vi.fn().mockResolvedValue(undefined);
    const load = vi.fn().mockResolvedValue(undefined);

    await expect(
      applyModelDraftNumPredict("glm5.3-flash:q6-vision", 0, true, {
        setDraft,
        unload,
        load,
      }),
    ).rejects.toThrow("disk full");
    expect(unload).not.toHaveBeenCalled();
    expect(load).not.toHaveBeenCalled();
  });

  it("reports a saved parameter when reload fails", async () => {
    const setDraft = vi.fn().mockResolvedValue(undefined);
    const unload = vi.fn().mockResolvedValue(undefined);
    const load = vi.fn().mockRejectedValue(new Error("runner failed"));

    await expect(
      applyModelDraftNumPredict("glm5.3-flash:q6-vision", 3, true, {
        setDraft,
        unload,
        load,
      }),
    ).rejects.toBeInstanceOf(ModelMTPReloadError);
    expect(setDraft).toHaveBeenCalledWith("glm5.3-flash:q6-vision", 3);
    expect(unload).toHaveBeenCalledTimes(1);
  });
});

describe("modelIsLoaded", () => {
  it("matches the running model by name or model field", () => {
    expect(
      modelIsLoaded(
        [{ name: "glm5.3-flash:q6-vision" }],
        "glm5.3-flash:q6-vision",
      ),
    ).toBe(true);
    expect(
      modelIsLoaded([{ model: "other:latest" }], "glm5.3-flash:q6-vision"),
    ).toBe(false);
  });
});
