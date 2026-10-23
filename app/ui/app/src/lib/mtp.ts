// Multi-token prediction is stored as the model's draft_num_predict parameter
// (the same Modelfile PARAMETER llama-server already reads). An app-only
// setting would not see a model that already has the parameter, and turning
// it off would leave the parameter in place.

export const defaultDraftNumPredict = 3;
export const draftNumPredictChoices = [1, 2, 3, 4];

const nextNKey = /(^|\.)nextn_predict_layers$/;

export interface ModelMTPInfo {
  /** True when GGUF metadata has nextn_predict_layers > 0. */
  supported: boolean;
  /** Present when the model parameter is set and positive, so the toggle is on. */
  draftNumPredict?: number;
}

/** GGUF /api/show model_info has MTP layers when some arch's nextn count is > 0. */
export function modelHasMTPLayers(
  modelInfo: Record<string, unknown> | undefined,
): boolean {
  if (!modelInfo) return false;
  for (const [key, value] of Object.entries(modelInfo)) {
    if (!nextNKey.test(key)) continue;
    const layers = typeof value === "number" ? value : Number(value);
    if (Number.isFinite(layers) && layers > 0) return true;
  }
  return false;
}

/**
 * Reads draft_num_predict from the text `parameters` block returned by
 * /api/show. That block is how an existing Modelfile PARAMETER is reported.
 * Missing and non-numeric values are undefined. Zero is off.
 */
export function parseDraftNumPredict(
  parameters: string | undefined,
): number | undefined {
  const match = parameters?.match(/^\s*draft_num_predict\s+(\S+)\s*$/m);
  if (!match) return undefined;
  const value = Number(match[1]);
  if (!Number.isFinite(value)) return undefined;
  return Math.round(value);
}

export function inspectModelMTP(
  parameters: string | undefined,
  modelInfo: Record<string, unknown> | undefined,
): ModelMTPInfo {
  const draft = parseDraftNumPredict(parameters);
  return {
    supported: modelHasMTPLayers(modelInfo),
    ...(draft !== undefined && draft > 0 ? { draftNumPredict: draft } : {}),
  };
}

export function modelIsLoaded(
  running: { name?: string; model?: string }[] | undefined,
  model: string,
): boolean {
  if (!model) return false;
  return (running ?? []).some(
    (entry) => entry.name === model || entry.model === model,
  );
}

/** Depths shown in the settings control. An unusual stored depth stays visible. */
export function draftDepthChoices(current: number | undefined): number[] {
  if (
    current !== undefined &&
    current > 0 &&
    !draftNumPredictChoices.includes(current)
  ) {
    return [current, ...draftNumPredictChoices];
  }
  return [...draftNumPredictChoices];
}

export class ModelMTPReloadError extends Error {
  constructor(cause: unknown) {
    super(
      "Multi-token prediction was saved, but the model could not be reloaded. It will apply the next time the model loads.",
    );
    this.name = "ModelMTPReloadError";
    void cause;
  }
}

/**
 * Writes only draft_num_predict, then reloads the model when it is already
 * loaded. llama-server reads the depth at process start, so a running model
 * has to be unloaded and loaded again. A model that is not loaded is left
 * unloaded; the parameter applies the next time it starts.
 */
export async function applyModelDraftNumPredict(
  model: string,
  draftNumPredict: number,
  loaded: boolean,
  deps: {
    setDraft: (model: string, draftNumPredict: number) => Promise<void>;
    unload: (model: string) => Promise<void>;
    load: (model: string) => Promise<void>;
  },
): Promise<void> {
  await deps.setDraft(model, draftNumPredict);
  if (!loaded) return;
  try {
    await deps.unload(model);
    await deps.load(model);
  } catch (error) {
    throw new ModelMTPReloadError(error);
  }
}
