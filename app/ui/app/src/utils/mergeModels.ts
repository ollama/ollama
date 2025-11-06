import { Model } from "@/gotypes";

// Featured models list (in priority order)
export const FEATURED_MODELS = [
  "gpt-oss:120b-cloud",
  "gpt-oss:20b-cloud",
  "deepseek-v3.1:671b-cloud",
  "qwen3-coder:480b-cloud",
  "qwen3-vl:235b-cloud",
  "minimax-m2:cloud",
  "glm-4.6:cloud",
  "gpt-oss:120b",
  "gpt-oss:20b",
  "gemma3:27b",
  "gemma3:12b",
  "gemma3:4b",
  "gemma3:1b",
  "deepseek-r1:8b",
  "qwen3-coder:30b",
  "qwen3-vl:30b",
  "qwen3-vl:8b",
  "qwen3-vl:4b",
  "qwen3:30b",
  "qwen3:8b",
  "qwen3:4b",
];

function alphabeticalSort(a: Model, b: Model): number {
  return a.model.toLowerCase().localeCompare(b.model.toLowerCase());
}

//Merges models, sorting cloud models first, then other models
export function mergeModels(
  localModels: Model[],
  airplaneMode: boolean = false,
): Model[] {
  const allModels = (localModels || []).map((model) => model);

  // 1. Get cloud models from local models and featured list
  const cloudModels = [...allModels.filter((m) => m.isCloud())];

  // Add any cloud models from FEATURED_MODELS that aren't in local models
  FEATURED_MODELS.filter((f) => f.endsWith("cloud")).forEach((cloudModel) => {
    if (!cloudModels.some((m) => m.model === cloudModel)) {
      cloudModels.push(new Model({ model: cloudModel }));
    }
  });

  // 2. Get other featured models (non-cloud)
  const featuredModels = FEATURED_MODELS.filter(
    (f) => !f.endsWith("cloud"),
  ).map((model) => {
    // Check if this model exists in local models
    const localMatch = allModels.find(
      (m) => m.model.toLowerCase() === model.toLowerCase(),
    );

    if (localMatch) return localMatch;

    return new Model({
      model,
    });
  });

  // 3. Get remaining local models that aren't featured and aren't cloud models
  const remainingModels = allModels.filter(
    (model) =>
      !model.isCloud() &&
      !FEATURED_MODELS.some(
        (f) => f.toLowerCase() === model.model.toLowerCase(),
      ),
  );

  cloudModels.sort((a, b) => {
    const aIndex = FEATURED_MODELS.indexOf(a.model);
    const bIndex = FEATURED_MODELS.indexOf(b.model);

    // If both are featured, sort by their position in FEATURED_MODELS
    if (aIndex !== -1 && bIndex !== -1) {
      return aIndex - bIndex;
    }

    // If only one is featured, featured model comes first
    if (aIndex !== -1 && bIndex === -1) return -1;
    if (aIndex === -1 && bIndex !== -1) return 1;

    // If neither is featured, sort alphabetically
    return a.model.toLowerCase().localeCompare(b.model.toLowerCase());
  });

  featuredModels.sort(
    (a, b) =>
      FEATURED_MODELS.indexOf(a.model) - FEATURED_MODELS.indexOf(b.model),
  );

  remainingModels.sort(alphabeticalSort);

  return airplaneMode
    ? [...featuredModels, ...remainingModels]
    : [...cloudModels, ...featuredModels, ...remainingModels];
}
