const TURBO_MIGRATION_BASE_MODELS = [
  "gpt-oss:20b",
  "gpt-oss:120b",
  "deepseek-v3.1:671b",
  "qwen3-coder:480b",
];

export function getTurboMigrationCloudModel(model: string): string | null {
  if (!TURBO_MIGRATION_BASE_MODELS.includes(model)) {
    return null;
  }

  return `${model}:cloud`;
}
