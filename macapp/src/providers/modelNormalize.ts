// Model normalization helpers
// Converts UI aliases to backend IDs.

export function normalizeModelId(id: string | undefined | null): string | undefined {
  if (!id) return id || undefined
  let m = id.trim()
  // strip :latest suffix
  m = m.replace(/:latest$/,'')
  // gpt-oss-20b -> gpt-oss:20b heuristic
  if (/^gpt-oss-\d+[a-z]+$/i.test(m)) {
    m = 'gpt-oss:' + m.substring('gpt-oss-'.length)
  }
  return m
}

// Returns true if original differs from normalized (alias case)
export function isAlias(original: string, normalized: string): boolean {
  return original.replace(/:latest$/,'') !== normalized
}
