// Lightweight fuzzy matching for project file paths.
// Higher scores are better; -1 means no match.
export function fuzzyScore(query: string, path: string): number {
  if (!query) {
    // With no query, prefer shallow, short paths
    return 1000 - path.length - path.split("/").length * 10;
  }

  const q = query.toLowerCase();
  const p = path.toLowerCase();
  const basename = p.slice(p.lastIndexOf("/") + 1);

  const basenameIdx = basename.indexOf(q);
  if (basenameIdx !== -1) {
    // Substring of the file name; earlier and tighter matches win
    return 10000 - basenameIdx * 10 - path.length;
  }

  const pathIdx = p.indexOf(q);
  if (pathIdx !== -1) {
    return 5000 - pathIdx * 5 - path.length;
  }

  // Subsequence match over the full path
  let pi = 0;
  let first = -1;
  let last = -1;
  for (let qi = 0; qi < q.length; qi++) {
    pi = p.indexOf(q[qi], pi);
    if (pi === -1) return -1;
    if (first === -1) first = pi;
    last = pi;
    pi++;
  }
  const spread = last - first;
  return 1000 - spread * 2 - path.length;
}

export function fuzzyFilter(
  query: string,
  paths: string[],
  limit: number,
): string[] {
  const scored: Array<{ path: string; score: number }> = [];
  for (const path of paths) {
    const score = fuzzyScore(query, path);
    if (score >= 0) {
      scored.push({ path, score });
    }
  }
  scored.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
  return scored.slice(0, limit).map((s) => s.path);
}
