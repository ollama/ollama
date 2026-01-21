export function parseParamsB(name: string): number | null {
  const lower = name.toLowerCase();
  const match = lower.match(/(^|:|\s)(\d+)\s*b(?![a-z])/);
  if (!match) return null;
  const num = Number(match[2]);
  if (Number.isNaN(num)) return null;
  return num;
}

export function isWithinParams(name: string, minB: number, maxB: number): boolean {
  const params = parseParamsB(name);
  if (params == null) return false;
  return params >= minB && params <= maxB;
}
