const GIB_FACTOR: Record<string, number> = {
  gib: 1, // 1 GiB = 1 GiB
  gb: 1000 / 1024, // 1 GB (decimal) = ~0.9765625 GiB
  mib: 1 / 1024, // 1 MiB = 1/1024 GiB
  mb: 1 / (1024 * 1024), // 1 MB (decimal) = 1,000,000 bytes = ~9.54e-7 GiB
};

export function parseVRAM(vramString: string): number | null {
  if (!vramString) return null;

  const match = vramString.match(/^(\d+(?:\.\d+)?)\s*(GiB|GB|MiB|MB)$/i);
  if (!match) return null;

  const value = parseFloat(match[1]);
  const unit = match[2].toLowerCase();

  return value * GIB_FACTOR[unit];
}

export function getTotalVRAM(inferenceComputes: { vram: string }[]): number {
  let totalVRAM = 0;
  for (const compute of inferenceComputes) {
    const parsed = parseVRAM(compute.vram);
    if (parsed !== null) {
      totalVRAM += parsed;
    }
  }
  return totalVRAM;
}
