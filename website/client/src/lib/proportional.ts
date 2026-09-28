/** Even split of `total` across `tags`, each getting at least 1 -- any
 *  remainder goes to the first tags in order (stable, not random) so the
 *  same inputs always produce the same split. */
export function evenSplit(tags: string[], total: number): Record<string, number> {
  const n = tags.length;
  const base = Math.floor(total / n);
  const remainder = total - base * n;
  const result: Record<string, number> = {};
  tags.forEach((tag, i) => {
    result[tag] = base + (i < remainder ? 1 : 0);
  });
  return result;
}

/** Rescales an existing count split to a new total, keeping each tag's
 *  share as close to its old ratio as integers allow (largest-remainder
 *  method), and never dropping a tag below 1. */
export function rescaleToTotal(
  counts: Record<string, number>,
  tags: string[],
  newTotal: number
): Record<string, number> {
  const currentTotal = tags.reduce((sum, t) => sum + (counts[t] ?? 0), 0);
  if (currentTotal <= 0) return evenSplit(tags, newTotal);

  const raw = tags.map((t) => ((counts[t] ?? 0) / currentTotal) * newTotal);
  const floors = raw.map((v) => Math.max(1, Math.floor(v)));
  let remaining = newTotal - floors.reduce((a, b) => a + b, 0);

  const byFraction = tags
    .map((_, i) => i)
    .sort((a, b) => raw[b] - Math.floor(raw[b]) - (raw[a] - Math.floor(raw[a])));

  let guard = 0;
  while (remaining !== 0 && guard < tags.length * 20) {
    const i = byFraction[guard % byFraction.length];
    if (remaining > 0) {
      floors[i]++;
      remaining--;
    } else if (floors[i] > 1) {
      floors[i]--;
      remaining++;
    }
    guard++;
  }

  const result: Record<string, number> = {};
  tags.forEach((t, i) => {
    result[t] = floors[i];
  });
  return result;
}
