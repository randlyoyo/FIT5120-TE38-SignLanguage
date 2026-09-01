/**
 * Classic Dynamic Time Warping distance between two sequences of feature
 * vectors. Replaces the reference repo's `fastdtw` (utils/dtw.py) — our
 * sequences are short (well under 100 frames) so the exact O(n*m) DP table
 * is fast enough in the browser and needs no extra dependency.
 */
function euclidean(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export function dtwDistance(a: number[][], b: number[][]): number {
  const n = a.length;
  const m = b.length;
  if (n === 0 || m === 0) return Infinity;

  const width = m + 1;
  const cost = new Float64Array((n + 1) * width).fill(Infinity);
  cost[0] = 0;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const d = euclidean(a[i - 1], b[j - 1]);
      const best = Math.min(
        cost[(i - 1) * width + j],
        cost[i * width + (j - 1)],
        cost[(i - 1) * width + (j - 1)],
      );
      cost[i * width + j] = d + best;
    }
  }

  return cost[n * width + m];
}
