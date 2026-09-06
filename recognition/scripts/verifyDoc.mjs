/**
 * Independent implementation of API.md §3, written from the document alone.
 *
 * Its job is to test the SPEC, not the model: if a competent reader can follow
 * API.md and land within 1e-5 of the golden fixtures, the document is complete.
 * If they cannot, the document has a hole, and that hole would otherwise have
 * been found by a teammate weeks later as unexplained accuracy loss.
 *
 *   node scripts/verifyDoc.mjs
 */
import { readFileSync } from "node:fs";

const MISSING = -999;
const T_OUT = 96;
const FEAT_DIM = 98;
const TARGET_FPS = 25;
const MIN_FRAMES = 8;

// §2: arms are entries 5..10 of the stored 11-point pose subset
// [0,2,5,7,8,11,12,13,14,15,16] -> [11,12,13,14,15,16]
const ARM = [5, 6, 7, 8, 9, 10];
const SH_L = 0, SH_R = 1, WR_L = 4, WR_R = 5;   // within ARM

const present = (pt) => pt[0] !== MISSING && pt[1] !== MISSING;

/** Round half to EVEN — numpy's rule, not Math.round's round-half-up.
 *  At 30 -> 25 fps every fifth source index lands exactly on .5, so the two
 *  rules disagree on 20% of frames. */
function rint(v) {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

function build({ width, height, fps, pose, left_hand, right_hand }) {
  const n = pose.length;
  if (!(fps >= 5 && fps <= 120)) throw new Error("fps out of range");
  if (n < MIN_FRAMES) throw new Error("too few frames");

  // §3.1 aspect-ratio correction, on a copy
  const ar = width / height;
  const fix = (g) => g.map((f) => f.map((p) => (present(p) ? [p[0] * ar, p[1]] : [MISSING, MISSING])));
  const P = fix(pose), L = fix(left_hand), R = fix(right_hand);

  // §3.2 nearest-neighbour resample to 25 fps
  const nOut = Math.max(1, rint((n / fps) * TARGET_FPS));
  const idx = Array.from({ length: nOut }, (_, j) => {
    const t = (j + 0.5) / TARGET_FPS;
    return Math.min(n - 1, Math.max(0, rint(t * fps - 0.5)));
  });
  const pick = (g) => idx.map((i) => g[i]);
  const p = pick(P), l = pick(L), r = pick(R);

  // §3.3 origin = shoulder midpoint, carried forward across dropouts
  const origins = [];
  let last = null;
  for (let t = 0; t < nOut; t++) {
    const a = p[t][ARM[SH_L]], b = p[t][ARM[SH_R]];
    if (present(a) && present(b)) last = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    origins.push(last);
  }
  if (last === null) throw new Error("no shoulders detected");
  for (let t = 0; t < nOut && origins[t] === null; t++) origins[t] = origins.find((o) => o !== null);

  // §3.4 scale = median shoulder width over the clip
  const widths = [];
  for (let t = 0; t < nOut; t++) {
    const a = p[t][ARM[SH_L]], b = p[t][ARM[SH_R]];
    if (present(a) && present(b)) widths.push(Math.hypot(a[0] - b[0], a[1] - b[1]));
  }
  widths.sort((x, y) => x - y);
  const m = widths.length;
  const scale = m % 2 ? widths[(m - 1) / 2] : (widths[m / 2 - 1] + widths[m / 2]) / 2;
  if (!(scale > 1e-4)) throw new Error("degenerate shoulder width");

  const norm = (pt, t) => (present(pt)
    ? [(pt[0] - origins[t][0]) / scale, (pt[1] - origins[t][1]) / scale]
    : null);

  // §3.5 trim still lead-in / lead-out, measured on the wrists
  const wrists = [];
  for (let t = 0; t < nOut; t++) {
    wrists.push([norm(p[t][ARM[WR_L]], t), norm(p[t][ARM[WR_R]], t)]);
  }
  const speed = [];
  for (let t = 0; t < nOut - 1; t++) {
    let s = 0;
    for (let k = 0; k < 2; k++) {
      const a = wrists[t][k], b = wrists[t + 1][k];
      if (a && b) s = Math.max(s, Math.hypot(b[0] - a[0], b[1] - a[1]));
    }
    speed.push(s);
  }
  const sorted = [...speed].sort((x, y) => x - y);
  // numpy.percentile default: linear interpolation between order statistics
  const pos = 0.95 * (sorted.length - 1);
  const lo = Math.floor(pos), frac = pos - lo;
  const p95 = sorted[lo] * (1 - frac) + sorted[Math.min(lo + 1, sorted.length - 1)] * frac;
  const thr = 0.15 * p95;
  const active = [];
  speed.forEach((s, i) => { if (s > thr) active.push(i); });
  if (!active.length) throw new Error("no motion");
  const pad = rint(0.1 * TARGET_FPS);
  const start = Math.max(0, active[0] - pad);
  const stop = Math.min(nOut, active[active.length - 1] + 2 + pad);
  if (stop - start < MIN_FRAMES) throw new Error("too short after trim");

  // §3.6 assemble 98-dim frames
  const out = [];
  for (let t = start; t < stop; t++) {
    const v = new Float64Array(FEAT_DIM);
    let c = 0;
    for (const a of ARM) { const q = norm(p[t][a], t); v[c++] = q ? q[0] : 0; v[c++] = q ? q[1] : 0; }
    const hl = present(l[t][0]), hr = present(r[t][0]);
    for (let i = 0; i < 21; i++) { const q = hl ? norm(l[t][i], t) : null; v[c++] = q ? q[0] : 0; v[c++] = q ? q[1] : 0; }
    for (let i = 0; i < 21; i++) { const q = hr ? norm(r[t][i], t) : null; v[c++] = q ? q[0] : 0; v[c++] = q ? q[1] : 0; }
    v[96] = hl ? 1 : 0;
    v[97] = hr ? 1 : 0;
    out.push(v);
  }
  return out;
}

// §3.7 linear resample to exactly T frames
function toFixed(feat, T = T_OUT) {
  const n = feat.length;
  const out = new Float64Array(T * FEAT_DIM);
  for (let j = 0; j < T; j++) {
    const pos = n === 1 ? 0 : (j * (n - 1)) / (T - 1);
    const i0 = Math.floor(pos), i1 = Math.min(i0 + 1, n - 1), w = pos - i0;
    for (let d = 0; d < FEAT_DIM; d++) {
      out[j * FEAT_DIM + d] = feat[i0][d] * (1 - w) + feat[i1][d] * w;
    }
  }
  return out;
}

const g = JSON.parse(readFileSync(new URL("../test/golden.json", import.meta.url)));
let bad = 0;
for (const c of g.cases) {
  const feat = build(c.input);
  const fixed = toFixed(feat, g.T);
  const sum = fixed.reduce((a, b) => a + b, 0);
  const dFirst = Math.max(...c.expect.feat_first_frame.map((v, i) => Math.abs(v - feat[0][i])));
  const dLast = Math.max(...c.expect.feat_last_frame.map((v, i) => Math.abs(v - feat[feat.length - 1][i])));
  const lenOk = feat.length === c.expect.feat_len;
  // Relative, not absolute: the checksum sums 96*98 = 9408 values, so float32
  // accumulation alone moves it far more than any single value moves. The
  // per-frame comparisons above are the binding check.
  const sumErr = Math.abs(sum - c.expect.fixed_checksum) / Math.max(1, Math.abs(c.expect.fixed_checksum));
  const ok = lenOk && dFirst < 1e-5 && dLast < 1e-5 && sumErr < 1e-4;
  if (!ok) bad++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${c.name}` +
    `  len ${feat.length}/${c.expect.feat_len}` +
    `  first ${dFirst.toExponential(1)}  last ${dLast.toExponential(1)}` +
    `  checksum rel ${sumErr.toExponential(1)}  (sum ${c.expect.fixed_checksum.toFixed(1)})`
  );
}
console.log(bad ? `\n${bad} case(s) failed — API.md is incomplete or wrong` : "\nAll cases match. API.md is implementable as written.");
process.exit(bad ? 1 : 0);
