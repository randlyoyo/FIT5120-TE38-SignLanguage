/**
 * Landmark sequence -> the (96, 98) tensor the encoder expects.
 *
 * This is a port of `signtest/dtw_features.py`. It is the single most
 * failure-prone piece of the recognition path: every step below has a
 * counterpart the encoder was trained against, and any divergence still runs,
 * still returns plausible numbers, and is silently less accurate.
 *
 * It lives on the server rather than in the browser so there is exactly one
 * copy to keep correct, and so the model can be changed without shipping a new
 * client.
 *
 * `npm run test:recognition` checks this file against fixtures exported from
 * the Python original. Run it after touching anything here.
 *
 * Contract: recognition/API.md section 3.
 */

const MISSING = -999;
const FEAT_DIM = 98;
const TARGET_FPS = 25;
const MIN_FRAMES = 8;
const FPS_MIN = 5;
const FPS_MAX = 120;

// Arms are entries 5..10 of the stored 11-point pose subset
// [0,2,5,7,8,11,12,13,14,15,16] -> pose indices [11,12,13,14,15,16].
const ARM = [5, 6, 7, 8, 9, 10];
const SH_L = 0, SH_R = 1, WR_L = 4, WR_R = 5;   // positions within ARM

const present = (pt) => pt && pt[0] !== MISSING && pt[1] !== MISSING;

/**
 * Round half to EVEN, matching numpy. Math.round rounds half UP, and at
 * 30 -> 25 fps the resample index lands on exactly x.5 every fifth frame, so
 * the two rules disagree on 20% of all frames.
 */
function rint(v) {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

class UnusableCapture extends Error {}

/**
 * @param {{width:number,height:number,fps:number,
 *          pose:number[][][], left_hand:number[][][], right_hand:number[][][]}} capture
 *        Coordinates are MediaPipe's normalised [0,1]; a missing landmark is
 *        [-999,-999]. `pose` carries the 11-point subset.
 * @returns {Float64Array[]} variable-length list of 98-dim frames
 */
function buildFeatures({ width, height, fps, pose, left_hand: lhIn, right_hand: rhIn }) {
  const n = pose?.length ?? 0;
  if (!(fps >= FPS_MIN && fps <= FPS_MAX)) throw new UnusableCapture(`fps out of range: ${fps}`);
  if (n < MIN_FRAMES) throw new UnusableCapture(`only ${n} frames`);
  if (lhIn.length !== n || rhIn.length !== n) throw new UnusableCapture("frame counts disagree");

  // 3.1 undo the anisotropic squash: MediaPipe divides x by width and y by
  // height independently, so a square gesture arrives stretched by the frame
  // aspect. The single shoulder-width divisor below cannot correct that.
  const ar = width / height;
  const fix = (g) => g.map((f) => f.map((p) => (present(p) ? [p[0] * ar, p[1]] : [MISSING, MISSING])));
  const P = fix(pose), L = fix(lhIn), R = fix(rhIn);

  // 3.2 nearest-neighbour resample onto a common 25 fps time base. Nearest,
  // not interpolated: a hand is present or absent for a whole frame, and
  // interpolating across a gap invents coordinates never observed.
  const nOut = Math.max(1, rint((n / fps) * TARGET_FPS));
  const idx = new Array(nOut);
  for (let j = 0; j < nOut; j++) {
    const t = (j + 0.5) / TARGET_FPS;
    idx[j] = Math.min(n - 1, Math.max(0, rint(t * fps - 0.5)));
  }
  const pick = (g) => idx.map((i) => g[i]);
  const p = pick(P), l = pick(L), r = pick(R);

  // 3.3 centre on the shoulder midpoint, carrying the last known origin
  // through frames where pose dropped out
  const origins = new Array(nOut).fill(null);
  let last = null;
  for (let t = 0; t < nOut; t++) {
    const a = p[t][ARM[SH_L]], b = p[t][ARM[SH_R]];
    if (present(a) && present(b)) last = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    origins[t] = last;
  }
  if (last === null) throw new UnusableCapture("no shoulders detected");
  const firstKnown = origins.find((o) => o !== null);
  for (let t = 0; t < nOut && origins[t] === null; t++) origins[t] = firstKnown;

  // 3.4 scale by ONE median shoulder width for the whole clip. A per-frame
  // divisor would also normalise away the signer leaning in and out, which is
  // real motion.
  const widths = [];
  for (let t = 0; t < nOut; t++) {
    const a = p[t][ARM[SH_L]], b = p[t][ARM[SH_R]];
    if (present(a) && present(b)) widths.push(Math.hypot(a[0] - b[0], a[1] - b[1]));
  }
  widths.sort((x, y) => x - y);
  const m = widths.length;
  const scale = m % 2 ? widths[(m - 1) / 2] : (widths[m / 2 - 1] + widths[m / 2]) / 2;
  if (!(scale > 1e-4)) throw new UnusableCapture("degenerate shoulder width");

  const norm = (pt, t) => (present(pt)
    ? [(pt[0] - origins[t][0]) / scale, (pt[1] - origins[t][1]) / scale]
    : null);

  // 3.5 trim the still lead-in and lead-out. Those frames carry no sign, vary
  // in length between captures, and would otherwise eat the fixed-length
  // budget in 3.7.
  const speed = new Array(Math.max(0, nOut - 1)).fill(0);
  for (let t = 0; t < nOut - 1; t++) {
    for (const w of [ARM[WR_L], ARM[WR_R]]) {
      const a = norm(p[t][w], t), b = norm(p[t + 1][w], t + 1);
      if (a && b) speed[t] = Math.max(speed[t], Math.hypot(b[0] - a[0], b[1] - a[1]));
    }
  }
  if (!speed.length) throw new UnusableCapture("no motion");
  const sorted = [...speed].sort((x, y) => x - y);
  const pos = 0.95 * (sorted.length - 1);          // numpy.percentile default:
  const lo = Math.floor(pos), frac = pos - lo;     // linear between order stats
  const p95 = sorted[lo] * (1 - frac) + sorted[Math.min(lo + 1, sorted.length - 1)] * frac;
  const thr = 0.15 * p95;
  let first = -1, lastActive = -1;
  speed.forEach((s, i) => { if (s > thr) { if (first < 0) first = i; lastActive = i; } });
  if (first < 0) throw new UnusableCapture("no motion");
  const pad = rint(0.1 * TARGET_FPS);
  const start = Math.max(0, first - pad);
  const stop = Math.min(nOut, lastActive + 2 + pad);
  if (stop - start < MIN_FRAMES) throw new UnusableCapture("too short after trim");

  // 3.6 assemble. A missing hand becomes zeros PLUS a cleared presence flag --
  // the flag is what carries "not here"; zero-filling alone would make a
  // missing hand nearly free, and one- vs two-handed is a strong cue.
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

/**
 * 3.7 linear resample to exactly T frames -> flat Float32Array(T * 98).
 * Linear here, unlike 3.2: by this point the presence flags are set, and a
 * fractional flag legitimately means "present for part of this window".
 */
function toFixedLength(feat, T) {
  const n = feat.length;
  const out = new Float32Array(T * FEAT_DIM);
  for (let j = 0; j < T; j++) {
    const pos = n === 1 ? 0 : (j * (n - 1)) / (T - 1);
    const i0 = Math.floor(pos), i1 = Math.min(i0 + 1, n - 1), w = pos - i0;
    const a = feat[i0], b = feat[i1];
    for (let d = 0; d < FEAT_DIM; d++) out[j * FEAT_DIM + d] = a[d] * (1 - w) + b[d] * w;
  }
  return out;
}

module.exports = { buildFeatures, toFixedLength, UnusableCapture, rint, FEAT_DIM, TARGET_FPS };
