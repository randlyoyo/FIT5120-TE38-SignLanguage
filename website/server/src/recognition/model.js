/**
 * Encoder + template bank, loaded once and shared.
 *
 * The bank ships int8 quantised -- 9.9 MB on disk instead of 39.5 MB, small
 * enough to live in git so that Vercel and Railway, which both deploy from the
 * repository, get it without a build-time fetch. Measured impact of the
 * quantisation on AUC, EER and top-4: none (differences below 0.01 points).
 *
 * It is dequantised to float32 once at load, so scoring is unchanged: one pass
 * over 3215 x 12 x 256 floats, about 1 ms, which is why identify needs no
 * index or vector database at this size.
 */

const fs = require("node:fs");
const path = require("node:path");
const ort = require("onnxruntime-node");

const { buildFeatures, toFixedLength, FEAT_DIM } = require("./features");

const ROOT = process.env.RECOGNITION_MODELS
  || path.join(__dirname, "..", "..", "..", "..", "recognition", "models");

let state = null;

/** Load on first use so a server without the assets still starts and serves
 *  the rest of the API; the recognition routes then answer 503. */
async function load() {
  if (state) return state;

  const index = JSON.parse(fs.readFileSync(path.join(ROOT, "bank_index.json"), "utf8"));
  const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, "manifest.json"), "utf8"));
  const raw = fs.readFileSync(path.join(ROOT, "bank.i8"));

  const { words, dim, n_template: nTpl } = index;
  const count = words.length * nTpl * dim;
  if (raw.length !== count) {
    throw new Error(
      `bank.i8 is ${raw.length} bytes, expected ${count} for `
      + `${words.length}x${nTpl}x${dim} int8 -- rebuild it with `
      + "recognition/scripts/buildBank.mjs"
    );
  }
  // Dequantise and re-normalise each template. The stored int8 values are
  // v*127 rounded, so dividing by 127 loses the unit norm slightly; the
  // encoder output is a unit vector, and the cosine rule below assumes both
  // sides are.
  const bank = new Float32Array(count);
  for (let t = 0; t < words.length * nTpl; t++) {
    const off = t * dim;
    let ss = 0;
    for (let d = 0; d < dim; d++) { const v = raw.readInt8(off + d) / 127; bank[off + d] = v; ss += v * v; }
    const inv = 1 / Math.sqrt(ss);
    for (let d = 0; d < dim; d++) bank[off + d] *= inv;
  }

  const session = await ort.InferenceSession.create(path.join(ROOT, "encoder.onnx"));

  state = {
    session, bank, words, dim, nTpl,
    T: manifest.T,
    tauVerify: index.tau_verify,
    // Confidence for identify is the gap between the best and second-best
    // word, not the absolute distance. Measured on the validation split:
    // margin predicts a correct top-1 at AUC 0.88, absolute distance only
    // 0.70 -- distance varies too much between words to threshold globally.
    marginConfident: manifest.margin_confident ?? 0.04,
    wordIndex: new Map(words.map((w, i) => [w, i])),
  };
  return state;
}

/** capture -> unit-norm Float32Array(dim) */
async function embed(capture) {
  const s = await load();
  const feat = buildFeatures(capture);
  const fixed = toFixedLength(feat, s.T);
  const out = await s.session.run({
    x: new ort.Tensor("float32", fixed, [1, s.T, FEAT_DIM]),
  });
  return { embedding: out.emb.data, frames: feat.length };
}

/** Cosine distance to one word: 1 - max similarity over its templates. */
function distanceToWord(s, embedding, wordIdx) {
  const base = wordIdx * s.nTpl * s.dim;
  let best = -Infinity;
  for (let k = 0; k < s.nTpl; k++) {
    const off = base + k * s.dim;
    let dot = 0;
    for (let d = 0; d < s.dim; d++) dot += embedding[d] * s.bank[off + d];
    if (dot > best) best = dot;
  }
  return 1 - best;
}

/** Distance to every word, in one pass over the bank. */
function distanceToAll(s, embedding) {
  const out = new Float64Array(s.words.length);
  for (let w = 0; w < s.words.length; w++) out[w] = distanceToWord(s, embedding, w);
  return out;
}

module.exports = { load, embed, distanceToWord, distanceToAll, ROOT };
