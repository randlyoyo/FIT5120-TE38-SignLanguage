/**
 * Parity against fixtures exported from the Python training pipeline.
 * A divergence here does not throw in production -- it silently costs
 * accuracy -- so this runs the whole path: features, fixed-length tensor,
 * and the encoder itself.
 */
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { buildFeatures, toFixedLength, mirrorCapture } = require("../src/recognition/features");
const { load, embed, distanceToWord } = require("../src/recognition/model");

const GOLDEN = path.join(__dirname, "..", "..", "..", "recognition", "test", "golden.json");

(async () => {
  const g = JSON.parse(fs.readFileSync(GOLDEN, "utf8"));
  const tol = g.tolerance ?? 1e-5;
  const s = await load();
  let failed = 0;

  for (const c of g.cases) {
    const feat = buildFeatures(c.input);
    const fixed = toFixedLength(feat, g.T);
    const maxAbs = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

    const dFirst = maxAbs(c.expect.feat_first_frame, feat[0]);
    const dLast = maxAbs(c.expect.feat_last_frame, feat[feat.length - 1]);
    const sum = fixed.reduce((a, b) => a + b, 0);
    const dSum = Math.abs(sum - c.expect.fixed_checksum) / Math.abs(c.expect.fixed_checksum);

    const { embedding } = await embed(c.input);
    const dEmb = maxAbs(c.expect.embedding, embedding);

    const ok = feat.length === c.expect.feat_len
      && dFirst < tol && dLast < tol && dSum < 1e-4 && dEmb < tol;
    if (!ok) failed++;
    console.log(
      `${ok ? "PASS" : "FAIL"}  ${c.name}  len ${feat.length}/${c.expect.feat_len}` +
      `  feat ${Math.max(dFirst, dLast).toExponential(1)}` +
      `  checksum ${dSum.toExponential(1)}  embedding ${dEmb.toExponential(1)}`
    );
  }
  console.log(failed ? `\n${failed} case(s) diverge from the Python pipeline` : "\nAll cases match Python.");
  assert.strictEqual(failed, 0);

  // Handedness-free verify: mirroring is its own inverse, and a mirrored
  // attempt scores the same as the original once verify takes the closer of both.
  for (const c of g.cases) {
    const m = mirrorCapture(c.input);
    assert.deepStrictEqual(mirrorCapture(m), c.input);
    const verifyDistance = async (cap) => Math.min(
      distanceToWord(s, (await embed(cap)).embedding, 0),
      distanceToWord(s, (await embed(mirrorCapture(cap))).embedding, 0),
    );
    const [dOrig, dMirror] = [await verifyDistance(c.input), await verifyDistance(m)];
    assert.ok(Math.abs(dOrig - dMirror) < 1e-6, `${c.name}: ${dOrig} vs ${dMirror}`);
    // ...and mirroring really changes what the model sees (not a no-op).
    assert.notDeepStrictEqual((await embed(m)).embedding, (await embed(c.input)).embedding);
  }
  console.log("Mirror check passed.");
})();
