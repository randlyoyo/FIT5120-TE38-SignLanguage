/**
 * Parity against fixtures exported from the Python training pipeline.
 * A divergence here does not throw in production -- it silently costs
 * accuracy -- so this runs the whole path: features, fixed-length tensor,
 * and the encoder itself.
 */
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { buildFeatures, toFixedLength } = require("../src/recognition/features");
const { load, embed, verifyScore } = require("../src/recognition/model");

// Verify scoring: a pass needs distance under tau AND no other word clearly
// closer; a fail shrinks with how much closer that other word was.
{
  const tau = 0.53;
  assert.deepStrictEqual(verifyScore(0.22, 0, tau), { matched: true, score: 83 });
  assert.strictEqual(verifyScore(0.40, 0.02, tau).matched, true);     // near-tie still passes
  assert.ok(verifyScore(0.40, 0.12, tau).score <= 35);                // one-handed A: HORN (CAR) clearly closer
  assert.strictEqual(verifyScore(0.67, 0.36, tau).score, 0);
  assert.strictEqual(verifyScore(0.60, 0, tau).matched, false);       // best match but too far
}

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
})();
