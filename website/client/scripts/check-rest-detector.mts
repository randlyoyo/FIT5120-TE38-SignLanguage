// Replays captures through the Practice auto-stop rule (src/lib/restDetector.ts).
//   node scripts/check-rest-detector.mts [dir of saved captures]
// Always checks the training fixtures: the rule must arm and must not stop
// inside them (they are cut tight to the sign). With a directory of real
// captures ({info, capture} JSON), it also prints where each would stop.
import assert from "node:assert";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { createRestDetector } from "../src/lib/restDetector.ts";

type Cap = { width: number; height: number; fps: number; pose: number[][][]; left_hand: number[][][]; right_hand: number[][][] };

function replay(c: Cap) {
  const step = createRestDetector(c.width, c.height);
  let armedAt: number | null = null;
  for (let i = 0; i < c.pose.length; i++) {
    const t = (i * 1000) / c.fps;
    const r = step(c.pose[i], t, c.left_hand[i][0][0] !== -999 || c.right_hand[i][0][0] !== -999);
    if (r.phase !== "waiting") armedAt ??= t;
    if (r.stop) return { armedAt, stopAt: t };
  }
  return { armedAt, stopAt: null };
}

const golden = JSON.parse(readFileSync(new URL("../../../recognition/test/golden.json", import.meta.url), "utf8"));
for (const g of golden.cases) {
  const r = replay(g.input);
  assert.notStrictEqual(r.armedAt, null, `${g.name}: never armed`);
  assert.strictEqual(r.stopAt, null, `${g.name}: stopped mid-sign at ${r.stopAt} ms`);
}
console.log(`training clips: ${golden.cases.length} armed, none stopped mid-sign`);

const dir = process.argv[2];
if (dir) {
  for (const f of readdirSync(dir).sort()) {
    const { info, capture } = JSON.parse(readFileSync(join(dir, f), "utf8"));
    const r = replay(capture);
    const verdict = r.armedAt === null ? "never armed" : r.stopAt === null ? "keeps recording" : `stops at ${(r.stopAt / 1000).toFixed(1)} s`;
    console.log(info.word, `recorded ${(capture.pose.length / capture.fps).toFixed(1)} s ->`, verdict);
  }
}
