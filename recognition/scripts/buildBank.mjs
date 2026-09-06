/**
 * Rebuild the deployed template bank from the training export.
 *
 *   node recognition/scripts/buildBank.mjs /path/to/signtest/export/bank.npy
 *
 * Output: recognition/models/bank.i8 -- (3215, 12, 256) int8.
 *
 * You should not normally need this. bank.i8 is committed, because Vercel and
 * Railway both deploy from the repository and a build-time fetch would be one
 * more thing to fail on two platforms. Run this only after retraining.
 *
 * int8 rather than the float32 the encoder emits: 9.9 MB instead of 39.5 MB,
 * with no measurable cost. Measured over three splits, quantising moved AUC by
 * 0.0000, EER by at most 0.01 points and top-4 not at all -- the templates are
 * unit vectors, so 127 levels per axis is ample.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const src = process.argv[2];
if (!src) {
  console.error("usage: node buildBank.mjs <path to bank.npy>");
  process.exit(1);
}

const buf = readFileSync(src);
if (buf.subarray(0, 6).toString("latin1") !== "\x93NUMPY") {
  console.error("not a .npy file");
  process.exit(1);
}
const headerLen = buf.readUInt16LE(8);
const header = buf.subarray(10, 10 + headerLen).toString("latin1");
const shape = [...header.matchAll(/(\d+)/g)].map((m) => Number(m[1])).slice(-3);
const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
if (descr !== "<f4") {
  console.error(`expected little-endian float32 ('<f4'), got ${descr}`);
  process.exit(1);
}
if (/'fortran_order':\s*True/.test(header)) {
  console.error("fortran_order arrays are not supported; re-save C-ordered");
  process.exit(1);
}

const [W, K, D] = shape;
const body = buf.subarray(10 + headerLen);
if (body.length !== W * K * D * 4) {
  console.error(`size mismatch: header says ${shape.join("x")}, body has ${body.length} bytes`);
  process.exit(1);
}
const f32 = new Float32Array(body.buffer, body.byteOffset, W * K * D);

// Symmetric quantisation; the decoder divides by 127 and re-normalises.
const out = Buffer.alloc(W * K * D);
let clipped = 0;
for (let i = 0; i < f32.length; i++) {
  let v = Math.round(f32[i] * 127);
  if (v > 127) { v = 127; clipped++; } else if (v < -127) { v = -127; clipped++; }
  out.writeInt8(v, i);
}

const dst = join(dirname(fileURLToPath(import.meta.url)), "..", "models", "bank.i8");
writeFileSync(dst, out);

const index = JSON.parse(readFileSync(join(dirname(dst), "bank_index.json")));
const ok = index.words.length === W && index.dim === D && index.n_template === K
  && index.bytes_per_word === K * D && index.dtype === "int8";
console.log(`wrote ${dst}`);
console.log(`  ${W} words x ${K} templates x ${D} dims int8, ${(out.length / 1e6).toFixed(1)} MB`);
console.log(`  offset(word) = index * ${K * D} bytes`);
console.log(`  clipped ${clipped} of ${f32.length} values`);
console.log(ok ? "  matches bank_index.json" : "  !! DISAGREES with bank_index.json -- do not deploy");
process.exit(ok ? 0 : 1);
