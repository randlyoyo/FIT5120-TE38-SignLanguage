/**
 * Convert the exported template bank (numpy .npy) to the flat float32 blob the
 * API serves.
 *
 *   node recognition/scripts/buildBank.mjs /path/to/signtest/export/bank.npy
 *
 * Output: recognition/models/bank.f32 -- (3215, 12, 256) float32, C order.
 * Kept out of git; see ASSETS.md.
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
// npy v1: magic(6) major(1) minor(1) headerLen(2 LE) then an ASCII dict
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

const body = buf.subarray(10 + headerLen);
const [W, K, D] = shape;
const expected = W * K * D * 4;
if (body.length !== expected) {
  console.error(`size mismatch: header says ${shape.join("x")} = ${expected} bytes, body has ${body.length}`);
  process.exit(1);
}

const out = join(dirname(fileURLToPath(import.meta.url)), "..", "models", "bank.f32");
writeFileSync(out, body);

const index = JSON.parse(readFileSync(join(dirname(out), "bank_index.json")));
const ok = index.words.length === W && index.dim === D && index.n_template === K
  && index.bytes_per_word === K * D * 4;
console.log(`wrote ${out}`);
console.log(`  shape ${W} words x ${K} templates x ${D} dims, ${(body.length / 1e6).toFixed(1)} MB`);
console.log(`  offset(word) = index * ${K * D * 4} bytes`);
console.log(ok ? "  matches bank_index.json" : "  !! DISAGREES with bank_index.json -- do not deploy");
process.exit(ok ? 0 : 1);
