#!/usr/bin/env node
// Generates a tiny set of SYNTHETIC placeholder hand-landmark sequences so the
// full learn/quiz/practice pipeline runs end-to-end before any real, licensed
// Auslan footage is available.
//
// IMPORTANT: these are procedurally generated hand motions, not real Auslan
// signs. They are deliberately labelled "Demo Sign A/B/..." rather than any
// real Auslan word, because the proposal itself flags "online self-learning
// could teach incorrect signs" as a named risk — every real demonstration
// must be checked by an Auslan-fluent signer before being presented to a
// learner as a real sign. Replace this dataset via the in-app /record tool
// (or a proper offline pipeline) using consented, verified recordings.
//
// Output: web/public/data/signs/manifest.json + one <id>.json per sign,
// matching the FrameLandmarks/SignTemplate shape consumed by
// src/lib/referenceData.ts.

import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, "..", "public", "data", "signs");

const FPS = 30;

// A plausible open-palm-facing-camera pose, normalized image coordinates
// (x right, y down, z ~depth). Indices follow the standard 21-point MediaPipe
// hand landmark layout (0 = wrist, 1-4 thumb, 5-8 index, 9-12 middle,
// 13-16 ring, 17-20 pinky).
const OPEN_HAND = [
  { x: 0.5, y: 0.85, z: 0 }, // 0 wrist
  { x: 0.42, y: 0.78, z: 0 }, // 1 thumb CMC
  { x: 0.36, y: 0.7, z: 0 }, // 2 thumb MCP
  { x: 0.32, y: 0.62, z: 0 }, // 3 thumb IP
  { x: 0.29, y: 0.55, z: 0 }, // 4 thumb TIP
  { x: 0.44, y: 0.55, z: 0 }, // 5 index MCP
  { x: 0.43, y: 0.42, z: 0 }, // 6 index PIP
  { x: 0.42, y: 0.32, z: 0 }, // 7 index DIP
  { x: 0.41, y: 0.23, z: 0 }, // 8 index TIP
  { x: 0.5, y: 0.53, z: 0 }, // 9 middle MCP
  { x: 0.5, y: 0.38, z: 0 }, // 10 middle PIP
  { x: 0.5, y: 0.27, z: 0 }, // 11 middle DIP
  { x: 0.5, y: 0.17, z: 0 }, // 12 middle TIP
  { x: 0.56, y: 0.55, z: 0 }, // 13 ring MCP
  { x: 0.57, y: 0.4, z: 0 }, // 14 ring PIP
  { x: 0.58, y: 0.29, z: 0 }, // 15 ring DIP
  { x: 0.58, y: 0.2, z: 0 }, // 16 ring TIP
  { x: 0.62, y: 0.58, z: 0 }, // 17 pinky MCP
  { x: 0.64, y: 0.46, z: 0 }, // 18 pinky PIP
  { x: 0.65, y: 0.37, z: 0 }, // 19 pinky DIP
  { x: 0.66, y: 0.29, z: 0 }, // 20 pinky TIP
];

// Finger tip/joint chains used when curling fingers toward the palm.
const FINGERS = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpPoint(a, b, t) {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), z: lerp(a.z, b.z, t) };
}

/** Curl one finger's PIP/DIP/TIP joints toward its MCP joint by fraction `t` (0=open, 1=fully curled). */
function curlFinger(hand, [mcp, pip, dip, tip], t) {
  const mcpPoint = hand[mcp];
  hand[pip] = lerpPoint(hand[pip], mcpPoint, t * 0.6);
  hand[dip] = lerpPoint(hand[dip], mcpPoint, t * 0.85);
  hand[tip] = lerpPoint(hand[tip], mcpPoint, t * 0.95);
}

function translateHand(hand, dx, dy) {
  return hand.map((p) => ({ x: p.x + dx, y: p.y + dy, z: p.z }));
}

function cloneHand(hand) {
  return hand.map((p) => ({ ...p }));
}

function mirrorHand(hand) {
  // Mirrors across the vertical center line, for a plausible "other hand" pose.
  return hand.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z }));
}

/** Demo Sign A — open hand pulses into a fist and back open, right hand only. */
function generateOpenClose(frameCount) {
  const frames = [];
  for (let i = 0; i < frameCount; i++) {
    const t = i / (frameCount - 1);
    const curl = Math.sin(Math.PI * t); // 0 -> 1 -> 0
    const hand = cloneHand(OPEN_HAND);
    for (const chain of Object.values(FINGERS)) curlFinger(hand, chain, curl);
    frames.push({ left: null, right: hand });
  }
  return frames;
}

/** Demo Sign B — open hand waves side to side, right hand only. */
function generateSideWave(frameCount) {
  const frames = [];
  for (let i = 0; i < frameCount; i++) {
    const t = i / (frameCount - 1);
    const dx = 0.15 * Math.sin(2 * Math.PI * t * 2);
    const hand = translateHand(OPEN_HAND, dx, 0);
    frames.push({ left: null, right: hand });
  }
  return frames;
}

/** Demo Sign C — index finger stays extended, other fingers curled, held with a slight bob. */
function generatePointHold(frameCount) {
  const frames = [];
  for (let i = 0; i < frameCount; i++) {
    const t = i / (frameCount - 1);
    const dy = 0.02 * Math.sin(2 * Math.PI * t * 3);
    const hand = cloneHand(OPEN_HAND);
    curlFinger(hand, FINGERS.thumb, 0.8);
    curlFinger(hand, FINGERS.middle, 1);
    curlFinger(hand, FINGERS.ring, 1);
    curlFinger(hand, FINGERS.pinky, 1);
    frames.push({ left: null, right: translateHand(hand, 0, dy) });
  }
  return frames;
}

/** Demo Sign D — both hands raised and waving together. */
function generateTwoHandWave(frameCount) {
  const frames = [];
  for (let i = 0; i < frameCount; i++) {
    const t = i / (frameCount - 1);
    const dx = 0.08 * Math.sin(2 * Math.PI * t * 2);
    const dy = -0.1;
    const right = translateHand(OPEN_HAND, dx, dy);
    const left = translateHand(mirrorHand(OPEN_HAND), -dx, dy);
    frames.push({ left, right });
  }
  return frames;
}

const SIGNS = [
  { id: "demo-sign-a", label: "Demo Sign A", generator: generateOpenClose, frameCount: 40 },
  { id: "demo-sign-b", label: "Demo Sign B", generator: generateSideWave, frameCount: 40 },
  { id: "demo-sign-c", label: "Demo Sign C", generator: generatePointHold, frameCount: 36 },
  { id: "demo-sign-d", label: "Demo Sign D", generator: generateTwoHandWave, frameCount: 40 },
];

mkdirSync(OUT_DIR, { recursive: true });

const manifest = SIGNS.map(({ id, label }) => ({ id, label }));
writeFileSync(join(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2));

for (const sign of SIGNS) {
  const frames = sign.generator(sign.frameCount);
  const template = { id: sign.id, label: sign.label, frames };
  writeFileSync(join(OUT_DIR, `${sign.id}.json`), JSON.stringify(template));
  console.log(`Wrote ${sign.id}.json (${frames.length} frames @ ${FPS}fps)`);
}

console.log(`\nDone. ${SIGNS.length} placeholder signs written to ${OUT_DIR}`);
