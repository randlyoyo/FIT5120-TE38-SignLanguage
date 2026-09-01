import { HandLandmarker } from "@mediapipe/tasks-vision";
import type { HandLandmarks, Point3D } from "./types";

/**
 * Port of the reference repo's `models/hand_model.py`: HAND_MODEL feature vector
 * is the angle between every pair of hand connection vectors, which makes it
 * invariant to hand position, orientation and scale (only relative joint
 * angles matter, not where the hand is in the frame).
 */

const CONNECTIONS: [number, number][] = HandLandmarker.HAND_CONNECTIONS.map(
  (c) => [c.start, c.end],
);

function connectionVector(landmarks: Point3D[], from: number, to: number): Point3D {
  const a = landmarks[from];
  const b = landmarks[to];
  return { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
}

function angleBetween(u: Point3D, v: Point3D): number {
  const dot = u.x * v.x + u.y * v.y + u.z * v.z;
  const normU = Math.hypot(u.x, u.y, u.z);
  const normV = Math.hypot(v.x, v.y, v.z);
  if (normU === 0 || normV === 0) return 0;
  // Clamp for floating point safety before acos (avoids NaN from |cos| slightly > 1).
  const cos = Math.min(1, Math.max(-1, dot / (normU * normV)));
  return Math.acos(cos);
}

/**
 * Returns a feature vector of length `CONNECTIONS.length ** 2` containing the
 * angle between every pair of hand connections for one hand in one frame.
 */
export function handFeatureVector(landmarks: HandLandmarks): number[] | null {
  if (!landmarks) return null;

  const vectors = CONNECTIONS.map(([from, to]) => connectionVector(landmarks, from, to));

  const angles: number[] = [];
  for (const u of vectors) {
    for (const v of vectors) {
      angles.push(angleBetween(u, v));
    }
  }
  return angles;
}
