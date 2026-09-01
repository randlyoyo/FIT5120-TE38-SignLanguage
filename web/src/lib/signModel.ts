import { handFeatureVector } from "./handModel";
import type { FrameLandmarks } from "./types";

/**
 * Port of the reference repo's `models/sign_model.py`: turns a whole recording
 * (sequence of per-frame landmarks) into two embeddings — one per hand — made
 * up of the feature vectors of only the frames where that hand was present.
 */
export interface SignEmbedding {
  hasLeft: boolean;
  hasRight: boolean;
  leftEmbedding: number[][];
  rightEmbedding: number[][];
}

export function buildSignEmbedding(frames: FrameLandmarks[]): SignEmbedding {
  const leftEmbedding: number[][] = [];
  const rightEmbedding: number[][] = [];

  for (const frame of frames) {
    const left = handFeatureVector(frame.left);
    if (left) leftEmbedding.push(left);

    const right = handFeatureVector(frame.right);
    if (right) rightEmbedding.push(right);
  }

  return {
    hasLeft: leftEmbedding.length > 0,
    hasRight: rightEmbedding.length > 0,
    leftEmbedding,
    rightEmbedding,
  };
}
