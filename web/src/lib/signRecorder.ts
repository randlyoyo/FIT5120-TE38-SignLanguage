import { dtwDistance } from "./dtw";
import type { ReferenceSign } from "./referenceData";
import { buildSignEmbedding } from "./signModel";
import type { FrameLandmarks } from "./types";

/**
 * Port of the reference repo's `sign_recorder.py`, adapted to the proposal's
 * "How the system works" slide: instead of the desktop app's manual
 * press-"r"-to-record / press-"q"-to-stop, recording starts the moment a hand
 * appears and ends via rest-pose segmentation (a run of consecutive frames
 * with no hand detected), so a learner just performs the sign in front of
 * the webcam with no keyboard interaction.
 */

const MAX_FRAMES = 60; // safety cap (~2s at 30fps) so a lost hand-absence signal can't buffer forever
const REST_FRAMES_TO_END = 8; // ~0.25s at 30fps of no hand marks the end of a sign
const VOTE_BATCH_SIZE = 5;
const VOTE_THRESHOLD = 0.5;

export interface RecognitionResult {
  predictedId: string | null;
  predictedLabel: string;
  confidence: number;
}

export class SignRecorder {
  private recordingState = false;
  private buffer: FrameLandmarks[] = [];
  private restStreak = 0;

  get isRecording(): boolean {
    return this.recordingState;
  }

  reset(): void {
    this.recordingState = false;
    this.buffer = [];
    this.restStreak = 0;
  }

  /**
   * Feed one frame's detection results in. Returns a RecognitionResult only
   * on the frame where a recording just finished and was scored, else null.
   */
  processFrame(frame: FrameLandmarks, referenceSigns: ReferenceSign[]): RecognitionResult | null {
    const handPresent = frame.left !== null || frame.right !== null;

    if (!this.recordingState) {
      if (handPresent) {
        this.recordingState = true;
        this.buffer = [frame];
        this.restStreak = 0;
      }
      return null;
    }

    this.buffer.push(frame);
    this.restStreak = handPresent ? 0 : this.restStreak + 1;

    if (this.restStreak >= REST_FRAMES_TO_END || this.buffer.length >= MAX_FRAMES) {
      const result = this.score(referenceSigns);
      this.reset();
      return result;
    }

    return null;
  }

  private score(referenceSigns: ReferenceSign[]): RecognitionResult {
    const recorded = buildSignEmbedding(this.buffer);

    const distances = referenceSigns.map((ref) => {
      let distance = Infinity;
      if (recorded.hasLeft === ref.hasLeft && recorded.hasRight === ref.hasRight) {
        distance = 0;
        if (recorded.hasLeft) distance += dtwDistance(recorded.leftEmbedding, ref.leftEmbedding);
        if (recorded.hasRight) distance += dtwDistance(recorded.rightEmbedding, ref.rightEmbedding);
      }
      return { id: ref.id, label: ref.label, distance };
    });

    distances.sort((a, b) => a.distance - b.distance);
    const batch = distances.slice(0, Math.min(VOTE_BATCH_SIZE, distances.length));

    const counts = new Map<string, number>();
    for (const d of batch) counts.set(d.id, (counts.get(d.id) ?? 0) + 1);

    let bestId: string | null = null;
    let bestCount = 0;
    for (const [id, count] of counts) {
      if (count > bestCount) {
        bestCount = count;
        bestId = id;
      }
    }

    const confidence = batch.length > 0 ? bestCount / batch.length : 0;

    if (!bestId || !Number.isFinite(batch[0]?.distance) || confidence < VOTE_THRESHOLD) {
      return { predictedId: null, predictedLabel: "Sign not recognised", confidence };
    }

    const label = referenceSigns.find((s) => s.id === bestId)?.label ?? bestId;
    return { predictedId: bestId, predictedLabel: label, confidence };
  }
}
