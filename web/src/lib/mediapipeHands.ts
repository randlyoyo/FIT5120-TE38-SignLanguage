import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import type { FrameLandmarks } from "./types";

/**
 * Thin wrapper around @mediapipe/tasks-vision's HandLandmarker, running fully
 * in-browser via WASM. This is the direct client-side replacement for the
 * reference repo's `mediapipe.solutions.holistic` (Python/OpenCV) call — no
 * video frame ever leaves the device, matching the proposal's privacy
 * requirement ("process frames in-browser, store no video").
 *
 * Only the hand landmarks are used (not full Holistic/pose): the ported
 * recognition math only ever consumes hand landmarks, so tracking pose here
 * would just cost extra model weight for no benefit.
 */

// Version pinned to match the installed @mediapipe/tasks-vision package (package.json).
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_ASSET =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let landmarkerPromise: Promise<HandLandmarker> | null = null;

function createLandmarker(): Promise<HandLandmarker> {
  landmarkerPromise ??= FilesetResolver.forVisionTasks(WASM_BASE).then((fileset) =>
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: MODEL_ASSET,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    }),
  );
  return landmarkerPromise;
}

export async function getHandLandmarker(): Promise<HandLandmarker> {
  return createLandmarker();
}

/**
 * Runs hand detection on the current video frame and maps the result into
 * our FrameLandmarks shape (left/right, or null when that hand isn't seen).
 *
 * Note: MediaPipe's handedness labels describe the subject's own hand (as if
 * looking through the camera at the viewer), which is why a typical mirrored
 * "selfie view" webcam preview still lines up with the reported label.
 */
export function detectFrame(
  landmarker: HandLandmarker,
  video: HTMLVideoElement,
  timestampMs: number,
): FrameLandmarks {
  const result = landmarker.detectForVideo(video, timestampMs);

  const frame: FrameLandmarks = { left: null, right: null };

  result.handedness.forEach((handedness, i) => {
    const label = handedness[0]?.categoryName;
    const landmarks = result.landmarks[i];
    if (!landmarks) return;

    const points = landmarks.map((l) => ({ x: l.x, y: l.y, z: l.z }));
    if (label === "Left") frame.left = points;
    else if (label === "Right") frame.right = points;
  });

  return frame;
}
