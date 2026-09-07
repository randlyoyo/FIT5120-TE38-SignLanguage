import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { Capture } from "../api/recognize";

// Pinned to match the exact bundle the training keypoints were extracted
// with (recognition/API.md §2, recognition/ASSETS.md) -- a different
// version shifts the landmark distribution in a way that looks like a
// model problem, not a version problem. Do not bump without re-reading
// both documents.
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.33/wasm";
// Loaded straight from Google's CDN, not self-hosted -- there's no local
// copy to deploy (recognition/ASSETS.md). Keep the "/1/" in the path: the
// same doc explicitly warns that "latest" would silently move to a
// different bundle and shift the landmark distribution without ever
// erroring. sha256 e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8
// is the exact bundle the training keypoints were extracted with.
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task";

// recognition/API.md §7: the stored feature layout carries an 11-point pose
// subset `[0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]` of MediaPipe's 33, of
// which only entries 5..10 (this array) are ever read by the model. The
// other five exist only to match that layout and are always the "missing"
// sentinel below.
const ARM_POSE_INDICES = [11, 12, 13, 14, 15, 16] as const;
const POSE_SUBSET_LENGTH = 11;
const HAND_LENGTH = 21;

// `(0, 0)` is a legal on-frame coordinate (top-left corner) that the model
// would read as a real position, so a missing landmark must be a sentinel
// the feature pipeline can recognise as absent, not zero (API.md §2, §6).
const MISSING: [number, number] = [-999, -999];

function filled(length: number): [number, number][] {
  return Array.from({ length }, () => MISSING);
}

// HolisticLandmarkerResult's fields are NormalizedLandmark[][] -- one entry
// per detected instance -- even though Holistic tracks a single subject, so
// the real landmark array is the first (only) element, not the field
// itself. Verified against the package's own vision.d.ts rather than
// assumed, since a flat-vs-nested mixup here would silently misread every
// point without ever throwing.
function posePoints(poseLandmarks: NormalizedLandmark[][]): [number, number][] {
  const pose = poseLandmarks[0];
  if (!pose || pose.length === 0) return filled(POSE_SUBSET_LENGTH);
  const points = filled(POSE_SUBSET_LENGTH);
  ARM_POSE_INDICES.forEach((fullIndex, i) => {
    const lm = pose[fullIndex];
    if (lm) points[5 + i] = [lm.x, lm.y];
  });
  return points;
}

function handPoints(handLandmarks: NormalizedLandmark[][]): [number, number][] {
  const hand = handLandmarks[0];
  if (!hand || hand.length === 0) return filled(HAND_LENGTH);
  return hand.map((lm) => [lm.x, lm.y]);
}

// Face landmarks are collected but NOT used for recognition -- the model reads
// arms and hands only (API.md §2). They are here because sign language marks
// questions, negation and topics with non-manual markers rather than with the
// hands, and that work needs real user captures to train on. Captures cannot be
// re-recorded after the fact, so the cheap moment to start collecting is
// before there is anything to use them for.
//
// A 32-point subset of the 478-point mesh, chosen for those markers rather than
// for appearance. Sending the full mesh would roughly triple the request body
// for points that carry no linguistic signal.
const FACE_INDICES = [
  // outer lip contour -- mouth shape (12)
  61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
  // eyebrows, inner/mid/outer each side -- raised for polar questions,
  // furrowed for wh-questions (6)
  107, 105, 70, 336, 334, 300,
  // eyelids, upper and lower each side -- aperture, squinting (4)
  159, 145, 386, 374,
  // eye corners (4). Brow height is only meaningful relative to the eye, and
  // the coarse single eye point in the body pose is not a precise enough
  // reference for a few pixels of brow movement.
  33, 133, 362, 263,
  // nose tip and bridge -- head pose anchor for nods and tilts (2)
  1, 4,
  // inner lips -- mouth opening, distinct from the outer contour (4)
  13, 14, 78, 308,
] as const;

function facePoints(faceLandmarks: NormalizedLandmark[][]): [number, number][] {
  const face = faceLandmarks[0];
  if (!face || face.length === 0) return filled(FACE_INDICES.length);
  return FACE_INDICES.map((i) => {
    const lm = face[i];
    return lm ? ([lm.x, lm.y] as [number, number]) : MISSING;
  });
}

// Indices of the left/right wrist within the 11-point pose array assembled
// above (positions 5..10 are the six arm points in the order §2 lists:
// shoulders, elbows, wrists -- so wrists are the last two).
const LEFT_WRIST_IDX = 9;
const RIGHT_WRIST_IDX = 10;

// Same signal the server's own offline trim uses (API.md §3.5: "speed[t] =
// max over both wrists of |p[t+1] - p[t]|"), just evaluated live frame by
// frame instead of over a whole recorded clip. Values are normalised [0,1]
// screen-fraction coordinates, so this is a per-frame displacement of ~1.5%
// of the frame's width/height -- a starting point, not a tuned constant
// (API.md §9: real webcam behaviour hasn't been measured yet, so keep
// thresholds configurable rather than trusting a value picked in a vacuum).
const MOTION_THRESHOLD = 0.015;
// How long the signer must hold still before capture auto-ends. Modelled on
// voice-activity-detection "hangover" periods: end-of-speech/end-of-gesture
// is never declared on the first quiet frame, since a real sign has brief
// internal pauses that must not be mistaken for "finished".
const SILENCE_HANGOVER_MS = 700;
// No auto-stop before this much has elapsed, even if the signer is already
// still -- covers the moment right after the countdown where hands haven't
// been raised into frame yet, which would otherwise read as "already done".
const MIN_CAPTURE_MS = 900;

function wristDisplacement(a: [number, number], b: [number, number]): number {
  if (a[0] === MISSING[0] || b[0] === MISSING[0]) return 0;
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

let landmarkerPromise: Promise<HolisticLandmarker> | null = null;

// detectForVideo's timestamps must be strictly increasing for the *whole
// lifetime of one landmarker instance*, not just within one capture -- the
// landmarker is a module-level singleton reused across every attempt, so
// this counter has to be too. Resetting it to ~0 at the start of each
// captureLandmarks() call (as if the landmarker were fresh each time) is
// exactly what produced "Packet timestamp mismatch ... expected 7967001 but
// received 0" on a second attempt: real bug, not an environment issue.
let lastFedTimestamp = -1;

function nextFedTimestamp(): number {
  const t = Math.max(Math.round(performance.now()), lastFedTimestamp + 1);
  lastFedTimestamp = t;
  return t;
}

/** Lazily creates the one shared landmarker (VIDEO mode, per API.md §2 --
 *  IMAGE mode gives jumpier coordinates and breaks train/inference
 *  consistency). Can reject if the CDN model fetch fails (offline, CDN
 *  outage, or a firewall blocking storage.googleapis.com); callers should
 *  surface that as "practice couldn't start" rather than a generic error. */
function getLandmarker(): Promise<HolisticLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = FilesetResolver.forVisionTasks(WASM_URL).then((fileset) =>
      HolisticLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: "VIDEO",
      })
    );
  }
  return landmarkerPromise;
}

export type CapturePhase = "waiting" | "active" | "settling";

export interface CaptureOptions {
  /** Hard upper bound, in milliseconds -- capture always stops here even if
   *  the signer never goes still (or the motion signal never trusts them
   *  as such). API.md §8 recommends keeping the whole window within 2-5s;
   *  this is deliberately more generous than that, since it's a safety net
   *  rather than the target duration -- see auto-stop below. */
  maxDurationMs: number;
  /** Escape hatch for a caller-driven early stop (e.g. a manual override
   *  button), polled once per frame alongside the automatic motion-based
   *  stop below. Optional -- auto-stop alone is the primary mechanism. */
  shouldStop?: () => boolean;
  /** Called once per detected frame, purely for UI feedback -- not used in
   *  the capture decision itself. `phase` mirrors the auto-stop state
   *  machine: "waiting" (no motion seen yet), "active" (currently moving,
   *  or too recently still to end), "settling" (still, and the auto-stop
   *  hangover countdown is running). */
  onFrame?: (elapsedMs: number, maxDurationMs: number, phase: CapturePhase) => void;
}

/**
 * Records landmarks from `video` and assembles them into the exact Capture
 * shape recognition/API.md §6 expects. Ends automatically once the signer
 * goes still after having moved (the same wrist-speed signal the server's
 * own offline trim uses, §3.5, run live instead of after the fact) rather
 * than a fixed recording length or a manual "done" action -- see the
 * threshold/hangover constants above for the exact rule. `video` must be
 * the raw, unmirrored camera feed -- if the on-screen preview is mirrored
 * for the user, that must be a CSS transform on the display only. A
 * mirrored tensor swaps left and right hands and the model scores a
 * correct sign as wrong (API.md §2).
 */
export async function captureLandmarks(
  video: HTMLVideoElement,
  { maxDurationMs, shouldStop, onFrame }: CaptureOptions
): Promise<Capture> {
  const landmarker = await getLandmarker();

  const pose: number[][][] = [];
  const leftHand: number[][][] = [];
  const rightHand: number[][][] = [];
  const face: number[][][] = [];

  const start = performance.now();
  let prevPoseFrame: [number, number][] | null = null;
  let hasMoved = false;
  let lastMotionAt = start;

  // The stop decision below reacts to every frame's raw motion reading, as
  // it should. But landmark jitter means that raw reading flips back and
  // forth across MOTION_THRESHOLD within a single real pause (a signer's
  // hand is never perfectly still), which used to feed straight into
  // `onFrame`'s phase and made the on-screen "Recording…"/"Got it…" text
  // flicker every couple of frames. Only forward a phase change to the
  // caller once the raw reading has held steady for PHASE_DISPLAY_DEBOUNCE_MS
  // -- display-only smoothing, the stop timing above is untouched.
  const PHASE_DISPLAY_DEBOUNCE_MS = 200;
  let displayPhase: CapturePhase = "waiting";
  let pendingPhase: CapturePhase | null = null;
  let pendingSince = start;

  for (;;) {
    const now = performance.now();
    const elapsed = now - start;
    if (elapsed >= maxDurationMs || shouldStop?.()) break;

    const result = landmarker.detectForVideo(video, nextFedTimestamp());
    const poseFrame = posePoints(result.poseLandmarks);
    pose.push(poseFrame);
    leftHand.push(handPoints(result.leftHandLandmarks));
    rightHand.push(handPoints(result.rightHandLandmarks));
    face.push(facePoints(result.faceLandmarks));

    if (prevPoseFrame) {
      const speed = Math.max(
        wristDisplacement(poseFrame[LEFT_WRIST_IDX], prevPoseFrame[LEFT_WRIST_IDX]),
        wristDisplacement(poseFrame[RIGHT_WRIST_IDX], prevPoseFrame[RIGHT_WRIST_IDX])
      );
      if (speed > MOTION_THRESHOLD) {
        hasMoved = true;
        lastMotionAt = now;
      }
    }
    prevPoseFrame = poseFrame;

    const stillFor = now - lastMotionAt;
    const rawPhase: CapturePhase = !hasMoved ? "waiting" : stillFor > 150 ? "settling" : "active";
    if (rawPhase === displayPhase) {
      pendingPhase = null;
    } else if (rawPhase !== pendingPhase) {
      pendingPhase = rawPhase;
      pendingSince = now;
    } else if (now - pendingSince >= PHASE_DISPLAY_DEBOUNCE_MS) {
      displayPhase = rawPhase;
      pendingPhase = null;
    }
    onFrame?.(elapsed, maxDurationMs, displayPhase);

    if (hasMoved && elapsed >= MIN_CAPTURE_MS && stillFor >= SILENCE_HANGOVER_MS) break;

    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }

  const elapsedSeconds = (performance.now() - start) / 1000;
  return {
    width: video.videoWidth,
    height: video.videoHeight,
    // Measured, not assumed -- API.md §3.2 rejects fps outside [5, 120] and
    // resamples against whatever the browser's real capture rate turned
    // out to be, which varies by device.
    fps: pose.length / elapsedSeconds,
    pose,
    left_hand: leftHand,
    right_hand: rightHand,
    face,
  };
}
