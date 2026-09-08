import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

/**
 * Framing guidance: is the signer far enough away, facing the camera, and
 * fully in shot before capture starts.
 *
 * Every band below was measured on the training corpus rather than chosen,
 * because the recogniser only ever saw one framing. Over 3,600 clips from
 * Train + Test_ITW + Test_STU (p2-p98):
 *
 *     shoulder span   0.177 - 0.257   of frame HEIGHT, aspect-corrected
 *     shoulder mid x  0.491 - 0.525   of frame width
 *     shoulder mid y  0.463 - 0.539   of frame height
 *     head/shoulder   0.357 - 0.423   frontal; 0.249 - 0.502 off-axis
 *
 * Three findings drive the rules, and each contradicts an obvious guess:
 *
 * 1. Distance itself is NOT what has to be matched. The feature pipeline
 *    divides by shoulder width (recognition/API.md §7), so scale is
 *    normalised out of the model's input. What breaks at close range is
 *    that the signing space stops fitting: hands reach 1.40 shoulder
 *    widths above the shoulder midpoint (p99), so with shoulders at
 *    y~0.494 a span above ~0.35 pushes a raised hand off the top edge.
 *    At 0.6 m from a typical webcam the span is 0.69 and the hands are
 *    well outside the frame -- signing into a laptop at desk distance
 *    cannot work, at any model quality.
 *
 * 2. Head-to-shoulder ratio cannot measure distance. It is a body
 *    constant, 0.39 at every distance. It measures TURN: both spans
 *    foreshorten as the signer rotates, so the ratio leaves its narrow
 *    frontal band in either direction.
 *
 * 3. The binding constraint at the far end is hand PIXELS, not span. The
 *    training clips are all 512x408 with hands measuring 42-47 px, and
 *    even there both hands are found in only 0.63-0.79 of frames --
 *    detection is the weak link already. Below ~37 px the measured rate
 *    drops to 0.64. Since pixels depend on stream resolution, the far
 *    limit has to as well, which is why `frame` is a parameter.
 */

const NOSE = 0;
const EAR_RIGHT = 7;
const EAR_LEFT = 8;
const SHOULDER_RIGHT = 11;
const SHOULDER_LEFT = 12;

export const BANDS = {
  /** Shoulder span as a fraction of frame height, aspect-corrected. */
  span: { ideal: [0.21, 0.3], accept: [0.17, 0.34] },
  centreX: [0.4, 0.6],
  centreY: [0.4, 0.6],
  headShoulder: [0.32, 0.46],
  /** A hand landmark this close to an edge counts as leaving frame. */
  edgeMargin: 0.03,
  minVisibility: 0.5,
  /** Hand bounding-box diagonal, in pixels. */
  handPixels: { min: 38 },
} as const;

/**
 * A default getUserMedia stream is often 640x480, which at a workable
 * standing distance leaves ~53 px of hand. Asking for 720p gives ~79 px --
 * more than the training data itself had. This is the cheapest single
 * improvement available to the capture path.
 */
export const VIDEO_CONSTRAINTS: MediaStreamConstraints = {
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 },
    facingMode: "user",
  },
};

export type FramingStatus =
  | "no_person"
  | "too_close"
  | "too_far"
  | "hands_out_of_frame"
  | "turned"
  | "off_centre"
  | "ok"
  | "ideal";

export interface FramingMetrics {
  span: number;
  headShoulder: number;
  centreX: number;
  centreY: number;
  handPixels: number;
  approxMetres: number;
}

export interface FramingResult {
  status: FramingStatus;
  severity: "ok" | "warn" | "block";
  message: string;
  metrics: FramingMetrics | null;
}

export interface FrameSize {
  width: number;
  height: number;
}

// Only used to turn a span into a friendly metre figure for the user. Real
// webcams run 60-80 deg horizontally and the browser never reports which,
// so this is an estimate by construction and the copy rounds it hard rather
// than presenting it as a reading.
const SHOULDER_METRES = 0.328; // measured on this corpus's world landmarks
const ASSUMED_HFOV_DEG = 70;

function verticalFovFactor(aspect: number): number {
  const h = (ASSUMED_HFOV_DEG * Math.PI) / 180;
  const v = 2 * Math.atan(Math.tan(h / 2) / aspect);
  return 2 * Math.tan(v / 2);
}

/** Largest hand bounding-box diagonal in pixels, or null if no hand is seen. */
function handDiagonalPx(
  hands: (NormalizedLandmark[] | null | undefined)[],
  frame: FrameSize,
): number | null {
  let best: number | null = null;
  for (const hand of hands) {
    if (!hand || hand.length === 0) continue;
    const xs = hand.map((p) => p.x);
    const ys = hand.map((p) => p.y);
    const dx = (Math.max(...xs) - Math.min(...xs)) * frame.width;
    const dy = (Math.max(...ys) - Math.min(...ys)) * frame.height;
    const diagonal = Math.hypot(dx, dy);
    if (best === null || diagonal > best) best = diagonal;
  }
  return best;
}

export function checkFraming(
  pose: NormalizedLandmark[] | null | undefined,
  frame: FrameSize,
  leftHand: NormalizedLandmark[] | null = null,
  rightHand: NormalizedLandmark[] | null = null,
): FramingResult {
  const required = [NOSE, EAR_RIGHT, EAR_LEFT, SHOULDER_RIGHT, SHOULDER_LEFT];
  const missing =
    !pose ||
    required.some(
      (i) => !pose[i] || (pose[i].visibility ?? 1) < BANDS.minVisibility,
    );
  if (missing) {
    return {
      status: "no_person",
      severity: "block",
      message: "没有检测到人，请正对摄像头",
      metrics: null,
    };
  }
  const points = pose as NormalizedLandmark[];

  // MediaPipe divides x by width and y by height independently, so a raw dx
  // is not comparable with a raw dy -- on a portrait stream the error is
  // over 2x. Scaling x by the aspect ratio puts both in units of frame
  // height, which is also what the feature pipeline does (API.md §7).
  const aspect = frame.width / frame.height;
  const distance = (a: NormalizedLandmark, b: NormalizedLandmark) =>
    Math.hypot((a.x - b.x) * aspect, a.y - b.y);

  const span = distance(points[SHOULDER_RIGHT], points[SHOULDER_LEFT]);
  const headWidth = distance(points[EAR_RIGHT], points[EAR_LEFT]);
  const centreX = (points[SHOULDER_RIGHT].x + points[SHOULDER_LEFT].x) / 2;
  const centreY = (points[SHOULDER_RIGHT].y + points[SHOULDER_LEFT].y) / 2;
  const headShoulder = span > 0 ? headWidth / span : NaN;

  // Hands measure 0.11 of frame height at the training framing of span 0.21,
  // so when no hand is detected yet the span still predicts their size.
  const handPixels =
    handDiagonalPx([leftHand, rightHand], frame) ??
    (0.11 / 0.21) * span * frame.height;

  const metrics: FramingMetrics = {
    span,
    headShoulder,
    centreX,
    centreY,
    handPixels,
    approxMetres: SHOULDER_METRES / (span * verticalFovFactor(aspect)),
  };

  const handOutOfFrame = [leftHand, rightHand].some(
    (hand) =>
      hand &&
      hand.some(
        (p) =>
          p.x < BANDS.edgeMargin ||
          p.x > 1 - BANDS.edgeMargin ||
          p.y < BANDS.edgeMargin ||
          p.y > 1 - BANDS.edgeMargin,
      ),
  );

  // Ordered by what the user should fix FIRST. Distance dominates: at 0.6 m
  // the hands are already out of frame and the pose is perspective-distorted,
  // so "move left a bit" would be a wasted instruction.
  const [nearest, farthest] = BANDS.span.accept;
  if (span > farthest) {
    const target = Math.max(1.5, metrics.approxMetres + 0.8);
    return {
      status: "too_close",
      severity: "block",
      message: `太近了，请后退到约 ${target.toFixed(1)} 米`,
      metrics,
    };
  }
  if (span < nearest || handPixels < BANDS.handPixels.min) {
    return {
      status: "too_far",
      severity: "block",
      message:
        handPixels < BANDS.handPixels.min
          ? "太远了，手看不清，请靠近一点"
          : "太远了，请靠近一点",
      metrics,
    };
  }
  if (handOutOfFrame) {
    return {
      status: "hands_out_of_frame",
      severity: "block",
      message: "手超出画面，请后退或把摄像头调远",
      metrics,
    };
  }
  if (
    headShoulder < BANDS.headShoulder[0] ||
    headShoulder > BANDS.headShoulder[1]
  ) {
    return {
      status: "turned",
      severity: "block",
      message: "请正对摄像头，不要侧身",
      metrics,
    };
  }
  if (centreX < BANDS.centreX[0] || centreX > BANDS.centreX[1]) {
    return {
      status: "off_centre",
      severity: "warn",
      message: centreX < 0.5 ? "请向右移动" : "请向左移动",
      metrics,
    };
  }
  if (centreY < BANDS.centreY[0] || centreY > BANDS.centreY[1]) {
    return {
      status: "off_centre",
      severity: "warn",
      message: centreY < 0.5 ? "摄像头请调低一点" : "摄像头请调高一点",
      metrics,
    };
  }

  const [idealLow, idealHigh] = BANDS.span.ideal;
  return {
    status: span >= idealLow && span <= idealHigh ? "ideal" : "ok",
    severity: "ok",
    message: "位置合适，可以开始",
    metrics,
  };
}

/**
 * Framing has to HOLD before capture starts: a single good frame happens
 * while someone is still walking backwards. Feed every frame in.
 */
export function makeStabiliser(framesRequired = 15) {
  let good = 0;
  return (result: FramingResult) => {
    good = result.severity === "ok" ? good + 1 : 0;
    return {
      ...result,
      ready: good >= framesRequired,
      progress: Math.min(1, good / framesRequired),
    };
  };
}
