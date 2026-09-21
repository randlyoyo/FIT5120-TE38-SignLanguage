/**
 * Framing guidance for sign capture.
 *
 * The recogniser was trained on 83,590 clips shot at essentially ONE framing.
 * Measured over Train + Test_ITW + Test_STU (3,600 clips, p2-p98):
 *
 *     shoulder span   0.177 - 0.257   (of frame HEIGHT, aspect-corrected)
 *     shoulder mid x  0.491 - 0.525   (of frame width)
 *     shoulder mid y  0.463 - 0.539   (of frame height)
 *     nose y          0.363 - 0.392
 *     head/shoulder   0.357 - 0.423   frontal;  0.249 - 0.502 on off-axis MTV
 *
 * Two things follow, and they are the reason this file exists:
 *
 * 1. That shoulder span corresponds to roughly 1.6-2.3 m from a typical webcam.
 *    A person sitting at a laptop is at 0.5-0.7 m -- about THREE TIMES too
 *    close, and completely outside anything the model has seen. Users must be
 *    told to step back; they will not guess this.
 *
 * 2. Head-to-shoulder ratio cannot measure distance -- it is a body constant
 *    (0.39 at every distance). It measures TURN: both spans foreshorten when
 *    the signer rotates, so the ratio leaves its narrow frontal band in either
 *    direction. On the multi-view split it runs 0.249-0.502 against a frontal
 *    0.357-0.423.
 *
 * The dataset bands above are studio-tight. The ACCEPT bands below are
 * deliberately wider: holding a real user to a controlled capture's p2-p98
 * would reject nearly everyone. The widening is an engineering judgement, not
 * a measurement -- revisit it once there is real user data to fit it to.
 */

// MediaPipe Pose landmark indices.
const NOSE = 0, EAR_R = 7, EAR_L = 8, SHOULDER_R = 11, SHOULDER_L = 12;

/** Largest hand bounding-box diagonal in pixels, or null if no hand is seen. */
function handDiagonalPx(hands, frame) {
  let best = null;
  for (const h of hands) {
    if (!h || h.length === 0) continue;
    const xs = h.map((p) => p.x), ys = h.map((p) => p.y);
    const dx = (Math.max(...xs) - Math.min(...xs)) * frame.width;
    const dy = (Math.max(...ys) - Math.min(...ys)) * frame.height;
    const d = Math.hypot(dx, dy);
    if (best === null || d > best) best = d;
  }
  return best;
}

export const BANDS = {
  // Closeness is bounded by the signing space leaving frame, NOT by matching
  // the capture distance. Measured over Train + Test_ITW, hands reach (p99)
  // 1.40 shoulder widths above the shoulder midpoint and 2.03 to the side;
  // with the shoulders at y~0.494 that allows a span of 0.35 before a hand
  // crosses the top edge and 0.44 before it crosses a 16:9 side edge.
  //
  // Being closer than the training capture is not itself a problem: the
  // feature pipeline divides by shoulder width, so distance is normalised out
  // of the model's input. Closer is in fact BETTER, because it is the only
  // way to buy hand pixels, and hand pixels are what detection runs on.
  span: { ideal: [0.21, 0.30], accept: [0.17, 0.34] },
  centreX: [0.40, 0.60],
  centreY: [0.40, 0.60],
  headShoulder: [0.32, 0.46],
  edgeMargin: 0.03,
  minVisibility: 0.5,

  // Hand landmark bounding-box diagonal, in pixels. The training clips are all
  // 512x408 and their hands measure 42-47 px, at which both hands are found in
  // only 0.63-0.79 of frames -- detection is already the weak link IN THE
  // STUDIO. Below ~37 px the measured rate falls to 0.64, so that is the floor.
  handPixels: { min: 38, good: 60 },
};

/**
 * The single highest-value line in this file. A default getUserMedia stream is
 * often 640x480, which at a sensible standing distance leaves ~53 px of hand.
 * Asking for 720p nearly doubles that, to more than the training data had.
 */
export const VIDEO_CONSTRAINTS = {
  video: { width: { ideal: 1280 }, height: { ideal: 720 },
           frameRate: { ideal: 30 }, facingMode: "user" },
};

// Only used to turn a span into a friendly metre figure. Real webcams run
// 60-80 deg horizontal; the number shown to a user is approximate by nature,
// so it is rounded hard in the copy below rather than presented as a reading.
const SHOULDER_METRES = 0.328;      // measured on this dataset's world landmarks
const ASSUMED_HFOV_DEG = 70;

function verticalFovFactor(aspect) {
  const h = (ASSUMED_HFOV_DEG * Math.PI) / 180;
  const v = 2 * Math.atan(Math.tan(h / 2) / aspect);
  return 2 * Math.tan(v / 2);
}

/**
 * @param {Array<{x:number,y:number,visibility?:number}>} pose  33 pose landmarks
 * @param {{width:number,height:number}} frame                  video pixel size
 * @param {Array|null} leftHand  21 hand landmarks, or null when not detected
 * @param {Array|null} rightHand
 */
export function checkFraming(pose, frame, leftHand = null, rightHand = null) {
  const aspect = frame.width / frame.height;
  const need = [NOSE, EAR_R, EAR_L, SHOULDER_R, SHOULDER_L];
  if (!pose || need.some((i) => !pose[i] ||
      (pose[i].visibility ?? 1) < BANDS.minVisibility)) {
    return { status: "no_person", message: "没有检测到人，请正对摄像头", metrics: null };
  }

  // MediaPipe divides x by width and y by height independently, so a raw dx is
  // not comparable with a raw dy. Scaling x by the aspect ratio puts both in
  // units of frame HEIGHT, which is also what the feature pipeline does.
  const dist = (a, b) => Math.hypot((a.x - b.x) * aspect, a.y - b.y);

  const span = dist(pose[SHOULDER_R], pose[SHOULDER_L]);
  const headWidth = dist(pose[EAR_R], pose[EAR_L]);
  const centreX = (pose[SHOULDER_R].x + pose[SHOULDER_L].x) / 2;
  const centreY = (pose[SHOULDER_R].y + pose[SHOULDER_L].y) / 2;
  const ratio = span > 0 ? headWidth / span : NaN;
  const metres = SHOULDER_METRES / (span * verticalFovFactor(aspect));

  // Hand size in pixels, from the hand landmarks when present, else predicted
  // from the span (hands measure 0.11 of frame height at the training framing
  // of span 0.21, so the ratio is 0.11/0.21 of the span).
  const handPx = handDiagonalPx([leftHand, rightHand], frame) ??
                 (0.11 / 0.21) * span * frame.height;

  const metrics = { span, headShoulder: ratio, centreX, centreY,
                    handPixels: handPx, approxMetres: metres };

  const handOutOfFrame = [leftHand, rightHand].some((h) =>
    h && h.some((p) => p.x < BANDS.edgeMargin || p.x > 1 - BANDS.edgeMargin ||
                       p.y < BANDS.edgeMargin || p.y > 1 - BANDS.edgeMargin));

  // Ordered by what the user should fix FIRST. Distance dominates: at 0.6 m the
  // hands leave frame and the pose is perspective-distorted, so telling someone
  // to centre themselves before they have stepped back wastes the instruction.
  const [loA, hiA] = BANDS.span.accept;
  if (span > hiA)
    return { status: "too_close", severity: "block",
             message: `太近了，请后退到约 ${Math.max(1.5, metres + 0.8).toFixed(1)} 米`,
             metrics };
  if (span < loA || handPx < BANDS.handPixels.min)
    return { status: "too_far", severity: "block",
             message: handPx < BANDS.handPixels.min
               ? "太远了，手看不清，请靠近一点"
               : "太远了，请靠近一点", metrics };
  if (handOutOfFrame)
    return { status: "hands_out_of_frame", severity: "block",
             message: "手超出画面，请后退或把摄像头调远", metrics };
  if (ratio < BANDS.headShoulder[0] || ratio > BANDS.headShoulder[1])
    return { status: "turned", severity: "block",
             message: "请正对摄像头，不要侧身", metrics };
  if (centreX < BANDS.centreX[0] || centreX > BANDS.centreX[1])
    return { status: "off_centre", severity: "warn",
             message: centreX < 0.5 ? "请向右移动" : "请向左移动", metrics };
  if (centreY < BANDS.centreY[0] || centreY > BANDS.centreY[1])
    return { status: "off_centre", severity: "warn",
             message: centreY < 0.5 ? "摄像头请调低一点" : "摄像头请调高一点", metrics };

  const [loI, hiI] = BANDS.span.ideal;
  return {
    status: span >= loI && span <= hiI ? "ideal" : "ok",
    severity: "ok",
    message: "位置合适，可以开始",
    metrics,
  };
}

/**
 * Framing must hold for a moment before capture starts -- a single good frame
 * happens while someone is still walking backwards. Feed every frame in.
 */
export function makeStabiliser(framesRequired = 15) {
  let good = 0;
  return (result) => {
    good = result.severity === "ok" ? good + 1 : 0;
    return { ...result, ready: good >= framesRequired, progress:
             Math.min(1, good / framesRequired) };
  };
}
