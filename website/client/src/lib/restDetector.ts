// When has the signer finished? recognition/API.md §8, measured on 800
// validation clips: the sign is over once the wrists are back down at rest
// (more than 1.2 shoulder widths below the shoulders) and nearly still
// (under 0.03 shoulder widths per 25 fps frame) for 8 frames in a row --
// armed only after a wrist has come up (above 0.8). It stopped mid-sign on
// 0.1% of those clips.
//
// Some learners finish with their hands still raised instead of lowering
// them, so a second, slower rule also ends the capture: the wrists held still
// at any height for HOLD_MS. A pause that long mid-sign is rare, and the
// server trims the still tail off before scoring (API.md §3.5).
//
// A third rule covers learners framed so the hands at rest fall out of the
// picture: the pose then only guesses the wrists, too jittery to ever read as
// still. Once a hand has been seen, both hands gone for HANDS_GONE_MS ends
// the capture. The rule this file replaced used 0.3 s here and cut signs
// whose hands briefly dipped out of frame; 1 s rides those out.
//
// It reads the wrists from the body pose, which MediaPipe estimates even when
// the hand itself slips out of the picture. The rule it replaces stopped
// 0.3 s after a *hand* was lost, and real webcam captures showed that cutting
// signs off half-way whenever a hand dipped below the frame.
//
// Works in the API's §3 normalised frame (shoulder midpoint at the origin,
// shoulder width as the unit, y down), so it is independent of how far the
// learner sits from the camera.

export type CapturePhase = "waiting" | "active" | "settling";

const RAISED = 0.8;
const REST_HEIGHT = 1.2;
const REST_SPEED = 0.03; // shoulder widths per frame at 25 fps
const REST_MS = 8 * 40; // 8 frames at 25 fps
const HOLD_MS = 1000;
const HANDS_GONE_MS = 1000;

const MISSING = -999;
// Indices in the 11-point pose subset the capture stores (landmarks.ts).
const L_SHOULDER = 5, R_SHOULDER = 6, L_WRIST = 9, R_WRIST = 10;

export interface RestStep {
  phase: CapturePhase;
  stop: boolean;
}

/** Feed it one pose frame at a time (with its time in ms, and whether either
 *  hand was detected); it says when to stop. */
export function createRestDetector(width: number, height: number) {
  let started = false;
  let restSince: number | null = null;
  let stillSince: number | null = null;
  let lastHandAt: number | null = null;
  let prev: { wrists: (number[] | null)[]; t: number } | null = null;

  return function step(pose: number[][], t: number, handVisible = true): RestStep {
    if (handVisible) lastHandAt = t;
    if (started && lastHandAt !== null && t - lastHandAt >= HANDS_GONE_MS) return { phase: "settling", stop: true };
    const ls = pose[L_SHOULDER];
    const rs = pose[R_SHOULDER];
    if (ls[0] === MISSING || rs[0] === MISSING) {
      prev = null;
      return { phase: started ? "active" : "waiting", stop: false };
    }
    const unit = Math.hypot((ls[0] - rs[0]) * width, (ls[1] - rs[1]) * height) || 1;
    const midX = ((ls[0] + rs[0]) / 2) * width;
    const midY = ((ls[1] + rs[1]) / 2) * height;
    const wrists = [pose[L_WRIST], pose[R_WRIST]].map((w) =>
      w[0] === MISSING ? null : [(w[0] * width - midX) / unit, (w[1] * height - midY) / unit]
    );

    const seen = wrists.filter((w): w is number[] => w !== null);
    const wristY = seen.length ? Math.min(...seen.map((w) => w[1])) : Infinity; // the higher wrist
    let speed = Infinity;
    if (prev && t > prev.t) {
      const perFrame = 40 / (t - prev.t); // rescale to a 25 fps frame
      speed = 0;
      wrists.forEach((w, i) => {
        const p = prev!.wrists[i];
        if (w && p) speed = Math.max(speed, Math.hypot(w[0] - p[0], w[1] - p[1]) * perFrame);
      });
    }
    prev = { wrists, t };

    if (!started && wristY < RAISED) started = true;
    if (!started) return { phase: "waiting", stop: false };

    const still = speed < REST_SPEED;
    stillSince = still ? (stillSince ?? t) : null;
    restSince = still && wristY > REST_HEIGHT ? (restSince ?? t) : null;
    if ((restSince !== null && t - restSince >= REST_MS) || (stillSince !== null && t - stillSince >= HOLD_MS)) {
      return { phase: "settling", stop: true };
    }
    return { phase: restSince !== null || stillSince !== null ? "settling" : "active", stop: false };
  };
}
