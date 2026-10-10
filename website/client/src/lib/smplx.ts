// SMPL-X joint layout, matching the bone names in public/smplx.glb.
// Contract: sign_chat_backend/README.md "Pose file (pose_url)".

/** The 22 body joints `body_pose` covers, in SMPL-X's fixed order (pelvis
 *  itself is `global_orient`, not in this list -- README: "21 body
 *  joints"). */
export const SMPLX_BODY_JOINTS = [
  "left_hip", "right_hip", "spine1",
  "left_knee", "right_knee", "spine2",
  "left_ankle", "right_ankle", "spine3",
  "left_foot", "right_foot", "neck",
  "left_collar", "right_collar", "head",
  "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow",
  "left_wrist", "right_wrist",
] as const;

/** Each hand's 15 joints, in SMPL-X/MANO order (`left_hand_pose` /
 *  `right_hand_pose`, 45 = 15 x 3 each). */
const HAND_JOINT_ORDER = ["index", "middle", "pinky", "ring", "thumb"] as const;
const HAND_JOINT_SUFFIX = ["1", "2", "3"] as const;

function handJoints(side: "left" | "right"): string[] {
  return HAND_JOINT_ORDER.flatMap((finger) => HAND_JOINT_SUFFIX.map((n) => `${side}_${finger}${n}`));
}

export const SMPLX_LEFT_HAND_JOINTS = handJoints("left");
export const SMPLX_RIGHT_HAND_JOINTS = handJoints("right");

/** One playable clip: axis-angle rotations per joint per frame, decoded
 *  from a `pose_url` response. `rotations[jointName]` is a flat
 *  Float32Array of length `frames * 3` (radians, SMPL-X convention). */
export interface SmplxClip {
  fps: number;
  frames: number;
  rotations: Record<string, Float32Array>;
}

export interface SmplxPoseJson {
  fps: number;
  frames: number;
  smplx: {
    global_orient: number[][];
    body_pose: number[][];
    left_hand_pose: number[][];
    right_hand_pose: number[][];
  };
}

/** Parses the backend's pose JSON (per-frame flat arrays) into one
 *  Float32Array per joint (per-joint is what the per-frame bone-update loop
 *  in SmplxAvatar actually indexes). */
export function parseSmplxPose(data: SmplxPoseJson): SmplxClip {
  const { fps, frames, smplx } = data;
  const rotations: Record<string, Float32Array> = {};

  function fill(jointNames: readonly string[], source: number[][], globalName?: string) {
    if (globalName) {
      const arr = new Float32Array(frames * 3);
      for (let t = 0; t < frames; t++) arr.set(source[t] ?? [0, 0, 0], t * 3);
      rotations[globalName] = arr;
      return;
    }
    jointNames.forEach((name, j) => {
      const arr = new Float32Array(frames * 3);
      for (let t = 0; t < frames; t++) {
        const frame = source[t];
        arr[t * 3] = frame?.[j * 3] ?? 0;
        arr[t * 3 + 1] = frame?.[j * 3 + 1] ?? 0;
        arr[t * 3 + 2] = frame?.[j * 3 + 2] ?? 0;
      }
      rotations[name] = arr;
    });
  }

  fill([], smplx.global_orient, "global_orient");
  fill(SMPLX_BODY_JOINTS, smplx.body_pose);
  fill(SMPLX_LEFT_HAND_JOINTS, smplx.left_hand_pose);
  fill(SMPLX_RIGHT_HAND_JOINTS, smplx.right_hand_pose);

  return { fps, frames, rotations };
}
