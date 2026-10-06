// SMPL-X joint layout and a retarget map onto the VRM 1.0 standard human
// bone names (VRMHumanBoneName) -- self-describing, so SmplxAvatar looks
// bones up through the avatar's own VRMHumanoid rather than guessing a
// specific rig's internal bone names. Contract:
// sign_chat_backend/README.md "Pose file (pose_url)" for the SMPL-X side,
// https://github.com/vrm-c/vrm-specification for the VRM side.

import type { VRMHumanBoneName } from "@pixiv/three-vrm";

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

/** Retarget map: SMPL-X joint name -> VRMHumanBoneName. Covers
 *  `global_orient` (-> hips) + body_pose + both hands; jaw/eyes are skipped
 *  (no blend-shape face on a generic VRM rig -- VRM's own jaw/eye bones
 *  are for look-at and lip sync, not SMPL-X's FLAME expression
 *  coefficients, so there's nothing sensible to map them to). VRM's thumb
 *  has 3 bones named Metacarpal/Proximal/Distal, every other finger
 *  Proximal/Intermediate/Distal -- both are 3 bones, matching SMPL-X's 3
 *  joints per finger, just named differently for the thumb. VRM calls the
 *  pinky "Little". */
export const HUMANOID_BONE_MAP: Record<string, VRMHumanBoneName> = {
  global_orient: "hips",
  left_hip: "leftUpperLeg",
  right_hip: "rightUpperLeg",
  spine1: "spine",
  left_knee: "leftLowerLeg",
  right_knee: "rightLowerLeg",
  spine2: "chest",
  left_ankle: "leftFoot",
  right_ankle: "rightFoot",
  spine3: "upperChest",
  left_foot: "leftToes",
  right_foot: "rightToes",
  neck: "neck",
  left_collar: "leftShoulder",
  right_collar: "rightShoulder",
  head: "head",
  left_shoulder: "leftUpperArm",
  right_shoulder: "rightUpperArm",
  left_elbow: "leftLowerArm",
  right_elbow: "rightLowerArm",
  left_wrist: "leftHand",
  right_wrist: "rightHand",
};

const THUMB_SUFFIX = ["Metacarpal", "Proximal", "Distal"];
const OTHER_FINGER_SUFFIX = ["Proximal", "Intermediate", "Distal"];
const VRM_FINGER_NAME: Record<string, string> = { index: "Index", middle: "Middle", pinky: "Little", ring: "Ring", thumb: "Thumb" };

function handBoneMap(side: "left" | "right", prefix: "left" | "right"): Record<string, VRMHumanBoneName> {
  const map: Record<string, VRMHumanBoneName> = {};
  for (const finger of HAND_JOINT_ORDER) {
    const suffixes = finger === "thumb" ? THUMB_SUFFIX : OTHER_FINGER_SUFFIX;
    HAND_JOINT_SUFFIX.forEach((n, i) => {
      map[`${side}_${finger}${n}`] = `${prefix}${VRM_FINGER_NAME[finger]}${suffixes[i]}` as VRMHumanBoneName;
    });
  }
  return map;
}

Object.assign(HUMANOID_BONE_MAP, handBoneMap("left", "left"), handBoneMap("right", "right"));

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
