import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { SmplxClip } from "../lib/smplx";

/** The SMPL-X neutral body itself (betas 0) as a skinned GLB, built by
 *  vrm_fit/smplx_web/build_glb.py: one bone per SMPL-X joint, named as in
 *  lib/smplx.ts, rest rotations all identity. The poses were fitted on this
 *  exact body, so they apply as-is -- no retargeting, contacts stay put.
 *  Lives in client/public/. */
const AVATAR_URL = "/smplx.glb";

const tmpAxis = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();

const DEG = Math.PI / 180;
/** SMPL-X's rest is a T-pose (arms straight out), which reads as a frozen
 *  error, not "waiting for input". Rotating each upper arm ~75° around the
 *  forward (z) axis swings it down to a relaxed stand, shown whenever
 *  there's no clip to animate (idle, or still generating). Keyed by SMPL-X
 *  joint name, like `clip.rotations`, so it drops into the same per-bone
 *  loop below. */
const IDLE_POSE: Record<string, [number, number, number]> = {
  left_shoulder: [0, 0, -75 * DEG],
  right_shoulder: [0, 0, 75 * DEG],
};

/** SMPL-X stores each joint as an axis-angle vector: direction = rotation
 *  axis, length = angle (radians). A zero vector (no rotation) has no
 *  direction to normalise, so it's left as the identity quaternion. */
function applyAxisAngle(bone: THREE.Object3D, x: number, y: number, z: number) {
  const angle = Math.sqrt(x * x + y * y + z * z);
  if (angle < 1e-8) {
    bone.quaternion.identity();
    return;
  }
  tmpAxis.set(x / angle, y / angle, z / angle);
  tmpQuat.setFromAxisAngle(tmpAxis, angle);
  bone.quaternion.copy(tmpQuat);
}

// Waist-up framing for the SMPL-X template's own coordinates (y up, origin
// near the chest: pelvis at y -0.35, top of head at y 0.42). The pelvis
// stays at rest (pose JSON carries no transl), so this framing holds.
const CAMERA_POSITION: [number, number, number] = [0, 0.02, 2.1];
const CAMERA_TARGET: [number, number, number] = [0, 0.0, 0];
// Vertical angle pinned at the tuned framing's own, so dragging only turns
// the avatar left/right -- looking from above/below adds nothing for
// reading a sign, and would expose the cropped-off framing.
const POLAR_ANGLE = Math.acos(
  (CAMERA_POSITION[1] - CAMERA_TARGET[1]) /
    Math.hypot(CAMERA_POSITION[1] - CAMERA_TARGET[1], CAMERA_POSITION[2] - CAMERA_TARGET[2])
);

/** Canvas gets mounted fresh every time the user switches messages (the
 *  stage only renders it for the selected reply), and react-three-fiber's
 *  auto-sizing (react-use-measure, watching the wrapper div) sometimes
 *  measures that div before the surrounding flex layout has settled to its
 *  final size -- the canvas is then stuck at the 300x150 HTML default,
 *  rendering into a tiny box in the corner instead of filling the stage.
 *  Dispatching a resize event a beat after mount forces a re-measure once
 *  layout has actually settled; this is the standard workaround for this
 *  known class of react-use-measure race. */
function ForceResizeAfterMount() {
  useEffect(() => {
    const id = requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    return () => cancelAnimationFrame(id);
  }, []);
  return null;
}

interface RiggedAvatarProps {
  url: string;
  clip: SmplxClip | null;
  playing: boolean;
  loop: boolean;
  speed: number;
}

function RiggedAvatar({ url, clip, playing, loop, speed }: RiggedAvatarProps) {
  const gltf = useLoader(GLTFLoader, url);

  useEffect(() => {
    // Three.js frustum-culls a SkinnedMesh by its rest-pose bounding box;
    // once the skeleton is driven away from rest pose, a signing gesture
    // can swing a hand outside that box and the mesh vanishes mid-frame.
    gltf.scene.traverse((obj) => {
      obj.frustumCulled = false;
    });
  }, [gltf]);

  // Bones carry the SMPL-X joint names themselves; global_orient drives the
  // root (pelvis). Each bone's rest rotation is identity, so a joint's
  // axis-angle is exactly that bone's local rotation.
  const bonesBySmplxName = useMemo(() => {
    const map = new Map<string, THREE.Object3D>();
    gltf.scene.traverse((obj) => {
      if (obj instanceof THREE.Bone) map.set(obj.name === "pelvis" ? "global_orient" : obj.name, obj);
    });
    return map;
  }, [gltf]);

  const elapsedRef = useRef(0);

  useEffect(() => {
    elapsedRef.current = 0;
  }, [clip]);

  useFrame((_state, delta) => {
    if (clip && playing) {
      elapsedRef.current += delta * speed;
      const rawFrame = Math.floor(elapsedRef.current * clip.fps);
      const frame = loop ? rawFrame % clip.frames : Math.min(clip.frames - 1, rawFrame);
      for (const [smplxName, bone] of bonesBySmplxName) {
        const arr = clip.rotations[smplxName];
        if (arr) applyAxisAngle(bone, arr[frame * 3], arr[frame * 3 + 1], arr[frame * 3 + 2]);
        else bone.quaternion.identity();
      }
    } else {
      for (const [smplxName, bone] of bonesBySmplxName) {
        const idle = IDLE_POSE[smplxName];
        if (idle) applyAxisAngle(bone, idle[0], idle[1], idle[2]);
        else bone.quaternion.identity();
      }
    }
  });

  return <primitive object={gltf.scene} />;
}

interface Props {
  clip: SmplxClip | null;
  playing: boolean;
  /** Restart the clip when it ends instead of holding the last frame. */
  loop?: boolean;
  speed?: number;
  className?: string;
}

/** 3D SMPL-X body driven directly by the backend's pose data (pose_url),
 *  instead of playing a rendered video. Renders nothing (caller should
 *  fall back to the video) only if the GLB path is ever cleared. */
export function SmplxAvatar({ clip, playing, loop = false, speed = 1, className }: Props) {
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  if (!AVATAR_URL) return null;

  function resetView() {
    const controls = controlsRef.current;
    if (!controls) return;
    controls.object.position.set(...CAMERA_POSITION);
    controls.target.set(...CAMERA_TARGET);
    controls.update();
  }

  return (
    <div className={`smplx-avatar ${className ?? ""}`}>
      <Canvas camera={{ position: CAMERA_POSITION, fov: 30 }}>
        <OrbitControls
          ref={controlsRef}
          target={CAMERA_TARGET}
          enablePan={false}
          enableZoom={false}
          minAzimuthAngle={-Math.PI / 2}
          maxAzimuthAngle={Math.PI / 2}
          minPolarAngle={POLAR_ANGLE}
          maxPolarAngle={POLAR_ANGLE}
        />
        <ForceResizeAfterMount />
        <ambientLight intensity={0.9} />
        <directionalLight position={[2, 4, 3]} intensity={1.1} />
        <Suspense fallback={null}>
          <RiggedAvatar url={AVATAR_URL} clip={clip} playing={playing} loop={loop} speed={speed} />
        </Suspense>
      </Canvas>
      <button type="button" className="smplx-avatar-reset" onClick={resetView}>
        Reset view
      </button>
    </div>
  );
}

export const smplxAvatarConfigured = Boolean(AVATAR_URL);
