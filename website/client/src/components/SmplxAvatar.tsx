import { Suspense, useEffect, useMemo, useRef, useState } from "react";
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
 *  Skin/clothes are a texture on the SMPL-X UV (BEDLAM skin + Meshcapade
 *  sample clothes, composed by vrm_fit/smplx_web/compose_texture.py); the
 *  hair is a separate mesh rigidly bound to the head bone.
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
  // Trunk/neck/head at the median of the fitted clips (643-clip sample, IQR
  // under 5°), so the head doesn't jump between idle and signing: ~5° nod at
  // both neck and head, a slight turn of the whole body.
  global_orient: [0, -0.046, 0],
  neck: [0.084, -0.003, -0.012],
  head: [0.084, -0.003, -0.012],
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
// near the chest: waist (spine1) at y -0.24, top of head at y 0.42). The pelvis
// stays at rest (pose JSON carries no transl), so this framing holds.
const CAMERA_POSITION: [number, number, number] = [0, 0.1, 1.5];
const CAMERA_TARGET: [number, number, number] = [0, 0.1, 0];
// Dragging turns the view around both axes: left/right up to 90°, and
// up/down (around x) within TILT of the tuned framing's own angle -- enough
// to look at the hands from above or below, not enough to flip over.
const POLAR_ANGLE = Math.acos(
  (CAMERA_POSITION[1] - CAMERA_TARGET[1]) /
    Math.hypot(CAMERA_POSITION[1] - CAMERA_TARGET[1], CAMERA_POSITION[2] - CAMERA_TARGET[2])
);
const TILT = 30 * DEG;

/** The pose data has no eye motion, so the eyes are driven here: they look
 *  at the camera (following the viewer as the view is turned), with small
 *  random saccades every second or few, like a person holding eye contact.
 *  Tracking is held to a small cone around the front view (±8°): past it the
 *  eyes stay at the edge of that cone instead of rolling toward the viewer. */
const EYE_MAX_YAW = 8 * DEG;
const EYE_MAX_PITCH = 8 * DEG;
const SACCADE = 2.5 * DEG;
const tmpEye = new THREE.Vector3();
const tmpEuler = new THREE.Euler();
const clampAngle = (a: number, max: number) => Math.max(-max, Math.min(max, a));

/** Seconds for the avatar to rise into place once its file has loaded (the
 *  fade itself is the loading overlay fading out on top of it -- fading the
 *  materials instead would make the body transparent, and three can't sort
 *  it against the already-transparent hair, so the hairline flickers). */
const APPEAR_SECONDS = 0.6;
const APPEAR_RISE = 0.04;

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
  smile: boolean;
  onReady: () => void;
}

function RiggedAvatar({ url, clip, playing, loop, speed, smile, onReady }: RiggedAvatarProps) {
  const gltf = useLoader(GLTFLoader, url);

  // The GLB carries a "smile" morph target (build_glb.py), off by default.
  // Set on every change, since the loaded scene is shared between pages.
  useEffect(() => {
    gltf.scene.traverse((obj) => {
      const i = (obj as THREE.Mesh).morphTargetDictionary?.smile;
      if (i !== undefined) (obj as THREE.Mesh).morphTargetInfluences![i] = smile ? 1 : 0;
    });
  }, [gltf, smile]);

  // Rise in on mount. The loaded scene is cached and shared between mounts,
  // so its offset is reset on unmount.
  const appearRef = useRef(0);
  useEffect(() => {
    onReady();
    appearRef.current = 0;
    return () => {
      gltf.scene.position.y = 0;
    };
  }, [gltf, onReady]);

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

  const eyes = useMemo(
    () => ["left_eye", "right_eye"].flatMap((name) => bonesBySmplxName.get(name) ?? []),
    [bonesBySmplxName]
  );
  const gazeRef = useRef({ offset: new THREE.Vector2(), target: new THREE.Vector2(), wait: 0 });

  const elapsedRef = useRef(0);

  useEffect(() => {
    elapsedRef.current = 0;
  }, [clip]);

  useFrame((state, delta) => {
    if (appearRef.current < 1) {
      appearRef.current = Math.min(1, appearRef.current + delta / APPEAR_SECONDS);
      const t = 1 - (1 - appearRef.current) ** 3; // ease-out
      gltf.scene.position.y = -APPEAR_RISE * (1 - t);
    }
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

    const gaze = gazeRef.current;
    gaze.wait -= delta;
    if (gaze.wait <= 0) {
      gaze.target.set((Math.random() * 2 - 1) * SACCADE, (Math.random() * 2 - 1) * SACCADE * 0.6);
      gaze.wait = 0.8 + Math.random() * 2.2;
    }
    gaze.offset.lerp(gaze.target, 1 - Math.exp(-delta * 25)); // saccades are quick
    for (const eye of eyes) {
      // Camera position in the eye's parent (head) space, relative to the eye
      // centre; the eye's rest gaze is +z, so yaw/pitch fall out directly.
      eye.parent!.updateWorldMatrix(true, false);
      eye.parent!.worldToLocal(tmpEye.copy(state.camera.position)).sub(eye.position);
      const yaw = clampAngle(Math.atan2(tmpEye.x, tmpEye.z) + gaze.offset.x, EYE_MAX_YAW);
      const pitch = clampAngle(Math.atan2(tmpEye.y, Math.hypot(tmpEye.x, tmpEye.z)) + gaze.offset.y, EYE_MAX_PITCH);
      eye.quaternion.setFromEuler(tmpEuler.set(-pitch, yaw, 0, "YXZ"));
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
  /** Gentle resting smile (Sign Chat's conversational avatar). */
  smile?: boolean;
  className?: string;
}

/** 3D SMPL-X body driven directly by the backend's pose data (pose_url),
 *  instead of playing a rendered video. Renders nothing (caller should
 *  fall back to the video) only if the GLB path is ever cleared. */
export function SmplxAvatar({ clip, playing, loop = false, speed = 1, smile = false, className }: Props) {
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const [ready, setReady] = useState(false);
  const markReady = useMemo(() => () => setReady(true), []);
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
          minPolarAngle={Math.max(0.05, POLAR_ANGLE - TILT)}
          maxPolarAngle={Math.min(Math.PI - 0.05, POLAR_ANGLE + TILT)}
        />
        <ForceResizeAfterMount />
        <ambientLight intensity={0.9} />
        <directionalLight position={[2, 4, 3]} intensity={1.1} />
        <Suspense fallback={null}>
          <RiggedAvatar url={AVATAR_URL} clip={clip} playing={playing} loop={loop} speed={speed} smile={smile} onReady={markReady} />
        </Suspense>
      </Canvas>
      <div className={`smplx-avatar-loading${ready ? " is-done" : ""}`} role="status" aria-hidden={ready}>
        {/* Indeterminate: three's loading manager counts whole files, not bytes,
            so a one-file avatar would sit at 0% until it is done. */}
        <span className="smplx-avatar-loading-label">Loading avatar</span>
        <span className="smplx-avatar-loading-track">
          <span className="smplx-avatar-loading-fill" />
        </span>
      </div>
      <button type="button" className="smplx-avatar-reset" onClick={resetView}>
        Reset view
      </button>
      {/* Required by the asset licences (CC BY 4.0 / CC BY-NC 4.0, SMPL-X non-commercial). */}
      <span className="smplx-avatar-credit">
        Body: SMPL-X (MPI-IS) · Textures: Meshcapade, CC BY / CC BY-NC 4.0
      </span>
    </div>
  );
}

export const smplxAvatarConfigured = Boolean(AVATAR_URL);
