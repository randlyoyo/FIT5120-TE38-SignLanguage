import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { VRMLoaderPlugin, type VRM } from "@pixiv/three-vrm";
import { HUMANOID_BONE_MAP, type SmplxClip } from "../lib/smplx";

/** Set by whoever drops the avatar file in (see AVATAR_SETUP.md) -- a VRM
 *  export (VRoid Studio or similar), placed in client/public/ and
 *  referenced by path (e.g. "/avatar.vrm"). */
const AVATAR_URL = import.meta.env.VITE_AVATAR_MODEL_URL as string | undefined;

const tmpAxis = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();

const DEG = Math.PI / 180;
/** VRM's T-pose rest holds both arms straight out to the sides, which reads
 *  as a frozen error, not "waiting for input". Rotating each upper arm
 *  ~85° around the forward axis swings it down to a relaxed stand, shown
 *  whenever there's no clip to animate (idle, or still generating).
 *  Keyed by SMPL-X joint name, like `clip.rotations`, so it drops into the
 *  same per-bone loop below. The rotation direction here is a best guess at
 *  VRM's axis convention -- if an arm swings up or through the body instead
 *  of down, flip that arm's sign. */
const IDLE_POSE: Record<string, [number, number, number]> = {
  left_shoulder: [0, 0, -85 * DEG],
  right_shoulder: [0, 0, 85 * DEG],
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

// Waist-up framing, tuned against this specific VRM's proportions (see
// AVATAR_SETUP.md); re-tune if a very differently-proportioned avatar ever
// replaces it.
const CAMERA_POSITION: [number, number, number] = [0, 1.25, 1.45];
const CAMERA_TARGET: [number, number, number] = [0, 1.28, 0];
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
  // three's own GLTFLoader, not drei's useGLTF: three-vrm's plugin is typed
  // against three's loader, and drei's (three-stdlib) GLTFParser doesn't match.
  const gltf = useLoader(GLTFLoader, url, (loader) => {
    loader.register((parser) => new VRMLoaderPlugin(parser));
  });
  const vrm = gltf.userData.vrm as VRM;

  useEffect(() => {
    // Three.js frustum-culls a SkinnedMesh by its rest-pose bounding box;
    // once the skeleton is driven away from rest pose, a signing gesture
    // can swing a hand outside that box and the mesh vanishes mid-frame.
    vrm.scene.traverse((obj) => {
      obj.frustumCulled = false;
    });
  }, [vrm]);

  // Looked up once per avatar via the VRM's own humanoid bone names
  // (self-describing per the VRM spec), not by guessing a specific rig's
  // internal node names the way a plain Mixamo/glTF rig would need.
  const bonesBySmplxName = useMemo(() => {
    const map = new Map<string, THREE.Object3D>();
    for (const [smplxName, vrmBoneName] of Object.entries(HUMANOID_BONE_MAP)) {
      const bone = vrm.humanoid.getNormalizedBoneNode(vrmBoneName);
      if (bone) map.set(smplxName, bone);
    }
    return map;
  }, [vrm]);

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
    // Propagates the normalized-bone rotations set above onto the VRM's
    // raw skeleton -- three-vrm's humanoid is a two-layer rig, and nothing
    // moves on screen until this runs (@pixiv/three-vrm docs, "Update").
    vrm.update(delta);
  });

  return <primitive object={vrm.scene} />;
}

interface Props {
  clip: SmplxClip | null;
  playing: boolean;
  /** Restart the clip when it ends instead of holding the last frame. */
  loop?: boolean;
  speed?: number;
  className?: string;
}

/** 3D avatar driven directly by the backend's keypoint data (pose_url),
 *  instead of playing a rendered video -- see lib/smplx.ts for the
 *  SMPL-X -> VRM human-bone retarget map. Renders nothing (caller should
 *  fall back to the video) until VITE_AVATAR_MODEL_URL is configured with
 *  an actual VRM avatar file; see AVATAR_SETUP.md. */
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
