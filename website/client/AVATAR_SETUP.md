# 3D avatar setup

The Sign Chat page can drive a real 3D avatar from the backend's keypoint
data (`pose_url`) instead of playing its rendered video — see
`src/lib/smplx.ts` and `src/components/SmplxAvatar.tsx`. It needs one file
we can't make or fetch ourselves: a **VRM** avatar (`.vrm`) with full
per-finger bones (hand shape is what carries meaning in a sign, so a rig
that collapses fingers to one bone isn't usable here — VRM's standard rig
always has them).

## Make an avatar: VRoid Studio

Ready Player Me (the obvious free option) shut down its public avatar
creator on 2026-01-31 (Netflix acquired it) — don't use it, the domain no
longer resolves. Use **[VRoid Studio](https://vroid.com/studio)** instead
(Pixiv, free, actively maintained):

1. Download and install it (Windows/Mac).
2. Create an avatar — full face, hair, clothing, body customization, no
   account needed to use the editor.
3. **Export → VRM 1.0** (not 0.x if the option exists — the bone names this
   code expects, `VRMHumanBoneName`, are the 1.0 spec's).
4. Save the result as `website/client/public/avatar.vrm`.

VRM's bones are self-describing (every VRM file declares which bone is
"leftHand", "leftIndexProximal", etc., regardless of what the file calls
them internally), so unlike a plain Mixamo/glTF rig there's no bone-name
guessing involved — `lib/smplx.ts`'s `HUMANOID_BONE_MAP` targets the VRM
spec's bone names directly and should work on anyone's VRM export.

## Wire it in

```
# website/client/.env
VITE_AVATAR_MODEL_URL=/avatar.vrm
```

Restart `npm run dev`. The avatar should appear on `/conversation` once a
reply comes back; `SmplxAvatar` renders nothing (silent fallback to the
video) until this env var is set.

## If something looks wrong

- **Avatar doesn't appear at all**: check the browser console for a 404 on
  the `.vrm` file (wrong path/filename) or a loader error (not actually a
  VRM file, or VRM 0.x exported without the humanoid bones this expects).
- **Body doesn't move but loads fine**: `vrm.update(delta)` not running, or
  `HUMANOID_BONE_MAP` not finding bones — log `bonesBySmplxName.size` in
  `SmplxAvatar.tsx` to check how many of the 52 mapped joints resolved.
- **Limbs rotate the wrong way**: a VRM 0.x file loaded as if it were 1.0
  (0.x's forward-facing axis is flipped) — re-export as VRM 1.0.
