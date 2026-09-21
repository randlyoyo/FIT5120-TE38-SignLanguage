#!/usr/bin/env python3
"""Map this rig's 49 joints onto the VRM and Mixamo humanoid conventions.

Both targets name the joint by the bone it drives, and both expect three
phalanx bones per finger, which is what this rig has once the fingertip is set
aside: MediaPipe stores four landmarks per finger, and the last is a tip with
nothing beyond it to rotate. So thumb1/2/3 drive proximal/intermediate/distal
and thumb4 carries no rotation at all -- it exists to give the distal bone a
direction and should not be written into an animation channel.

Sides carry straight across. MediaPipe's "left" is the signer's left, and so
is VRM's and Mixamo's, so no mirroring is needed. (The mirroring risk in this
project is elsewhere: MediaPipe's pose indices put the LEFT ear at 7 and the
right at 8, the opposite of what slr_common.py's comment claims.)

Two joints do not map cleanly and are flagged rather than fudged:

  chest   this rig goes hips -> chest in one bone, where both targets expect a
          spine chain (VRM spine/chest/upperChest). The single rotation is
          written to `spine`; a character with a segmented torso will bend at
          one joint instead of curving.
  neck    the joint sits at the NOSE, not at the base of the skull, because
          that is the landmark available. Its rotation therefore mixes neck
          and head. It is written to `head`, which is where a viewer reads it,
          and `neck` is left for the renderer to leave at rest.
"""

FINGERS = [("thumb", "Thumb"), ("index", "Index"), ("middle", "Middle"),
           ("ring", "Ring"), ("pinky", "Little")]
VRM_PHALANX = ["Proximal", "Intermediate", "Distal"]


def vrm_map():
    """-> {our joint name: VRM humanoid bone}. Unmapped joints are absent."""
    m = {"hips": "hips", "chest": "spine", "neck": "head"}
    for s, side in (("L", "left"), ("R", "right")):
        m[f"shoulder_{s}"] = f"{side}UpperArm"
        m[f"elbow_{s}"] = f"{side}LowerArm"
        m[f"wrist_{s}"] = f"{side}Hand"
        for ours, vrm in FINGERS:
            name = "Little" if vrm == "Little" else vrm
            for i, ph in enumerate(VRM_PHALANX, start=1):
                m[f"{ours}{i}_{s}"] = f"{side}{name}{ph}"
    return m


def mixamo_map(prefix="mixamorig:"):
    """-> {our joint name: Mixamo node name}."""
    m = {"hips": f"{prefix}Hips", "chest": f"{prefix}Spine1",
         "neck": f"{prefix}Head"}
    for s, side in (("L", "Left"), ("R", "Right")):
        m[f"shoulder_{s}"] = f"{prefix}{side}Arm"
        m[f"elbow_{s}"] = f"{prefix}{side}ForeArm"
        m[f"wrist_{s}"] = f"{prefix}{side}Hand"
        for ours, mix in FINGERS:
            name = "Pinky" if mix == "Little" else mix
            for i in (1, 2, 3):
                m[f"{ours}{i}_{s}"] = f"{prefix}{side}Hand{name}{i}"
    return m


# Joints that must NOT be written as animation channels: fingertips have no
# bone past them, and driving them rotates nothing while confusing a retargeter
# into thinking the chain is one bone longer than it is.
def tip_joints():
    return [f"{f}4_{s}" for f, _ in FINGERS for s in ("L", "R")]


if __name__ == "__main__":
    import json
    import retarget as RT
    v, x = vrm_map(), mixamo_map()
    tips = set(tip_joints())
    print(f"{'ours':<14s}{'VRM':<24s}{'Mixamo':<34s}")
    print("-" * 72)
    for n in RT.NAMES:
        if n in tips:
            print(f"{n:<14s}{'(tip, no rotation)':<24s}{'-':<34s}")
        else:
            print(f"{n:<14s}{v.get(n,'-'):<24s}{x.get(n,'-'):<34s}")
    unmapped = [n for n in RT.NAMES if n not in v and n not in tips]
    print(f"\nmapped {len([n for n in RT.NAMES if n in v])}, "
          f"tips {len(tips)}, unmapped {len(unmapped)}: {unmapped}")
    json.dump({"vrm": v, "mixamo": x, "tips": sorted(tips)},
              open("renders/joint_names.json", "w"), indent=1)
    print("-> renders/joint_names.json")
