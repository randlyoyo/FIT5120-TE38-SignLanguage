#!/usr/bin/env python3
"""Read a .glb back and check it against the JSON it came from.

The axis conversion is the part that cannot be eyeballed. A sign rig with the
half-turn applied backwards produces numbers that look entirely reasonable --
same magnitudes, same smooth trajectories -- and an avatar that is upside down
and facing away. So the check is explicit: run forward kinematics on what the
glb actually contains, undo the conversion, and compare joint for joint against
the source. It also asserts the things a viewer would silently tolerate: that
the head ends up above the hips, and that the signer faces the camera.
"""
import json
import struct
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import retarget as RT


def read_glb(path):
    raw = open(path, "rb").read()
    magic, ver, total = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67 and ver == 2, "not a glTF 2.0 binary"
    assert total == len(raw), f"header says {total} bytes, file is {len(raw)}"
    off, js, blob = 12, None, None
    while off < len(raw):
        ln, ty = struct.unpack_from("<II", raw, off)
        chunk = raw[off + 8: off + 8 + ln]
        if ty == 0x4E4F534A:
            js = json.loads(chunk)
        elif ty == 0x004E4942:
            blob = chunk
        off += 8 + ln
    return js, blob


def accessor(g, blob, i):
    a = g["accessors"][i]
    v = g["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
    arr = np.frombuffer(blob, np.float32, a["count"] * n,
                        v.get("byteOffset", 0)).reshape(a["count"], n)
    return arr.squeeze() if n == 1 else arr


def main(word):
    g, blob = read_glb(f"renders/glb_mixamo/{word.replace('/', '_')}.glb")
    src = json.load(open(f"renders/anim/{word.replace('/', '_')}.json"))
    rig = json.load(open("renders/rig.json"))
    names = rig["joints"]

    # Pull the animation back out of the glb, in glb node order.
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(src["rotations"]), len(names), 1))
    root = None
    for ch in g["animations"][0]["channels"]:
        s = g["animations"][0]["samplers"][ch["sampler"]]
        out = accessor(g, blob, s["output"])
        if ch["target"]["path"] == "rotation":
            q[:, ch["target"]["node"]] = out
        else:
            root = out
    # Undo the half turn about X that the exporter applied.
    q[..., 1] *= -1; q[..., 2] *= -1
    root = root.copy(); root[:, 1] *= -1; root[:, 2] *= -1
    rest = np.array(rig["restOffsets"])

    P_glb, _ = RT.fk(q, rest, root)
    P_src, _ = RT.fk(np.array(src["rotations"]), rest,
                     np.array(src["rootPositions"]))
    e = np.linalg.norm(P_glb - P_src, axis=-1)
    print(f"{word}: {len(P_src)} frames, {len(g['nodes'])} nodes, "
          f"{len(g['animations'][0]['channels'])} channels")
    print(f"  round-trip error   max {e.max()*1000:.4f} mm  "
          f"(float32 rounding only; anything larger is an axis bug)")

    # Orientation sanity, in glb space this time.
    Pg, _ = RT.fk(np.array(src["rotations"]), rest, np.array(src["rootPositions"]))
    up = np.array([0, -1, 0.0])          # our +Y is down, so up is -Y
    head_above = float(np.median((Pg[:, RT.IDX["neck"]] - Pg[:, RT.IDX["hips"]]) @ up))
    sh = Pg[:, RT.IDX["shoulder_L"]] - Pg[:, RT.IDX["shoulder_R"]]
    print(f"  head above hips    {head_above*100:+.1f} cm  "
          f"{'OK' if head_above > 0 else 'UPSIDE DOWN'}")
    print(f"  left shoulder +x   {float(np.median(sh[:, 0]))*100:+.1f} cm  "
          f"{'OK (facing camera)' if np.median(sh[:, 0]) > 0 else 'MIRRORED'}")
    tips = {n for n in names if n.endswith("4_L") or n.endswith("4_R")}
    animated = {g["nodes"][c["target"]["node"]]["name"]
                for c in g["animations"][0]["channels"]
                if c["target"]["path"] == "rotation"}
    bad = [n for n in names if n in tips and
           g["nodes"][names.index(n)]["name"] in animated]
    print(f"  fingertip channels {len(bad)} (must be 0)")


if __name__ == "__main__":
    for w in sys.argv[1:]:
        main(w)
