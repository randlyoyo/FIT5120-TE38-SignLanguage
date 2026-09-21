#!/usr/bin/env python3
"""Write a word's animation as a self-contained .glb (skeleton + clip, no mesh).

    python export_gltf.py WHAT --naming mixamo
    python export_gltf.py --all --naming vrm

A skeleton-only glb is what a retargeter wants: Three.js, Blender and Unity all
accept it and can drive a character rig from it, and shipping no mesh keeps the
file to the animation itself.

Axes. MediaPipe world coordinates are +X right, +Y DOWN, +Z away from camera;
glTF is +Y up with the camera looking down -Z. The two differ by a half turn
about X, so positions become (x, -y, -z) and -- conjugating by that same
rotation -- quaternions become (x, -y, -z, w). Skipping this is the classic way
to end up with an avatar signing upside down and facing backwards.

Fingertips are deliberately given no animation channel: there is no bone past
them, so a channel there rotates nothing and tells a retargeter the chain is
one joint longer than it is.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

import joint_names as JN
import retarget as RT

OUT = Path("renders")


def to_gltf_vec(v):
    v = np.asarray(v, np.float32).copy()
    v[..., 1] *= -1.0
    v[..., 2] *= -1.0
    return v


def to_gltf_quat(q):
    q = np.asarray(q, np.float32).copy()
    q[..., 1] *= -1.0
    q[..., 2] *= -1.0
    return q


class Buf:
    """Accumulates binary data and the accessors that point into it."""

    def __init__(self):
        self.blob = bytearray()
        self.views, self.accessors = [], []

    def add(self, arr, type_, comp=5126):
        arr = np.ascontiguousarray(arr, np.float32)
        while len(self.blob) % 4:
            self.blob.append(0)
        off = len(self.blob)
        self.blob += arr.tobytes()
        self.views.append({"buffer": 0, "byteOffset": off,
                           "byteLength": arr.nbytes})
        acc = {"bufferView": len(self.views) - 1, "componentType": comp,
               "count": int(arr.shape[0]), "type": type_}
        flat = arr.reshape(arr.shape[0], -1)
        acc["min"] = flat.min(axis=0).tolist()
        acc["max"] = flat.max(axis=0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1


def build(word, naming="ours", fps_override=None):
    rig = json.load(open(OUT / "rig.json"))
    anim = json.load(open(OUT / f"anim/{word.replace('/', '_')}.json"))
    names, parents = rig["joints"], rig["parents"]
    rest = to_gltf_vec(np.array(rig["restOffsets"], np.float32))
    q = to_gltf_quat(np.array(anim["rotations"], np.float32))       # (T,J,4)
    root = to_gltf_vec(np.array(anim["rootPositions"], np.float32))  # (T,3)
    T = q.shape[0]
    fps = float(fps_override or anim["fps"] or 25.0)
    times = (np.arange(T, dtype=np.float32) / fps)

    table = {"ours": {n: n for n in names},
             "vrm": JN.vrm_map(), "mixamo": JN.mixamo_map()}[naming]
    tips = set(JN.tip_joints())

    nodes = []
    for i, n in enumerate(names):
        nodes.append({"name": table.get(n, n),
                      "translation": rest[i].tolist(),
                      "rotation": [0.0, 0.0, 0.0, 1.0]})
    for i, p in enumerate(parents):
        if p >= 0:
            nodes[p].setdefault("children", []).append(i)
    nodes[0]["translation"] = root[0].tolist()

    buf = Buf()
    t_acc = buf.add(times.reshape(-1, 1), "SCALAR")
    channels, samplers = [], []
    for i, n in enumerate(names):
        if n in tips:
            continue          # no bone past a fingertip; a channel here is a lie
        out = buf.add(q[:, i], "VEC4")
        samplers.append({"input": t_acc, "output": out, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": i, "path": "rotation"}})
    out = buf.add(root, "VEC3")
    samplers.append({"input": t_acc, "output": out, "interpolation": "LINEAR"})
    channels.append({"sampler": len(samplers) - 1,
                     "target": {"node": 0, "path": "translation"}})

    gltf = {
        "asset": {"version": "2.0",
                  "generator": f"signtest retarget ({naming} naming)"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes,
        "buffers": [{"byteLength": len(buf.blob)}],
        "bufferViews": buf.views, "accessors": buf.accessors,
        "animations": [{"name": word, "channels": channels,
                        "samplers": samplers}],
        "extras": {"word": word, "source": anim.get("source"),
                   "fps": fps, "frames": T,
                   "note": "status/face channels are in the sibling JSON; "
                           "joints whose status is 3 must not be rendered"},
    }
    return gltf, bytes(buf.blob)


def write_glb(gltf, blob, path):
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    bl = blob + b"\0" * ((4 - len(blob) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bl)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(bl), 0x004E4942)); f.write(bl)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--naming", choices=["ours", "vrm", "mixamo"], default="mixamo")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    outdir = Path(a.outdir or OUT / f"glb_{a.naming}")
    outdir.mkdir(parents=True, exist_ok=True)
    words = a.words
    if a.all:
        words = [w["word"] for w in json.load(open(OUT / "manifest.json"))["words"]]
    total = 0
    for w in words:
        gltf, blob = build(w, a.naming)
        n = write_glb(gltf, blob, outdir / f"{w.replace('/', '_')}.glb")
        total += n
        if len(words) <= 8:
            print(f"  {w:<16s} {n/1024:7.1f} KB  {gltf['extras']['frames']} frames")
    print(f"{len(words)} files -> {outdir}  ({total/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
