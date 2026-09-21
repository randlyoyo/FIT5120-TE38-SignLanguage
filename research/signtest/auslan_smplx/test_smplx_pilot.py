"""Checks for smplx_pilot.py that need neither NLF, WiLoR nor the SMPL-X/MANO files.

    python auslan_smplx/test_smplx_pilot.py --signspark /path/to/SignSparK

With --signspark, the 6D and left-hand conventions are checked against
SignSparK's own loader code, and a written LMDB is read back through it.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
import types

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smplx_pilot as sp  # noqa: E402

rng = np.random.default_rng(0)


def rand_R(*shape):
    return Rotation.random(int(np.prod(shape)), random_state=rng).as_matrix().reshape(*shape, 3, 3)


def check(name, ok, detail=''):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        check.failed += 1


check.failed = 0


def test_6d_roundtrip():
    R = rand_R(50)
    err = np.abs(sp.sixd_to_mat(sp.mat_to_6d(R)) - R).max()
    check('6D <-> matrix round trip', err < 1e-9, f'max err {err:.1e}')


def test_mirror_is_axis_angle_yz_flip():
    # WiLoR's flip_mano_params negates the y and z axis-angle components; that must be conjugation by M.
    aa = rng.normal(size=(100, 3))
    R = sp.aa_to_mat(aa)
    conj = sp.M_MIRROR @ R @ sp.M_MIRROR
    ref = sp.aa_to_mat(aa * np.array([1, -1, -1]))
    check('M-conjugation == WiLoR axis-angle (x, -y, -z) flip', np.abs(conj - ref).max() < 1e-9)
    check('F-conjugation == M-conjugation', np.abs(sp.F_FLIP @ R @ sp.F_FLIP - conj).max() < 1e-12)


def constant_hands(T, go, hp, valid=None):
    return {'global_orient': np.repeat(go[None], T, 0), 'hand_pose': np.repeat(hp[None], T, 0),
            'valid': np.ones(T, bool) if valid is None else valid}


def test_assembly_geometry():
    T = 9
    body = np.repeat(rng.normal(scale=0.4, size=(1, 55, 3)), T, 0)
    # a true left hand (SMPL-X left) as it sits in the camera, and what WiLoR sees on the mirrored crop
    L_go, L_hp = rand_R(), rand_R(15)
    raw_L_go, raw_L_hp = sp.M_MIRROR @ L_go @ sp.M_MIRROR, sp.M_MIRROR @ L_hp @ sp.M_MIRROR
    R_go, R_hp = rand_R(), rand_R(15)
    res = sp.assemble(body, {'left': constant_hands(T, raw_L_go, raw_L_hp), 'right': constant_hands(T, R_go, R_hp)})
    G = sp.global_rotations(res['local'], 22)
    check('right wrist global == WiLoR hand orientation', np.abs(G[:, sp.R_WRIST] - R_go).max() < 1e-6)
    check('left wrist global == un-mirrored WiLoR orientation', np.abs(G[:, sp.L_WRIST] - L_go).max() < 1e-6)
    check('left fingers stored as true SMPL-X left', np.abs(res['local'][:, sp.L_HAND] - L_hp).max() < 1e-6)
    check('right fingers == WiLoR hand_pose', np.abs(res['local'][:, sp.R_HAND] - R_hp).max() < 1e-6)
    body_R = sp.aa_to_mat(body)
    same = [j for j in range(1, 22) if j not in (sp.L_WRIST, sp.R_WRIST)]
    check('other body joints untouched (NLF)', np.abs(res['local'][:, same] - body_R[:, same]).max() < 1e-6)
    f = res['features']
    shapes = {k: v.shape for k, v in f.items()}
    check('feature shapes', shapes == {'left_features': (T, 90), 'right_features': (T, 90),
                                       'body_features': (T, 126), 'face_features': (T, 56)}, str(shapes))
    return res


def test_gaps_and_missing_hand():
    T = 12
    body = rng.normal(scale=0.3, size=(T, 55, 3))
    go, hp = rand_R(), rand_R(15)
    valid = np.ones(T, bool); valid[3:6] = False
    h = constant_hands(T, go, hp, valid)
    h['global_orient'][~valid] = np.nan                       # must never be read
    h['hand_pose'][~valid] = np.nan
    res = sp.assemble(body, {'right': h})
    check('gap frames filled, finite', np.isfinite(res['local']).all())
    check('missing left hand keeps NLF fingers',
          np.abs(res['local'][:, sp.L_HAND] - sp.sixd_to_mat(sp.smooth_6d(sp.mat_to_6d(sp.aa_to_mat(body[:, sp.L_HAND]))))).max() < 1e-6)
    check('coverage reported', res['coverage'] == {'left': 0.0, 'right': 0.75}, str(res['coverage']))


def test_segments():
    T = 80
    kp = np.zeros((T, 133, 2)); sc = np.full((T, 133), 0.9)
    kp[:, 5] = [100, 100]; kp[:, 6] = [200, 100]             # shoulders, width 100
    rest = np.array([150, 400.0])
    wr = np.repeat(rest[None], T, 0)
    # two signs: 15..35 and 40..60 with a hold at ~37
    for a, b in ((15, 36), (38, 61)):
        t = np.linspace(0, np.pi, b - a)
        wr[a:b] = np.stack([150 + 60 * np.sin(2 * t), 180 + 20 * np.cos(t)], 1)
    wr[36:38] = wr[35]
    kp[:, 10] = wr; kp[:, 9] = rest
    lab = sp.segments_from_wrists(kp, sc)
    starts = np.flatnonzero(lab == 2)
    check('resting frames are non-sign', lab[:10].max() == 0 and lab[-10:].max() == 0, str(lab))
    check('at least one sign start inside the active span', len(starts) >= 1 and starts.min() >= 10, f'starts {starts.tolist()}')
    check('labels in {0,1,2}', set(np.unique(lab)) <= {0, 1, 2})
    check('every start follows a 0 frame (SignSparK only splits on 0 -> sign)',
          all(t == 0 or lab[t - 1] == 0 for t in starts), f'starts {starts.tolist()}')
    return lab


def test_hand_box():
    kp = rng.uniform(10, 50, size=(21, 2)); sc = np.full(21, 0.9)
    b = sp.hand_box(kp, sc)
    check('hand box encloses points', b[0] <= kp[:, 0].min() and b[2] >= kp[:, 0].max())
    sc[:15] = 0.1
    check('hand box None when unreliable', sp.hand_box(kp, sc) is None)


def test_mano_stub():
    # Build a pickle that references chumpy classes, the way MANO_RIGHT.pkl does.
    chumpy = types.ModuleType('chumpy'); ch = types.ModuleType('chumpy.ch')

    class Ch:
        def __reduce__(self):
            return (object.__new__, (Ch,), {'x': self.x, 'dterms': []})
    Ch.__module__ = 'chumpy.ch'
    Ch.__qualname__ = 'Ch'
    ch.Ch = Ch; chumpy.ch = ch
    sys.modules['chumpy'] = chumpy; sys.modules['chumpy.ch'] = ch
    import scipy.sparse as sps
    c = Ch(); c.x = rng.normal(size=(778, 3, 10))
    data = {'shapedirs': c, 'J_regressor': sps.csc_matrix(rng.normal(size=(16, 778))),
            'f': np.arange(12).reshape(4, 3), 'kintree_table': np.zeros((2, 16), np.int64)}
    with tempfile.TemporaryDirectory() as d:
        pkl = os.path.join(d, 'MANO_RIGHT.pkl')
        with open(pkl, 'wb') as fh:
            pickle.dump(data, fh, protocol=2)
        del sys.modules['chumpy'], sys.modules['chumpy.ch']   # as on Colab: chumpy is not importable
        keys = sp.mano_pkl_to_npz(pkl, os.path.join(d, 'MANO_RIGHT.npz'))
        z = np.load(os.path.join(d, 'MANO_RIGHT.npz'))
        check('MANO pkl -> npz without chumpy', keys == sorted(data) and np.allclose(z['shapedirs'], c.x)
              and z['J_regressor'].shape == (16, 778))


def test_mano_real_chumpy(chumpy_root):
    # The official MANO_RIGHT.pkl holds chumpy *operations* (state {'a', 'idxs'}, no 'x');
    # with the chumpy source on sys.path the converter must evaluate them.
    sys.path.insert(0, chumpy_root)
    import chumpy as ch
    import scipy.sparse as sps
    base = ch.Ch(rng.normal(size=(778, 3, 20)))
    data = {'shapedirs': base[:, :, :10], 'v_template': ch.Ch(rng.normal(size=(778, 3))),
            'J_regressor': sps.csc_matrix(rng.normal(size=(16, 778))), 'f': np.arange(12).reshape(4, 3)}
    with tempfile.TemporaryDirectory() as d:
        pkl = os.path.join(d, 'MANO_RIGHT.pkl')
        with open(pkl, 'wb') as fh:
            pickle.dump(data, fh, protocol=2)
        sp.mano_pkl_to_npz(pkl, os.path.join(d, 'MANO_RIGHT.npz'))
        z = np.load(os.path.join(d, 'MANO_RIGHT.npz'))
        check('MANO pkl with chumpy operations -> npz (real chumpy)',
              np.allclose(z['shapedirs'], base.r[:, :, :10]) and np.allclose(z['v_template'], data['v_template'].r))


def test_against_signspark(ssk_root, res, seg):
    sys.path.insert(0, ssk_root)
    sys.modules.setdefault('wandb', types.ModuleType('wandb'))
    from signspark import pose_datasets_lmdb as pdl
    R = rand_R(20)
    check('6D matches SignSparK _matrix_to_rot6d_np', np.abs(pdl._matrix_to_rot6d_np(R) - sp.mat_to_6d(R)).max() < 1e-12)
    d6 = rng.normal(size=(20, 6))
    check('Gram-Schmidt matches SignSparK _rot6d_to_matrix_np', np.abs(pdl._rot6d_to_matrix_np(d6) - sp.sixd_to_mat(d6)).max() < 1e-9)
    # the left hand stored by assemble(), after SignSparK's loader flip, must be the raw WiLoR (right-convention) pose
    stored = res['features']['left_features']
    raw_back = pdl.flip_left_hand_features(stored)
    L_hp = res['local'][0, sp.L_HAND]
    raw = sp.M_MIRROR @ L_hp @ sp.M_MIRROR
    check('SignSparK flip of stored left == raw WiLoR left', np.abs(raw_back[0].reshape(15, 6) - sp.mat_to_6d(raw)).max() < 1e-5)

    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test', 'AuslanDaily-pilot_test.lmdb')
        os.makedirs(os.path.dirname(path))
        T = len(seg)
        feats = {k: np.repeat(v[:1], T, 0) for k, v in res['features'].items()}
        sp.write_lmdb(path, [('clip_a', sp.record_bytes('Auslan', 'hello .', seg, feats))], map_size=1 << 26)
        for feat, concat, dim in (('hand', True, 180), ('body', False, 60), ('face', False, 56)):
            args = SimpleNamespace(dataset_feat=feat, specify_lang=True, flip_left_hand=True,
                                   keyframe_selection_mode=3, custom_keyframe_file='null')
            ds = pdl.PoseDataset_Lmdb(args, split='test', seq_len=304, data=path, concat_hands=concat)
            x, L, text, kfs, name = ds[0]
            ds.env.close(); ds._env = None
            if feat == 'hand':
                n_seg = len(pdl.PoseDataset_Lmdb._segment_bounds(seg))
                check('SignSparK sees every segment', n_seg == int((seg == 2).sum()) and len(kfs) >= 3 * n_seg - n_seg,
                      f'{n_seg} segments, {int((seg == 2).sum())} starts, keyframes {kfs}')
            check(f'SignSparK loader reads {feat}', x.shape == (304, dim) and np.isfinite(x.numpy()).all()
                  and L == T and text == '<Auslan> hello .', f'{tuple(x.shape)} L={L} kf={kfs}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--signspark', default=None)
    ap.add_argument('--chumpy', default=None, help='chumpy source checkout (github.com/mattloper/chumpy)')
    a = ap.parse_args()
    test_6d_roundtrip()
    test_mirror_is_axis_angle_yz_flip()
    res = test_assembly_geometry()
    test_gaps_and_missing_hand()
    seg = test_segments()
    test_hand_box()
    test_mano_stub()
    if a.chumpy:
        test_mano_real_chumpy(a.chumpy)
    if a.signspark:
        test_against_signspark(a.signspark, res, seg)
    print(f'\n{check.failed} failed')
    sys.exit(1 if check.failed else 0)
