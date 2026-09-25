"""Text-only SignSparK generations on Auslan, rendered as skeletons and measured for jitter.

Sits on top of signspark_ft.py (model building, text cache, loaders). For a set of sentences it
samples each stream (hand / body / face) from the released and the fine-tuned weights with the same
noise, runs SMPL-X forward kinematics through SignSparK's own build_smplx_input, and writes videos:
ground truth | released | fine-tuned, each with a wrist-centred close-up of the right hand, where
finger jitter is easiest to see.

Jitter is measured on 3D joint positions, as the mean magnitude of the second difference
(acceleration) per frame:
  - fingers relative to their wrist (arm motion removed), both hands;
  - the wrists themselves.
It is reported for the ground truth too, because the question is whether the fine-tuned model has
picked up frame-to-frame noise that is in our fitted Auslan clips.
"""

from __future__ import annotations

import os
import re
import subprocess

import numpy as np
import torch

import signspark_ft as ft

ID6 = np.array([1, 0, 0, 0, 1, 0], np.float32)
FPS = 25                                   # Auslan-Daily is 25 fps

# SMPL-X joint indices (127-joint output)
L_WRIST, R_WRIST = 20, 21
L_FINGERS = list(range(25, 40)) + list(range(66, 71))
R_FINGERS = list(range(40, 55)) + list(range(71, 76))
LEGS = {1, 2, 4, 5, 7, 8, 10, 11}
FEET = set(range(60, 66))
LEFT = {16, 18, 20} | set(range(25, 40)) | set(range(66, 71))     # drawn blue
RIGHT = {17, 19, 21} | set(range(40, 55)) | set(range(71, 76))    # drawn orange; front view, so on the viewer's left
TIPS = [(39, 66), (27, 67), (30, 68), (36, 69), (33, 70), (54, 71), (42, 72), (45, 73), (51, 74), (48, 75)]
FACE_LMK = range(76, 127)


# --------------------------------------------------------------------------- text and length
def normalise(sentence: str) -> str:
    """Auslan-Daily style: lower case, punctuation split off by a space ('i am sally .')."""
    s = sentence.strip().lower()
    s = re.sub(r"\s*([.,!?;:])", r" \1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if re.search(r"[.!?]$", s) else s + " ."


def fit_length(texts, lengths):
    """frames = a + b * words, least squares on real clips (text without the language tag)."""
    w = np.array([len(t.split()) for t in texts], float)
    b, a = np.polyfit(w, np.asarray(lengths, float), 1)
    return float(a), float(b)


def est_length(text, a, b, lo=20, hi=300):
    return int(np.clip(round(a + b * len(text.split())), lo, hi))


# --------------------------------------------------------------------------- batches
def text_batch(stream, texts, lengths, T=304):
    """A batch for sentences that have no clip: placeholder poses (identity rotations), a length
    mask, no keyframes. Hand batches are two sequences per sentence, L0 R0 L1 R1 ..., as in the
    loader's split_hands order."""
    C = {'hand': 90, 'body': 60, 'face': 56}[stream]
    n = len(texts)
    rot = np.tile(ID6, C // 6)
    feat = np.concatenate([rot, np.zeros(C - len(rot), np.float32)]) if stream == 'face' else rot
    reps = 2 if stream == 'hand' else 1
    x = torch.from_numpy(np.tile(feat, (n * reps, T, 1))).transpose(1, 2).contiguous()
    L = torch.tensor([l for l in lengths for _ in range(reps)])
    y = {'mask': (torch.arange(T)[None] < L[:, None])[:, None], 'lengths': L,
         'text': [t for t in texts for _ in range(reps)], 'keyframes': [[] for _ in range(n * reps)],
         'video_names': [f'sentence_{i}' for i in range(n) for _ in range(reps)]}
    return x, y


@torch.no_grad()
def sample_text_only(model, flow, batches, amp, device, steps=20, text_scale=2.5, seed=0):
    """Classifier-free guided, keyframe-free sampling (the text-to-pose setting). Batch bi uses the
    noise seed seed*1000+bi, the same scheme as ft.evaluate, so two models given the same batches
    start from identical noise. Returns a list of (B, C, T) arrays."""
    ft.set_mode(model, False)
    fwd = ft.Amp(model, amp)

    def guided(x, t, y=None, obs_x0=None, obs_mask=None):
        out = fwd(x, t, y=y, obs_x0=obs_x0, obs_mask=obs_mask)
        out_u = fwd(x, t, y=dict(y, uncond=True), obs_x0=obs_x0, obs_mask=obs_mask)
        return out_u + text_scale * (out - out_u)

    outs = []
    for bi, (x, y) in enumerate(batches):
        x = x.to(device)
        y = dict(y, mask=y['mask'].to(device))
        none = torch.zeros(x.shape, dtype=torch.bool, device=device)
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            torch.manual_seed(seed * 1000 + bi)
            noise = torch.randn_like(x)
            out, _ = flow.decode(guided, noise=noise, keyframe_mask=none, x_embed=x,
                                 model_kwargs={'y': y, 'obs_x0': x, 'obs_mask': none},
                                 ode_package='torchdiffeq', ode_stepnum=steps)
        outs.append(out.float().cpu().numpy())
    return outs


@torch.no_grad()
def sample_keyframed(model, flow, batches, amp, device, steps=20, seed=0):
    """Keyframe-conditioned sampling (the paper's setting, no guidance), exactly as
    signspark_ft.evaluate scores kf_err: the real frames at the segment keyframes are given, the model
    fills in the rest. Same noise scheme as sample_text_only. Returns (list of (B, C, T) arrays, list
    of per-sequence keyframe index lists) - the keyframes are copies of the real pose, so a fair
    comparison leaves them out."""
    ft.set_mode(model, False)
    fwd = ft.Amp(model, amp)
    outs, kfs = [], []
    for bi, (x, y) in enumerate(batches):
        x = x.to(device)
        B, J, T = x.shape
        y = dict(y, mask=y['mask'].to(device))
        obs = ft.kf_mask(y['keyframes'], B, J, T).to(device).contiguous()
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            torch.manual_seed(seed * 1000 + bi)
            noise = torch.randn_like(x)
            out, _ = flow.decode(fwd, noise=noise, keyframe_mask=obs, x_embed=x,
                                 model_kwargs={'y': y, 'obs_x0': x, 'obs_mask': obs},
                                 ode_package='torchdiffeq', ode_stepnum=steps)
        outs.append(out.float().cpu().numpy())
        kfs += [list(k) for k in y['keyframes']]
    return outs, kfs


def per_clip(stream, arrays, lengths):
    """Batch outputs -> one (T, C) array per clip, cut to its length. Hand: (T, 180) = L | R."""
    x = np.concatenate(arrays)                       # (N or 2N, C, T)
    x = x.transpose(0, 2, 1)
    if stream == 'hand':
        return [np.concatenate([x[2 * i, :n], x[2 * i + 1, :n]], 1) for i, n in enumerate(lengths)]
    return [x[i, :n] for i, n in enumerate(lengths)]


# --------------------------------------------------------------------------- SMPL-X
class Skeleton:
    """SMPL-X forward kinematics for the three stream outputs of one clip."""

    def __init__(self, smplx_npz, ssk, device):
        import smplx
        ft.setup_paths(ssk)
        from tools.visualize import build_smplx_input
        self.build = build_smplx_input
        self.device = device
        self.model = smplx.create(smplx_npz, model_type='smplx', gender='neutral', use_pca=False,
                                  flat_hand_mean=True, num_betas=10, num_expression_coeffs=50).to(device).eval()
        self.parents = self.model.parents.cpu().numpy()

    @torch.no_grad()
    def joints(self, body, hands, face):
        """body (T, 60), hands (T, 180) left in the loader's right-hand convention, face (T, 56)
        -> (T, 127, 3) joints."""
        kw = self.build(torch.from_numpy(body).float(), torch.from_numpy(hands[:, :90]).float(),
                        torch.from_numpy(hands[:, 90:]).float(), face=torch.from_numpy(face).float(),
                        device=self.device)
        return self.model(**kw).joints.cpu().numpy()


# --------------------------------------------------------------------------- jitter
def accel(P):
    """Mean |second difference| over frames and joints, (T, J, 3) -> float (units of P per frame^2)."""
    if len(P) < 3:
        return float('nan')
    return float(np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1).mean())


def speed(P):
    if len(P) < 2:
        return float('nan')
    return float(np.linalg.norm(P[1:] - P[:-1], axis=-1).mean())


def jitter(J):
    """Per clip, in millimetres: finger acceleration relative to the wrist, wrist acceleration,
    and finger speed relative to the wrist (how much the fingers articulate at all)."""
    fl = J[:, L_FINGERS] - J[:, [L_WRIST]]
    fr = J[:, R_FINGERS] - J[:, [R_WRIST]]
    fingers = np.concatenate([fl, fr], 1) * 1000
    wrists = J[:, [L_WRIST, R_WRIST]] * 1000
    return {'finger_accel_mm': accel(fingers), 'wrist_accel_mm': accel(wrists), 'finger_speed_mm': speed(fingers)}


# --------------------------------------------------------------------------- body diagnostics
# The body stream drives these (legs and spine are held neutral by build_smplx_input).
BODY_JOINTS = {12: 'neck', 13: 'L_collar', 14: 'R_collar', 15: 'head', 16: 'L_shoulder', 17: 'R_shoulder',
               18: 'L_elbow', 19: 'R_elbow', 20: 'L_wrist', 21: 'R_wrist'}


def joint_accel(J):
    """Per body joint, millimetres per frame^2: mean |acceleration| in 3D and its vertical part alone."""
    P = J * 1000
    out = {}
    for j, name in BODY_JOINTS.items():
        a = P[2:, j] - 2 * P[1:-1, j] + P[:-2, j]
        out[name] = (float(np.linalg.norm(a, axis=-1).mean()), float(np.abs(a[:, 1]).mean())) if len(a) else (np.nan, np.nan)
    return out


def shoulder_height(J):
    """Height of the mid-point between the shoulders, millimetres, per frame."""
    return J[:, [16, 17], 1].mean(1) * 1000


def bob_spectrum(J, win=64, detrend=9):
    """Power spectrum of the shoulder height's fast part over the first `win` frames: the slow motion
    (a `detrend`-frame moving average) is removed, a Hann window applied. A periodic up-and-down bob
    shows as a peak; frequencies are np.fft.rfftfreq(win, 1 / FPS). None for clips shorter than win."""
    if len(J) < win:
        return None
    y = shoulder_height(J[:win])
    y = y - np.convolve(np.pad(y, detrend // 2, mode='edge'), np.ones(detrend) / detrend, 'valid')
    return np.abs(np.fft.rfft(y * np.hanning(win))) ** 2


def hf_power(J, lo_hz=6.0, win=64):
    """Mean power of the shoulder-height spectrum at and above lo_hz: the high-frequency floor that
    makes the fine-tuned models bob. Real (smoothed) signing has almost none there. None if T < win."""
    p = bob_spectrum(J, win)
    if p is None:
        return None
    f = np.fft.rfftfreq(win, 1 / FPS)
    return float(p[f >= lo_hz].mean())


# --------------------------------------------------------------------------- post-hoc smoothing and scores
N_ROT = {'hand': 30, 'body': 10, 'face': 1}           # 6D rotations per stream feature row (hand = L | R)


def _orthonormal_6d(d6):
    """(..., 6) -> nearest proper rotation's first two rows (Gram-Schmidt, SignSparK's row layout)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.concatenate([b1, b2], -1)


def smooth_stream(stream, x, sigma):
    """Gaussian low-pass along time on one clip of generated features (T, C): the 6D rotations are
    filtered and re-orthonormalised, the face's 50 expression coefficients filtered as numbers. The
    same filter smooth_lmdb applies to the training data. sigma in frames; 0 returns x."""
    if sigma <= 0 or len(x) < 2:
        return x
    from scipy.ndimage import gaussian_filter1d
    f = gaussian_filter1d(x.astype(np.float64), sigma, axis=0, mode='nearest')
    k = N_ROT[stream] * 6
    rot = _orthonormal_6d(f[:, :k].reshape(len(x), -1, 6)).reshape(len(x), k)
    return np.concatenate([rot, f[:, k:]], 1).astype(x.dtype)


def stream_scores(stream, preds, gts, device='cpu', chunk=64):
    """Text-only scores of generated clips against the real ones, as in signspark_ft.evaluate:
    txt_dtw (DTW distance: degrees, face expression L1) and txt_motion (generated / real mean
    frame-to-frame change). preds, gts: lists of (T, C) arrays, same clips in the same order; hand
    clips are (T, 180) and are scored as two 90-dim sequences, as the model generates them."""
    import torch
    P, G = [], []
    for p, g in zip(preds, gts):
        if stream == 'hand':
            P += [p[:, :90], p[:, 90:]]
            G += [g[:, :90], g[:, 90:]]
        else:
            P.append(p)
            G.append(g)
    dtws, mp, mg = [], [0.0, 0.0], [0.0, 0.0]
    for s in range(0, len(P), chunk):
        ps, gs = P[s:s + chunk], G[s:s + chunk]
        L = torch.tensor([len(p) for p in ps])
        T = int(L.max())
        pad = lambda a: np.concatenate([a, np.repeat(a[-1:], T - len(a), 0)])
        pt = torch.from_numpy(np.stack([pad(p) for p in ps])).float().transpose(1, 2).to(device)
        gt = torch.from_numpy(np.stack([pad(g) for g in gs])).float().transpose(1, 2).to(device)
        dtws.append(ft.dtw(ft.cost_matrix(stream, pt, gt), L).cpu())
        valid = (torch.arange(T)[None] < L[:, None]).to(device)
        for acc, x in ((mp, pt), (mg, gt)):
            s_, n_ = ft.motion(x, valid)
            acc[0] += s_
            acc[1] += n_
    return {'txt_dtw': float(torch.cat(dtws).mean()), 'txt_motion': (mp[0] / mp[1]) / (mg[0] / mg[1])}


# --------------------------------------------------------------------------- handshape variety
# SMPL-X / MANO hand pose order: index 1-3, middle 1-3, pinky 1-3, ring 1-3, thumb 1-3. With
# flat_hand_mean=True (SignSparK's setting) the identity rotation is the flat hand, so a joint's rotation
# angle is how far it is bent away from flat.
FINGERS = ('index', 'middle', 'pinky', 'ring', 'thumb')


def finger_curl(h90):
    """(T, 90) 6D hand pose -> (T, 5) degrees: per finger, the summed rotation angles of its 3 joints."""
    d = h90.reshape(len(h90), 15, 6).astype(np.float64)
    a1, a2 = d[..., :3], d[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    tr = b1[..., 0] + b2[..., 1] + b3[..., 2]
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    return ang.reshape(len(h90), 5, 3).sum(-1)


def hand_curls(clips):
    """Hand-stream clips (T, 180) = left | right in the loader's right-hand convention ->
    {'left': (N, 5), 'right': (N, 5)} finger curls of every frame."""
    return {'left': np.concatenate([finger_curl(c[:, :90]) for c in clips]),
            'right': np.concatenate([finger_curl(c[:, 90:]) for c in clips])}


def fit_shapes(curls, k=16, seed=0):
    """k handshape clusters on the real frames' finger curls (the reference vocabulary)."""
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit(curls)


def shape_stats(curls, km, ref_hist=None, flat_deg=30.0):
    """How varied the handshapes are, from (N, 5) finger curls:
    curl_mean   mean curl of the four fingers (thumb excluded), degrees - how bent the hand is;
    curl_std    spread of each finger's curl over all frames, averaged over the five - variety;
    flat_frac   frames with all four fingers under flat_deg - the open flat hand;
    coverage    clusters holding at least 1% of the frames;
    js          Jensen-Shannon divergence (bits, 0..1) of the cluster histogram from ref_hist."""
    hist = np.bincount(km.predict(curls), minlength=km.n_clusters) / len(curls)
    out = {'curl_mean': float(curls[:, :4].mean()), 'curl_std': float(curls.std(0).mean()),
           'flat_frac': float((curls[:, :4] < flat_deg).all(1).mean()), 'coverage': int((hist >= 0.01).sum())}
    if ref_hist is not None:
        m = (hist + ref_hist) / 2
        kl = lambda p, q: float(np.sum(p[p > 0] * np.log2(p[p > 0] / q[p > 0])))
        out['js'] = 0.5 * kl(hist, m) + 0.5 * kl(ref_hist, m)
    return out, hist


# --------------------------------------------------------------------------- retrieval of keyframes
def load_bank(path):
    """Every clip of an LMDB (the training set) as a retrieval bank: its text as the loader builds it,
    its length, its segment keyframes (the loader's first / middle / last frame of each segment) and
    its per-stream features in the loader's convention (hand = left, flipped to right-hand convention,
    | right; body = the 10 joints the body stream models; face = jaw 6D + 50 expression)."""
    import io
    import pickle
    import lmdb
    from signspark.pose_datasets_lmdb import PoseDataset_Lmdb, flip_left_hand_features
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    bank = []
    with env.begin() as txn:
        for k in pickle.loads(txn.get(b'__meta__'))['clip_ids']:
            with np.load(io.BytesIO(txn.get(k.encode())), allow_pickle=True) as z:
                seg = z['segment']
                kf = sorted({f for a, b in PoseDataset_Lmdb._segment_bounds(seg.tolist()) for f in (a, (a + b) // 2, b)})
                bank.append({'name': k, 'text': f"<{z['language'][0]}> {z['translation'][0]}", 'T': len(seg), 'keyframes': kf,
                             'hand': np.concatenate([flip_left_hand_features(z['left_features'])[:, :90],
                                                     z['right_features'][:, :90]], 1).astype(np.float32),
                             'body': z['body_features'][:, 66:].astype(np.float32),
                             'face': z['face_features'].astype(np.float32)})
    env.close()
    return bank


def retrieve(query_emb, bank_emb):
    """Nearest bank entry by cosine similarity of text embeddings: (indices, similarities)."""
    import torch
    q = torch.nn.functional.normalize(query_emb.float(), dim=-1)
    b = torch.nn.functional.normalize(bank_emb.float(), dim=-1)
    sim = q @ b.T
    best = sim.max(1)
    return best.indices.cpu().numpy(), best.values.cpu().numpy()


def _untag(t):
    """'<Auslan> i am sally .' -> 'i am sally .'"""
    return t.split('> ', 1)[1] if t.startswith('<') and '> ' in t else t


def word_sim(queries, bank_texts, stop_words=None):
    """(Q, N) cosine similarity of TF-IDF word vectors (language tag removed): words shared by the two
    sentences count, common words (in many sentences) count less than rare content words.
    stop_words='english' drops English function words, which Auslan often does not sign."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stop_words, sublinear_tf=True)
    B = vec.fit_transform([_untag(t) for t in bank_texts])
    Q = vec.transform([_untag(t) for t in queries])
    return (Q @ B.T).toarray().astype(np.float32)


def mix_sim(a, b, alpha):
    """alpha * a + (1 - alpha) * b after standardising each query's row of scores (z-scores over the
    bank), so that scores on different scales (M-CLIP cosine ~0.9-1, TF-IDF 0-1) mix fairly."""
    z = lambda m: (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-8)
    return alpha * z(a) + (1 - alpha) * z(b)


def warp_index(n_src, n_dst):
    """Frame indices that stretch or squeeze an n_src-frame clip to n_dst frames (nearest frame)."""
    return np.round(np.linspace(0, n_src - 1, n_dst)).astype(int)


def warped_keyframes(entry, n_dst):
    """The entry's keyframe positions, rescaled to a clip of n_dst frames: [(dst_frame, src_frame)]."""
    if entry['T'] < 2 or n_dst < 2:
        return [(0, 0)]
    out = {}
    for f in entry['keyframes']:
        out.setdefault(int(round(f * (n_dst - 1) / (entry['T'] - 1))), f)
    return sorted(out.items())


def keyframe_batch(stream, texts, lengths, entries, T=304):
    """A batch whose keyframes come from retrieved clips: at each rescaled keyframe position the
    retrieved clip's pose, elsewhere an identity placeholder that the model never sees (only the
    keyframes are observed). Same layout as text_batch: hand clips are two sequences, L then R."""
    x, y = text_batch(stream, texts, lengths, T)
    reps = 2 if stream == 'hand' else 1
    kfs = []
    for i, (e, n) in enumerate(zip(entries, lengths)):
        pairs = warped_keyframes(e, n)
        feat = e[stream]
        for r in range(reps):
            seq = feat[:, 90 * r:90 * (r + 1)] if stream == 'hand' else feat
            for dst, src in pairs:
                x[i * reps + r, :, dst] = torch.from_numpy(seq[src])
            kfs.append([d for d, _ in pairs])
    y['keyframes'] = kfs
    return x, y


def copy_retrieved(stream, entry, n_dst):
    """The retrieved clip itself, time-warped to n_dst frames: the no-model baseline."""
    return entry[stream][warp_index(entry['T'], n_dst)]


# --------------------------------------------------------------------------- the final pipeline
def generate(sentences, bank, weights, cfgs, device, amp=True, steps=20, text_scale=2.5, seed=0, sigma=1.0, log=print):
    """English sentences -> Auslan signing features, the pipeline chosen in EXPERIMENTS.md E26-E33:

    1. normalised to Auslan-Daily style and tagged <Auslan>;
    2. length from the word count (a words -> frames fit on the bank, i.e. the training set);
    3. the most similar training sentence by TF-IDF word overlap (ties settled by the model's M-CLIP text
       embedding) - `seen` when the exact sentence is in the training set;
    4. hands: that sentence's keyframes, rescaled to the target length, given to the hand stream;
    5. body and face: the same keyframes if seen, text only otherwise (another sentence's arms and face
       made them worse);
    6. Gaussian smoothing sigma frames after generation (removes the fine-tuned models' bob).

    bank: load_bank(training LMDB); weights / cfgs: {stream: weights path / signspark config}.
    Returns one dict per sentence: text, retrieved (the training sentence), seen, T, and hand (T, 180),
    body (T, 60), face (T, 56) in the loader's convention (what Skeleton.joints takes)."""
    import gc
    texts = [f'<Auslan> {normalise(s)}' for s in sentences]
    bank_texts = [e['text'] for e in bank]
    a, b = fit_length([_untag(t) for t in bank_texts], [e['T'] for e in bank])
    lengths = [est_length(_untag(t), a, b) for t in texts]
    seen = [t in set(bank_texts) for t in texts]

    out = {}
    for stream in ('hand', 'body', 'face'):
        prime = texts + sorted(set(bank_texts)) if stream == 'hand' else texts
        model, flow = ft.build(cfgs[stream], weights[stream], device, texts=prime)
        if stream == 'hand':
            enc = model.encode_text
            q = torch.nn.functional.normalize(enc(texts).float(), dim=-1)
            bk = torch.nn.functional.normalize(enc(bank_texts).float(), dim=-1)
            score = word_sim(texts, bank_texts) + 1e-3 * (q @ bk.T).cpu().numpy()
            entries = [bank[j] for j in score.argmax(1)]
        keyed = [i for i in range(len(texts)) if stream == 'hand' or seen[i]]
        free = [i for i in range(len(texts)) if i not in keyed]
        res = [None] * len(texts)
        if keyed:
            arr, _ = sample_keyframed(model, flow, [keyframe_batch(stream, [texts[i] for i in keyed], [lengths[i] for i in keyed],
                                                                   [entries[i] for i in keyed])], amp, device, steps, seed)
            for i, c in zip(keyed, per_clip(stream, arr, [lengths[i] for i in keyed])):
                res[i] = c
        if free:
            arr = sample_text_only(model, flow, [text_batch(stream, [texts[i] for i in free], [lengths[i] for i in free])],
                                   amp, device, steps, text_scale, seed)
            for i, c in zip(free, per_clip(stream, arr, [lengths[i] for i in free])):
                res[i] = c
        out[stream] = [smooth_stream(stream, c, sigma) for c in res]
        del model, flow
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        log(f'{stream}: {len(keyed)} with keyframes, {len(free)} from text only')
    return [{'sentence': s, 'text': t, 'retrieved': e['text'], 'seen': sn, 'T': n,
             'hand': out['hand'][i], 'body': out['body'][i], 'face': out['face'][i]}
            for i, (s, t, e, sn, n) in enumerate(zip(sentences, texts, entries, seen, lengths))]


# --------------------------------------------------------------------------- drawing
def bones(parents):
    return [(j, int(parents[j])) for j in range(1, 55)
            if j not in LEGS and int(parents[j]) not in LEGS and parents[j] >= 0] + TIPS


def body_box(clips, size):
    keep = [j for j in range(127) if j not in LEGS | FEET | {0}]
    P = np.concatenate([c[:, keep, :2].reshape(-1, 2) for c in clips])
    lo, hi = P.min(0), P.max(0)
    return (lo + hi) / 2, 0.9 * size / max(hi - lo)


def hand_box(clips, size):
    """Wrist-relative right-hand view. The panel is centred on the wrist plus the mean finger offset
    over all clips and frames - a constant, so arm motion stays removed - and the scale fits the
    furthest finger from that centre in any panel."""
    rel = [c[:, R_FINGERS, :2] - c[:, [R_WRIST], :2] for c in clips]
    off = np.concatenate([r.reshape(-1, 2) for r in rel]).mean(0)
    reach_x = max(np.abs(r[..., 0] - off[0]).max() for r in rel)
    reach_y = max(np.abs(r[..., 1] - off[1]).max() for r in rel)
    # the close-up is size wide but only size // 2 tall
    return min(0.42 * size / max(reach_x, 1e-6), 0.42 * (size // 2) / max(reach_y, 1e-6)), off


def _draw(img, P, bone_list, face=True, width=2):
    import cv2
    for a, b in bone_list:
        colour = (0, 90, 230) if a in RIGHT else (230, 120, 0) if a in LEFT else (60, 60, 60)   # BGR: orange, blue, grey
        cv2.line(img, tuple(P[a]), tuple(P[b]), colour, width, cv2.LINE_AA)
    if face:
        for k in FACE_LMK:
            cv2.circle(img, tuple(P[k]), 1, (40, 40, 40), -1, cv2.LINE_AA)


def panel(J, size, bbox, hbox, bone_list, title):
    """Upper body on top, right hand close-up below, one column of the video."""
    import cv2
    top = np.full((size, size, 3), 255, np.uint8)
    (x0, y0), s = bbox
    P = np.stack([(J[:, 0] - x0) * s + size / 2, (y0 - J[:, 1]) * s + size / 2], 1).astype(int)
    _draw(top, P, bone_list)
    scale = 0.6                                  # shrink a long title until it fits its own panel
    while scale > 0.3 and cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > size - 12:
        scale -= 0.05
    cv2.putText(top, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    h = size // 2
    low = np.full((h, size, 3), 248, np.uint8)
    hscale, off = hbox
    w = J[R_WRIST, :2] + off
    Q = np.stack([(J[:, 0] - w[0]) * hscale + size / 2, (w[1] - J[:, 1]) * hscale + h / 2], 1).astype(int)
    hb = [(a, b) for a, b in bone_list if a in R_FINGERS]           # fingers only: the arm would run off the panel
    _draw(low, Q, hb, face=False, width=2)
    cv2.circle(low, tuple(Q[R_WRIST]), 3, (0, 90, 230), -1, cv2.LINE_AA)
    cv2.putText(low, 'right hand (wrist-centred)', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return np.concatenate([top, low], 0)


def write_video(path, clips, titles, parents, size=400, caption=''):
    """clips: list of (T_i, 127, 3); shorter clips hold their last frame."""
    import cv2
    bbox = body_box(clips, size)
    hbox = hand_box(clips, size)
    bl = bones(parents)
    T = max(len(c) for c in clips)
    frames = []
    for t in range(T):
        row = np.concatenate([panel(c[min(t, len(c) - 1)], size, bbox, hbox, bl, lab) for c, lab in zip(clips, titles)], 1)
        if caption:
            bar = np.full((28, row.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, caption[:110], (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            row = np.concatenate([bar, row], 0)
        frames.append(row)
    H, W = frames[0].shape[:2]
    tmp = path + '.tmp.mp4'
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    try:            # browsers and Colab need H.264; mp4v is kept if ffmpeg is missing
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', path],
                       check=True)
        os.remove(tmp)
    except (FileNotFoundError, subprocess.CalledProcessError):
        os.replace(tmp, path)
    return path
