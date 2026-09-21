"""Masked pose losses.

All terms are computed in signer space (units: shoulder widths) and use
    m[b, t, j] = frame_mask[b, t] & joint_valid[b, t, j]
so padded frames and missing / long-gap keypoints contribute exactly zero.

L_total = lambda_pose    * L_pose      SmoothL1 (or L1/MSE) on coordinates, per-part weights,
                                       + hand_relative_weight * wrist-relative finger term
        + lambda_vel     * L_velocity  |dx_hat - dx|, both frames valid
        + lambda_acc     * L_accel     |d2x_hat - d2x|, three frames valid
        + lambda_bone    * L_bone      | |bone_hat| - |bone| |, both endpoints valid
        + lambda_counter * L_counter   MSE on the progress counter (Progressive Transformers)
        + lambda_feat    * L_feat      relative MSE ||f_hat - f||^2 / ||f||^2 of frozen Uni-Sign
                                       pose-encoder features (optional)

Progressive Transformers' own loss was MSE on joints + counter; set
`pose_loss_type: mse` to reproduce it. SmoothL1 is the default because
keypoint-detector outliers dominate MSE on small data.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from slp_models import skeleton as sk


def _elem(pred, gt, kind: str, beta: float):
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, gt, reduction="none", beta=beta)
    if kind == "l1":
        return (pred - gt).abs()
    if kind == "mse":
        return (pred - gt) ** 2
    raise ValueError(f"unknown pose_loss_type {kind!r}")


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.to(x.dtype)
    while m.dim() < x.dim():
        m = m[..., None]
    denom = (m.expand_as(x)).sum().clamp(min=1.0)
    return (x * m).sum() / denom


class PoseLoss(torch.nn.Module):
    def __init__(self, lcfg: dict):
        super().__init__()
        self.cfg = lcfg
        w = sk.part_weights(hand=float(lcfg.get("hand_weight", 2.0)), body=float(lcfg.get("body_weight", 1.0)),
                            face=float(lcfg.get("face_weight", 0.5)), brows=float(lcfg.get("brow_weight", 0.5)))
        self.register_buffer("joint_w", torch.from_numpy(w))
        e = torch.tensor(sk.BONE_LOSS_EDGES, dtype=torch.long)
        self.register_buffer("bone_a", e[:, 0])
        self.register_buffer("bone_b", e[:, 1])

    def forward(self, pred, gt, joint_valid, frame_mask, counter_pred=None, counter_gt=None,
                feat_pred=None, feat_gt=None) -> dict[str, torch.Tensor]:
        if pred.shape != gt.shape:
            raise ValueError(f"pred {tuple(pred.shape)} vs gt {tuple(gt.shape)}")
        c = self.cfg
        pred, gt = pred.float(), gt.float()
        m = joint_valid & frame_mask[..., None]                      # (B,T,J)
        kind, beta = c.get("pose_loss_type", "smooth_l1"), float(c.get("smooth_l1_beta", 0.1))
        out: dict[str, torch.Tensor] = {}

        e = _elem(pred, gt, kind, beta).mean(-1) * self.joint_w      # (B,T,J)
        l_pose = (e * m).sum() / (m.to(e.dtype) * self.joint_w).sum().clamp(min=1.0)
        hw = float(c.get("hand_relative_weight", 1.0))
        if hw > 0:
            terms = []
            for part, root in (("left", sk.L_HAND_ROOT), ("right", sk.R_HAND_ROOT)):
                sl = sk.PART_SLICES[part]
                pr = pred[:, :, sl] - pred[:, :, root:root + 1]
                gr = gt[:, :, sl] - gt[:, :, root:root + 1]
                mm = m[:, :, sl] & m[:, :, root:root + 1]
                terms.append(_masked_mean(_elem(pr, gr, kind, beta).mean(-1), mm))
            l_pose = l_pose + hw * sum(terms) / len(terms)
        out["pose"] = l_pose

        mv = m[:, 1:] & m[:, :-1]
        out["velocity"] = _masked_mean((torch.diff(pred, dim=1) - torch.diff(gt, dim=1)).abs().mean(-1), mv)
        if pred.shape[1] >= 3:
            ma = m[:, 2:] & m[:, 1:-1] & m[:, :-2]
            out["acceleration"] = _masked_mean(
                (torch.diff(pred, n=2, dim=1) - torch.diff(gt, n=2, dim=1)).abs().mean(-1), ma)
        else:
            out["acceleration"] = pred.new_zeros(())

        bp = (pred[:, :, self.bone_a] - pred[:, :, self.bone_b]).norm(dim=-1)
        bg = (gt[:, :, self.bone_a] - gt[:, :, self.bone_b]).norm(dim=-1)
        mb = m[:, :, self.bone_a] & m[:, :, self.bone_b]
        out["bone"] = _masked_mean((bp - bg).abs(), mb)

        if counter_pred is not None:
            out["counter"] = _masked_mean((counter_pred.float() - counter_gt.float()) ** 2, frame_mask)
        if feat_pred is not None:
            # Relative MSE: scale-free, ~1 when the prediction carries no signal.
            fg = feat_gt.float()
            rel = ((feat_pred.float() - fg) ** 2).mean(-1) / (fg ** 2).mean(-1).detach().clamp(min=1e-6)
            out["feat"] = _masked_mean(rel, frame_mask)

        lam = {"pose": c.get("lambda_pose", 1.0), "velocity": c.get("lambda_vel", 1.0),
               "acceleration": c.get("lambda_acc", 0.5), "bone": c.get("lambda_bone", 0.1),
               "counter": c.get("lambda_counter", 1.0), "feat": c.get("lambda_feat", 0.1)}
        total = pred.new_zeros(())
        for k, v in out.items():
            total = total + float(lam[k]) * v
        out["total"] = total
        return out
