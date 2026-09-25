"""Sign-chat backend: an avatar that holds a conversation in Auslan.

    user text  ─────────────────────────┐
                                        ├─> dialogue model ─> reply text ─> SignSparK (fine-tuned) ─> avatar pose + video
    user video ─> rtmlib ─> Uni-Sign ───┘

The two sign-language models are the ones trained in this repository:
  * sign -> text: Uni-Sign pose-only, Arm A from openasl_pose_only_slt.pth, fine-tuned on Auslan-Daily (unisign/)
  * text -> sign: SignSparK fine-tuned on Auslan-Daily Communication, round 2 (auslan_smplx/)
"""

__version__ = "0.1.0"
