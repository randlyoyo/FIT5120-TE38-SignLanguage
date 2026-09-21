Shared notes for every arm config.

`data.manifest` / `data.npz_dir` must be the SAME pair across all arms. That is
what makes an arm-to-arm comparison a comparison. If you rebuild the manifest,
rerun every arm.

`backend.name: smoke` runs the toy model in smoke_model.py. It exists to test
the harness, never to produce a number. Switch to `unisign` once the checkpoint
is downloaded and UniSignBackend's two methods are wired up.

`group_patterns` decides what "the pose encoders" means for the freeze policy.
Verify it against the real checkpoint before trusting Arm B:

    python model_adapter.py --checkpoint <ckpt> --repo <clone> --report
