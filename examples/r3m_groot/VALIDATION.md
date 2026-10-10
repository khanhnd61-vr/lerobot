# Validation — 2026-10-10

Runtime: `/mnt/khanh/work/r3m/.venv`, Python 3.12.13, PyTorch 2.11.0+cu128,
Transformers 4.57.3, Diffusers 0.35.1. Inference ran on CPU with four threads.
Isaac-GR00T revision: `d2b7e75b937e3ec9aa5dbc798f08b89692c49734`.

- 13 regression tests passed, including construction of the real SO-101 driver
  without connecting, and verification of its arm/gripper normalization modes.
- Ruff lint and formatting checks, `git diff --check`, and launcher syntax passed.
- `uv pip check` found all 99 installed packages compatible.
- All four encoders produced finite features of the expected width:
  R3M/VIGIL 2048, SigLIP-2 1536, VC-1 768.
- All four checkpoints strict-loaded the action head and projectors and completed
  two inference calls through the documented launcher, using cached episode 2,
  seed 0, `--device cpu --replan 1 --max-steps 2`.

| Encoder | Raw executed-action MAE | Raw future-chunk MAE | Mean inference seconds |
|---|---:|---:|---:|
| VIGIL | 3.236 | 3.254 | 1.348 |
| R3M | 6.938 | 5.799 | 1.372 |
| SigLIP-2 | 2.448 | 2.433 | 1.673 |
| VC-1 | 2.590 | 2.756 | 1.832 |

These are **two-frame smoke results**, not full validation scores, policy rankings,
or robot success rates. Aggregate MAE mixes arm degrees and gripper percentage
points. Guarded MAE was approximately 19 for all four: the dataset contains a
large action/state mismatch, especially the almost constant wrist-roll readback.
The adapter limits motion against the readback instead of bypassing that limit
to make replay error look better.

No physical rollout or GPU benchmark was performed. `nvidia-smi` cannot
communicate with the NVIDIA driver on this host. Robot calibration ID and actual
camera devices were not supplied. See [README.md](README.md) for those arguments.

To repeat the regression suite from the checkout:

```bash
PYTHONPATH=src uv run --no-project \
  --python /mnt/khanh/work/r3m/.venv/bin/python -m pytest tests/r3m_groot -q
```
