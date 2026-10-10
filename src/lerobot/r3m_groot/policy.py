"""Inference for the protocol-B checkpoints, without loading the GR00T VLM.

The encoder implementation is supplied by the training bundle. Its preprocessing
is deliberately preserved: replacing it with a generic HF pooler changes features.
"""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARMS = ("official", "semvggtglobaldelta_corpusnoreg-tcn01_bs128x2", "siglipfrozen", "vc1")


def vector(value, name):
    """Validate the six values at the policy/hardware boundary."""
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain six finite joint values, got {result}")
    return result


def prepare_image(image):
    """Resize raw RGB pixels exactly as training frame extraction did."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Camera frames must be HWC RGB uint8, without policy normalization")
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


def load_checkpoint(path):
    """Read a trusted local training checkpoint and validate its normalization."""
    # These local training artifacts contain numpy arrays, not just tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint["arm"] not in ARMS:
        raise ValueError(f"Unsupported encoder: {checkpoint['arm']}")
    if not isinstance(checkpoint["horizon"], int) or checkpoint["horizon"] < 1:
        raise ValueError("Checkpoint horizon must be positive")
    for name in ("action_mean", "action_std", "state_mean", "state_std"):
        value = vector(checkpoint[name], name)
        if name.endswith("std") and (value <= 0).any():
            raise ValueError(f"{name} must be positive")
    if not checkpoint["text_features"]:
        raise ValueError("Checkpoint has no task embeddings")
    return checkpoint


def load_encoder(bundle, arm, device):
    """Load the supplied implementation with explicit paths, without changing HOME."""
    spec = importlib.util.spec_from_file_location("so101_training_encoders", bundle / "so101_encoders.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUNS = str(bundle / "encoders/runs")
    # The original loader reads BASELINE_CKPTS and calls an R3M HOME-based loader.
    # Give this private module an os proxy; never mutate the process environment.
    import os
    from types import SimpleNamespace

    module.os = SimpleNamespace(
        path=os.path,
        environ={**os.environ, "BASELINE_CKPTS": str(bundle / "encoders/baseline_ckpts")},
    )
    if arm == "official":
        import hydra
        from omegaconf import OmegaConf
        from r3m import cleanup_config, remove_language_head

        root = bundle / "encoders/r3m_home/.r3m/original_r3m"
        cfg = cleanup_config(OmegaConf.load(root / "config.yaml"))
        cfg.device = device
        model = nn.DataParallel(hydra.utils.instantiate(cfg))
        weights = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(remove_language_head(weights["r3m"]), strict=True)
        model = model.module.to(device).eval().requires_grad_(False)
        module.load_r3m_arm = lambda *args: model
    return module.get_encoder(arm, 93749, device)[0]


class R3MGrootPolicy:
    """Raw RGB + joint state + task -> absolute action chunk in training units.

    Arm joints use degrees; the LeRobot gripper uses its calibrated 0..100 range.
    The checkpoint performs mean/std normalization itself. No relative-action or
    percentile processor should be applied to its inputs or outputs.
    """

    def __init__(self, checkpoint, bundle, groot, device="cuda:0"):
        """Strict-load the action head and projectors, plus the training encoder."""
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Use --device cpu for offline checks or fix the GPU driver."
            )
        c = load_checkpoint(checkpoint)
        self.arm, self.horizon = c["arm"], c["horizon"]
        self.tasks = tuple(c["text_features"])
        self.state_mean, self.state_std = c["state_mean"], c["state_std"]
        self.action_mean, self.action_std = c["action_mean"], c["action_std"]
        self.pca, self.text = c["pca"], c["text_features"]
        self.state_dim = c["max_state_dim"]
        self.encode = load_encoder(Path(bundle), self.arm, str(self.device))

        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

        cfg = Gr00tN1d7Config.from_pretrained(str(groot), local_files_only=True)
        if self.horizon > cfg.action_horizon or self.state_dim != cfg.max_state_dim:
            raise ValueError("GR00T config and checkpoint dimensions disagree")
        # Meta construction avoids keeping two full copies of the action head.
        # Convert to fp32 to match the supplied deployment implementation.
        with torch.device("meta"):
            self.head = Gr00tN1d7ActionHead(cfg)
        weights = {key: value.to(device=self.device, dtype=torch.float32) for key, value in c["head"].items()}
        self.head.load_state_dict(weights, strict=True, assign=True)
        self.head.eval().requires_grad_(False)
        self.projectors = nn.ModuleList(
            [nn.Linear(c["proj"][f"{i}.weight"].shape[1], cfg.backbone_embedding_dim) for i in range(3)]
        ).to(self.device)
        self.projectors.load_state_dict(c["proj"], strict=True)
        self.projectors.eval().requires_grad_(False)

    def _feature(self, image, camera):
        feature = self.encode(prepare_image(image)[None])
        if self.pca is not None:
            mean, basis = self.pca[camera]
            feature = (feature - mean) @ basis
        return torch.as_tensor(feature, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def predict_action_chunk(self, front, wrist, state, task):
        """Predict an Hx6 absolute target chunk from one unnormalized observation."""
        from transformers.feature_extraction_utils import BatchFeature

        if task not in self.tasks:
            raise ValueError(f"Unknown task {task!r}; choose one of {self.tasks}")
        state = vector(state, "state")
        tokens = torch.stack(
            [
                self.projectors[0](self._feature(front, "front")),
                self.projectors[1](self._feature(wrist, "wrist")),
                self.projectors[2](
                    torch.as_tensor(self.text[task], dtype=torch.float32, device=self.device)[None]
                ),
            ],
            dim=1,
        )
        padded_state = torch.zeros(1, 1, self.state_dim, device=self.device)
        padded_state[0, 0, :6] = torch.as_tensor(
            (state - self.state_mean) / self.state_std, device=self.device
        )
        backbone = BatchFeature(
            data={
                "backbone_features": tokens,
                "backbone_attention_mask": torch.ones(1, 3, dtype=torch.bool, device=self.device),
                "image_mask": torch.tensor([[True, True, False]], device=self.device),
            }
        )
        # Never supply `action`: GR00T interprets it as a denoising prefix (RTC).
        inputs = BatchFeature(
            data={
                "state": padded_state,
                "embodiment_id": torch.zeros(1, dtype=torch.long, device=self.device),
            }
        )
        result = self.head.get_action(backbone, inputs)["action_pred"]
        action = result[0, : self.horizon, :6].float().cpu().numpy()
        action = action * self.action_std + self.action_mean
        if action.shape != (self.horizon, 6) or not np.isfinite(action).all():
            raise ValueError("Policy produced invalid actions")
        return action
