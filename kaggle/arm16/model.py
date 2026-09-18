"""Arm C: global MIL with target-specific local-ROI residuals.

This module deliberately keeps the proven Arm 12/15 global path intact:

    ConvNeXt -> LayerNorm -> 12 finding-specific attention queries -> 12 logits

Five weak targets additionally receive a small residual from an anatomy-specific
bag of local 2.5-D windows.  The same image encoder is used for the global and
local bags.  A learned sigmoid gate is initialized at sigmoid(-2) ~= 0.119 so
the new path cannot dominate a warm-started global model at initialization.

Input contract
--------------
global_windows: [B, W, 3, H, W]
global_mask:    [B, W] boolean; True means a real window
roi_windows:    mapping from ROI target name to [B, R, 3, H, W]
roi_masks:      mapping from ROI target name to [B, R] boolean

The allowed ROI names are the five entries in ROI_TARGETS.  A missing ROI bag,
or a sample whose entire ROI mask is false, contributes exactly zero residual.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


LABELS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

ROI_TARGETS = (
    "MCL",
    "Lateral Meniscus",
    "Lateral OA",
    "PF OA",
    "Synovitis",
)
ROI_TARGET_INDICES = {name: LABELS.index(name) for name in ROI_TARGETS}


@dataclass(frozen=True)
class ArmCOutput:
    """Optional diagnostic output from :class:`GlobalLocalResidualMIL`."""

    logits: torch.Tensor
    global_logits: torch.Tensor
    roi_residuals: dict[str, torch.Tensor]
    roi_gates: dict[str, torch.Tensor]
    global_attention: torch.Tensor | None = None
    roi_attention: dict[str, torch.Tensor] | None = None


def build_backbone(
    arch: str = "convnext_small.in12k_ft_in1k_384",
    pretrained: bool = True,
) -> nn.Module:
    """Build the same feature-only timm backbone used by Arm 12/15."""

    if arch == "tiny_test":
        net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        net.num_features = 16
        return net

    import timm

    return timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=0,
        in_chans=3,
        global_pool="avg",
    )


def _validate_bag(name: str, windows: torch.Tensor, mask: torch.Tensor) -> None:
    if windows.ndim != 5:
        raise ValueError(f"{name} windows must be [B,K,C,H,W], got {tuple(windows.shape)}")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(windows.shape[:2]):
        raise ValueError(
            f"{name} mask must be [B,K]={tuple(windows.shape[:2])}, got {tuple(mask.shape)}"
        )
    if windows.shape[2] != 3:
        raise ValueError(f"{name} windows must contain 3-slice channels, got C={windows.shape[2]}")


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Stable masked softmax which returns all-zero rows for empty bags."""

    mask = mask.to(device=scores.device, dtype=torch.bool)
    masked = scores.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    weights = torch.softmax(masked, dim=dim).to(scores.dtype)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class TargetQueryMIL(nn.Module):
    """The existing Raptor/Arm 12 per-finding attention MIL head.

    Attribute names intentionally remain ``att``, ``clsW`` and ``clsb`` so
    those weights can be copied directly from an Arm 12/15 checkpoint.
    """

    def __init__(self, feature_dim: int, n_targets: int = len(LABELS), drop: float = 0.2):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, n_targets),
        )
        self.clsW = nn.Parameter(torch.zeros(n_targets, feature_dim))
        self.clsb = nn.Parameter(torch.zeros(n_targets))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        scores = self.att(features)
        weights = _masked_softmax(scores, mask.unsqueeze(-1), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", weights, features)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        if return_attention:
            return logits, weights
        return logits


class GatedAttentionResidual(nn.Module):
    """One target-specific Ilse-style gated attention pooler and readout."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, drop: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.value = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Sigmoid())
        self.score = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden_dim, 1))
        self.readout = nn.Linear(feature_dim, 1)
        nn.init.trunc_normal_(self.readout.weight, std=0.02)
        nn.init.zeros_(self.readout.bias)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mask = mask.to(device=features.device, dtype=torch.bool)
        h = self.norm(features)
        scores = self.score(self.value(h) * self.gate(h)).squeeze(-1)
        weights = _masked_softmax(scores, mask, dim=1)
        pooled = torch.einsum("bk,bkf->bf", weights, h)
        residual = self.readout(pooled).squeeze(-1)
        # Empty bags must be a true no-op, including the readout bias.
        residual = residual * mask.any(dim=1).to(residual.dtype)
        if return_attention:
            return residual, weights
        return residual


class GlobalLocalResidualMIL(nn.Module):
    """ConvNeXt global MIL plus five bounded target-specific ROI residuals."""

    def __init__(
        self,
        backbone: nn.Module,
        n_targets: int = len(LABELS),
        roi_targets: tuple[str, ...] = ROI_TARGETS,
        attention_dim: int = 256,
        drop: float = 0.2,
        beta_init: float = -2.0,
        encoder_chunk_size: int | None = None,
    ):
        super().__init__()
        if n_targets != len(LABELS):
            raise ValueError(f"Arm C expects {len(LABELS)} targets, got {n_targets}")
        unknown = set(roi_targets) - set(ROI_TARGETS)
        if unknown:
            raise ValueError(f"unsupported ROI targets: {sorted(unknown)}")

        feature_dim = int(backbone.num_features)
        self.backbone = backbone
        # Preserve the exact global normalization and parameter names from Arm 12/15.
        self.norm = nn.LayerNorm(feature_dim)
        self.global_head = TargetQueryMIL(feature_dim, n_targets=n_targets, drop=drop)
        self.roi_targets = tuple(roi_targets)
        self.roi_poolers = nn.ModuleDict(
            {
                name: GatedAttentionResidual(feature_dim, hidden_dim=attention_dim, drop=drop)
                for name in self.roi_targets
            }
        )
        self.beta = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(float(beta_init))) for name in self.roi_targets}
        )
        self.encoder_chunk_size = encoder_chunk_size

    # Compatibility aliases for the established global checkpoint vocabulary.
    @property
    def att(self) -> nn.Module:
        return self.global_head.att

    @property
    def clsW(self) -> nn.Parameter:
        return self.global_head.clsW

    @property
    def clsb(self) -> nn.Parameter:
        return self.global_head.clsb

    def _backbone_forward(self, images: torch.Tensor) -> torch.Tensor:
        chunk = self.encoder_chunk_size
        if chunk is None or images.shape[0] <= chunk:
            return self.backbone(images)
        return torch.cat([self.backbone(x) for x in images.split(chunk, dim=0)], dim=0)

    def encode_bag(self, windows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode only valid windows and scatter them back into a padded bag."""

        _validate_bag("input", windows, mask)
        mask = mask.to(device=windows.device, dtype=torch.bool)
        batch, count = mask.shape
        flat_mask = mask.reshape(-1)
        flat_windows = windows.flatten(0, 1)
        feature_dim = int(self.backbone.num_features)
        features = windows.new_zeros((batch * count, feature_dim))
        if flat_mask.any():
            valid_features = self._backbone_forward(flat_windows[flat_mask])
            features[flat_mask] = valid_features.to(features.dtype)
        return features.view(batch, count, feature_dim)

    def forward(
        self,
        global_windows: torch.Tensor,
        global_mask: torch.Tensor | None = None,
        roi_windows: Mapping[str, torch.Tensor] | None = None,
        roi_masks: Mapping[str, torch.Tensor] | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | ArmCOutput:
        if global_mask is None:
            global_mask = torch.ones(
                global_windows.shape[:2], device=global_windows.device, dtype=torch.bool
            )
        _validate_bag("global", global_windows, global_mask)
        if not global_mask.to(torch.bool).any(dim=1).all():
            raise ValueError("every study must contain at least one valid global window")

        global_features = self.norm(self.encode_bag(global_windows, global_mask))
        if return_details:
            global_logits, global_attention = self.global_head(
                global_features, global_mask, return_attention=True
            )
        else:
            global_logits = self.global_head(global_features, global_mask)
            global_attention = None
        logits = global_logits.clone()

        roi_windows = {} if roi_windows is None else dict(roi_windows)
        roi_masks = {} if roi_masks is None else dict(roi_masks)
        unknown = (set(roi_windows) | set(roi_masks)) - set(self.roi_targets)
        if unknown:
            raise KeyError(f"unknown ROI bags: {sorted(unknown)}")
        if set(roi_windows) != set(roi_masks):
            raise ValueError("roi_windows and roi_masks must contain the same target keys")

        residuals: dict[str, torch.Tensor] = {}
        gates: dict[str, torch.Tensor] = {}
        attentions: dict[str, torch.Tensor] = {}
        for name in self.roi_targets:
            if name not in roi_windows:
                continue
            windows, mask = roi_windows[name], roi_masks[name]
            _validate_bag(name, windows, mask)
            if windows.shape[0] != global_windows.shape[0]:
                raise ValueError(f"{name} ROI batch size differs from global batch size")
            roi_features = self.encode_bag(windows, mask)
            if return_details:
                residual, attention = self.roi_poolers[name](
                    roi_features, mask, return_attention=True
                )
                attentions[name] = attention
            else:
                residual = self.roi_poolers[name](roi_features, mask)
            gate = torch.sigmoid(self.beta[name])
            target_index = ROI_TARGET_INDICES[name]
            logits[:, target_index] = logits[:, target_index] + gate * residual
            residuals[name] = residual
            gates[name] = gate

        if return_details:
            return ArmCOutput(
                logits=logits,
                global_logits=global_logits,
                roi_residuals=residuals,
                roi_gates=gates,
                global_attention=global_attention,
                roi_attention=attentions,
            )
        return logits


def make_model(
    arch: str = "convnext_small.in12k_ft_in1k_384",
    pretrained: bool = True,
    **kwargs: Any,
) -> GlobalLocalResidualMIL:
    return GlobalLocalResidualMIL(build_backbone(arch, pretrained), **kwargs)


def load_global_checkpoint(
    model: GlobalLocalResidualMIL,
    checkpoint: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Warm-start Arm C's global path from an Arm 12/15 checkpoint.

    ROI modules and beta gates remain freshly initialized.  DDP's optional
    ``module.`` prefix is accepted.  Shape mismatches fail loudly.
    """

    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    remap = {
        "backbone.": "backbone.",
        "norm.": "norm.",
        "att.": "global_head.att.",
        "clsW": "global_head.clsW",
        "clsb": "global_head.clsb",
    }
    global_state: dict[str, torch.Tensor] = {}
    for old_key, value in state.items():
        for old_prefix, new_prefix in remap.items():
            if old_key == old_prefix or old_key.startswith(old_prefix):
                suffix = old_key[len(old_prefix) :]
                global_state[new_prefix + suffix] = value
                break
    if not global_state:
        raise ValueError("checkpoint contains no recognized Arm 12/15 global model weights")
    incompatible = model.load_state_dict(global_state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("roi_")]
    if unexpected:
        raise RuntimeError(f"unexpected global checkpoint keys: {unexpected}")
    loaded = set(global_state)
    required = {
        key
        for key in model.state_dict()
        if key.startswith(("backbone.", "norm.", "global_head."))
    }
    missing_global = sorted(required - loaded)
    if missing_global:
        raise RuntimeError(f"checkpoint is missing global weights: {missing_global[:10]}")
    return {
        "loaded_keys": len(loaded),
        "fresh_roi_targets": list(model.roi_targets),
        "initial_roi_gate": float(torch.sigmoid(next(iter(model.beta.values()))).detach()),
    }
