"""Loss and target utilities for Arm C training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def blend_report_and_oof_targets(
    report_posterior: torch.Tensor,
    oof_teacher: torch.Tensor,
    reliability: torch.Tensor,
    gold_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-pass cross-fitted target refinement from the finalized strategy.

    ``reliability`` can be [targets] or broadcastable to [batch, targets].
    Gold cells, when provided, retain the report/gold value exactly.
    """

    if report_posterior.shape != oof_teacher.shape:
        raise ValueError("report_posterior and oof_teacher must have identical shapes")
    reliability = reliability.to(report_posterior).clamp(0, 1)
    target = reliability * report_posterior + (1 - reliability) * oof_teacher.detach()
    if gold_mask is not None:
        target = torch.where(gold_mask.to(torch.bool), report_posterior, target)
    return target


def masked_weighted_soft_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    cell_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft-label BCE with partial labels and per-cell weights.

    This supports the positive/negative/equivocal/unmentioned label pipeline:
    unmentioned cells are masked rather than forced to zero, and verified gold
    cells can receive (for example) an 8x ``cell_weight``.
    """

    if logits.shape != targets.shape or logits.shape != label_mask.shape:
        raise ValueError("logits, targets, and label_mask must have identical shapes")
    loss = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none", pos_weight=pos_weight
    )
    weight = label_mask.to(loss.dtype)
    if cell_weight is not None:
        weight = weight * cell_weight.to(loss.dtype)
    denominator = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denominator

