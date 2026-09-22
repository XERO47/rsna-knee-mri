import copy
import unittest

import torch

from losses import blend_report_and_oof_targets, masked_weighted_soft_bce
from model import LABELS, ROI_TARGETS, GlobalLocalResidualMIL, build_backbone, load_global_checkpoint


class ArmCModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GlobalLocalResidualMIL(build_backbone("tiny_test", pretrained=False), drop=0)
        self.model.eval()
        self.global_x = torch.randn(2, 4, 3, 24, 24)
        self.global_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)

    def test_shapes_gates_and_target_isolation(self):
        roi_x = {name: torch.randn(2, 3, 3, 24, 24) for name in ROI_TARGETS}
        roi_mask = {name: torch.ones(2, 3, dtype=torch.bool) for name in ROI_TARGETS}
        out = self.model(
            self.global_x, self.global_mask, roi_x, roi_mask, return_details=True
        )
        self.assertEqual(tuple(out.logits.shape), (2, len(LABELS)))
        for name in ROI_TARGETS:
            self.assertAlmostEqual(
                float(out.roi_gates[name].detach()),
                0.1192029,
                places=5,
            )
        untouched = [i for i, name in enumerate(LABELS) if name not in ROI_TARGETS]
        torch.testing.assert_close(out.logits[:, untouched], out.global_logits[:, untouched])

    def test_missing_and_empty_roi_are_no_ops(self):
        global_only = self.model(self.global_x, self.global_mask)
        empty_mask = torch.zeros(2, 3, dtype=torch.bool)
        with_empty = self.model(
            self.global_x,
            self.global_mask,
            {"MCL": torch.randn(2, 3, 3, 24, 24)},
            {"MCL": empty_mask},
        )
        torch.testing.assert_close(global_only, with_empty)

    def test_fresh_roi_heads_preserve_warm_started_global_logits_exactly(self):
        global_only = self.model(self.global_x, self.global_mask)
        roi_x = {name: torch.randn(2, 3, 3, 24, 24) for name in ROI_TARGETS}
        roi_mask = {name: torch.ones(2, 3, dtype=torch.bool) for name in ROI_TARGETS}
        with_fresh_roi = self.model(self.global_x, self.global_mask, roi_x, roi_mask)
        torch.testing.assert_close(global_only, with_fresh_roi, rtol=0, atol=0)

    @staticmethod
    def legacy_forward(model, global_x, global_mask, roi_x, roi_mask):
        """Reference the former one-backbone-call-per-ROI execution path."""

        global_features = model.norm(model.encode_bag(global_x, global_mask))
        global_logits = model.global_head(global_features, global_mask)
        logits = global_logits.clone()
        for name in model.roi_targets:
            features = model.encode_bag(roi_x[name], roi_mask[name])
            residual = model.roi_poolers[name](features, roi_mask[name])
            index = LABELS.index(name)
            logits[:, index] = logits[:, index] + torch.sigmoid(model.beta[name]) * residual
        return logits, global_logits

    def test_packed_roi_matches_legacy_logits_masks_and_gradients(self):
        packed = copy.deepcopy(self.model).train()
        legacy = copy.deepcopy(self.model).train()
        # Fresh ROI readouts are exactly zero. Give them identical nonzero
        # weights so the comparison also exercises backbone/attention grads.
        for candidate in (packed, legacy):
            for pooler in candidate.roi_poolers.values():
                pooler.readout.weight.data.fill_(0.025)
                pooler.readout.bias.data.fill_(0.01)

        roi_x = {name: torch.randn(2, 3, 3, 24, 24) for name in ROI_TARGETS}
        roi_mask = {
            name: torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
            for name in ROI_TARGETS
        }
        # Include a completely empty bag and prove it stays a true no-op.
        roi_mask["Synovitis"][0].zero_()

        packed_logits = packed(self.global_x, self.global_mask, roi_x, roi_mask)
        legacy_logits, legacy_global = self.legacy_forward(
            legacy, self.global_x, self.global_mask, roi_x, roi_mask
        )
        torch.testing.assert_close(packed_logits, legacy_logits, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            packed(self.global_x, self.global_mask),
            legacy_global,
            rtol=1e-6,
            atol=1e-6,
        )

        packed_logits.square().mean().backward()
        legacy_logits.square().mean().backward()
        for (packed_name, packed_parameter), (legacy_name, legacy_parameter) in zip(
            packed.named_parameters(), legacy.named_parameters()
        ):
            self.assertEqual(packed_name, legacy_name)
            if packed_parameter.grad is None or legacy_parameter.grad is None:
                self.assertIsNone(packed_parameter.grad)
                self.assertIsNone(legacy_parameter.grad)
                continue
            torch.testing.assert_close(
                packed_parameter.grad,
                legacy_parameter.grad,
                rtol=2e-5,
                atol=2e-6,
            )

    def test_packed_roi_reduces_backbone_invocations_to_two(self):
        calls = []
        handle = self.model.backbone.register_forward_hook(
            lambda _module, _inputs, _output: calls.append(1)
        )
        try:
            roi_x = {name: torch.randn(1, 3, 3, 24, 24) for name in ROI_TARGETS}
            roi_mask = {name: torch.ones(1, 3, dtype=torch.bool) for name in ROI_TARGETS}
            self.model(
                self.global_x[:1],
                self.global_mask[:1],
                roi_x,
                roi_mask,
            )
        finally:
            handle.remove()
        self.assertEqual(len(calls), 2, "expected one global and one packed ROI call")

    def test_masked_global_pixels_do_not_change_prediction(self):
        changed = self.global_x.clone()
        changed[0, 3] = 10000
        a = self.model(self.global_x, self.global_mask)
        b = self.model(changed, self.global_mask)
        torch.testing.assert_close(a, b)

    def test_arm13_checkpoint_mapping(self):
        state = self.model.state_dict()
        old = {}
        for key, value in state.items():
            if key.startswith("backbone.") or key.startswith("norm."):
                old[key] = value.clone()
            elif key.startswith("global_head.att."):
                old[key.replace("global_head.", "", 1)] = value.clone()
            elif key == "global_head.clsW":
                old["clsW"] = value.clone()
            elif key == "global_head.clsb":
                old["clsb"] = value.clone()
        fresh = GlobalLocalResidualMIL(build_backbone("tiny_test", pretrained=False), drop=0)
        summary = load_global_checkpoint(fresh, {"model": old})
        self.assertGreater(summary["loaded_keys"], 0)
        torch.testing.assert_close(
            self.model(self.global_x, self.global_mask),
            fresh(self.global_x, self.global_mask),
        )

    def test_loss_and_target_refinement(self):
        report = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
        teacher = torch.tensor([[0.0, 1.0], [0.6, 0.4]])
        reliability = torch.tensor([0.75, 0.25])
        refined = blend_report_and_oof_targets(report, teacher, reliability)
        expected = torch.tensor([[0.75, 0.75], [0.30, 0.50]])
        torch.testing.assert_close(refined, expected)
        logits = torch.zeros_like(refined, requires_grad=True)
        loss = masked_weighted_soft_bce(
            logits,
            refined,
            torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            pos_weight=torch.tensor([1.0, 2.0]),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(float(logits.grad[1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
