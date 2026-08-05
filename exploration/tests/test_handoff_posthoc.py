"""Focused verification for archive-only handoff post-hoc analysis."""

from pathlib import Path
import unittest

import numpy as np
import torch

from exploration.sampled_tabular_mdp.posthoc import (
    enumerated_fishers,
    exact_policy_metrics,
    exact_reward_continuation,
    fixed_reference_fisher,
    load_training_archive,
    sha256_file,
    tensor_metrics_to_numpy,
)
from exploration.sampled_tabular_mdp.run_handoff_posthoc import (
    CHECKPOINT_OFFSETS,
    FINAL_UPDATE,
    SOURCE_ROOT,
    SWITCH_TIMES,
    discover_switch_archives,
)
from exploration.tabular_mdp.geometry import reduced_categorical_fisher
from exploration.tabular_mdp.model import TwoStepTrap, probabilities_from_reduced_logits


class HandoffPosthocMathTests(unittest.TestCase):
    def setUp(self):
        self.phi = torch.tensor(
            ((0.4, -0.7, -0.2, 0.8), (1.1, -0.3, 0.6, -0.4)),
            dtype=torch.float64,
        )

    def test_behavior_and_reward_gradient_match_autograd(self):
        metrics = exact_policy_metrics(self.phi)
        mdp = TwoStepTrap()
        value = self.phi.detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(mdp.exact_return(value).sum(), value)[0]
        stored = torch.stack([metrics[f"reward_gradient_{i}"] for i in range(4)], dim=-1)
        torch.testing.assert_close(stored, gradient, atol=1e-12, rtol=1e-12)
        pi0, pi1 = probabilities_from_reduced_logits(self.phi)
        v1 = pi1[:, 0] + 0.2 * pi1[:, 1]
        torch.testing.assert_close(metrics["v1"], v1, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(metrics["delta_safe"], v1 - 0.5, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(metrics["population_return"], mdp.exact_return(self.phi), atol=1e-14, rtol=1e-14)

    def test_exact_fishers_match_score_enumeration(self):
        pi0, pi1 = probabilities_from_reduced_logits(self.phi)
        f0_enum, f1_enum = enumerated_fishers(self.phi)
        torch.testing.assert_close(f0_enum, reduced_categorical_fisher(pi0), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(f1_enum, reduced_categorical_fisher(pi1), atol=1e-14, rtol=1e-14)
        fref = fixed_reference_fisher(self.phi)
        torch.testing.assert_close(fref[:, :2, :2], 0.5 * f0_enum, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(fref[:, 2:, 2:], 0.5 * f1_enum, atol=1e-14, rtol=1e-14)

    def test_spectra_reconstruct_trace_logdet_and_gradient_norm(self):
        metrics = exact_policy_metrics(self.phi)
        dimensions = {"f0": 2, "f1": 2, "f_pool": 4, "f_ref": 4}
        for name, dimension in dimensions.items():
            eigenvalues = torch.stack(
                [metrics[f"{name}_eigenvalue_{i}"] for i in range(1, dimension + 1)],
                dim=-1,
            )
            torch.testing.assert_close(eigenvalues.sum(dim=-1), metrics[f"{name}_trace"], atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(torch.log(eigenvalues).sum(dim=-1), metrics[f"{name}_logdet"], atol=1e-11, rtol=1e-11)
        for name in ("f_pool", "f_ref"):
            projection_sq = sum(metrics[f"{name}_reward_projection_sq_{i}"] for i in range(1, 5))
            torch.testing.assert_close(projection_sq, metrics["reward_gradient_norm"].square(), atol=1e-12, rtol=1e-12)

    def test_exact_continuation_is_deterministic(self):
        first_steps, first = exact_reward_continuation(
            self.phi, start_update=10, final_update=30, alpha=0.05, record_interval=5
        )
        second_steps, second = exact_reward_continuation(
            self.phi, start_update=10, final_update=30, alpha=0.05, record_interval=5
        )
        np.testing.assert_array_equal(first_steps, second_steps)
        np.testing.assert_array_equal(first, second)


@unittest.skipUnless(SOURCE_ROOT.exists(), "handoff robustness archives are not available")
class HandoffPosthocArchiveTests(unittest.TestCase):
    def test_required_checkpoints_exist_without_rerun(self):
        archives = discover_switch_archives()
        self.assertEqual(len(archives), 10)
        for (_, switch), archive in archives.items():
            for step in (switch, *(switch + offset for offset in CHECKPOINT_OFFSETS[1:]), FINAL_UPDATE):
                archive.index_for_step(step)

    def test_loader_is_read_only(self):
        path = sorted(SOURCE_ROOT.glob("*.npz"))[0]
        before_stat = path.stat()
        before_hash = sha256_file(path)
        load_training_archive(path)
        after_stat = path.stat()
        self.assertEqual(before_hash, sha256_file(path))
        self.assertEqual(before_stat.st_size, after_stat.st_size)
        self.assertEqual(before_stat.st_mtime_ns, after_stat.st_mtime_ns)

    def test_reconstructed_endpoint_matches_archive(self):
        archive = discover_switch_archives()[("adverse", 2000)]
        index = archive.index_for_step(FINAL_UPDATE)
        reconstructed = tensor_metrics_to_numpy(exact_policy_metrics(archive.phi[index]))
        np.testing.assert_allclose(
            reconstructed["population_return"],
            archive.metrics["population_return"][index],
            atol=1e-12,
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
