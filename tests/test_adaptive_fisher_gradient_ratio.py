import pytest

from fisher_log_barrier.continuous_mountain_car_experiment import (
    ContinuousMountainCarConfig,
    gradient_balanced_beta,
)


def test_gradient_balanced_beta_hits_requested_ratio():
    reward_norm = 12.0
    unscaled_fisher_norm = 30.0

    beta = gradient_balanced_beta(reward_norm, unscaled_fisher_norm, 0.05)

    assert beta == pytest.approx(0.02)
    assert beta * unscaled_fisher_norm / reward_norm == pytest.approx(0.05)


@pytest.mark.parametrize("reward_norm,fisher_norm", [(0.0, 3.0), (3.0, 0.0)])
def test_gradient_balanced_beta_disables_zero_norm_component(reward_norm, fisher_norm):
    assert gradient_balanced_beta(reward_norm, fisher_norm, 0.05) == 0.0


def test_target_gradient_ratio_is_validated():
    with pytest.raises(ValueError, match="target_fisher_gradient_ratio"):
        ContinuousMountainCarConfig(target_fisher_gradient_ratio=-0.01).validate()
