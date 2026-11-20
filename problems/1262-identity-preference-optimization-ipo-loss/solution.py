import numpy as np


def ipo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float = 0.1,
) -> float:
    """
    Compute the Identity Preference Optimization (IPO) loss.

    Args:
        policy_chosen_logps: Sequence log-probabilities of chosen responses under policy.
        policy_rejected_logps: Sequence log-probabilities of rejected responses under policy.
        ref_chosen_logps: Sequence log-probabilities of chosen responses under reference model.
        ref_rejected_logps: Sequence log-probabilities of rejected responses under reference model.
        beta: Regularization parameter (> 0) controlling divergence from reference.

    Returns:
        Mean batch IPO loss as a Python float rounded to 6 decimal places.
    """
    policy_chosen = np.asarray(policy_chosen_logps, dtype=np.float64)
    policy_rejected = np.asarray(policy_rejected_logps, dtype=np.float64)
    ref_chosen = np.asarray(ref_chosen_logps, dtype=np.float64)
    ref_rejected = np.asarray(ref_rejected_logps, dtype=np.float64)

    # Preference margin h_theta(y_w, y_l)
    chosen_logratios = policy_chosen - ref_chosen
    rejected_logratios = policy_rejected - ref_rejected
    h_theta = chosen_logratios - rejected_logratios

    # Target log-odds margin 1 / (2 * beta)
    target = 1.0 / (2.0 * beta)

    # Mean squared error against the target margin
    loss = np.mean((h_theta - target) ** 2)

    return round(float(loss), 6)