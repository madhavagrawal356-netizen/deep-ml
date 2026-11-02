import numpy as np

def compute_log_prob_ratios(
    policy_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
    mask: np.ndarray
) -> dict:
    """
    Compute log-probability ratios between policy and reference models.
    """

    # Difference between policy and reference log-probabilities
    per_token_log_ratios = (
        policy_log_probs - reference_log_probs
    ) * mask

    # Sum over valid tokens for each sequence
    sequence_log_ratios = np.sum(per_token_log_ratios, axis=1)

    # Average across sequences
    mean_sequence_log_ratio = float(np.mean(sequence_log_ratios))

    return {
        'per_token_log_ratios': per_token_log_ratios,
        'sequence_log_ratios': sequence_log_ratios,
        'mean_sequence_log_ratio': mean_sequence_log_ratio
    }