
import torch

def cross_entropy(logits, targets):

    # Calculate log(sum(exp(logits))) for each sample
    log_sum_exp = torch.logsumexp(logits, dim=1)

    # Extract the logits corresponding to the correct classes
    correct_logits = logits.gather(
        1, targets.unsqueeze(1)
    ).squeeze(1)

    # Calculate negative log likelihood
    losses = log_sum_exp - correct_logits

    # Average across the batch
    return losses.mean()

    
