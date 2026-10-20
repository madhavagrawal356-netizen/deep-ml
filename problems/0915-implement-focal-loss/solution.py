import torch
import torch.nn.functional as F
def focal_loss(logits, targets, gamma=2.0):
    log_probs = F.log_softmax(logits, dim=1)

    # Select the log probability of the correct class
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Convert log probability into probability
    pt = log_pt.exp()

    # Calculate focal loss for each example
    loss = -((1 - pt) ** gamma) * log_pt

    # Return the average loss across all examples
    return loss.mean()
    # TODO: mean focal loss over the batch
    
