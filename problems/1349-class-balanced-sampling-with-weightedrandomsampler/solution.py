import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(labels, num_samples):
    # TODO: count how many items belong to each class
    # TODO: weight each ITEM by 1 / count of its own class
    # TODO: return WeightedRandomSampler(weights, num_samples, replacement=True)
    counts = torch.bincount(labels)
    weights = []
    for item in labels:
        weights.append(1/counts[item])
    weights = torch.tensor(weights)
    return WeightedRandomSampler(weights, num_samples, replacement=True)

