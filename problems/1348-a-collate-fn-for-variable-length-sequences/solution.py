import torch
import torch.nn as nn

def pad_collate(batch, pad_value=0):
    # 1. Split sequences and labels
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    # 2. Compute lengths and maximum length
    lengths = torch.tensor([len(seq) for seq in sequences])
    B = len(sequences)
    T_max = lengths.max().item()

    # 3. Create padded tensor
    padded = torch.full((B, T_max), pad_value, dtype=torch.long)

    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = torch.as_tensor(seq, dtype=torch.long)

    # 4. Create boolean mask: True for real tokens
    mask = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)

    # 5. Return results
    return padded, lengths, mask, labels
    
