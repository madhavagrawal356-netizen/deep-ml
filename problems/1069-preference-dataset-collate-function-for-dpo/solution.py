import numpy as np

def collate_preference_batch(batch, pad_token_id=0):
    """
    Collate a list of preference items into a padded batch for DPO training.

    Args:
        batch: list of dicts with keys 'prompt', 'chosen', 'rejected'
               (each value is a list of int token ids)
        pad_token_id: int used to pad sequences to a common length

    Returns:
        dict with 'chosen_input_ids', 'rejected_input_ids',
                  'chosen_mask', 'rejected_mask' (all nested Python lists)
    """
    max_len=0
    chosen_input_ids = []
    rejected_input_ids = []
    chosen_mask = []
    rejected_mask = []
    prompt_lengths=[]
    for item in batch:
        chosen = item['prompt']+item['chosen']
        rejected = item['prompt']+item['rejected']
        prompt_lengths.append(len(item['prompt']))
        if len(chosen)>max_len:
            max_len = len(chosen)
        if len(rejected)>max_len:
            max_len = len(rejected)
        chosen_input_ids.append(chosen)
        rejected_input_ids.append(rejected)
    for item in chosen_input_ids:
        chosen=[]
        if len(item)<max_len:
            i=len(item)
            for _ in range(i, max_len):
                item.append(pad_token_id) 
        for i in range(len(item)):
            chosen.append(item[i]!=pad_token_id and i>=prompt_lengths[len(chosen_mask)])
        chosen_mask.append(chosen)
    for item in rejected_input_ids:
        rejected=[]
        if len(item)<max_len:
            i = len(item)
            for _ in range(i, max_len):
                item.append(pad_token_id)
        for i in range(len(item)): 
            rejected.append(item[i]!=pad_token_id and i>=prompt_lengths[len(rejected_mask)])
        rejected_mask.append(rejected)
    return {'chosen_input_ids': chosen_input_ids, 'rejected_input_ids': rejected_input_ids, 'chosen_mask': chosen_mask, 'rejected_mask': rejected_mask}
