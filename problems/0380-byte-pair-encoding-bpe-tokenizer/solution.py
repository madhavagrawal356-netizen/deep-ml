def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]

    """
    out = []
    for _ in range(num_merges):
        frq = {}
        for item, value in corpus.items():
            tokens = item.split()
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                frq[pair] = frq.get(pair, 0) + value
        max_freq = 0
        if not frq:
            break
        max_item = max(frq, key=frq.get)
        out.append(max_item)
        new_corpus = {}
        for item, value in corpus.items():
            tokens = item.split()
            merged = []
            i = 0

            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == max_item:
                    merged.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1

            new_item = " ".join(merged)
            new_corpus[new_item] = new_corpus.get(new_item, 0) + value

        corpus = new_corpus
    return out


        
