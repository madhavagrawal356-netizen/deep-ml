
def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    reference = reference.lower().split()
    candidate = candidate.lower().split()

    if not reference or not candidate:
        return 0.0

    # Match each candidate unigram to one unused reference occurrence.
    used = set()
    positions = []

    for word in candidate:
        for i, ref_word in enumerate(reference):
            if word == ref_word and i not in used:
                used.add(i)
                positions.append(i)
                break

    matches = len(positions)

    if matches == 0:
        return 0.0

    precision = matches / len(candidate)
    recall = matches / len(reference)

    # METEOR weighted harmonic mean
    f_mean = (precision * recall) / (
        alpha * precision + (1 - alpha) * recall
    )

    # Count contiguous chunks in the matched position sequence
    chunks = 0
    for i, pos in enumerate(positions):
        if i == 0 or pos != positions[i - 1] + 1:
            chunks += 1

    # Fragmentation penalty
    penalty = gamma * (chunks / matches) ** beta

    return f_mean * (1 - penalty)


    