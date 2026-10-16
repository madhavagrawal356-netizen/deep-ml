import math
from collections import Counter


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    n = len(labels)
    if n == 0:
        return 0.0
    h = 0.0
    for count in Counter(labels).values():
        p = count / n
        h -= p * math.log2(p)
    return h


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    n = len(examples)
    if n == 0:
        return 0.0

    base = calculate_entropy([e[target_attr] for e in examples])

    # partition examples by the value of attr
    subsets = {}
    for e in examples:
        subsets.setdefault(e[attr], []).append(e[target_attr])

    remainder = sum(
        (len(sub) / n) * calculate_entropy(sub) for sub in subsets.values()
    )
    return base - remainder


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(e[target_attr] for e in examples)
    # highest count first, then alphabetical
    return min(counts, key=lambda label: (-counts[label], label))


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm.

    Tree format: an internal node is {attr: {value: subtree, ...}},
    and a leaf is the predicted class label itself.
    """
    labels = [e[target_attr] for e in examples]

    # all examples share one class -> leaf
    if len(set(labels)) == 1:
        return labels[0]

    # no attributes left to split on -> majority leaf
    if not attributes:
        return majority_class(examples, target_attr)

    # pick the attribute with the highest gain (ties go to the earliest in `attributes`)
    best = max(attributes, key=lambda a: calculate_information_gain(examples, a, target_attr))
    remaining = [a for a in attributes if a != best]

    tree = {best: {}}
    for value in sorted({e[best] for e in examples}, key=str):
        subset = [e for e in examples if e[best] == value]
        tree[best][value] = learn_decision_tree(subset, remaining, target_attr)
    return tree