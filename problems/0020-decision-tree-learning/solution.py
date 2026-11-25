import math
from collections import Counter


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    total = len(labels)
    if total == 0:
        return 0.0

    counts = Counter(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    total = len(examples)
    if total == 0:
        return 0.0

    parent_entropy = calculate_entropy([ex[target_attr] for ex in examples])

    subsets = {}
    for ex in examples:
        subsets.setdefault(ex[attr], []).append(ex[target_attr])

    weighted_child_entropy = sum(
        (len(child_labels) / total) * calculate_entropy(child_labels)
        for child_labels in subsets.values()
    )

    return parent_entropy - weighted_child_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(ex[target_attr] for ex in examples)
    # Sort by (-count, label) so highest frequency wins and alphabetical order breaks ties
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    labels = [ex[target_attr] for ex in examples]

    # Base case 1: All examples share the same target label (pure node)
    if len(set(labels)) == 1:
        return labels[0]

    # Base case 2: No remaining attributes to split on
    if not attributes:
        return majority_class(examples, target_attr)

    # Select best attribute using strictly greater comparison so earlier attributes win ties
    best_attr = attributes[0]
    best_gain = -1.0
    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    # Build subtree by splitting on sorted distinct values of best_attr
    tree = {best_attr: {}}
    remaining_attrs = [attr for attr in attributes if attr != best_attr]
    distinct_values = sorted({ex[best_attr] for ex in examples})

    for val in distinct_values:
        subset = [ex for ex in examples if ex[best_attr] == val]
        if not subset:
            tree[best_attr][val] = majority_class(examples, target_attr)
        else:
            tree[best_attr][val] = learn_decision_tree(subset, remaining_attrs, target_attr)

    return tree