"""Instance-level dataset splitting ensuring 0% search node leakage."""

from __future__ import annotations
from typing import List, Tuple, Sequence
import random
import hashlib
from generators.base import CSPInstance


def split_instances(
    instances: Sequence[CSPInstance],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[CSPInstance], List[CSPInstance], List[CSPInstance]]:
    """
    Split benchmark instances strictly by instance identifier.
    
    Guarantees:
    - Zero data leakage: states from the same instance never appear in different splits.
    - Fully deterministic and reproducible given a seed.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # Sort instances deterministically by name
    sorted_instances = sorted(list(instances), key=lambda inst: inst.name)
    rng = random.Random(seed)
    indices = list(range(len(sorted_instances)))
    rng.shuffle(indices)

    n = len(sorted_instances)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))

    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train : n_train + n_val])
    test_idx = set(indices[n_train + n_val :])

    train = [sorted_instances[i] for i in sorted(train_idx)]
    val = [sorted_instances[i] for i in sorted(val_idx)]
    test = [sorted_instances[i] for i in sorted(test_idx)]

    # Assert no overlap
    train_names = set(inst.name for inst in train)
    val_names = set(inst.name for inst in val)
    test_names = set(inst.name for inst in test)

    assert not (train_names & val_names), "Data leakage: overlap between train and val splits!"
    assert not (train_names & test_names), "Data leakage: overlap between train and test splits!"
    assert not (val_names & test_names), "Data leakage: overlap between val and test splits!"

    return train, val, test
