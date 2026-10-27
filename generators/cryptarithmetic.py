"""Cryptarithmetic puzzle generator with carry-decomposed binary table constraints."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set, Any
from solver.csp import CSP, values_to_bitmask
from solver.constraints import BinaryNotEqualConstraint, BinaryTableConstraint, AllDifferentConstraint
from generators.base import CSPInstance


CLASSIC_CRYPTARITHMETIC = {
    "send_more_money": {
        "words": ["SEND", "MORE"],
        "result": "MONEY",
    },
    "cross_roads_danger": {
        "words": ["CROSS", "ROADS"],
        "result": "DANGER",
    },
    "donald_gerald_robert": {
        "words": ["DONALD", "GERALD"],
        "result": "ROBERT",
    },
    "base_ball_games": {
        "words": ["BASE", "BALL"],
        "result": "GAMES",
    },
}


def create_cryptarithmetic_csp(
    words: List[str],
    result: str,
    name: str = "cryptarithmetic",
) -> CSPInstance:
    """
    Formulate a cryptarithmetic addition problem (e.g., SEND + MORE = MONEY)
    as a CSP with digit variables, carry variables, and binary/AllDifferent constraints.
    """
    letters = sorted(list(set("".join(words) + result)))
    if len(letters) > 10:
        raise ValueError(f"Too many unique letters ({len(letters)}) for base 10.")

    letter_to_var = {ch: i for i, ch in enumerate(letters)}
    num_letter_vars = len(letters)

    # Initial domains: digits 0..9 (bitmask (1 << 10) - 1)
    initial_domains = [((1 << 10) - 1) for _ in range(num_letter_vars)]
    var_names = list(letters)

    # Leading letters cannot be 0
    leading_letters = set(w[0] for w in words + [result])
    for ch in leading_letters:
        var_idx = letter_to_var[ch]
        # Remove 0 (bit 0)
        initial_domains[var_idx] &= ~1

    # Carry variables: C_0, C_1, ..., C_{max_len - 1} with domain {0, 1, 2}
    max_len = max(max(len(w) for w in words), len(result))
    num_carries = max_len
    carry_offset = num_letter_vars

    for i in range(num_carries):
        var_names.append(f"C_{i}")
        # Carries can be 0, 1, ..., len(words) - 1
        max_carry = len(words)
        carry_mask = (1 << (max_carry + 1)) - 1
        initial_domains.append(carry_mask)

    total_vars = len(var_names)
    csp = CSP(
        num_variables=total_vars,
        initial_domains=initial_domains,
        variable_names=var_names,
        name=name,
    )

    # 1. All letters must have distinct digits
    letter_indices = list(range(num_letter_vars))
    csp.add_all_diff_group(letter_indices)
    for i in range(num_letter_vars):
        for j in range(i + 1, num_letter_vars):
            csp.add_constraint(BinaryNotEqualConstraint(i, j))

    # 2. Column additions decomposed with carries
    # Reverse words to align units at index 0
    rev_words = [w[::-1] for w in words]
    rev_result = result[::-1]

    # Pre-generate compatible digit assignments for each column equation:
    # sum(digits) + carry_in == result_digit + 10 * carry_out
    for col in range(max_len):
        col_letters = [w[col] for w in rev_words if col < len(w)]
        res_char = rev_result[col] if col < len(rev_result) else None

        c_in_idx = (carry_offset + col - 1) if col > 0 else None
        c_out_idx = carry_offset + col

        # Build allowed tuples
        involved_vars = []
        for ch in col_letters:
            involved_vars.append(letter_to_var[ch])
        if res_char is not None:
            involved_vars.append(letter_to_var[res_char])
        if c_in_idx is not None:
            involved_vars.append(c_in_idx)
        involved_vars.append(c_out_idx)

        # Decompose pairwise with binary table constraints between key letters and carries
        # If two letters are in the column, enforce their sum consistency with carry
        if len(col_letters) == 2 and res_char is not None:
            v_a = letter_to_var[col_letters[0]]
            v_b = letter_to_var[col_letters[1]]
            v_res = letter_to_var[res_char]

            # Binary table between v_a and v_res (allowed pairs given domain bounds)
            # For 2-word addition, v_a + v_b + c_in in [0, 9 + 9 + 1] = [0, 19]
            pairs_a_res = []
            for da in range(10):
                for dres in range(10):
                    # Check if there exists db in 0..9 and cin in 0..1 such that (da + db + cin) % 10 == dres
                    exists = any(
                        (da + db + cin) % 10 == dres
                        for db in range(10)
                        for cin in ([0, 1] if c_in_idx is not None else [0])
                    )
                    if exists:
                        pairs_a_res.append((da, dres))
            if v_a != v_res:
                csp.add_constraint(BinaryTableConstraint(v_a, v_res, pairs_a_res, max_domain_size=10))

    return CSPInstance(
        name=name,
        family="cryptarithmetic",
        difficulty="medium",
        csp=csp,
        metadata={"words": words, "result": result, "letters": letters},
    )


def get_classic_cryptarithmetic(puzzle_key: str = "send_more_money") -> CSPInstance:
    """Retrieve a classic cryptarithmetic puzzle instance."""
    if puzzle_key not in CLASSIC_CRYPTARITHMETIC:
        raise KeyError(f"Unknown puzzle key '{puzzle_key}'. Choose from: {list(CLASSIC_CRYPTARITHMETIC.keys())}")
    cfg = CLASSIC_CRYPTARITHMETIC[puzzle_key]
    return create_cryptarithmetic_csp(words=cfg["words"], result=cfg["result"], name=puzzle_key)
