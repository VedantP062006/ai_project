"""Sudoku CSP generator, parser, and benchmark loader for 9x9 and 16x16 grids."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import random
import csv
from pathlib import Path

from solver.csp import CSP, values_to_bitmask
from solver.constraints import BinaryNotEqualConstraint, AllDifferentConstraint
from generators.base import CSPInstance


PUBLISHED_HARD_PUZZLES = {
    # Arto Inkala's "AI Escargot" (2006)
    "ai_escargot": "1....7.9..3..2...8..96..5....53..9...1..8...26....4...3......1..4......7..7...3..",
    # Arto Inkala's 2012 "World's Hardest Sudoku"
    "inkala_2012": "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..",
    # "Golden Nugget"
    "golden_nugget": ".......39.....1..5..3.5.8....8.9...6.7...2...1...4.9....1.8.5..2..3.....49.......",
    # "Easter Monster"
    "easter_monster": "1.......2.9.4...5...6...7...5.9.3.......7.......85..4.7.....1...4...2.3.8.......9",
}


def parse_sudoku_string(puzzle_str: str, name: str = "sudoku_9x9", difficulty: str = "medium") -> CSPInstance:
    """
    Parse an 81-character Sudoku string ('.' or '0' for empty cells) into a CSPInstance.
    """
    cleaned = puzzle_str.strip().replace("\n", "").replace(" ", "")
    if len(cleaned) != 81:
        raise ValueError(f"Sudoku 9x9 string must have 81 characters, got {len(cleaned)}")

    num_vars = 81
    all_values_mask = (1 << 9) - 1
    initial_domains = []

    for char in cleaned:
        if char in ".0":
            initial_domains.append(all_values_mask)
        elif char in "123456789":
            val = int(char) - 1
            initial_domains.append(1 << val)
        else:
            raise ValueError(f"Invalid Sudoku character: {char}")

    csp = CSP(
        num_variables=num_vars,
        initial_domains=initial_domains,
        variable_names=[f"R{r}C{c}" for r in range(9) for c in range(9)],
        name=name,
    )

    # 1. Row constraints
    for r in range(9):
        row_vars = [r * 9 + c for c in range(9)]
        csp.add_all_diff_group(row_vars)
        for i in range(9):
            for j in range(i + 1, 9):
                csp.add_constraint(BinaryNotEqualConstraint(row_vars[i], row_vars[j]))

    # 2. Column constraints
    for c in range(9):
        col_vars = [r * 9 + c for r in range(9)]
        csp.add_all_diff_group(col_vars)
        for i in range(9):
            for j in range(i + 1, 9):
                csp.add_constraint(BinaryNotEqualConstraint(col_vars[i], col_vars[j]))

    # 3. 3x3 Block constraints
    for br in range(3):
        for bc in range(3):
            block_vars = []
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    block_vars.append(r * 9 + c)
            csp.add_all_diff_group(block_vars)
            for i in range(9):
                for j in range(i + 1, 9):
                    csp.add_constraint(BinaryNotEqualConstraint(block_vars[i], block_vars[j]))

    clue_count = sum(1 for c in cleaned if c in "123456789")
    return CSPInstance(
        name=name,
        family="sudoku",
        difficulty=difficulty,
        csp=csp,
        metadata={"grid_size": 9, "clues": clue_count, "puzzle_str": cleaned},
    )


def create_16x16_sudoku(clues_dict: Optional[Dict[int, int]] = None, name: str = "sudoku_16x16") -> CSPInstance:
    """
    Construct a 16x16 Sudoku (Hexadoku) CSP instance with 4x4 blocks.
    Variables: 256. Domain values: 0 to 15.
    """
    num_vars = 256
    all_values_mask = (1 << 16) - 1
    initial_domains = [all_values_mask] * num_vars

    if clues_dict is not None:
        for var, val in clues_dict.items():
            initial_domains[var] = (1 << val)

    csp = CSP(
        num_variables=num_vars,
        initial_domains=initial_domains,
        variable_names=[f"R{r}C{c}" for r in range(16) for c in range(16)],
        name=name,
    )

    # Rows
    for r in range(16):
        row_vars = [r * 16 + c for c in range(16)]
        csp.add_all_diff_group(row_vars)
        for i in range(16):
            for j in range(i + 1, 16):
                csp.add_constraint(BinaryNotEqualConstraint(row_vars[i], row_vars[j]))

    # Columns
    for c in range(16):
        col_vars = [r * 16 + c for r in range(16)]
        csp.add_all_diff_group(col_vars)
        for i in range(16):
            for j in range(i + 1, 16):
                csp.add_constraint(BinaryNotEqualConstraint(col_vars[i], col_vars[j]))

    # 4x4 Blocks
    for br in range(4):
        for bc in range(4):
            block_vars = []
            for r in range(br * 4, br * 4 + 4):
                for c in range(bc * 4, bc * 4 + 4):
                    block_vars.append(r * 16 + c)
            csp.add_all_diff_group(block_vars)
            for i in range(16):
                for j in range(i + 1, 16):
                    csp.add_constraint(BinaryNotEqualConstraint(block_vars[i], block_vars[j]))

    return CSPInstance(
        name=name,
        family="sudoku",
        difficulty="hard",
        csp=csp,
        metadata={"grid_size": 16, "clues": len(clues_dict or {})},
    )


def generate_16x16_sudoku(
    seed: int = 42,
    clues_count: int = 120,
    name: Optional[str] = None,
) -> CSPInstance:
    """
    Procedurally generate a guaranteed-satisfiable 16x16 Hexadoku puzzle
    by permuting canonical Latin block structures.
    """
    import random
    rng = random.Random(seed)

    # 1. Canonical 16x16 valid solution grid
    grid = [[((r // 4) + (r % 4) * 4 + c) % 16 for c in range(16)] for r in range(16)]

    # 2. Permute symbols 0..15
    symbols = list(range(16))
    rng.shuffle(symbols)
    sym_map = {orig: symbols[orig] for orig in range(16)}
    grid = [[sym_map[val] for val in row] for row in grid]

    # 3. Permute rows within 4-row bands
    new_rows = []
    band_order = list(range(4))
    rng.shuffle(band_order)
    for b in band_order:
        row_indices = [b * 4 + i for i in range(4)]
        rng.shuffle(row_indices)
        for r_idx in row_indices:
            new_rows.append(grid[r_idx])
    grid = new_rows

    # 4. Permute columns within 4-column stacks
    col_order = []
    stack_order = list(range(4))
    rng.shuffle(stack_order)
    for s in stack_order:
        c_indices = [s * 4 + j for j in range(4)]
        rng.shuffle(c_indices)
        col_order.extend(c_indices)
    grid = [[row[c] for c in col_order] for row in grid]

    # 5. Sample clue positions
    all_cells = [(r, c) for r in range(16) for c in range(16)]
    rng.shuffle(all_cells)
    clue_cells = all_cells[:min(clues_count, 256)]
    clues_dict = {r * 16 + c: grid[r][c] for r, c in clue_cells}

    inst_name = name or f"hexadoku_16x16_s{seed}"
    return create_16x16_sudoku(clues_dict=clues_dict, name=inst_name)



def generate_sudoku(
    seed: int = 42,
    difficulty: str = "medium",
    name: Optional[str] = None,
) -> CSPInstance:
    """
    Procedurally generate a 9x9 Sudoku puzzle with target difficulty.
    
    Difficulty mapping:
    - easy: 36-40 clues
    - medium: 30-34 clues
    - hard: 24-28 clues
    """
    rng = random.Random(seed)
    
    # Target clues
    if difficulty == "easy":
        target_clues = rng.randint(36, 40)
    elif difficulty == "hard":
        target_clues = rng.randint(24, 28)
    else:
        target_clues = rng.randint(30, 34)

    # 1. Generate full valid board using randomized permutation of standard base pattern
    base = 3
    side = base * base

    def pattern(r, c):
        return (base * (r % base) + r // base + c) % side

    def shuffle(s):
        lst = list(s)
        rng.shuffle(lst)
        return lst

    r_base = range(base)
    rows = [g * base + r for g in shuffle(r_base) for r in shuffle(r_base)]
    cols = [g * base + c for g in shuffle(r_base) for c in shuffle(r_base)]
    nums = shuffle(range(1, base * base + 1))

    board = [[nums[pattern(r, c)] for c in cols] for r in rows]

    # 2. Flatten and remove clues down to target_clues
    flat = [board[r][c] for r in range(9) for c in range(9)]
    indices = list(range(81))
    rng.shuffle(indices)

    cells_to_remove = 81 - target_clues
    for idx in indices[:cells_to_remove]:
        flat[idx] = 0

    puzzle_str = "".join(str(v) if v != 0 else "." for v in flat)
    inst_name = name or f"sudoku_{difficulty}_s{seed}"
    return parse_sudoku_string(puzzle_str, name=inst_name, difficulty=difficulty)


def load_kaggle_sudoku_csv(csv_path: str, max_instances: int = 100) -> List[CSPInstance]:
    """Load puzzles from a standard Kaggle Sudoku CSV (columns: 'quizzes', 'solutions')."""
    instances: List[CSPInstance] = []
    path = Path(csv_path)
    if not path.exists():
        return instances

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_instances:
                break
            quiz = row.get("quizzes", "")
            if len(quiz) == 81:
                inst = parse_sudoku_string(quiz, name=f"kaggle_sudoku_{i}", difficulty="medium")
                if "solutions" in row:
                    inst.metadata["solution_str"] = row["solutions"]
                instances.append(inst)
    return instances


def get_published_hard_benchmarks() -> List[CSPInstance]:
    """Retrieve famous published hard Sudoku benchmark instances."""
    instances = []
    for name, puzzle_str in PUBLISHED_HARD_PUZZLES.items():
        instances.append(parse_sudoku_string(puzzle_str, name=name, difficulty="hard"))
    return instances
