"""
puzzle_utils.py – Puzzle parsing, category detection, and difficulty estimation.

Provides utilities to:
- Parse raw CSV rows into structured Puzzle objects
- Detect puzzle category from prompt text (keyword + heuristic rules)
- Estimate difficulty from prompt length, example count, and complexity cues
- Extract input-output example pairs from prompt strings
"""

import re
import math
from typing import List, Optional, Tuple
from collections import Counter

from src.config import (
    Puzzle,
    PuzzleExample,
    PUZZLE_CATEGORIES,
    CATEGORY_ALIASES,
)


# ───────────────────────────────────────────────────────────────────────────
# Category detection
# ───────────────────────────────────────────────────────────────────────────

# Keyword weights per category  (keyword → list of categories it signals)
_CATEGORY_KEYWORDS = {
    "sequence_completion": [
        "sequence", "next number", "next letter", "continue the",
        "what comes next", "series", "progression", "arithmetic sequence",
        "geometric sequence", "find the next", "complete the sequence",
        "pattern in the", "nth term", "missing number", "missing term",
    ],
    "pattern_transformation": [
        "transform", "apply the rule", "input becomes", "output is",
        "map", "convert", "replace", "swap", "reverse", "shift",
        "encoding", "decoding", "cipher", "code", "function",
        "if we apply", "following rule", "same rule",
    ],
    "analogy_reasoning": [
        "analogy", "is to", "as", "similar", "corresponds",
        "relationship", "parallel", "proportion", "A is to B",
        "same relationship", "corresponding", "matching",
    ],
    "logic_grid": [
        "grid", "table", "matrix", "sudoku", "constraint",
        "each row", "each column", "each house", "logic puzzle",
        "clue", "neighbour", "adjacent", "who lives", "who owns",
        "einstein", "zebra",
    ],
    "causal_inference": [
        "cause", "effect", "because", "therefore", "consequence",
        "if...then", "leads to", "results in", "influences",
        "due to", "as a result", "depends on", "impact",
    ],
    "spatial_reasoning": [
        "rotate", "flip", "mirror", "fold", "shape", "pattern",
        "grid", "2-d", "2d", "tile", "arrangement", "position",
        "direction", "left", "right", "above", "below", "overlap",
        "reflect", "symmetry", "spatial",
    ],
    "math_word_problem": [
        "how many", "calculate", "total", "sum", "difference",
        "product", "divided", "percent", "fraction", "ratio",
        "average", "mean", "price", "cost", "distance", "speed",
        "time", "area", "volume", "perimeter", "equation",
    ],
}


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def detect_category(prompt: str) -> str:
    """
    Score each category by keyword overlap and return the highest-scoring one.

    Parameters
    ----------
    prompt : str
        The full puzzle prompt text.

    Returns
    -------
    str
        Category name from PUZZLE_CATEGORIES.
    """
    normed = _normalise(prompt)
    scores: dict = {cat: 0.0 for cat in PUZZLE_CATEGORIES}

    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            # Weight by keyword length (longer = more specific)
            weight = len(kw.split())
            if kw in normed:
                scores[cat] += weight

    # Also check alias map
    for alias, canonical in CATEGORY_ALIASES.items():
        if alias in normed:
            scores[canonical] += 1.0

    best_cat = max(scores, key=lambda k: scores[k])
    # If nothing matched, fall back to sequence_completion (most common)
    if scores[best_cat] == 0.0:
        best_cat = "sequence_completion"
    return best_cat


# ───────────────────────────────────────────────────────────────────────────
# Example pair extraction
# ───────────────────────────────────────────────────────────────────────────

# Regex patterns for common delimiters separating input/output pairs
_INPUT_OUTPUT_PATTERNS = [
    # "Input: ... Output: ..."
    re.compile(r"Input:\s*(.*?)\s*Output:\s*(.*?)(?=\n|$)", re.IGNORECASE | re.DOTALL),
    # "input = ... output = ..."
    re.compile(r"input\s*=\s*(.*?)\s*output\s*=\s*(.*?)(?=\n|$)", re.IGNORECASE | re.DOTALL),
    # "->" arrow notation
    re.compile(r"(.*?)\s*->\s*(.*?)(?=\n|$)"),
    # "=>" arrow notation
    re.compile(r"(.*?)\s*=>\s*(.*?)(?=\n|$)"),
]


def extract_examples(prompt: str) -> List[PuzzleExample]:
    """
    Attempt to extract input-output example pairs from a puzzle prompt.

    Uses multiple regex strategies and returns all unique pairs found.
    """
    pairs: List[PuzzleExample] = []
    seen = set()

    for pattern in _INPUT_OUTPUT_PATTERNS:
        for match in pattern.finditer(prompt):
            inp = match.group(1).strip()
            out = match.group(2).strip()
            if inp and out and (inp, out) not in seen:
                seen.add((inp, out))
                pairs.append(PuzzleExample(input_str=inp, output_str=out))

    return pairs


# ───────────────────────────────────────────────────────────────────────────
# Difficulty estimation
# ───────────────────────────────────────────────────────────────────────────

_DIFFICULTY_CUES_HARD = [
    "multi-step", "nested", "compound", "complex", "advanced",
    "requires", "derive", "proof", "optimal", "minimum",
    "maximum", "impossible", "contradiction", "paradox",
]

_DIFFICULTY_CUES_MEDIUM = [
    "two", "both", "then", "after that", "second step",
    "intermediate", "multiple", "several", "each",
]


def estimate_difficulty(prompt: str) -> str:
    """
    Heuristic difficulty estimation based on prompt characteristics.

    Returns one of 'easy', 'medium', 'hard'.
    """
    normed = _normalise(prompt)
    word_count = len(normed.split())
    num_examples = len(extract_examples(prompt))

    score = 0.0

    # Length heuristics
    if word_count > 300:
        score += 2
    elif word_count > 150:
        score += 1

    # Example count
    if num_examples >= 5:
        score += 1.5
    elif num_examples >= 3:
        score += 0.5

    # Keyword cues
    for cue in _DIFFICULTY_CUES_HARD:
        if cue in normed:
            score += 1.0
    for cue in _DIFFICULTY_CUES_MEDIUM:
        if cue in normed:
            score += 0.5

    # Mathematical expression complexity
    math_symbols = len(re.findall(r"[+\-*/^=<>≠≥≤√∫∑∏]", prompt))
    score += min(math_symbols * 0.2, 2.0)

    if score >= 4.0:
        return "hard"
    elif score >= 2.0:
        return "medium"
    return "easy"


# ───────────────────────────────────────────────────────────────────────────
# Parsing from CSV row
# ───────────────────────────────────────────────────────────────────────────

def parse_puzzle(row: dict, has_solution: bool = True) -> Puzzle:
    """
    Build a Puzzle dataclass from a CSV row dict.

    Parameters
    ----------
    row : dict
        Must contain keys 'puzzle_id' and 'prompt'. May contain 'solution'.
    has_solution : bool
        Whether the row includes a ground-truth solution.

    Returns
    -------
    Puzzle
    """
    puzzle_id = str(row.get("puzzle_id", row.get("id", "unknown")))
    prompt = str(row.get("prompt", ""))
    solution = str(row["solution"]) if has_solution and "solution" in row else None

    examples = extract_examples(prompt)
    category = detect_category(prompt)
    difficulty = estimate_difficulty(prompt)

    return Puzzle(
        puzzle_id=puzzle_id,
        prompt=prompt,
        solution=solution,
        examples=examples,
        category=category,
        difficulty=difficulty,
    )


def parse_dataframe(df, has_solution: bool = True) -> List[Puzzle]:
    """Parse all rows of a pandas DataFrame into Puzzle objects."""
    records = df.to_dict(orient="records")
    return [parse_puzzle(r, has_solution=has_solution) for r in records]


# ───────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ───────────────────────────────────────────────────────────────────────────

def category_distribution(puzzles: List[Puzzle]) -> Counter:
    """Count puzzles per detected category."""
    return Counter(p.category for p in puzzles)


def difficulty_distribution(puzzles: List[Puzzle]) -> Counter:
    """Count puzzles per estimated difficulty."""
    return Counter(p.difficulty for p in puzzles)


def prompt_length_stats(puzzles: List[Puzzle]) -> dict:
    """Compute basic statistics about prompt lengths."""
    lengths = [len(p.prompt.split()) for p in puzzles]
    if not lengths:
        return {"count": 0, "mean": 0, "min": 0, "max": 0, "median": 0}
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    return {
        "count": n,
        "mean": sum(lengths) / n,
        "min": lengths_sorted[0],
        "max": lengths_sorted[-1],
        "median": lengths_sorted[n // 2] if n % 2 == 1 else (
            lengths_sorted[n // 2 - 1] + lengths_sorted[n // 2]) / 2,
    }


def examples_per_puzzle_stats(puzzles: List[Puzzle]) -> dict:
    """Compute stats about the number of example pairs per puzzle."""
    counts = [len(p.examples) for p in puzzles]
    return Counter(counts)
