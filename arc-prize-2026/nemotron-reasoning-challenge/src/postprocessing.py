"""
postprocessing.py – Clean and normalise model outputs into final answers.

Handles:
- Extracting the answer from chain-of-thought output
- Stripping whitespace, markdown, and formatting artefacts
- Normalising numbers, sequences, and common answer formats
- Validation / sanity checks for different puzzle categories
"""

import re
import json
from typing import Optional, List


# ───────────────────────────────────────────────────────────────────────────
# Core answer extraction
# ───────────────────────────────────────────────────────────────────────────

_ANSWER_PATTERNS = [
    # "Answer: 42" or "Answer: forty-two"
    re.compile(r"(?:^|\n)\s*Answer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE),
    # "The answer is 42."
    re.compile(r"(?:the |final )?answer\s*(?:is|:)\s*(.+?)[.\n]", re.IGNORECASE),
    # "=> 42" or "-> 42"
    re.compile(r"(?:=>|->)\s*(.+?)(?:\n|$)"),
    # Last non-empty line (fallback)
    re.compile(r"\n\s*(.+?)\s*$"),
]


def extract_answer(raw_output: str) -> str:
    """
    Extract the final answer from a model's raw text output.

    Tries multiple patterns in priority order and returns the first match.
    """
    if not raw_output:
        return ""

    # Strip markdown code fences
    cleaned = re.sub(r"```[\w]*\n?", "", raw_output)
    cleaned = re.sub(r"```", "", cleaned)

    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            answer = match.group(1).strip()
            if answer:
                return answer

    # Last resort: return stripped output
    return cleaned.strip().split("\n")[-1].strip()


# ───────────────────────────────────────────────────────────────────────────
# Category-specific normalisation
# ───────────────────────────────────────────────────────────────────────────

def _normalise_number(answer: str) -> str:
    """Convert number words to digits and strip punctuation."""
    word_to_digit = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16",
        "seventeen": "17", "eighteen": "18", "nineteen": "19",
        "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
        "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
        "hundred": "100", "thousand": "1000", "million": "1000000",
    }
    tokens = answer.lower().replace("-", " ").split()
    converted = []
    for tok in tokens:
        converted.append(word_to_digit.get(tok, tok))
    result = " ".join(converted)
    # Remove non-digit/non-letter/non-comma/non-dot characters
    result = re.sub(r"[^\w\s,.\-/]", "", result)
    return result.strip()


def _normalise_sequence(answer: str) -> str:
    """For sequence answers, ensure consistent formatting."""
    # Remove trailing punctuation
    answer = answer.rstrip(".,;:")
    # Ensure comma-separated lists have no spaces after commas (or consistent spacing)
    answer = re.sub(r",\s*", ", ", answer)
    return answer.strip()


def _normalise_choice(answer: str) -> str:
    """Extract a single letter choice (A, B, C, ...)."""
    match = re.search(r"\b([A-Z])\b", answer.upper())
    if match:
        return match.group(1)
    return answer.strip().upper()[:1]


def normalise_answer(answer: str, category: Optional[str] = None) -> str:
    """
    Normalise a raw answer string based on the puzzle category.

    Parameters
    ----------
    answer : str
        Raw extracted answer.
    category : str, optional
        Puzzle category for category-specific normalisation.

    Returns
    -------
    str
        Cleaned, normalised answer.
    """
    if not answer:
        return ""

    # Remove leading/trailing whitespace and quotes
    answer = answer.strip().strip('"').strip("'").strip()

    if category in ("sequence_completion", "math_word_problem"):
        return _normalise_number(answer)
    elif category == "pattern_transformation":
        return _normalise_sequence(answer)
    elif category in ("analogy_reasoning", "logic_grid"):
        return _normalise_choice(answer)
    elif category == "spatial_reasoning":
        # Could be a grid reference like "A3" or a description
        return answer.upper().strip()

    # Generic fallback
    return answer.strip()


# ───────────────────────────────────────────────────────────────────────────
# Validation
# ───────────────────────────────────────────────────────────────────────────

def validate_answer(answer: str, category: Optional[str] = None) -> bool:
    """Basic sanity check – reject clearly malformed answers."""
    if not answer or answer in ("", "null", "none", "undefined", "nan"):
        return False
    if len(answer) > 500:
        return False
    # Reject if it's just punctuation / whitespace
    if re.match(r"^[^\w]+$", answer):
        return False
    return True


# ───────────────────────────────────────────────────────────────────────────
# Full pipeline
# ───────────────────────────────────────────────────────────────────────────

def postprocess(
    raw_output: str,
    category: Optional[str] = None,
) -> Optional[str]:
    """
    Full post-processing pipeline: extract → normalise → validate.

    Returns None if the output is invalid.
    """
    answer = extract_answer(raw_output)
    if not answer:
        return None
    answer = normalise_answer(answer, category)
    if not validate_answer(answer, category):
        return None
    return answer


# ───────────────────────────────────────────────────────────────────────────
# Self-consistency helpers
# ───────────────────────────────────────────────────────────────────────────

def majority_vote(answers: List[str], weights: Optional[List[float]] = None) -> Optional[str]:
    """
    Return the most common answer.  If *weights* are provided, use
    weighted majority voting.

    Parameters
    ----------
    answers : list[str]
        List of candidate answers (may contain None / empty).
    weights : list[float], optional
        Per-answer weights (lower temperature → higher weight).

    Returns
    -------
    str or None
        The winning answer, or None if all answers were empty.
    """
    valid = [(a, w) for a, w in zip(answers, weights or [1.0] * len(answers))
             if a and validate_answer(a)]
    if not valid:
        return None

    if weights is None:
        # Simple frequency count
        counts: dict = {}
        for a, _ in valid:
            counts[a] = counts.get(a, 0) + 1
        return max(counts, key=counts.get)
    else:
        # Weighted frequency
        weighted_counts: dict = {}
        for a, w in valid:
            weighted_counts[a] = weighted_counts.get(a, 0.0) + w
        return max(weighted_counts, key=weighted_counts.get)
