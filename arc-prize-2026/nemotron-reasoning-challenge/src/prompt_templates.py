"""
prompt_templates.py – Prompt construction for each puzzle category and style.

Supports three prompting strategies:
- zero_shot: Direct question without examples.
- few_shot: Include k example pairs from the same puzzle.
- chain_of_thought: Instruct the model to reason step-by-step.
"""

from typing import List, Optional

from src.config import (
    Puzzle,
    PuzzleExample,
    PromptConfig,
    PUZZLE_CATEGORIES,
)


# ───────────────────────────────────────────────────────────────────────────
# Category-specific instruction fragments
# ───────────────────────────────────────────────────────────────────────────

_CATEGORY_INSTRUCTIONS = {
    "sequence_completion": (
        "Identify the pattern or rule governing the sequence. "
        "Determine what comes next by continuing the same rule."
    ),
    "pattern_transformation": (
        "Study the input-output example pairs carefully. "
        "Identify the transformation rule that maps each input to its output, "
        "then apply the same rule to the final input."
    ),
    "analogy_reasoning": (
        "Analyze the relationship between the first pair. "
        "Find the option that has the same relationship with the third element."
    ),
    "logic_grid": (
        "Use the given clues to eliminate impossibilities. "
        "Deduce the correct assignment for each cell in the grid."
    ),
    "causal_inference": (
        "Trace the chain of cause and effect. "
        "Determine what logically follows from the given conditions."
    ),
    "spatial_reasoning": (
        "Visualise the spatial arrangement and apply the described "
        "transformation (rotate, flip, fold, etc.) to determine the result."
    ),
    "math_word_problem": (
        "Translate the word problem into a mathematical expression. "
        "Solve step-by-step and state the final numerical answer clearly."
    ),
}


# ───────────────────────────────────────────────────────────────────────────
# Style-specific wrappers
# ───────────────────────────────────────────────────────────────────────────

def _format_example(ex: PuzzleExample, idx: int) -> str:
    return f"Example {idx + 1}:\n  Input:  {ex.input_str}\n  Output: {ex.output_str}"


def build_zero_shot_prompt(puzzle: Puzzle, cfg: PromptConfig) -> str:
    """
    Zero-shot: just the puzzle + a direct question.
    """
    parts = []
    if cfg.include_category_hint and puzzle.category:
        hint = _CATEGORY_INSTRUCTIONS.get(puzzle.category, "")
        parts.append(hint)
    parts.append(puzzle.prompt)
    parts.append("\nProvide the answer directly:")
    return "\n\n".join(parts)


def build_few_shot_prompt(
    puzzle: Puzzle,
    cfg: PromptConfig,
    extra_examples: Optional[List[PuzzleExample]] = None,
) -> str:
    """
    Few-shot: include up to max_examples_in_prompt example pairs
    from the puzzle itself, plus any additional examples passed in.
    """
    parts = []
    if cfg.include_category_hint and puzzle.category:
        hint = _CATEGORY_INSTRUCTIONS.get(puzzle.category, "")
        parts.append(hint)

    # Gather examples
    examples = list(puzzle.examples[:cfg.max_examples_in_prompt])
    if extra_examples:
        remaining = cfg.max_examples_in_prompt - len(examples)
        examples.extend(extra_examples[:remaining])

    if examples:
        parts.append("Here are some example input-output pairs:")
        for i, ex in enumerate(examples):
            parts.append(_format_example(ex, i))
        parts.append("")  # blank line

    parts.append(puzzle.prompt)
    parts.append("\nFollowing the same rule, provide the answer:")
    return "\n\n".join(parts)


def build_chain_of_thought_prompt(
    puzzle: Puzzle,
    cfg: PromptConfig,
    extra_examples: Optional[List[PuzzleExample]] = None,
) -> str:
    """
    Chain-of-thought: explicitly request step-by-step reasoning.
    """
    parts = []
    if cfg.include_category_hint and puzzle.category:
        hint = _CATEGORY_INSTRUCTIONS.get(puzzle.category, "")
        parts.append(hint)
        parts.append("")

    # Gather examples
    examples = list(puzzle.examples[:cfg.max_examples_in_prompt])
    if extra_examples:
        remaining = cfg.max_examples_in_prompt - len(examples)
        examples.extend(extra_examples[:remaining])

    if examples:
        parts.append("Here are some example input-output pairs with reasoning:")
        for i, ex in enumerate(examples):
            parts.append(_format_example(ex, i))
            # Encourage CoT reasoning on examples too
            parts.append("  Reasoning: [apply the rule step by step]")
        parts.append("")

    parts.append(puzzle.prompt)
    parts.append(
        "\nThink step-by-step to find the answer. "
        "Show your reasoning clearly, then state the final answer "
        "on the last line in the format:  Answer: <your answer>"
    )
    return "\n\n".join(parts)


# ───────────────────────────────────────────────────────────────────────────
# Dispatch
# ───────────────────────────────────────────────────────────────────────────

def build_prompt(
    puzzle: Puzzle,
    cfg: Optional[PromptConfig] = None,
    extra_examples: Optional[List[PuzzleExample]] = None,
) -> str:
    """
    Build a prompt for the given puzzle using the style specified in *cfg*.

    Parameters
    ----------
    puzzle : Puzzle
    cfg : PromptConfig, optional
        Uses defaults if None.
    extra_examples : list of PuzzleExample, optional
        Additional examples to include in few-shot / CoT prompts.

    Returns
    -------
    str
        The formatted prompt string (without system message).
    """
    if cfg is None:
        cfg = PromptConfig()

    builders = {
        "zero_shot": build_zero_shot_prompt,
        "few_shot": build_few_shot_prompt,
        "chain_of_thought": build_chain_of_thought_prompt,
    }

    builder = builders.get(cfg.style, build_chain_of_thought_prompt)
    return builder(puzzle, cfg, extra_examples)


def build_chat_messages(
    puzzle: Puzzle,
    cfg: Optional[PromptConfig] = None,
    extra_examples: Optional[List[PuzzleExample]] = None,
) -> List[dict]:
    """
    Build chat-style message list (system + user) for chat-tuned models.

    Returns
    -------
    list[dict]
        Each dict has 'role' and 'content' keys.
    """
    if cfg is None:
        cfg = PromptConfig()

    user_content = build_prompt(puzzle, cfg, extra_examples)
    messages = [
        {"role": "system", "content": cfg.system_message},
        {"role": "user", "content": user_content},
    ]
    return messages
