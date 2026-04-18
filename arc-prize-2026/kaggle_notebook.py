#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC Prize 2026 - ARC-AGI-2 Kaggle Competition Notebook
========================================================

Strategy: LLM code generation using Qwen3-4B (4-bit quantized) to synthesize
Python transform() functions from ARC task examples. Generated code is verified
against all training pairs before acceptance. Falls back to heuristic solvers
when LLM fails.

Kaggle constraints respected:
  - No internet access (uses pre-installed transformers / torch / bitsandbytes)
  - GPU: T4 (16 GB) or P100 (16 GB)
  - Time limit: 9 hours
  - Packages: transformers, torch, numpy, scipy, cv2, tqdm

Usage:
  Enable GPU in Kaggle notebook settings, then run all cells.
  Output: /kaggle/working/submission.json
"""

# ============================================================================
# Cell 1: Imports & Configuration
# ============================================================================

import json
import os
import copy
import time
import sys
import traceback
import textwrap
import threading
import re
import hashlib
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------
# Paths on Kaggle
DATA_DIR = "/kaggle/input/arc-prize-2026-arc-agi-2"
OUTPUT_PATH = "/kaggle/working/submission.json"

# LLM settings
MODEL_NAME = "Qwen/Qwen3-4B"  # fallback: "Qwen/Qwen2.5-7B-Instruct"
QUANTIZATION_4BIT = True
MAX_NEW_TOKENS = 2048
NUM_ATTEMPTS = 5  # code-generation attempts per task (with different temps)
TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9]  # varied creativity

# Execution safety
CODE_TIMEOUT_SECONDS = 10  # per code execution attempt
MAX_GEN_TIME_SECONDS = 300  # max wall-clock for all LLM attempts on one task

# Submission: we submit two attempts per test input
NUM_SUBMISSION_ATTEMPTS = 2

# ============================================================================
# Cell 2: Grid Utility Functions
# ============================================================================

def grid_to_str(grid: List[List[int]], label: str = "") -> str:
    """Convert a 2-D grid to a human-readable multi-line string.

    Example output::

        Input (5x5):
        0 0 1 0 0
        0 1 1 1 0
        0 0 1 0 0
    """
    if not grid:
        return f"{label}: (empty)"
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    header = f"{label} ({rows}x{cols}):" if label else f"({rows}x{cols}):"
    lines = [header]
    for row in grid:
        lines.append(" ".join(str(v) for v in row))
    return "\n".join(lines)


def print_grid(grid: List[List[int]], label: str = "") -> None:
    """Print a grid as ASCII art to stdout."""
    print(grid_to_str(grid, label))


def grids_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    """Check whether two grids are exactly equal."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for va, vb in zip(ra, rb):
            if va != vb:
                return False
    return True


def grid_to_numpy(grid: List[List[int]]) -> np.ndarray:
    """Convert a grid list to a numpy int array."""
    return np.array(grid, dtype=np.int64)


def numpy_to_grid(arr: np.ndarray) -> List[List[int]]:
    """Convert a numpy array back to a nested list grid."""
    return arr.tolist()


def normalize_grid(result: Any) -> Optional[List[List[int]]]:
    """Attempt to coerce an arbitrary return value into a proper grid.

    Handles: list-of-lists, numpy arrays, tuples, and flat lists that
    need reshaping.
    """
    # Already a list of lists of ints
    if isinstance(result, list):
        if all(isinstance(row, list) for row in result):
            if all(isinstance(v, int) for row in result for v in row):
                return result
        # Numpy-like: list of numpy arrays
        if all(isinstance(row, (list, np.ndarray)) for row in result):
            try:
                return [[int(v) for v in row] for row in result]
            except (TypeError, ValueError):
                pass
    # Numpy array
    if isinstance(result, np.ndarray):
        if result.ndim == 2:
            return result.astype(int).tolist()
        elif result.ndim == 1:
            # Try to keep as-is (single row)
            return [result.astype(int).tolist()]
    # Tuple of tuples
    if isinstance(result, tuple):
        return normalize_grid(list(result))
    return None


# ============================================================================
# Cell 3: Data Loading
# ============================================================================

def load_task(filepath: str) -> Dict[str, Any]:
    """Load a single ARC task JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_all_tasks(data_dir: str, split: str = "evaluation") -> Dict[str, Dict]:
    """Load every task in *split* (evaluation or training).

    Returns dict mapping task_id -> task dict.
    """
    split_dir = os.path.join(data_dir, "data", split)
    tasks = {}
    if not os.path.isdir(split_dir):
        print(f"[WARN] Directory not found: {split_dir}")
        return tasks
    for fname in sorted(os.listdir(split_dir)):
        if fname.endswith(".json"):
            task_id = fname.replace(".json", "")
            tasks[task_id] = load_task(os.path.join(split_dir, fname))
    return tasks


def load_arc_data(data_dir: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Load both evaluation and training splits.

    Returns (eval_tasks, train_tasks).
    """
    eval_tasks = load_all_tasks(data_dir, "evaluation")
    train_tasks = load_all_tasks(data_dir, "training")
    print(f"Loaded {len(eval_tasks)} evaluation tasks, "
          f"{len(train_tasks)} training tasks.")
    return eval_tasks, train_tasks


# ============================================================================
# Cell 4: Prompt Construction
# ============================================================================

SYSTEM_PROMPT = (
    "You are an expert programmer who solves ARC (Abstraction and Reasoning "
    "Corpus) puzzles. You will be given input-output grid examples that "
    "demonstrate a pattern. Your job is to write a single Python function "
    "`transform(input_grid)` that takes a 2-D list of integers (the input "
    "grid) and returns a 2-D list of integers (the output grid).\n\n"
    "Rules:\n"
    "1. `input_grid` is a list of lists of ints. Each row is a list.\n"
    "2. Your function must return a list of lists of ints.\n"
    "3. Values are always between 0 and 9 inclusive.\n"
    "4. Grid sizes range from 1x1 to 30x30.\n"
    "5. Do NOT print anything. Just return the grid.\n"
    "6. Only output the function code. No extra explanation.\n"
    "7. Think step by step about the pattern before writing code.\n"
    "8. You may use numpy as `import numpy as np` if helpful.\n"
    "9. Ensure your function handles variable-sized inputs correctly.\n"
)


def build_task_prompt(task: Dict, test_input_idx: int = 0) -> str:
    """Build the user prompt for a task.

    Includes all training example pairs and the chosen test input.
    """
    parts = []
    # Training examples
    for i, pair in enumerate(task["train"]):
        parts.append(grid_to_str(pair["input"], f"Example {i+1} Input"))
        parts.append(grid_to_str(pair["output"], f"Example {i+1} Output"))
        parts.append("")  # blank line

    # Test input (no output - that's what we need to predict)
    test_input = task["test"][test_input_idx]["input"]
    parts.append(grid_to_str(test_input, "Test Input"))
    parts.append("")
    parts.append(
        "Write ONLY a Python function. Start with `def transform(input_grid):`.\n"
        "Think carefully about what transformation maps each input to its output.\n"
        "Return the predicted output grid as a list of lists of ints."
    )
    return "\n".join(parts)


def build_code_extraction_regex() -> re.Pattern:
    """Regex to extract `def transform(...): ...` from LLM output."""
    return re.compile(
        r"(def\s+transform\s*\(\s*input_grid\s*\)\s*:.*?)"
        r"(?=\n(?:def |class |$))",
        re.DOTALL,
    )


def extract_function_code(llm_output: str) -> Optional[str]:
    """Extract the transform function from raw LLM text.

    Tries multiple strategies:
    1. Find ```python ... ``` block
    2. Find def transform( directly
    3. Take everything after the function definition
    """
    # Strategy 1: fenced code block
    fence_match = re.search(
        r"```(?:python)?\s*\n?(.*?)```", llm_output, re.DOTALL
    )
    if fence_match:
        code_block = fence_match.group(1).strip()
        if "def transform" in code_block:
            # Extract just the function
            fn_match = build_code_extraction_regex().search(code_block)
            if fn_match:
                return fn_match.group(1).strip()
            return code_block

    # Strategy 2: direct function definition
    fn_match = build_code_extraction_regex().search(llm_output)
    if fn_match:
        return fn_match.group(1).strip()

    # Strategy 3: find first def transform and take rest
    idx = llm_output.find("def transform")
    if idx >= 0:
        return llm_output[idx:].strip()

    return None


# ============================================================================
# Cell 5: Safe Code Execution
# ============================================================================

class CodeExecutionError(Exception):
    """Raised when generated code fails to execute."""


class CodeTimeoutError(Exception):
    """Raised when code execution exceeds the timeout."""


def _run_code_in_thread(
    code: str,
    input_grid: List[List[int]],
    result_container: list,
    error_container: list,
):
    """Worker that executes code and populates result/error containers."""
    try:
        local_ns: Dict[str, Any] = {"np": np}
        exec(code, local_ns)
        transform_fn = local_ns.get("transform")
        if transform_fn is None:
            error_container.append("No 'transform' function defined.")
            return
        result = transform_fn(copy.deepcopy(input_grid))
        result_container.append(result)
    except Exception as e:
        error_container.append(str(e))


def safe_execute_code(
    code: str,
    input_grid: List[List[int]],
    timeout: float = CODE_TIMEOUT_SECONDS,
) -> Optional[List[List[int]]]:
    """Execute generated code safely with a timeout.

    Returns the output grid on success, or None on failure.
    """
    result_container: list = []
    error_container: list = []
    thread = threading.Thread(
        target=_run_code_in_thread,
        args=(code, input_grid, result_container, error_container),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None  # timeout

    if error_container:
        return None  # execution error

    if not result_container:
        return None  # no result

    raw = result_container[0]
    normalized = normalize_grid(raw)
    if normalized is None:
        return None
    return normalized


def verify_code_on_training(
    code: str,
    train_pairs: List[Dict],
) -> Tuple[bool, int]:
    """Verify that *code* passes ALL training pairs.

    Returns (all_passed, num_passed).
    """
    num_passed = 0
    for pair in train_pairs:
        result = safe_execute_code(code, pair["input"])
        if result is not None and grids_equal(result, pair["output"]):
            num_passed += 1
        else:
            return False, num_passed
    return True, num_passed


# ============================================================================
# Cell 6: Fallback Heuristic Solvers
# ============================================================================

def heuristic_identity(grid):
    return copy.deepcopy(grid)


def heuristic_rotate90(grid):
    g = np.array(grid)
    return g.rot90().tolist()


def heuristic_rotate180(grid):
    g = np.array(grid)
    return g.rot90(2).tolist()


def heuristic_rotate270(grid):
    g = np.array(grid)
    return g.rot90(3).tolist()


def heuristic_flip_h(grid):
    g = np.array(grid)
    return g[:, ::-1].tolist()


def heuristic_flip_v(grid):
    g = np.array(grid)
    return g[::-1, :].tolist()


def heuristic_transpose(grid):
    g = np.array(grid)
    return g.T.tolist()


def heuristic_crop(grid):
    """Remove all-zero border rows and columns."""
    g = np.array(grid)
    if g.size == 0:
        return grid
    rows_nonzero = np.where(g.any(axis=1))[0]
    cols_nonzero = np.where(g.any(axis=0))[0]
    if len(rows_nonzero) == 0 or len(cols_nonzero) == 0:
        return [[0]]
    return g[rows_nonzero[0]:rows_nonzero[-1]+1,
              cols_nonzero[0]:cols_nonzero[-1]+1].tolist()


def heuristic_fill_most_common(grid):
    """Replace all non-zero with most common non-zero value."""
    g = np.array(grid)
    vals = g[g > 0].tolist()
    if not vals:
        return copy.deepcopy(grid)
    most_common = Counter(vals).most_common(1)[0][0]
    result = np.where(g > 0, most_common, 0)
    return result.tolist()


def heuristic_invert_colors(grid):
    """Map each non-zero value to a different value based on position."""
    g = np.array(grid)
    unique = sorted(set(g.flat) - {0})
    if not unique:
        return copy.deepcopy(grid)
    mapping = {}
    for i, v in enumerate(unique):
        mapping[v] = unique[(i + 1) % len(unique)]
    result = np.copy(g)
    for old, new in mapping.items():
        result[result == old] = new
    return result.tolist()


# All heuristic functions to try
HEURISTIC_SOLVERS = [
    ("identity", heuristic_identity),
    ("rotate90", heuristic_rotate90),
    ("rotate180", heuristic_rotate180),
    ("rotate270", heuristic_rotate270),
    ("flip_h", heuristic_flip_h),
    ("flip_v", heuristic_flip_v),
    ("transpose", heuristic_transpose),
    ("crop", heuristic_crop),
    ("fill_most_common", heuristic_fill_most_common),
    ("invert_colors", heuristic_invert_colors),
]


def try_heuristics(train_pairs: List[Dict]) -> List[Tuple[str, Any]]:
    """Return list of (heuristic_name, function) that pass all training pairs."""
    passing = []
    for name, fn in HEURISTIC_SOLVERS:
        all_ok = True
        for pair in train_pairs:
            try:
                result = fn(pair["input"])
                norm = normalize_grid(result)
                if norm is None or not grids_equal(norm, pair["output"]):
                    all_ok = False
                    break
            except Exception:
                all_ok = False
                break
        if all_ok:
            passing.append((name, fn))
    return passing


def apply_heuristic_fn(fn, input_grid):
    """Apply a heuristic function and normalize output."""
    try:
        result = fn(input_grid)
        return normalize_grid(result)
    except Exception:
        return None


# ============================================================================
# Cell 7: LLM Model Loading
# ============================================================================

def load_llm_model(model_name: str = MODEL_NAME, use_4bit: bool = True):
    """Load a quantized language model for code generation.

    Returns (model, tokenizer) or (None, None) on failure.
    """
    print(f"[INFO] Loading model: {model_name} ...")
    t0 = time.time()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if use_4bit:
            try:
                import bitsandbytes  # noqa: F401
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                print("[INFO] Using 4-bit quantization (bitsandbytes).")
            except ImportError:
                print("[WARN] bitsandbytes not available, loading without quantization.")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

        elapsed = time.time() - t0
        print(f"[INFO] Model loaded in {elapsed:.1f}s")
        return model, tokenizer

    except Exception as e:
        print(f"[ERROR] Failed to load model {model_name}: {e}")
        traceback.print_exc()
        return None, None


def generate_code(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.4,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Optional[str]:
    """Generate code from the LLM given a prompt string.

    Returns raw LLM text output or None on failure.
    """
    if model is None or tokenizer is None:
        return None
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return raw_text

    except Exception as e:
        print(f"[ERROR] Code generation failed: {e}")
        return None


# ============================================================================
# Cell 8: Task Solver
# ============================================================================

def solve_task_with_llm(
    task: Dict,
    model,
    tokenizer,
    test_input_idx: int = 0,
) -> List[List[List[int]]]:
    """Solve a single task's test input using LLM code generation.

    Returns a list of candidate output grids (up to NUM_SUBMISSION_ATTEMPTS).
    """
    candidates: List[List[List[int]]] = []
    train_pairs = task["train"]
    test_input = task["test"][test_input_idx]["input"]
    prompt = build_task_prompt(task, test_input_idx)

    # First: try heuristics (fast, no GPU needed)
    heuristics = try_heuristics(train_pairs)
    if heuristics:
        print(f"  [HEURISTIC] {len(heuristics)} heuristic(s) passed training!")
        for name, fn in heuristics[:NUM_SUBMISSION_ATTEMPTS]:
            result = apply_heuristic_fn(fn, test_input)
            if result is not None:
                candidates.append(result)
                print(f"    -> {name} applied successfully")
        if len(candidates) >= NUM_SUBMISSION_ATTEMPTS:
            return candidates[:NUM_SUBMISSION_ATTEMPTS]

    # If LLM not available, return whatever heuristics gave us
    if model is None or tokenizer is None:
        print("  [SKIP] LLM not available, returning heuristic results only.")
        return candidates if candidates else _zero_grid(test_input)

    # LLM code generation with multiple attempts
    num_temps = min(NUM_ATTEMPTS, len(TEMPERATURES))
    t_start = time.time()

    for attempt_idx in range(num_temps):
        # Check time budget
        if time.time() - t_start > MAX_GEN_TIME_SECONDS:
            print(f"  [TIMEOUT] Exceeded {MAX_GEN_TIME_SECONDS}s budget, stopping.")
            break

        temp = TEMPERATURES[attempt_idx % len(TEMPERATURES)]
        print(f"  [ATTEMPT {attempt_idx+1}/{num_temps}] temp={temp:.1f} ...")

        raw_output = generate_code(model, tokenizer, prompt, temperature=temp)
        if raw_output is None:
            print(f"    -> Generation failed")
            continue

        code = extract_function_code(raw_output)
        if code is None:
            print(f"    -> Could not extract function from output")
            continue

        print(f"    -> Extracted {len(code)} chars of code")

        # Verify against ALL training pairs
        all_passed, num_passed = verify_code_on_training(code, train_pairs)
        if all_passed:
            print(f"    -> PASSED all {num_passed} training pairs!")
            # Run on test input
            test_result = safe_execute_code(code, test_input)
            if test_result is not None:
                candidates.append(test_result)
                print(f"    -> Test output shape: "
                      f"{len(test_result)}x{len(test_result[0]) if test_result else 0}")
                if len(candidates) >= NUM_SUBMISSION_ATTEMPTS:
                    break
            else:
                print(f"    -> Test execution failed or returned invalid result")
        else:
            print(f"    -> Failed verification: {num_passed}/{len(train_pairs)} passed")

    # Fill remaining slots with heuristic or zero grid
    while len(candidates) < NUM_SUBMISSION_ATTEMPTS:
        if heuristics:
            _, fn = heuristics[0]
            result = apply_heuristic_fn(fn, test_input)
            if result is not None:
                candidates.append(result)
                continue
        candidates.append(_zero_grid(test_input))
        break

    return candidates[:NUM_SUBMISSION_ATTEMPTS]


def _zero_grid(input_grid: List[List[int]]) -> List[List[int]]:
    """Generate a zero grid matching the input dimensions as last resort."""
    rows = len(input_grid)
    cols = len(input_grid[0]) if input_grid else 0
    return [[0] * cols for _ in range(rows)]


# ============================================================================
# Cell 9: Main Solving Pipeline
# ============================================================================

def solve_all_tasks(
    eval_tasks: Dict[str, Dict],
    model=None,
    tokenizer=None,
    max_tasks: Optional[int] = None,
) -> Dict[str, List]:
    """Solve all evaluation tasks and produce the submission dict.

    submission.json format:
    {
        "task_id": [
            [[row1_attempt1], [row2_attempt1], ...],  // attempt 1
            [[row1_attempt2], [row2_attempt2], ...],  // attempt 2
        ],
        ...
    }
    """
    submission: Dict[str, List] = {}
    task_ids = list(eval_tasks.keys())
    if max_tasks is not None:
        task_ids = task_ids[:max_tasks]

    total = len(task_ids)
    print(f"\n{'='*60}")
    print(f"Solving {total} evaluation tasks ...")
    print(f"{'='*60}\n")

    stats = {"llm_success": 0, "heuristic_only": 0, "zero_fallback": 0}

    for idx, task_id in enumerate(tqdm(task_ids, desc="Solving tasks")):
        task = eval_tasks[task_id]
        num_test = len(task["test"])

        task_attempts = []
        for test_idx in range(num_test):
            candidates = solve_task_with_llm(
                task, model, tokenizer, test_input_idx=test_idx
            )
            task_attempts.extend(candidates)

        submission[task_id] = task_attempts

        # Quick stats
        is_zero = all(
            all(v == 0 for row in attempt for v in row)
            for attempt in task_attempts
        )
        if is_zero:
            stats["zero_fallback"] += 1
        else:
            stats["llm_success"] += 1

        # Progress
        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"\n  Progress: {idx+1}/{total} | "
                  f"LLM/heuristic: {stats['llm_success']} | "
                  f"Zero fallback: {stats['zero_fallback']}")

    print(f"\n{'='*60}")
    print(f"FINAL STATS: LLM/heuristic={stats['llm_success']}, "
          f"Zero fallback={stats['zero_fallback']}, "
          f"Total={total}")
    print(f"{'='*60}\n")

    return submission


def save_submission(submission: Dict, path: str) -> None:
    """Write submission.json to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(submission, f)
    print(f"[INFO] Submission saved to {path}")
    print(f"[INFO] Tasks in submission: {len(submission)}")


# ============================================================================
# Cell 10: Training Validation (scoring cell)
# ============================================================================

def validate_on_training(
    train_tasks: Dict[str, Dict],
    model=None,
    tokenizer=None,
    max_tasks: int = 20,
) -> Dict[str, Any]:
    """Score the solver on training tasks where we have ground truth.

    Uses training tasks' test outputs for validation.
    Returns accuracy stats.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATION: Testing on {max_tasks} training tasks ...")
    print(f"{'='*60}\n")

    task_ids = list(train_tasks.keys())[:max_tasks]
    total_pairs = 0
    correct_pairs = 0
    per_task = {}

    for task_id in tqdm(task_ids, desc="Validating"):
        task = train_tasks[task_id]
        num_test = len(task["test"])
        task_correct = 0

        for test_idx in range(num_test):
            total_pairs += 1
            candidates = solve_task_with_llm(
                task, model, tokenizer, test_input_idx=test_idx
            )
            expected = task["test"][test_idx]["output"]

            for candidate in candidates:
                if grids_equal(candidate, expected):
                    correct_pairs += 1
                    task_correct += 1
                    break  # one correct attempt is enough per pair

        per_task[task_id] = task_correct

    accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
    print(f"\n  Training Validation Results:")
    print(f"    Pairs correct: {correct_pairs}/{total_pairs}")
    print(f"    Accuracy: {accuracy:.1%}")
    print(f"    Tasks with at least 1 correct: "
          f"{sum(1 for v in per_task.values() if v > 0)}/{len(task_ids)}")

    # Show some example tasks
    print(f"\n  Per-task breakdown (first 10):")
    for tid in list(task_ids)[:10]:
        num_test = len(train_tasks[tid]["test"])
        print(f"    {tid}: {per_task[tid]}/{num_test} correct")

    return {
        "accuracy": accuracy,
        "correct_pairs": correct_pairs,
        "total_pairs": total_pairs,
        "per_task": per_task,
    }


# ============================================================================
# Cell 11: Demo - Grid Visualization
# ============================================================================

def demo_visualization():
    """Print sample grids to verify the visualization works."""
    print("=" * 50)
    print("GRID VISUALIZATION DEMO")
    print("=" * 50)
    sample = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ]
    print_grid(sample, "Diamond")
    print()
    print_grid(sample, "Rotated 90")
    print_grid(heuristic_rotate90(sample), "")
    print()
    print_grid(sample, "Flipped H")
    print_grid(heuristic_flip_h(sample), "")


# ============================================================================
# Cell 12: Main Entry Point
# ============================================================================

def main():
    """Main entry point for the Kaggle notebook."""
    print("=" * 60)
    print("ARC PRIZE 2026 - ARC-AGI-2 SOLVER")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print()

    # -------------------------------------------------------------------------
    # Step 0: Demo visualization
    # -------------------------------------------------------------------------
    demo_visualization()

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading ARC data ...")
    eval_tasks, train_tasks = load_arc_data(DATA_DIR)

    if not eval_tasks:
        print("[ERROR] No evaluation tasks found. Check DATA_DIR path.")
        print(f"  DATA_DIR = {DATA_DIR}")
        print(f"  Contents: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else 'N/A'}")
        return

    # -------------------------------------------------------------------------
    # Step 2: Load LLM model
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Loading LLM model ...")
    model, tokenizer = load_llm_model(MODEL_NAME, QUANTIZATION_4BIT)

    # -------------------------------------------------------------------------
    # Step 3: Validate on training data (optional, but recommended)
    # -------------------------------------------------------------------------
    if train_tasks and model is not None:
        print("\n[STEP 3] Validating on training tasks ...")
        val_results = validate_on_training(
            train_tasks, model, tokenizer, max_tasks=10
        )
    else:
        print("\n[STEP 3] Skipping training validation "
              "(no model or no training data).")

    # -------------------------------------------------------------------------
    # Step 4: Solve evaluation tasks
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Solving evaluation tasks ...")
    submission = solve_all_tasks(eval_tasks, model, tokenizer)

    # -------------------------------------------------------------------------
    # Step 5: Save submission
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Saving submission ...")
    save_submission(submission, OUTPUT_PATH)

    # -------------------------------------------------------------------------
    # Step 6: Final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("NOTEBOOK COMPLETE")
    print("=" * 60)
    total_attempts = sum(len(v) for v in submission.values())
    print(f"  Tasks solved: {len(submission)}")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Output file: {OUTPUT_PATH}")
    print()

    # Memory cleanup
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache()
    print("[INFO] GPU memory freed.")


# ============================================================================
# Conditional execution: only run main() when executed as a script.
# In Kaggle, this runs automatically. In notebooks, import this file.
# ============================================================================

if __name__ == "__main__":
    main()
