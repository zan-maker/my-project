"""
ARC-AGI-2 Solver — AnnotateX ARC Edition
=========================================
Multi-strategy solver for ARC Prize 2026 ARC-AGI-2 competition.

Strategies:
  1. LLM Code Generation — Generate Python functions to solve tasks
  2. Heuristic Rules — Fast geometric/color heuristics as fallback
  3. ICL Grid Reasoning — Direct grid prediction via in-context learning

Usage:
  python arc_solver.py --data_dir ./ARC-AGI-2/data/evaluation --output submission.json
  python arc_solver.py --data_dir ./ARC-AGI-2/data/training --eval --output eval_results.json
"""

import json
import os
import glob
import sys
import time
import argparse
import random
import re
import traceback
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================
class SolverConfig:
    # LLM settings (Qwen3-4B or similar, available on Kaggle GPUs)
    MODEL_NAME = "Qwen/Qwen3-4B"
    USE_4BIT = True
    MAX_NEW_TOKENS = 2048
    TEMPERATURE = 0.3
    TOP_P = 0.9
    MAX_INPUT_TOKENS = 30000

    # Code generation settings
    NUM_CODE_ATTEMPTS = 6  # Generate multiple code candidates
    CODE_TIMEOUT = 10      # seconds per code execution

    # Heuristic settings
    USE_HEURISTICS = True
    HEURISTIC_VERIFICATION = True

    # ICL settings
    USE_ICL = False  # Enable if we want direct grid prediction

    # General
    SEED = 42
    MAX_TASKS = None  # None = all tasks


# ============================================================
# GRID UTILITIES
# ============================================================
COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "cyan", 9: "brown"
}

COLOR_SYMBOLS = {
    0: ".", 1: "B", 2: "R", 3: "G", 4: "Y",
    5: "#", 6: "M", 7: "O", 8: "C", 9: "W"
}


def grid_to_text(grid: List[List[int]], use_symbols: bool = False) -> str:
    """Convert grid to readable text format."""
    rows = []
    for row in grid:
        if use_symbols:
            rows.append(" ".join(COLOR_SYMBOLS.get(v, str(v)) for v in row))
        else:
            rows.append(" ".join(str(v) for v in row))
    return "\n".join(rows)


def grid_to_python(grid: List[List[int]]) -> str:
    """Convert grid to Python list string."""
    return json.dumps(grid)


def grid_shape(grid: List[List[int]]) -> Tuple[int, int]:
    return (len(grid), len(grid[0]))


def verify_output(expected: List[List[int]], actual: List[List[int]]) -> bool:
    """Check if two grids are exactly equal."""
    if expected is None or actual is None:
        return False
    if len(expected) != len(actual):
        return False
    for er, ar in zip(expected, actual):
        if len(er) != len(ar):
            return False
        for ev, av in zip(er, ar):
            if ev != av:
                return False
    return True


def task_to_prompt(task: Dict) -> str:
    """Convert a full ARC task into a text prompt for LLM code generation."""
    train_pairs = task['train']
    test_inputs = task['test']

    prompt = "You are an expert ARC puzzle solver. Look at the input-output examples and figure out the transformation rule.\n"
    prompt += "Write a Python function `transform(input_grid)` that applies this rule.\n\n"

    for i, pair in enumerate(train_pairs):
        inp = pair['input']
        out = pair['output']
        prompt += f"Example {i+1}:\n"
        prompt += f"Input ({len(inp)}x{len(inp[0])}):\n{grid_to_text(inp)}\n"
        prompt += f"Output ({len(out)}x{len(out[0])}):\n{grid_to_text(out)}\n\n"

    prompt += f"Now solve this test input:\n"
    for i, test in enumerate(test_inputs):
        inp = test['input']
        prompt += f"Test Input ({len(inp)}x{len(inp[0])}):\n{grid_to_text(inp)}\n"

    prompt += "\nWrite only the Python function. The function signature must be:\n"
    prompt += "def transform(input_grid):\n"
    prompt += "    # Your code here\n"
    prompt += "    return output_grid\n"

    return prompt


def task_to_icl_prompt(task: Dict) -> str:
    """Convert ARC task into an ICL prompt for direct grid prediction."""
    train_pairs = task['train']
    test_inputs = task['test']

    prompt = "Solve these ARC puzzles by predicting the output grid for the test input.\n\n"

    for i, pair in enumerate(train_pairs):
        inp = pair['input']
        out = pair['output']
        prompt += f"Input:\n{grid_to_text(inp, use_symbols=True)}\n"
        prompt += f"Output:\n{grid_to_text(out, use_symbols=True)}\n\n"

    for i, test in enumerate(test_inputs):
        inp = test['input']
        prompt += f"Input:\n{grid_to_text(inp, use_symbols=True)}\n"
        prompt += f"Output:\n"

    return prompt


# ============================================================
# STRATEGY 1: LLM CODE GENERATION
# ============================================================
class LLMCodeGenerator:
    """Generate Python code to solve ARC tasks using LLMs."""

    def __init__(self, config: SolverConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load_model(self):
        """Lazy load model."""
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading {self.config.MODEL_NAME}...")
        t0 = time.time()

        bnb_config = None
        if self.config.USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16, "device_map": "auto"}
        if bnb_config:
            kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(self.config.MODEL_NAME, **kwargs)
        self.model.eval()
        self._loaded = True
        print(f"Model loaded in {time.time()-t0:.1f}s")

    def generate_code(self, task: Dict) -> List[str]:
        """Generate multiple Python code candidates for a task."""
        self._load_model()
        import torch

        prompt = task_to_prompt(task)

        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Write only valid Python code. No explanations."},
            {"role": "user", "content": prompt}
        ]

        try:
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            full_prompt = f"<|im_start|>system\nYou are an expert Python programmer.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        codes = []
        for attempt in range(self.config.NUM_CODE_ATTEMPTS):
            try:
                inputs = self.tokenizer(
                    full_prompt, return_tensors="pt",
                    truncation=True, max_length=self.config.MAX_INPUT_TOKENS
                ).to(self.model.device)

                temp = self.config.TEMPERATURE + attempt * 0.1
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.MAX_NEW_TOKENS,
                        temperature=min(temp, 0.8),
                        top_p=self.config.TOP_P,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                generated = outputs[0][inputs.input_ids.shape[1]:]
                code = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                code = self._extract_function(code)
                if code:
                    codes.append(code)

            except Exception as e:
                print(f"  Code gen attempt {attempt+1} failed: {e}")

        return codes

    def _extract_function(self, text: str) -> Optional[str]:
        """Extract transform function from LLM output."""
        # Try to find def transform(
        patterns = [
            r'(def transform\(.*?\n(?:.*?\n)*?)(?=\ndef |\n\S|\Z)',
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if 'def transform' in code:
                    return code

        # If no function found but there's code-like content
        if 'def transform' in text:
            idx = text.index('def transform')
            return text[idx:].strip()

        return None


# ============================================================
# STRATEGY 2: EXECUTION ENGINE
# ============================================================
class ExecutionEngine:
    """Safely execute generated code against training examples."""

    @staticmethod
    def execute_and_verify(code: str, task: Dict, timeout: int = 10) -> Optional[str]:
        """Execute code and verify against all training pairs. Returns code if verified."""
        train_pairs = task['train']

        # Prepare execution environment
        exec_globals = {
            'np': np,
            '__builtins__': __builtins__,
        }

        try:
            exec(code, exec_globals)
        except Exception as e:
            return None

        if 'transform' not in exec_globals:
            return None

        transform_fn = exec_globals['transform']

        # Verify against all training pairs
        for pair in train_pairs:
            try:
                import signal
                import threading

                result = [None]
                error = [None]

                def run():
                    try:
                        inp = [row[:] for row in pair['input']]  # Deep copy
                        out = transform_fn(inp)
                        result[0] = out
                    except Exception as e:
                        error[0] = e

                t = threading.Thread(target=run)
                t.start()
                t.join(timeout=timeout)

                if t.is_alive() or error[0] is not None:
                    return None

                output = result[0]
                if not isinstance(output, list):
                    return None

                if not verify_output(pair['output'], output):
                    return None

            except Exception:
                return None

        return code  # Verified!

    @staticmethod
    def run_on_test(code: str, test_input: List[List[int]], timeout: int = 10) -> Optional[List[List[int]]]:
        """Run verified code on a test input."""
        exec_globals = {'np': np, '__builtins__': __builtins__}

        try:
            exec(code, exec_globals)
        except Exception:
            return None

        if 'transform' not in exec_globals:
            return None

        transform_fn = exec_globals['transform']

        try:
            import threading
            result = [None]
            error = [None]

            def run():
                try:
                    inp = [row[:] for row in test_input]
                    out = transform_fn(inp)
                    result[0] = out
                except Exception as e:
                    error[0] = e

            t = threading.Thread(target=run)
            t.start()
            t.join(timeout=timeout)

            if t.is_alive() or error[0] is not None:
                return None

            output = result[0]
            if isinstance(output, list) and len(output) > 0:
                return output
            return None

        except Exception:
            return None


# ============================================================
# STRATEGY 3: HEURISTIC RULES
# ============================================================
class HeuristicSolver:
    """Fast heuristic-based solver for common ARC patterns."""

    @staticmethod
    def get_all_attempts(task: Dict) -> List[List[List[int]]]:
        """Generate multiple heuristic attempts for a task."""
        attempts = []
        train_pairs = task['train']
        test_pairs = task['test']

        if not test_pairs:
            return attempts

        test_input = test_pairs[0]['input']
        inp_arr = np.array(test_input)

        # Analyze train pairs for patterns
        patterns = HeuristicSolver._analyze_patterns(train_pairs)

        for strategy_name, strategy_fn in [
            ("same_size_identity", HeuristicSolver._same_size_identity),
            ("rotate_90", HeuristicSolver._rotate_90),
            ("rotate_180", HeuristicSolver._rotate_180),
            ("rotate_270", HeuristicSolver._rotate_270),
            ("flip_h", HeuristicSolver._flip_horizontal),
            ("flip_v", HeuristicSolver._flip_vertical),
            ("transpose", HeuristicSolver._transpose),
            ("crop_nonzero", HeuristicSolver._crop_to_nonzero),
            ("extract_objects", HeuristicSolver._extract_object_grid),
            ("color_map_most_common", HeuristicSolver._color_map_frequency),
            ("tile_2x2", HeuristicSolver._tile_2x2),
            ("tile_3x3", HeuristicSolver._tile_3x3),
            ("fill_contour", HeuristicSolver._fill_contour),
            ("center_object", HeuristicSolver._center_object),
            ("complement_colors", HeuristicSolver._complement_colors),
        ]:
            try:
                result = strategy_fn(test_input, train_pairs, patterns)
                if result and HeuristicSolver._verify_against_train(strategy_fn, train_pairs, patterns):
                    attempts.append(result)
                    if len(attempts) >= 2:
                        break
            except Exception:
                continue

        return attempts if attempts else [HeuristicSolver._safe_fallback(test_input)]

    @staticmethod
    def _analyze_patterns(train_pairs: List[Dict]) -> Dict:
        """Analyze training pairs to detect transformation patterns."""
        patterns = {
            'size_changes': [],
            'shape_matches': True,
            'color_maps': [],
            'inout_same_size': True,
            'rotation_detected': None,
            'flip_detected': None,
        }

        for pair in train_pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])
            patterns['size_changes'].append((inp.shape, out.shape))
            if inp.shape != out.shape:
                patterns['inout_same_size'] = False

        # Check for rotations
        for pair in train_pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])
            if inp.shape == out.shape:
                if np.array_equal(np.rot90(inp, 1), out):
                    patterns['rotation_detected'] = 90
                elif np.array_equal(np.rot90(inp, 2), out):
                    patterns['rotation_detected'] = 180
                elif np.array_equal(np.rot90(inp, 3), out):
                    patterns['rotation_detected'] = 270
                elif np.array_equal(np.fliplr(inp), out):
                    patterns['flip_detected'] = 'horizontal'
                elif np.array_equal(np.flipud(inp), out):
                    patterns['flip_detected'] = 'vertical'

        return patterns

    @staticmethod
    def _verify_against_train(strategy_fn, train_pairs, patterns) -> bool:
        """Check if a heuristic works on ALL training pairs."""
        for pair in train_pairs:
            try:
                result = strategy_fn(pair['input'], train_pairs, patterns)
                if not verify_output(pair['output'], result):
                    return False
            except Exception:
                return False
        return True

    # --- Individual heuristic strategies ---

    @staticmethod
    def _same_size_identity(test_input, train_pairs, patterns):
        return [row[:] for row in test_input]

    @staticmethod
    def _rotate_90(test_input, train_pairs, patterns):
        if patterns.get('rotation_detected') == 90:
            return np.rot90(np.array(test_input), 1).tolist()
        return None

    @staticmethod
    def _rotate_180(test_input, train_pairs, patterns):
        if patterns.get('rotation_detected') == 180:
            return np.rot90(np.array(test_input), 2).tolist()
        return None

    @staticmethod
    def _rotate_270(test_input, train_pairs, patterns):
        if patterns.get('rotation_detected') == 270:
            return np.rot90(np.array(test_input), 3).tolist()
        return None

    @staticmethod
    def _flip_horizontal(test_input, train_pairs, patterns):
        if patterns.get('flip_detected') == 'horizontal':
            return np.fliplr(np.array(test_input)).tolist()
        return None

    @staticmethod
    def _flip_vertical(test_input, train_pairs, patterns):
        if patterns.get('flip_detected') == 'vertical':
            return np.flipud(np.array(test_input)).tolist()
        return None

    @staticmethod
    def _transpose(test_input, train_pairs, patterns):
        return np.array(test_input).T.tolist()

    @staticmethod
    def _crop_to_nonzero(test_input, train_pairs, patterns):
        arr = np.array(test_input)
        rows = np.any(arr != 0, axis=1)
        cols = np.any(arr != 0, axis=0)
        if not np.any(rows) or not np.any(cols):
            return [row[:] for row in test_input]
        return arr[np.ix_(rows, cols)].tolist()

    @staticmethod
    def _extract_object_grid(test_input, train_pairs, patterns):
        """Extract the largest non-zero object from the grid."""
        arr = np.array(test_input)
        if not np.any(arr != 0):
            return arr.tolist()

        from scipy import ndimage
        labeled, num_features = ndimage.label(arr != 0)
        if num_features == 0:
            return arr.tolist()

        # Find largest object
        sizes = ndimage.sum(arr != 0, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        result = arr[np.ix_(rows, cols)].tolist()
        return result

    @staticmethod
    def _color_map_frequency(test_input, train_pairs, patterns):
        """Map colors based on frequency analysis."""
        inp = np.array(test_input)
        if len(train_pairs) < 1:
            return inp.tolist()

        # Simple approach: if train outputs have different color distribution
        first_in = np.array(train_pairs[0]['input'])
        first_out = np.array(train_pairs[0]['output'])

        # Build a simple color map from train pair
        unique_in = np.unique(first_in[first_in != 0]) if np.any(first_in != 0) else []
        unique_out = np.unique(first_out[first_out != 0]) if np.any(first_out != 0) else []

        if len(unique_in) == len(unique_out) and len(unique_in) > 0:
            color_map = {int(i): int(o) for i, o in zip(sorted(unique_in), sorted(unique_out))}
            result = np.zeros_like(inp)
            for old_c, new_c in color_map.items():
                result[inp == old_c] = new_c
            return result.tolist()

        return inp.tolist()

    @staticmethod
    def _tile_2x2(test_input, train_pairs, patterns):
        """Tile the input 2x2."""
        arr = np.array(test_input)
        return np.tile(arr, (2, 2)).tolist()

    @staticmethod
    def _tile_3x3(test_input, train_pairs, patterns):
        """Tile the input 3x3."""
        arr = np.array(test_input)
        return np.tile(arr, (3, 3)).tolist()

    @staticmethod
    def _fill_contour(test_input, train_pairs, patterns):
        """Fill the enclosed region (flood fill from edges)."""
        arr = np.array(test_input)
        if arr.size == 0:
            return arr.tolist()

        from scipy import ndimage
        # Label background connected from edges
        mask = np.zeros_like(arr, dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        mask[arr != 0] = False

        labeled, _ = ndimage.label(mask)
        # Keep only the background region connected to edges
        bg_mask = labeled == labeled[0, 0] if labeled[0, 0] > 0 else np.zeros_like(arr, dtype=bool)

        # Fill non-background, non-object cells with most common non-zero color
        non_zero_colors = arr[arr != 0]
        if len(non_zero_colors) > 0:
            fill_color = int(np.bincount(non_zero_colors).argmax())
            result = arr.copy()
            result[(~bg_mask) & (arr == 0)] = fill_color
            return result.tolist()
        return arr.tolist()

    @staticmethod
    def _center_object(test_input, train_pairs, patterns):
        """Center the largest object in the grid."""
        arr = np.array(test_input)
        from scipy import ndimage

        labeled, num_features = ndimage.label(arr != 0)
        if num_features == 0:
            return arr.tolist()

        sizes = ndimage.sum(arr != 0, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        obj_mask = labeled == largest

        rows = np.any(obj_mask, axis=1)
        cols = np.any(obj_mask, axis=0)
        obj = arr[np.ix_(rows, cols)]

        h, w = obj.shape
        out_h, out_w = arr.shape
        y_off = (out_h - h) // 2
        x_off = (out_w - w) // 2

        result = np.zeros_like(arr)
        result[y_off:y_off+h, x_off:x_off+w] = obj
        return result.tolist()

    @staticmethod
    def _complement_colors(test_input, train_pairs, patterns):
        """Replace each color with its 'complement' (10 - color)."""
        arr = np.array(test_input)
        return (9 - arr).tolist()

    @staticmethod
    def _safe_fallback(test_input):
        """Return a copy of input as safe fallback."""
        return [row[:] for row in test_input]


# ============================================================
# MAIN SOLVER
# ============================================================
class ARCSolver:
    """Main solver that orchestrates all strategies."""

    def __init__(self, config: SolverConfig):
        self.config = config
        self.code_gen = LLMCodeGenerator(config) if config.MODEL_NAME else None
        self.executor = ExecutionEngine()
        self.heuristic = HeuristicSolver()
        self.stats = Counter()

    def solve_task(self, task_id: str, task: Dict) -> List[List[List[int]]]:
        """Solve a single task, returning 2 attempt grids."""
        test_pairs = task['test']
        train_pairs = task['train']

        if not test_pairs:
            self.stats['no_test_pairs'] += 1
            return [[[0]], [[0]]]

        attempts = []
        verified_codes = []

        # Strategy 1: LLM Code Generation
        if self.code_gen:
            try:
                codes = self.code_gen.generate_code(task)
                self.stats['code_generated'] += len(codes)

                for code in codes:
                    verified = self.executor.execute_and_verify(code, task, self.config.CODE_TIMEOUT)
                    if verified:
                        verified_codes.append(verified)

                self.stats['code_verified'] += len(verified_codes)
            except Exception as e:
                print(f"  LLM code gen failed for {task_id}: {e}")

        # Run verified codes on test inputs
        for code in verified_codes:
            for test_pair in test_pairs:
                test_input = test_pair['input']
                result = self.executor.run_on_test(code, test_input, self.config.CODE_TIMEOUT)
                if result:
                    attempts.append(result)
                    if len(attempts) >= 2:
                        break
            if len(attempts) >= 2:
                break

        # Strategy 2: Heuristic Rules (fallback)
        if len(attempts) < 2 and self.config.USE_HEURISTICS:
            heuristic_attempts = self.heuristic.get_all_attempts(task)
            for ha in heuristic_attempts:
                if ha not in attempts:
                    attempts.append(ha)
                    if len(attempts) >= 2:
                        break

        # Strategy 3: Safe fallback
        while len(attempts) < 2:
            test_input = test_pairs[0]['input']
            attempts.append([row[:] for row in test_input])

        self.stats['tasks_solved'] += 1
        return attempts[:2]

    def solve_all(self, data_dir: str, output_path: str):
        """Solve all tasks in a directory and save submission."""
        json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

        if self.config.MAX_TASKS:
            json_files = json_files[:self.config.MAX_TASKS]

        submission = {}
        total = len(json_files)
        start = time.time()

        print(f"Solving {total} tasks from {data_dir}")
        print("=" * 60)

        for i, fpath in enumerate(json_files):
            task_id = os.path.splitext(os.path.basename(fpath))[0]

            with open(fpath, 'r') as f:
                task = json.load(f)

            if i % 5 == 0 or i == total - 1:
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
                print(f"[{i+1}/{total}] {task_id} | ETA: {eta/60:.1f}m")

            try:
                attempts = self.solve_task(task_id, task)
                submission[task_id] = attempts
            except Exception as e:
                print(f"  ERROR {task_id}: {e}")
                traceback.print_exc()
                submission[task_id] = [[[0]], [[0]]]

        elapsed = time.time() - start
        print(f"\nDone! {total} tasks in {elapsed/60:.1f}m ({elapsed/total:.1f}s/task)")

        # Save submission
        with open(output_path, 'w') as f:
            json.dump(submission, f, indent=2)
        print(f"Submission saved: {output_path}")

        # Print stats
        print(f"\nStats: {dict(self.stats)}")

        return submission


# ============================================================
# EVALUATION MODE
# ============================================================
def evaluate(data_dir: str, solver: ARCSolver) -> Dict:
    """Evaluate solver on training data (where we have answers)."""
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    total_tasks = 0
    solved_tasks = 0
    total_test_pairs = 0
    correct_pairs = 0

    for fpath in json_files:
        task_id = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath, 'r') as f:
            task = json.load(f)

        if not task.get('test') or 'output' not in task['test'][0]:
            continue

        total_tasks += 1
        attempts = solver.solve_task(task_id, task)

        for j, test_pair in enumerate(task['test']):
            if 'output' not in test_pair:
                continue

            total_test_pairs += 1
            expected = test_pair['output']

            # Check attempt 1
            if j < len(attempts[0]) and verify_output(expected, attempts[0][j]):
                correct_pairs += 1
                continue

            # Check attempt 2
            if len(attempts) > 1 and j < len(attempts[1]) and verify_output(expected, attempts[1][j]):
                correct_pairs += 1
                continue

        # Track per-task solve rate
        if total_tasks % 50 == 0:
            print(f"  [{total_tasks} tasks] Solved: {correct_pairs}/{total_test_pairs} pairs ({100*correct_pairs/max(1,total_test_pairs):.1f}%)")

    return {
        'total_tasks': total_tasks,
        'solved_tasks': solved_tasks,
        'total_test_pairs': total_test_pairs,
        'correct_pairs': correct_pairs,
        'accuracy': correct_pairs / max(1, total_test_pairs)
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Solver")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to task data directory')
    parser.add_argument('--output', type=str, default='submission.json', help='Output submission file')
    parser.add_argument('--eval', action='store_true', help='Evaluate on training data')
    parser.add_argument('--max_tasks', type=int, default=None, help='Max tasks to solve')
    parser.add_argument('--no_heuristics', action='store_true', help='Disable heuristic solver')
    parser.add_argument('--model', type=str, default=None, help='Model name (default: Qwen/Qwen3-4B)')
    args = parser.parse_args()

    random.seed(SolverConfig.SEED)
    np.random.seed(SolverConfig.SEED)

    config = SolverConfig()
    if args.model:
        config.MODEL_NAME = args.model
    if args.no_heuristics:
        config.USE_HEURISTICS = False
    if args.max_tasks:
        config.MAX_TASKS = args.max_tasks

    solver = ARCSolver(config)

    if args.eval:
        print("=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)
        results = evaluate(args.data_dir, solver)
        print(f"\nResults: {json.dumps(results, indent=2)}")
    else:
        solver.solve_all(args.data_dir, args.output)


if __name__ == "__main__":
    main()
