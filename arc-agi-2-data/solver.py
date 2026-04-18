#!/usr/bin/env python3
"""
ARC-AGI-2 Solver — AnnotateX ICL Methodology
Multi-strategy approach: heuristics + LLM in-context learning
"""

import json
import numpy as np
from copy import deepcopy
from collections import Counter
from itertools import product

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def grid_to_array(grid):
    return np.array(grid, dtype=int)

def array_to_grid(arr):
    return arr.tolist()

def grid_shape(grid):
    if not grid or not grid[0]:
        return (0, 0)
    return (len(grid), len(grid[0]))

def print_grid(grid, label=""):
    color_map = {0: '.', 1: '#', 2: 'R', 3: 'G', 4: 'B', 5: 'Y', 6: 'P', 7: 'O', 8: 'C', 9: 'M'}
    if label:
        print(f"  {label} ({len(grid)}x{len(grid[0])}):")
    for row in grid:
        print('    ' + ' '.join(color_map.get(c, str(c)) for c in row))

def grid_to_text(grid):
    """Convert 2D grid to compact text representation for LLM prompting."""
    if not grid:
        return "empty grid"
    rows = []
    for row in grid:
        rows.append(' '.join(str(c) for c in row))
    return '\n'.join(rows)

def task_to_text(task):
    """Convert full ARC task to text for ICL prompting."""
    parts = []
    for i, ex in enumerate(task.get('train', [])):
        parts.append(f"Example {i+1} Input:")
        parts.append(grid_to_text(ex['input']))
        parts.append(f"Example {i+1} Output:")
        parts.append(grid_to_text(ex['output']))
        parts.append("")
    return '\n'.join(parts)


# ============================================================
# HEURISTIC SOLVERS
# ============================================================

class HeuristicSolver:
    """Collection of pattern-matching heuristic solvers for ARC tasks."""

    def solve(self, task):
        """Try all heuristics, return best prediction or None."""
        best_pred = None
        best_conf = -1

        solvers = [
            self.color_mapping,
            self.tile_repeat,
            self.mirror_horizontal,
            self.mirror_vertical,
            self.mirror_diagonal,
            self.extract_object,
            self.fill_region,
            self.detect_borders,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.scale_2x,
            self.scale_3x,
            self.remove_color,
            self.replace_color,
            self.compress_rows,
            self.compress_cols,
            self.detect_and_transform_pattern,
            self.same_as_input,
        ]

        for solver in solvers:
            try:
                pred, conf = solver(task)
                if pred is not None and conf > best_conf:
                    best_pred = pred
                    best_conf = conf
            except Exception:
                continue

        return best_pred

    def _validate(self, task, pred_fn):
        """Validate a prediction function against all train examples.
        Returns (avg_confidence, predictions) or (0, None)."""
        preds = []
        total_conf = 0
        for ex in task['train']:
            try:
                pred = pred_fn(ex['input'], ex)
                expected = ex['output']
                if pred is None:
                    return 0, None
                pred_arr = grid_to_array(pred)
                exp_arr = grid_to_array(expected)
                if pred_arr.shape == exp_arr.shape:
                    match = np.sum(pred_arr == exp_arr) / pred_arr.size
                    total_conf += match
                    preds.append(pred)
                else:
                    return 0, None
            except Exception:
                return 0, None
        return total_conf / len(task['train']), preds

    def same_as_input(self, task):
        """Identity: output = input."""
        def fn(inp, ex):
            return deepcopy(inp)
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def color_mapping(self, task):
        """Detect a consistent color-to-color mapping across all train examples."""
        def fn(inp, ex):
            mapping = self._infer_color_map(ex['input'], ex['output'])
            if mapping is None:
                return None
            return self._apply_color_map(inp, mapping)

        # Build mapping from first train example
        ex0 = task['train'][0]
        mapping = self._infer_color_map(ex0['input'], ex0['output'])
        if mapping is None:
            return None, 0

        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def _infer_color_map(self, inp, out):
        """Infer a color mapping from input to output grids."""
        inp_arr = grid_to_array(inp)
        out_arr = grid_to_array(out)
        if inp_arr.shape != out_arr.shape:
            return None
        mapping = {}
        for i in range(inp_arr.shape[0]):
            for j in range(inp_arr.shape[1]):
                src = int(inp_arr[i, j])
                dst = int(out_arr[i, j])
                if src in mapping:
                    if mapping[src] != dst:
                        return None  # inconsistent mapping
                else:
                    mapping[src] = dst
        if len(mapping) == 0 or all(k == v for k, v in mapping.items()):
            return None  # trivial mapping
        return mapping

    def _apply_color_map(self, grid, mapping):
        result = deepcopy(grid)
        for i in range(len(result)):
            for j in range(len(result[0])):
                if result[i][j] in mapping:
                    result[i][j] = mapping[result[i][j]]
        return result

    def tile_repeat(self, task):
        """Detect tiling pattern: input is tiled to form larger output."""
        def fn(inp, ex):
            factor = self._infer_tile_factor(ex['input'], ex['output'])
            if factor is None:
                return None
            return self._tile_grid(inp, factor)

        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def _infer_tile_factor(self, inp, out):
        in_h, in_w = grid_shape(inp)
        out_h, out_w = grid_shape(out)
        if in_h == 0 or in_w == 0:
            return None
        if out_h % in_h != 0 or out_w % in_w != 0:
            return None
        fy = out_h // in_h
        fx = out_w // in_w
        if fy == fx and fy > 1:
            return fy
        if fy > 1 or fx > 1:
            # Check if tiling works
            return (fy, fx)
        return None

    def _tile_grid(self, grid, factor):
        if isinstance(factor, tuple):
            fy, fx = factor
        else:
            fy, fx = factor, factor
        result = []
        for r in range(fy):
            for row in grid:
                new_row = []
                for c in range(fx):
                    new_row.extend(row)
                result.append(new_row)
        return result

    def mirror_horizontal(self, task):
        """Mirror grid horizontally (left-right flip)."""
        def fn(inp, ex):
            return [row[::-1] for row in inp]
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def mirror_vertical(self, task):
        """Mirror grid vertically (top-bottom flip)."""
        def fn(inp, ex):
            return inp[::-1]
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def mirror_diagonal(self, task):
        """Transpose grid (diagonal mirror)."""
        def fn(inp, ex):
            arr = grid_to_array(inp)
            return array_to_grid(arr.T)
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def extract_object(self, task):
        """Extract the non-background object from the grid."""
        def fn(inp, ex):
            return self._extract_main_object(inp)
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def _extract_main_object(self, grid):
        arr = grid_to_array(grid)
        # Find bounding box of non-zero elements
        nonzero = np.nonzero(arr)
        if len(nonzero[0]) == 0:
            return None
        min_r, max_r = nonzero[0].min(), nonzero[0].max()
        min_c, max_c = nonzero[1].min(), nonzero[1].max()
        return arr[min_r:max_r+1, min_c:max_c+1].tolist()

    def fill_region(self, task):
        """Fill the interior of a shape with the border color."""
        def fn(inp, ex):
            return self._fill_interior(inp)
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def _fill_interior(self, grid):
        arr = grid_to_array(grid).astype(int)
        nonzero = np.nonzero(arr)
        if len(nonzero[0]) == 0:
            return None
        # Find most common non-zero color
        vals = arr[arr > 0]
        if len(vals) == 0:
            return None
        fill_color = int(Counter(vals.flatten()).most_common(1)[0][0])
        min_r, max_r = nonzero[0].min(), nonzero[0].max()
        min_c, max_c = nonzero[1].min(), nonzero[1].max()
        for r in range(min_r+1, max_r):
            for c in range(min_c+1, max_c):
                arr[r, c] = fill_color
        return arr.tolist()

    def detect_borders(self, task):
        """Extract the border/outline of objects."""
        def fn(inp, ex):
            return self._extract_border(inp)
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def _extract_border(self, grid):
        arr = grid_to_array(grid)
        result = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] != 0:
                    # Check if on border (adjacent to 0 or edge)
                    is_border = False
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if ni < 0 or ni >= arr.shape[0] or nj < 0 or nj >= arr.shape[1]:
                            is_border = True
                            break
                        if arr[ni, nj] == 0:
                            is_border = True
                            break
                    if is_border:
                        result[i, j] = arr[i, j]
        return result.tolist()

    def rotate_90(self, task):
        """Rotate grid 90 degrees clockwise."""
        def fn(inp, ex):
            arr = grid_to_array(inp)
            return array_to_grid(np.rot90(arr, -1))
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def rotate_180(self, task):
        """Rotate grid 180 degrees."""
        def fn(inp, ex):
            arr = grid_to_array(inp)
            return array_to_grid(np.rot90(arr, 2))
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def rotate_270(self, task):
        """Rotate grid 270 degrees clockwise (90 CCW)."""
        def fn(inp, ex):
            arr = grid_to_array(inp)
            return array_to_grid(np.rot90(arr, 1))
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def scale_2x(self, task):
        """Scale grid by 2x using nearest neighbor."""
        return self._scale(task, 2)

    def scale_3x(self, task):
        """Scale grid by 3x using nearest neighbor."""
        return self._scale(task, 3)

    def _scale(self, task, factor):
        def fn(inp, ex):
            arr = grid_to_array(inp)
            return array_to_grid(np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1))
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def remove_color(self, task):
        """Remove a specific color from the grid (set to 0)."""
        ex0 = task['train'][0]
        inp_colors = set(c for row in ex0['input'] for c in row)
        out_colors = set(c for row in ex0['output'] for c in row)
        removed = inp_colors - out_colors
        if len(removed) != 1:
            return None, 0
        remove_c = removed.pop()

        def fn(inp, ex):
            return [[0 if c == remove_c else c for c in row] for row in inp]
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def replace_color(self, task):
        """Replace one color with another (keeping grid shape same)."""
        ex0 = task['train'][0]
        if grid_shape(ex0['input']) != grid_shape(ex0['output']):
            return None, 0
        mapping = self._infer_color_map(ex0['input'], ex0['output'])
        if mapping is None:
            return None, 0
        return self.color_mapping(task)

    def compress_rows(self, task):
        """Remove empty (all-zero) rows."""
        def fn(inp, ex):
            return [row for row in inp if any(c != 0 for c in row)]
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def compress_cols(self, task):
        """Remove empty (all-zero) columns."""
        def fn(inp, ex):
            if not inp or not inp[0]:
                return None
            ncols = len(inp[0])
            non_empty = [j for j in range(ncols) if any(row[j] != 0 for row in inp)]
            return [[row[j] for j in non_empty] for row in inp]
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def detect_and_transform_pattern(self, task):
        """More complex pattern: detect object properties and transform."""
        ex0 = task['train'][0]
        inp = grid_to_array(ex0['input'])
        out = grid_to_array(ex0['output'])

        # Try: move object to different position while preserving shape
        inp_obj = self._find_object_properties(inp)
        out_obj = self._find_object_properties(out)

        if inp_obj and out_obj:
            # Check if same shape but different position
            if (inp_obj['shape'] == out_obj['shape'] and
                inp_obj['color'] == out_obj['color'] and
                (inp_obj['bbox'] != out_obj['bbox'])):
                offset_r = out_obj['bbox'][0] - inp_obj['bbox'][0]
                offset_c = out_obj['bbox'][1] - inp_obj['bbox'][1]

                # Validate on all train examples
                valid = True
                for ex in task['train'][1:]:
                    i_arr = grid_to_array(ex['input'])
                    o_arr = grid_to_array(ex['output'])
                    i_obj = self._find_object_properties(i_arr)
                    o_obj = self._find_object_properties(o_arr)
                    if not i_obj or not o_obj:
                        valid = False
                        break
                    o_r = o_obj['bbox'][0] - i_obj['bbox'][0]
                    o_c = o_obj['bbox'][1] - i_obj['bbox'][1]
                    if o_r != offset_r or o_c != offset_c:
                        valid = False
                        break

                if valid:
                    test_inp = grid_to_array(task['test'][0]['input'])
                    test_out = np.zeros_like(test_inp)
                    for r in range(test_inp.shape[0]):
                        for c in range(test_inp.shape[1]):
                            nr, nc = r + offset_r, c + offset_c
                            if (0 <= nr < test_out.shape[0] and
                                0 <= nc < test_out.shape[1]):
                                test_out[nr, nc] = test_inp[r, c]
                    return test_out.tolist(), 0.9

        return None, 0

    def _find_object_properties(self, arr):
        nonzero = np.nonzero(arr)
        if len(nonzero[0]) == 0:
            return None
        min_r, max_r = nonzero[0].min(), nonzero[0].max()
        min_c, max_c = nonzero[1].min(), nonzero[1].max()
        obj = arr[min_r:max_r+1, min_c:max_c+1]
        color_counts = Counter(obj[obj > 0].flatten().tolist())
        most_common_color = color_counts.most_common(1)[0][0] if color_counts else 0
        return {
            'bbox': (min_r, min_c, max_r, max_c),
            'shape': obj.shape,
            'color': most_common_color,
            'pixels': int(np.sum(arr > 0))
        }


# ============================================================
# ICL GRID-TO-TEXT ENCODER (for LLM-based approach)
# ============================================================

class GridToTextEncoder:
    """Encodes ARC grids as structured text for LLM in-context learning."""

    COLOR_NAMES = {
        0: 'black', 1: 'blue', 2: 'red', 3: 'green',
        4: 'yellow', 5: 'gray', 6: 'magenta', 7: 'orange',
        8: 'cyan', 9: 'brown'
    }

    @classmethod
    def encode_grid(cls, grid, label=""):
        """Encode a grid as visual text with color names."""
        lines = []
        if label:
            lines.append(label)
        for row in grid:
            line = ' '.join(cls.COLOR_NAMES.get(c, str(c)) for c in row)
            lines.append(line)
        return '\n'.join(lines)

    @classmethod
    def encode_task(cls, task):
        """Encode an entire ARC task as a text prompt."""
        parts = []
        parts.append("You are given input-output pairs that demonstrate a pattern.")
        parts.append("Apply the same pattern to the test input to produce the output.\n")

        for i, ex in enumerate(task['train']):
            parts.append(f"--- Training Example {i+1} ---")
            parts.append(f"Input grid ({len(ex['input'])}x{len(ex['input'][0])}):")
            parts.append(cls.encode_grid(ex['input']))
            parts.append(f"Output grid ({len(ex['output'])}x{len(ex['output'][0])}):")
            parts.append(cls.encode_grid(ex['output']))
            parts.append("")

        parts.append("--- Test Input ---")
        test_ex = task['test'][0]
        parts.append(f"Input grid ({len(test_ex['input'])}x{len(test_ex['input'][0])}):")
        parts.append(cls.encode_grid(test_ex['input']))
        parts.append("\nOutput the predicted grid as a 2D array. Use the same format.")

        return '\n'.join(parts)

    @classmethod
    def encode_grid_compact(cls, grid):
        """Compact numeric encoding for efficient LLM context."""
        lines = []
        for row in grid:
            lines.append(' '.join(str(c) for c in row))
        return '\n'.join(lines)


# ============================================================
# MAIN SOLVER PIPELINE
# ============================================================

def generate_submission(test_data, output_path):
    """Generate submission.json from test challenges."""
    solver = HeuristicSolver()
    submission = {}

    total_tasks = len(test_data)
    heuristic_solved = 0

    for i, (task_id, task) in enumerate(test_data.items()):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Processing task {i+1}/{total_tasks} ({task_id})...")

        test_input = task['test'][0]['input']

        # Try heuristic solvers
        prediction = solver.solve(task)

        if prediction is not None:
            heuristic_solved += 1
            attempt_1 = prediction
            # For attempt_2, try same-as-input as fallback
            attempt_2 = deepcopy(test_input)
        else:
            # Fallback: use test input as both attempts (will score 0 but valid)
            attempt_1 = deepcopy(test_input)
            attempt_2 = deepcopy(test_input)

        submission[task_id] = [{"attempt_1": attempt_1, "attempt_2": attempt_2}]

    print(f"\n=== RESULTS ===")
    print(f"Total tasks: {total_tasks}")
    print(f"Heuristic-solved: {heuristic_solved}")
    print(f"Fallback (copy input): {total_tasks - heuristic_solved}")

    save_json(output_path, submission)
    print(f"\nSubmission saved to: {output_path}")
    return submission


def validate_on_training():
    """Validate the heuristic solver on training data."""
    solver = HeuristicSolver()
    train_challenges = load_json('/home/z/my-project/arc-agi-2-data/arc-agi_training_challenges.json')
    train_solutions = load_json('/home/z/my-project/arc-agi-2-data/arc-agi_training_solutions.json')

    solved = 0
    total = len(train_challenges)

    for task_id, task in train_challenges.items():
        prediction = solver.solve(task)
        if prediction is None:
            continue

        # Check against solution
        if task_id in train_solutions:
            solution = train_solutions[task_id][0]  # First test solution
            pred_arr = grid_to_array(prediction)
            sol_arr = grid_to_array(solution)

            if pred_arr.shape == sol_arr.shape and np.array_equal(pred_arr, sol_arr):
                solved += 1
                print(f"  SOLVED: {task_id}")

    print(f"\n=== TRAINING VALIDATION ===")
    print(f"Solved: {solved}/{total} ({100*solved/total:.1f}%)")
    return solved


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        print("Validating on training data...")
        validate_on_training()
    else:
        print("Loading test challenges...")
        test_data = load_json('/home/z/my-project/arc-agi-2-data/arc-agi_test_challenges.json')

        print("Generating submission...")
        submission = generate_submission(
            test_data,
            '/home/z/my-project/arc-agi-2-data/submission.json'
        )
        print("Done!")
