#!/usr/bin/env python3
"""
AnnotateX ARC-AGI-2 Solver v1
Multi-heuristic baseline solver for ARC Prize 2026 ARC-AGI-2 competition.
Adapts in-context learning methodology with pattern-matching heuristics.
"""

import json
import numpy as np
from copy import deepcopy
from collections import Counter

import os
import glob

print("Loading ARC-AGI-2 test data...")

# Find the competition data directory
data_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'arc-agi_test_challenges.json' in files:
        data_dir = root
        break

if data_dir is None:
    # Fallback: try common patterns
    candidates = glob.glob('/kaggle/input/*/arc-agi_test_challenges.json')
    if candidates:
        data_dir = os.path.dirname(candidates[0])
    else:
        # List all input directories for debugging
        print("Available input directories:")
        for d in os.listdir('/kaggle/input/'):
            print(f"  {d}")
        raise FileNotFoundError("Could not find arc-agi_test_challenges.json")

print(f"Data directory: {data_dir}")

with open(os.path.join(data_dir, 'arc-agi_test_challenges.json')) as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test tasks")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def grid_to_array(grid):
    return np.array(grid, dtype=int)

def array_to_grid(arr):
    return arr.tolist()

def grid_shape(grid):
    if not grid or not grid[0]:
        return (0, 0)
    return (len(grid), len(grid[0]))


# ============================================================
# HEURISTIC SOLVER
# ============================================================

class HeuristicSolver:
    """Multi-strategy heuristic solver for ARC grid puzzles."""

    def solve(self, task):
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
            self.compress_rows,
            self.compress_cols,
            self.move_object,
            self.same_as_input,
        ]

        for solver_fn in solvers:
            try:
                pred, conf = solver_fn(task)
                if pred is not None and conf > best_conf:
                    best_pred = pred
                    best_conf = conf
            except Exception:
                continue

        return best_pred

    def _validate(self, task, pred_fn):
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

    # --- Individual Heuristics ---

    def same_as_input(self, task):
        def fn(inp, ex):
            return deepcopy(inp)
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def color_mapping(self, task):
        ex0 = task['train'][0]
        mapping = self._infer_color_map(ex0['input'], ex0['output'])
        if mapping is None:
            return None, 0

        def fn(inp, ex):
            return self._apply_color_map(inp, mapping)

        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def _infer_color_map(self, inp, out):
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
                        return None
                else:
                    mapping[src] = dst
        if len(mapping) == 0 or all(k == v for k, v in mapping.items()):
            return None
        return mapping

    def _apply_color_map(self, grid, mapping):
        result = deepcopy(grid)
        for i in range(len(result)):
            for j in range(len(result[0])):
                if result[i][j] in mapping:
                    result[i][j] = mapping[result[i][j]]
        return result

    def tile_repeat(self, task):
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
        fy, fx = out_h // in_h, out_w // in_w
        if fy == fx and fy > 1:
            return fy
        if fy > 1 or fx > 1:
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
        def fn(inp, ex):
            return [row[::-1] for row in inp]
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def mirror_vertical(self, task):
        def fn(inp, ex):
            return inp[::-1]
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def mirror_diagonal(self, task):
        def fn(inp, ex):
            return grid_to_array(inp).T.tolist()
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def extract_object(self, task):
        def fn(inp, ex):
            arr = grid_to_array(inp)
            nonzero = np.nonzero(arr)
            if len(nonzero[0]) == 0:
                return None
            min_r, max_r = nonzero[0].min(), nonzero[0].max()
            min_c, max_c = nonzero[1].min(), nonzero[1].max()
            return arr[min_r:max_r+1, min_c:max_c+1].tolist()
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def fill_region(self, task):
        def fn(inp, ex):
            arr = grid_to_array(inp).astype(int)
            nonzero = np.nonzero(arr)
            if len(nonzero[0]) == 0:
                return None
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
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def detect_borders(self, task):
        def fn(inp, ex):
            arr = grid_to_array(inp)
            result = np.zeros_like(arr)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if arr[i, j] != 0:
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
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def rotate_90(self, task):
        def fn(inp, ex):
            return np.rot90(grid_to_array(inp), -1).tolist()
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def rotate_180(self, task):
        def fn(inp, ex):
            return np.rot90(grid_to_array(inp), 2).tolist()
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def rotate_270(self, task):
        def fn(inp, ex):
            return np.rot90(grid_to_array(inp), 1).tolist()
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def _scale(self, task, factor):
        def fn(inp, ex):
            arr = grid_to_array(inp)
            return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1).tolist()
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def scale_2x(self, task):
        return self._scale(task, 2)

    def scale_3x(self, task):
        return self._scale(task, 3)

    def remove_color(self, task):
        ex0 = task['train'][0]
        inp_colors = set(c for row in ex0['input'] for c in row)
        out_colors = set(c for row in ex0['output'] for c in row)
        removed = inp_colors - out_colors
        if len(removed) != 1:
            return None, 0
        remove_c = removed.pop()

        def fn(inp, ex):
            return [[0 if c == remove_c else c for c in row] for row in inp]
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def compress_rows(self, task):
        def fn(inp, ex):
            return [row for row in inp if any(c != 0 for c in row)]
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def compress_cols(self, task):
        def fn(inp, ex):
            if not inp or not inp[0]:
                return None
            ncols = len(inp[0])
            non_empty = [j for j in range(ncols) if any(row[j] != 0 for row in inp)]
            return [[row[j] for j in non_empty] for row in inp]
        conf, _ = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def move_object(self, task):
        ex0 = task['train'][0]
        inp = grid_to_array(ex0['input'])
        out = grid_to_array(ex0['output'])
        inp_obj = self._find_object_props(inp)
        out_obj = self._find_object_props(out)

        if inp_obj and out_obj:
            if (inp_obj['shape'] == out_obj['shape'] and
                inp_obj['color'] == out_obj['color'] and
                inp_obj['bbox'] != out_obj['bbox']):
                offset_r = out_obj['bbox'][0] - inp_obj['bbox'][0]
                offset_c = out_obj['bbox'][1] - inp_obj['bbox'][1]

                valid = True
                for ex in task['train'][1:]:
                    i_arr = grid_to_array(ex['input'])
                    o_arr = grid_to_array(ex['output'])
                    i_obj = self._find_object_props(i_arr)
                    o_obj = self._find_object_props(o_arr)
                    if not i_obj or not o_obj:
                        valid = False
                        break
                    if (o_obj['bbox'][0] - i_obj['bbox'][0] != offset_r or
                        o_obj['bbox'][1] - i_obj['bbox'][1] != offset_c):
                        valid = False
                        break

                if valid:
                    test_inp = grid_to_array(task['test'][0]['input'])
                    test_out = np.zeros_like(test_inp)
                    for r in range(test_inp.shape[0]):
                        for c in range(test_inp.shape[1]):
                            nr, nc = r + offset_r, c + offset_c
                            if 0 <= nr < test_out.shape[0] and 0 <= nc < test_out.shape[1]:
                                test_out[nr, nc] = test_inp[r, c]
                    return test_out.tolist(), 0.9

        return None, 0

    def _find_object_props(self, arr):
        nonzero = np.nonzero(arr)
        if len(nonzero[0]) == 0:
            return None
        min_r, max_r = nonzero[0].min(), nonzero[0].max()
        min_c, max_c = nonzero[1].min(), nonzero[1].max()
        obj = arr[min_r:max_r+1, min_c:max_c+1]
        color_counts = Counter(obj[obj > 0].flatten().tolist())
        most_common = color_counts.most_common(1)[0][0] if color_counts else 0
        return {'bbox': (min_r, min_c, max_r, max_c), 'shape': obj.shape, 'color': most_common}


# ============================================================
# MAIN: Generate Submission
# ============================================================

print("Initializing AnnotateX solver...")
solver = HeuristicSolver()
submission = {}
total_tasks = len(test_data)
heuristic_solved = 0

for i, (task_id, task) in enumerate(test_data.items()):
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{total_tasks}...")

    test_input = task['test'][0]['input']

    # Try heuristic solvers
    prediction = solver.solve(task)

    if prediction is not None:
        heuristic_solved += 1
        attempt_1 = prediction
        # attempt_2: use different heuristic if possible, else copy input
        attempt_2 = deepcopy(test_input)
    else:
        # Fallback: copy input as both attempts
        attempt_1 = deepcopy(test_input)
        attempt_2 = deepcopy(test_input)

    submission[task_id] = [{"attempt_1": attempt_1, "attempt_2": attempt_2}]

print(f"\n{'='*50}")
print(f"AnnotateX ARC-AGI-2 Solver v1 Results")
print(f"{'='*50}")
print(f"Total tasks: {total_tasks}")
print(f"Heuristic-solved: {heuristic_solved} ({100*heuristic_solved/total_tasks:.1f}%)")
print(f"Fallback (copy input): {total_tasks - heuristic_solved}")

# Save submission
output_path = '/kaggle/working/submission.json'
with open(output_path, 'w') as f:
    json.dump(submission, f)

print(f"\nSubmission saved to: {output_path}")
print(f"File size: {len(json.dumps(submission))} bytes")
print("Done!")
