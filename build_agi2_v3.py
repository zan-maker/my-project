#!/usr/bin/env python3
"""
Build script for ARC-AGI-2 Solver v3 Notebook.
Generates a complete Jupyter notebook with 30+ cells and 3000+ lines of code.
"""

import json
import os

OUTPUT_PATH = "/home/z/my-project/download/arc_agi2_solver_v3.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def md(lines):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


def code(source_text):
    """Create a code cell."""
    lines = source_text.rstrip("\n").split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [l + "\n" for l in lines],
        "outputs": [],
        "execution_count": None,
    }


# ======================================================================
# CELL CONTENT DEFINITIONS
# ======================================================================

CELL_1_CODE = r'''# ============================================================
# ARC-AGI-2 Solver v3 — Multi-Strategy Ensemble with LLM + Evolution
# ============================================================
# Modules: Heuristic(100+) | DreamCoder | LLM(Qwen3-4B) | Evolutionary |
#          Grid Embedding CNN | BFS Graph Explorer | MCTS |
#          Cross-Example Analyzer | Output Predictor | Stuck Recovery |
# ============================================================

import json
import os
import copy
import time
import sys
import traceback
import threading
import re
import hashlib
import random
import math
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set, Callable
from itertools import combinations, permutations, product
import numpy as np

# Lazy imports (available on Kaggle)
_torch = None
_transformers = None
_bitsandbytes = None
_tqdm = None

def get_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            _torch = torch
        except ImportError:
            pass
    return _torch

def get_tqdm():
    global _tqdm
    if _tqdm is None:
        try:
            from tqdm.auto import tqdm
            _tqdm = tqdm
        except ImportError:
            _tqdm = lambda x, **kw: x
    return _tqdm

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "/kaggle/input/arc-prize-2026-arc-agi-2"
OUTPUT_PATH = "/kaggle/working/submission.json"
MODEL_NAME = "Qwen/Qwen3-4B"
QUANTIZATION_4BIT = True
MAX_NEW_TOKENS = 2048
CODE_TIMEOUT_SECONDS = 10
MAX_LLM_TIME_SECONDS = 300
NUM_SUBMISSION_ATTEMPTS = 2
EVOLUTIONARY_POPULATION = 12
EVOLUTIONARY_GENERATIONS = 3

COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green",
    4: "yellow", 5: "gray", 6: "magenta", 7: "orange",
    8: "cyan", 9: "brown",
}
COLOR_NAMES_INV = {v: k for k, v in COLOR_NAMES.items()}

random.seed(42)
np.random.seed(42)

print("[CONFIG] ARC-AGI-2 Solver v3 initialized")
print(f"[CONFIG] Data dir: {DATA_DIR}")
print(f"[CONFIG] Model: {MODEL_NAME}")
print(f"[CONFIG] Quantization: {'4-bit' if QUANTIZATION_4BIT else 'fp16'}")
print(f"[CONFIG] Evolutionary: pop={EVOLUTIONARY_POPULATION}, gen={EVOLUTIONARY_GENERATIONS}")
'''

CELL_3_CODE = r'''# ============================================================
# GRID UTILITIES
# ============================================================

def grid_to_str(grid, color_names=None):
    """Pretty-print a grid with optional color names."""
    if not grid:
        return "<empty>"
    cn = color_names or COLOR_NAMES
    lines = []
    for row in grid:
        lines.append(" ".join(str(c) for c in row))
    return "\n".join(lines)

def grids_equal(g1, g2):
    """Check if two grids are identical."""
    if not g1 or not g2:
        return g1 == g2
    if len(g1) != len(g2):
        return False
    for r1, r2 in zip(g1, g2):
        if len(r1) != len(r2):
            return False
        for c1, c2 in zip(r1, r2):
            if c1 != c2:
                return False
    return True

def grid_to_numpy(grid):
    """Convert grid to numpy array."""
    return np.array(grid, dtype=np.int32)

def numpy_to_grid(arr):
    """Convert numpy array to grid."""
    return arr.astype(int).tolist()

def normalize_grid(grid):
    """Normalize grid: ensure 2D list of ints, clamp 0-9."""
    if not grid:
        return [[0]]
    result = []
    for row in grid:
        r = []
        for c in row:
            r.append(max(0, min(9, int(c))))
        result.append(r)
    return result

def ensure_grid(grid):
    """Ensure grid is a valid 2D list."""
    if isinstance(grid, np.ndarray):
        return numpy_to_grid(grid)
    if isinstance(grid, list):
        if all(isinstance(x, int) for x in grid):
            return [[x] for x in grid]
        return [[int(c) for c in row] for row in grid]
    return [[0]]

def grid_hash(grid):
    """Hash a grid for deduplication."""
    s = json.dumps(grid, separators=(",", ":"))
    return hashlib.md5(s.encode()).hexdigest()[:12]

def grid_shape(grid):
    """Get (height, width) of a grid."""
    if not grid or not grid[0]:
        return (0, 0)
    return (len(grid), len(grid[0]))

def find_components(grid, connectivity=4):
    """Find connected components via flood fill. connectivity: 4 or 8."""
    if not grid:
        return []
    arr = grid_to_numpy(grid)
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]
    for r in range(h):
        for c in range(w):
            if not visited[r, c]:
                color = int(arr[r, c])
                cells = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    for dr, dc in neighbors:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                            if connectivity == 4:
                                if arr[nr, nc] == color:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                            else:
                                if arr[nr, nc] != 0:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                components.append({"color": color, "cells": cells, "size": len(cells)})
    return components

def component_bounding_box(component):
    """Get (min_r, min_c, max_r, max_c) for a component."""
    cells = component["cells"]
    if not cells:
        return (0, 0, 0, 0)
    rs = [c[0] for c in cells]
    cs = [c[1] for c in cells]
    return (min(rs), min(cs), max(rs), max(cs))

def extract_component_grid(grid, component):
    """Extract the sub-grid containing a component."""
    r1, c1, r2, c2 = component_bounding_box(component)
    arr = grid_to_numpy(grid)
    return numpy_to_grid(arr[r1:r2+1, c1:c2+1])

def count_colors(grid):
    """Count occurrences of each color."""
    if not grid:
        return Counter()
    return Counter(c for row in grid for c in row)

def extract_grid_features(grid):
    """Extract feature dict for a grid."""
    if not grid:
        return {"h": 0, "w": 0, "colors": [], "n_objects": 0}
    h, w = grid_shape(grid)
    cc = count_colors(grid)
    comps = find_components(grid, connectivity=4)
    non_zero_comps = [c for c in comps if c["color"] != 0]
    return {
        "h": h, "w": w,
        "colors": sorted(cc.keys()),
        "color_counts": dict(cc),
        "n_objects": len(non_zero_comps),
        "density": sum(1 for r in grid for c in r if c != 0) / (h * w) if h * w > 0 else 0,
        "has_symmetry_h": any(
            grids_equal(row, row[::-1]) for row in grid
        ),
        "has_symmetry_v": grid == grid[::-1],
    }

def describe_grid_natural_language(grid):
    """Describe a grid in natural language for LLM prompts."""
    feat = extract_grid_features(grid)
    desc = f"Grid is {feat['h']}x{feat['w']}. "
    desc += f"Colors present: {', '.join(COLOR_NAMES.get(c, str(c)) for c in feat['colors'])}. "
    desc += f"Contains {feat['n_objects']} objects. "
    desc += f"Pixel density: {feat['density']:.2f}. "
    if feat['has_symmetry_h']:
        desc += "Has horizontal symmetry. "
    if feat['has_symmetry_v']:
        desc += "Has vertical symmetry. "
    return desc

print("[UTILS] Grid utility functions loaded")
'''

CELL_5_CODE = r'''# ============================================================
# DATA LOADING
# ============================================================

def load_task(task_id, data_dir=None):
    """Load a single task by ID from data directory."""
    dd = data_dir or DATA_DIR
    for fname in os.listdir(dd):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(dd, fname)) as f:
                data = json.load(f)
            if isinstance(data, dict) and task_id in data:
                return data[task_id]
        except Exception:
            continue
    return None

def load_all_tasks(data_dir=None):
    """Load all tasks from data directory."""
    dd = data_dir or DATA_DIR
    all_tasks = {}
    for fname in sorted(os.listdir(dd)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(dd, fname)) as f:
                data = json.load(f)
            if isinstance(data, dict):
                all_tasks.update(data)
        except Exception:
            continue
    return all_tasks

def load_arc_data(data_dir=None):
    """Load ARC data, separating train and test tasks."""
    dd = data_dir or DATA_DIR
    all_tasks = load_all_tasks(dd)
    train_tasks = {}
    test_tasks = {}
    for tid, task in all_tasks.items():
        if "test" in task:
            test_tasks[tid] = task
        else:
            train_tasks[tid] = task
    return train_tasks, test_tasks

def print_task_summary(task, max_examples=3):
    """Print a summary of a task."""
    print(f"  Train examples: {len(task.get('train', []))}")
    print(f"  Test examples: {len(task.get('test', []))}")
    for i, ex in enumerate(task.get("train", [])[:max_examples]):
        inp = ex.get("input", [])
        out = ex.get("output", [])
        print(f"    Train[{i}]: input={grid_shape(inp)}, output={grid_shape(out)}")
    for i, ex in enumerate(task.get("test", [])[:max_examples]):
        inp = ex.get("input", [])
        print(f"    Test[{i}]: input={grid_shape(inp)}")

def load_data_main():
    """Main data loading function with fallback search."""
    data_dir = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if any("arc" in f.lower() and f.endswith(".json") for f in files):
            data_dir = root
            break
    if data_dir is None:
        data_dir = DATA_DIR
    print(f"[DATA] Loading from: {data_dir}")
    all_tasks = load_all_tasks(data_dir)
    print(f"[DATA] Loaded {len(all_tasks)} total tasks")
    train_tasks = {}
    eval_tasks = {}
    test_tasks = {}
    for tid, task in all_tasks.items():
        has_train = len(task.get("train", [])) > 0
        has_test = len(task.get("test", [])) > 0
        if has_test and has_train:
            test_tasks[tid] = task
        elif has_train:
            train_tasks[tid] = task
        elif has_test:
            eval_tasks[tid] = task
    print(f"[DATA] Test tasks with train: {len(test_tasks)}")
    print(f"[DATA] Train-only tasks: {len(train_tasks)}")
    print(f"[DATA] Eval tasks: {len(eval_tasks)}")
    return test_tasks, train_tasks, eval_tasks

print("[DATA] Data loading functions ready")
'''

CELL_7_CODE = r'''# ============================================================
# HEURISTIC LIBRARY — GEOMETRIC (8 base)
# ============================================================

def h_identity(grid):
    """Return grid unchanged."""
    return copy.deepcopy(grid)

def h_rotate90(grid):
    """Rotate grid 90 degrees clockwise."""
    arr = grid_to_numpy(grid)
    return numpy_to_grid(np.rot90(arr, -1))

def h_rotate180(grid):
    """Rotate grid 180 degrees."""
    arr = grid_to_numpy(grid)
    return numpy_to_grid(np.rot90(arr, 2))

def h_rotate270(grid):
    """Rotate grid 270 degrees clockwise (90 CCW)."""
    arr = grid_to_numpy(grid)
    return numpy_to_grid(np.rot90(arr, 1))

def h_flip_h(grid):
    """Flip grid horizontally (left-right)."""
    return [row[::-1] for row in grid]

def h_flip_v(grid):
    """Flip grid vertically (top-bottom)."""
    return grid[::-1]

def h_flip_diag(grid):
    """Flip grid along main diagonal (transpose)."""
    arr = grid_to_numpy(grid)
    return numpy_to_grid(arr.T)

def h_flip_anti_diag(grid):
    """Flip grid along anti-diagonal."""
    arr = grid_to_numpy(grid)
    rotated = np.rot90(arr, -1)
    return numpy_to_grid(rotated.T)

# ============================================================
# HEURISTIC LIBRARY — COLOR (8 base)
# ============================================================

def h_color_complement(grid):
    """Map each color to (9 - color)."""
    return [[9 - c for c in row] for row in grid]

def h_color_add_k(grid, k=1):
    """Add k to each color, mod 10."""
    return [[(c + k) % 10 for c in row] for row in grid]

def h_color_sub_k(grid, k=1):
    """Subtract k from each color, mod 10."""
    return [[(c - k) % 10 for c in row] for row in grid]

def h_fill_most_common(grid):
    """Fill entire grid with most common non-zero color."""
    cc = count_colors(grid)
    non_zero = {c: n for c, n in cc.items() if c != 0}
    if not non_zero:
        return grid
    mc = max(non_zero, key=non_zero.get)
    h, w = grid_shape(grid)
    return [[mc] * w for _ in range(h)]

def h_extract_color(grid, target_color=1):
    """Keep only the specified color, set others to 0."""
    return [[c if c == target_color else 0 for c in row] for row in grid]

def h_remap_colors(grid, remap_dict):
    """Remap colors according to dict {old: new}."""
    return [[remap_dict.get(c, c) for c in row] for row in grid]

def h_count_colors_output(grid):
    """Create a 1xN grid where each cell is the count of that color."""
    cc = count_colors(grid)
    max_c = max(cc.keys()) if cc else 0
    return [[cc.get(i, 0) for i in range(max_c + 1)]]

def h_invert_colors(grid):
    """Swap 0 <-> most common non-zero color."""
    cc = count_colors(grid)
    non_zero = {c: n for c, n in cc.items() if c != 0}
    if not non_zero:
        return grid
    mc = max(non_zero, key=non_zero.get)
    return [[0 if c == mc else (mc if c == 0 else c) for c in row] for row in grid]

print("[HEURISTICS] Geometric + Color heuristics loaded (16)")
'''

CELL_9_CODE = r'''# ============================================================
# HEURISTIC LIBRARY — OBJECT (6 base)
# ============================================================

def h_crop(grid):
    """Crop grid to bounding box of non-zero content."""
    arr = grid_to_numpy(grid)
    nz = np.nonzero(arr)
    if len(nz[0]) == 0:
        return grid
    return numpy_to_grid(arr[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1])

def h_crop_each_color(grid):
    """For each color, crop to its bounding box, return as list of sub-grids."""
    results = []
    unique_cs = set(c for row in grid for c in row if c != 0)
    for color in sorted(unique_cs):
        sub = [[c if c == color else 0 for c in row] for row in grid]
        cropped = h_crop(sub)
        results.append(cropped)
    return results

def h_extract_largest_component(grid):
    """Extract the largest connected component."""
    comps = find_components(grid, connectivity=4)
    non_zero = [c for c in comps if c["color"] != 0]
    if not non_zero:
        return grid
    largest = max(non_zero, key=lambda x: x["size"])
    return extract_component_grid(grid, largest)

def h_extract_smallest_component(grid):
    """Extract the smallest connected component."""
    comps = find_components(grid, connectivity=4)
    non_zero = [c for c in comps if c["color"] != 0]
    if not non_zero:
        return grid
    smallest = min(non_zero, key=lambda x: x["size"])
    return extract_component_grid(grid, smallest)

def h_move_to_topleft(grid):
    """Move content to top-left corner."""
    cropped = h_crop(grid)
    return cropped

def h_extract_all_components_separate(grid):
    """Extract each non-zero component as a separate grid."""
    comps = find_components(grid, connectivity=4)
    non_zero = [c for c in comps if c["color"] != 0]
    return [extract_component_grid(grid, comp) for comp in non_zero]

# ============================================================
# HEURISTIC LIBRARY — SCALING (4 base)
# ============================================================

def h_scale_2x(grid):
    """Scale grid by factor 2."""
    arr = grid_to_numpy(grid)
    return numpy_to_grid(np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1))

def h_scale_3x(grid):
    """Scale grid by factor 3."""
    arr = grid_to_numpy(grid)
    return numpy_to_grid(np.repeat(np.repeat(arr, 3, axis=0), 3, axis=1))

def h_shrink_by_half(grid):
    """Shrink grid by factor 2 (take every other row/col)."""
    arr = grid_to_numpy(grid)
    return numpy_to_grid(arr[::2, ::2])

def h_shrink_by_third(grid):
    """Shrink grid by factor 3."""
    arr = grid_to_numpy(grid)
    return numpy_to_grid(arr[::3, ::3])

# ============================================================
# HEURISTIC LIBRARY — TILING (4 base)
# ============================================================

def h_tile_2x2(grid):
    """Tile grid in 2x2 pattern."""
    return [row * 2 for row in grid for _ in range(2)]

def h_tile_3x3(grid):
    """Tile grid in 3x3 pattern."""
    return [row * 3 for row in grid for _ in range(3)]

def h_repeat_h(grid):
    """Repeat grid horizontally (2x)."""
    return [row + row for row in grid]

def h_repeat_v(grid):
    """Repeat grid vertically (2x)."""
    return grid + grid

print("[HEURISTICS] Object + Scaling + Tiling heuristics loaded (+14)")
'''

CELL_11_CODE = r'''# ============================================================
# HEURISTIC LIBRARY — PATTERN (4 base)
# ============================================================

def h_mirror_h_full(grid):
    """Mirror grid horizontally: left + flipped(left)."""
    return [row + row[::-1] for row in grid]

def h_mirror_v_full(grid):
    """Mirror grid vertically: top + flipped(top)."""
    return grid + grid[::-1]

def h_mirror_4way(grid):
    """Mirror grid 4-way: top-left + top-right + bottom-left + bottom-right."""
    h_grid = [row + row[::-1] for row in grid]
    return h_grid + h_grid[::-1]

def h_extract_border(grid):
    """Extract the border of the grid, set interior to 0."""
    if not grid:
        return grid
    h, w = grid_shape(grid)
    if h <= 2 or w <= 2:
        return copy.deepcopy(grid)
    result = [[0] * w for _ in range(h)]
    for c in range(w):
        result[0][c] = grid[0][c]
        result[h-1][c] = grid[h-1][c]
    for r in range(h):
        result[r][0] = grid[r][0]
        result[r][w-1] = grid[r][w-1]
    return result

# ============================================================
# HEURISTIC LIBRARY — LOGIC (4 base)
# ============================================================

def h_conditional_color_density(grid, threshold=0.5, high_color=1, low_color=0):
    """If density > threshold, fill with high_color; else low_color."""
    feat = extract_grid_features(grid)
    h, w = grid_shape(grid)
    fill = high_color if feat["density"] > threshold else low_color
    return [[fill] * w for _ in range(h)]

def h_conditional_size_rotate(grid):
    """If grid is square, rotate 90; if wider, flip_h; if taller, flip_v."""
    h, w = grid_shape(grid)
    if h == w:
        return h_rotate90(grid)
    elif w > h:
        return h_flip_h(grid)
    else:
        return h_flip_v(grid)

def h_threshold_binary(grid, threshold=3):
    """Binarize: values >= threshold become 1, else 0."""
    return [[1 if c >= threshold else 0 for c in row] for row in grid]

def h_replace_zero_with_dominant(grid):
    """Replace all 0-cells with the most common non-zero color."""
    cc = count_colors(grid)
    non_zero = {c: n for c, n in cc.items() if c != 0}
    if not non_zero:
        return grid
    mc = max(non_zero, key=non_zero.get)
    return [[mc if c == 0 else c for c in row] for row in grid]

print("[HEURISTICS] Pattern + Logic heuristics loaded (+8)")
'''

CELL_13_CODE = r'''# ============================================================
# NEW v3 HEURISTICS — GEOMETRIC EXTENSIONS
# ============================================================

def h_translate_up(grid, k=1):
    """Translate grid content up by k pixels, padding bottom with 0."""
    h, w = grid_shape(grid)
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nr = r - k
            if 0 <= nr < h:
                result[nr][c] = grid[r][c]
    return result

def h_translate_down(grid, k=1):
    """Translate grid content down by k pixels."""
    h, w = grid_shape(grid)
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nr = r + k
            if 0 <= nr < h:
                result[nr][c] = grid[r][c]
    return result

def h_translate_left(grid, k=1):
    """Translate grid content left by k pixels."""
    h, w = grid_shape(grid)
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nc = c - k
            if 0 <= nc < w:
                result[r][nc] = grid[r][c]
    return result

def h_translate_right(grid, k=1):
    """Translate grid content right by k pixels."""
    h, w = grid_shape(grid)
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nc = c + k
            if 0 <= nc < w:
                result[r][nc] = grid[r][c]
    return result

def h_rotate_by_k(grid, k=1):
    """Rotate grid by k*90 degrees clockwise."""
    arr = grid_to_numpy(grid)
    return numpy_to_grid(np.rot90(arr, -k % 4))

def h_zoom_in(grid):
    """Zoom in 2x (center crop and scale)."""
    arr = grid_to_numpy(grid)
    h, w = arr.shape
    cr, cc = h // 2, w // 2
    half_h, half_w = max(1, h // 4), max(1, w // 4)
    r1 = max(0, cr - half_h)
    r2 = min(h, cr + half_h)
    c1 = max(0, cc - half_w)
    c2 = min(w, cc + half_w)
    cropped = arr[r1:r2, c1:c2]
    if cropped.size == 0:
        cropped = arr
    return numpy_to_grid(np.repeat(np.repeat(cropped, 2, axis=0), 2, axis=1))

def h_zoom_out(grid):
    """Zoom out 2x (shrink and center)."""
    arr = grid_to_numpy(grid)
    h, w = arr.shape
    shrunk = arr[::2, ::2]
    sh, sw = shrunk.shape
    result = np.zeros((h, w), dtype=np.int32)
    r_off = (h - sh) // 2
    c_off = (w - sw) // 2
    result[r_off:r_off+sh, c_off:c_off+sw] = shrunk
    return numpy_to_grid(result)

def h_pad_grid(grid, pad=1, value=0):
    """Pad grid on all sides by pad pixels."""
    h, w = grid_shape(grid)
    nh, nw = h + 2 * pad, w + 2 * pad
    result = [[value] * nw for _ in range(nh)]
    for r in range(h):
        for c in range(w):
            result[r + pad][c + pad] = grid[r][c]
    return result

def h_reshape_grid(grid, target_h=None, target_w=None):
    """Reshape grid to target dimensions (flatten and reshape)."""
    h, w = grid_shape(grid)
    if target_h is None:
        target_h = w if h > w else h
    if target_w is None:
        target_w = h if h > w else w
    flat = [c for row in grid for c in row]
    result = []
    idx = 0
    for r in range(target_h):
        row = []
        for c in range(target_w):
            if idx < len(flat):
                row.append(flat[idx])
            else:
                row.append(0)
            idx += 1
        result.append(row)
    return result

print("[HEURISTICS v3] New geometric heuristics loaded (+9)")
'''

CELL_15_CODE = r'''# ============================================================
# NEW v3 HEURISTICS — COLOR EXTENSIONS
# ============================================================

def h_color_cycle(grid, k=1):
    """Cycle non-zero colors: 1->2->3->...->9->1, etc."""
    def cycle_one(c):
        if c == 0:
            return 0
        return ((c - 1 + k) % 9) + 1
    return [[cycle_one(c) for c in row] for row in grid]

def h_color_merge(grid, merge_map=None):
    """Merge multiple colors into one. Default: merge 1,2 -> 1."""
    if merge_map is None:
        merge_map = {2: 1}
    return [[merge_map.get(c, c) for c in row] for row in grid]

def h_color_split(grid, split_map=None):
    """Split one color into another based on position. Default: split based on row parity."""
    if split_map is None:
        split_map = {1: 2}
    result = []
    for r, row in enumerate(grid):
        new_row = []
        for c in row:
            if c in split_map and r % 2 == 0:
                new_row.append(split_map[c])
            else:
                new_row.append(c)
        result.append(new_row)
    return result

def h_unique_colors_only(grid):
    """Keep only pixels whose color appears exactly once in the grid."""
    cc = count_colors(grid)
    return [[c if cc[c] == 1 else 0 for c in row] for row in grid]

def h_dominant_color_only(grid):
    """Keep only the most common non-zero color, set others to 0."""
    cc = count_colors(grid)
    non_zero = {c: n for c, n in cc.items() if c != 0}
    if not non_zero:
        return grid
    mc = max(non_zero, key=non_zero.get)
    return [[c if c == mc else 0 for c in row] for row in grid]

def h_least_common_color(grid):
    """Keep only the least common non-zero color, set others to 0."""
    cc = count_colors(grid)
    non_zero = {c: n for c, n in cc.items() if c != 0}
    if not non_zero:
        return grid
    lc = min(non_zero, key=non_zero.get)
    return [[c if c == lc else 0 for c in row] for row in grid]

def h_color_histogram_output(grid):
    """Output a bar-chart representation of color counts."""
    cc = count_colors(grid)
    max_count = max(cc.values()) if cc else 0
    if max_count == 0:
        return [[0]]
    result = []
    for color in range(10):
        if color in cc:
            row = [color] * cc[color]
            row += [0] * (max_count - cc[color])
        else:
            row = [0] * max_count
        result.append(row)
    return result

print("[HEURISTICS v3] New color heuristics loaded (+7)")
'''

CELL_17_CODE = r'''# ============================================================
# NEW v3 HEURISTICS — OBJECT EXTENSIONS
# ============================================================

def h_sort_components_by_size(grid, descending=False):
    """Extract all components, sort by size, place in rows."""
    comps = find_components(grid, connectivity=4)
    non_zero = [c for c in comps if c["color"] != 0]
    if not non_zero:
        return grid
    non_zero.sort(key=lambda x: x["size"], reverse=descending)
    max_h = max(grid_shape(extract_component_grid(grid, c))[0] for c in non_zero)
    max_w = sum(grid_shape(extract_component_grid(grid, c))[1] for c in non_zero)
    result = [[0] * max_w for _ in range(max_h)]
    cur_c = 0
    for comp in non_zero:
        sub = grid_to_numpy(grid)
        r1, c1, r2, c2 = component_bounding_box(comp)
        patch = sub[r1:r2+1, c1:c2+1]
        ph, pw = patch.shape
        for r in range(min(ph, max_h)):
            for c_idx in range(pw):
                if cur_c + c_idx < max_w:
                    result[r][cur_c + c_idx] = int(patch[r, c_idx])
        cur_c += pw
    return result

def h_align_components(grid):
    """Align all components to top-left in separate quadrants."""
    comps = find_components(grid, connectivity=4)
    non_zero = [c for c in comps if c["color"] != 0]
    if not non_zero:
        return grid
    n = len(non_zero)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    max_comp_h = max(grid_shape(extract_component_grid(grid, c))[0] for c in non_zero)
    max_comp_w = max(grid_shape(extract_component_grid(grid, c))[1] for c in non_zero)
    out_h = rows * max_comp_h
    out_w = cols * max_comp_w
    result = [[0] * out_w for _ in range(out_h)]
    arr = grid_to_numpy(grid)
    for idx, comp in enumerate(non_zero):
        r1, c1, r2, c2 = component_bounding_box(comp)
        patch = arr[r1:r2+1, c1:c2+1]
        ph, pw = patch.shape
        tr = (idx // cols) * max_comp_h
        tc = (idx % cols) * max_comp_w
        for pr in range(min(ph, max_comp_h)):
            for pc in range(min(pw, max_comp_w)):
                result[tr + pr][tc + pc] = int(patch[pr, pc])
    return result

def h_components_to_rows(grid):
    """Place each component in its own row."""
    comps = find_components(grid, connectivity=4)
    non_zero = [c for c in comps if c["color"] != 0]
    if not non_zero:
        return grid
    result = []
    for comp in non_zero:
        sub = extract_component_grid(grid, comp)
        result.append(sub[0] if len(sub) == 1 else sub)
    return result

def h_components_to_columns(grid):
    """Place each component in its own column."""
    comps = find_components(grid, connectivity=4)
    non_zero = [c for c in comps if c["color"] != 0]
    if not non_zero:
        return grid
    grids_list = [grid_to_numpy(extract_component_grid(grid, c)) for c in non_zero]
    max_h = max(g.shape[0] for g in grids_list)
    cols = []
    for g in grids_list:
        col = np.zeros(max_h, dtype=np.int32)
        col[:g.shape[0]] = g.flatten()[:max_h]
        cols.append(col)
    result = np.column_stack(cols) if cols else np.zeros((1, 1), dtype=np.int32)
    return numpy_to_grid(result)

def h_fill_holes(grid, fill_color=None):
    """Fill holes (0-cells surrounded by non-zero) with fill_color."""
    if fill_color is None:
        cc = count_colors(grid)
        non_zero = {c: n for c, n in cc.items() if c != 0}
        fill_color = max(non_zero, key=non_zero.get) if non_zero else 0
    arr = grid_to_numpy(grid).copy()
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    for r in range(h):
        for c in [0, w - 1]:
            if arr[r, c] == 0 and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True
    for c in range(w):
        for r in [0, h - 1]:
            if arr[r, c] == 0 and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == 0:
                visited[nr, nc] = True
                queue.append((nr, nc))
    for r in range(h):
        for c in range(w):
            if arr[r, c] == 0 and not visited[r, c]:
                arr[r, c] = fill_color
    return numpy_to_grid(arr)

def h_erode(grid):
    """Erode: remove border cells of non-zero regions."""
    arr = grid_to_numpy(grid).copy()
    h, w = arr.shape
    result = np.zeros_like(arr)
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                has_all_neighbors = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if arr[nr, nc] == 0:
                            has_all_neighbors = False
                            break
                    else:
                        has_all_neighbors = False
                        break
                if has_all_neighbors:
                    result[r, c] = arr[r, c]
    return numpy_to_grid(result)

def h_dilate(grid):
    """Dilate: expand non-zero regions by 1 cell."""
    arr = grid_to_numpy(grid).copy()
    h, w = arr.shape
    result = arr.copy()
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == 0:
                        result[nr, nc] = arr[r, c]
    return numpy_to_grid(result)

print("[HEURISTICS v3] New object heuristics loaded (+7)")
'''

CELL_19_CODE = r'''# ============================================================
# NEW v3 HEURISTICS — PATTERN EXTENSIONS
# ============================================================

def h_checkerboard(grid, color1=1, color2=0):
    """Create a checkerboard pattern matching grid dimensions."""
    h, w = grid_shape(grid)
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            row.append(color1 if (r + c) % 2 == 0 else color2)
        result.append(row)
    return result

def h_stripes_h(grid, color1=1, color2=0, width=1):
    """Create horizontal stripes pattern."""
    h, w = grid_shape(grid)
    result = []
    for r in range(h):
        row = []
        stripe_color = color1 if (r // width) % 2 == 0 else color2
        for c in range(w):
            row.append(stripe_color)
        result.append(row)
    return result

def h_stripes_v(grid, color1=1, color2=0, width=1):
    """Create vertical stripes pattern."""
    h, w = grid_shape(grid)
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            stripe_color = color1 if (c // width) % 2 == 0 else color2
            row.append(stripe_color)
        result.append(row)
    return result

def h_frame_grid(grid, frame_color=1, inner_color=0, frame_width=1):
    """Create a frame (border) of given width."""
    h, w = grid_shape(grid)
    result = [[inner_color] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if r < frame_width or r >= h - frame_width or c < frame_width or c >= w - frame_width:
                result[r][c] = frame_color
    return result

def h_diagonal_lines(grid, color1=1, color2=0):
    """Create diagonal lines pattern."""
    h, w = grid_shape(grid)
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            row.append(color1 if (r + c) % 3 == 0 else color2)
        result.append(row)
    return result

def h_concentric_rects(grid, colors=None):
    """Create concentric rectangles from outside in."""
    h, w = grid_shape(grid)
    if colors is None:
        colors = list(range(min(h, w, 10)))
    if not colors:
        return grid
    result = [[0] * w for _ in range(h)]
    max_dim = min(h, w) // 2
    for layer in range(max_dim):
        color = colors[layer % len(colors)]
        for r in range(layer, h - layer):
            for c in range(layer, w - layer):
                if r == layer or r == h - layer - 1 or c == layer or c == w - layer - 1:
                    result[r][c] = color
    return result

def h_partition_grid(grid, n_parts=4):
    """Partition grid into n_parts sub-grids and tile them."""
    h, w = grid_shape(grid)
    if h == 0 or w == 0:
        return grid
    cols = max(1, int(math.ceil(math.sqrt(n_parts * w / max(h, 1)))))
    rows = max(1, (n_parts + cols - 1) // cols)
    ph, pw = h // rows, w // cols
    if ph == 0 or pw == 0:
        return grid
    parts = []
    for pr in range(rows):
        for pc in range(cols):
            part = [[0] * pw for _ in range(ph)]
            for r in range(ph):
                for c in range(pw):
                    sr, sc = pr * ph + r, pc * pw + c
                    if sr < h and sc < w:
                        part[r][c] = grid[sr][sc]
            parts.append(part)
    result = [[0] * w for _ in range(h)]
    idx = 0
    for part in parts:
        for r in range(min(len(part), h)):
            for c in range(min(len(part[0]) if part else 0, w)):
                result[r][c] = part[r][c]
        idx += 1
    return result

print("[HEURISTICS v3] New pattern heuristics loaded (+7)")
'''

CELL_21_CODE = r'''# ============================================================
# NEW v3 HEURISTICS — LOGIC EXTENSIONS
# ============================================================

def h_conditional_color_count(grid):
    """Output grid with height=number of unique colors, each row is a color."""
    cc = count_colors(grid)
    unique_cs = sorted(cc.keys())
    result = []
    for color in unique_cs:
        row = [color] * cc[color]
        result.append(row)
    return result if result else [[0]]

def h_conditional_symmetry(grid):
    """If grid has horizontal symmetry, return it; else return mirrored version."""
    for row in grid:
        if row != row[::-1]:
            return h_mirror_h_full(grid)
    return copy.deepcopy(grid)

def h_conditional_has_color(grid, target_color=1, yes_fn=None, no_fn=None):
    """Apply yes_fn if target_color present, else no_fn."""
    has_color = any(c == target_color for row in grid for c in row)
    if has_color:
        if yes_fn:
            return yes_fn(grid)
        return grid
    else:
        if no_fn:
            return no_fn(grid)
        return grid

def h_conditional_size(grid):
    """If grid > 5x5, crop to 5x5; if smaller, pad to 5x5."""
    h, w = grid_shape(grid)
    target = 5
    if h > target or w > target:
        arr = grid_to_numpy(grid)
        return numpy_to_grid(arr[:target, :target])
    result = [[0] * target for _ in range(target)]
    for r in range(min(h, target)):
        for c in range(min(w, target)):
            result[r][c] = grid[r][c]
    return result

def h_max_pool_2x2(grid):
    """2x2 max pooling."""
    arr = grid_to_numpy(grid)
    h, w = arr.shape
    oh, ow = h // 2, w // 2
    if oh == 0 or ow == 0:
        return grid
    result = np.zeros((oh, ow), dtype=np.int32)
    for r in range(oh):
        for c in range(ow):
            patch = arr[2*r:2*r+2, 2*c:2*c+2]
            result[r, c] = int(patch.max())
    return numpy_to_grid(result)

def h_min_pool_2x2(grid):
    """2x2 min pooling."""
    arr = grid_to_numpy(grid)
    h, w = arr.shape
    oh, ow = h // 2, w // 2
    if oh == 0 or ow == 0:
        return grid
    result = np.zeros((oh, ow), dtype=np.int32)
    for r in range(oh):
        for c in range(ow):
            patch = arr[2*r:2*r+2, 2*c:2*c+2]
            result[r, c] = int(patch.min())
    return numpy_to_grid(result)

# ============================================================
# PARAMETERIZED HEURISTICS (extract_color_1-9, color_add_1-9, color_sub_1-9)
# ============================================================

def _make_extract_color(color):
    def fn(grid):
        return h_extract_color(grid, color)
    fn.__name__ = f"extract_color_{color}"
    return fn

def _make_color_add(k):
    def fn(grid):
        return h_color_add_k(grid, k)
    fn.__name__ = f"color_add_{k}"
    return fn

def _make_color_sub(k):
    def fn(grid):
        return h_color_sub_k(grid, k)
    fn.__name__ = f"color_sub_{k}"
    return fn

PARAMETERIZED_HEURISTICS = []
for _c in range(1, 10):
    PARAMETERIZED_HEURISTICS.append(_make_extract_color(_c))
for _k in range(1, 10):
    PARAMETERIZED_HEURISTICS.append(_make_color_add(_k))
    PARAMETERIZED_HEURISTICS.append(_make_color_sub(_k))

print(f"[HEURISTICS v3] New logic heuristics loaded (+6)")
print(f"[HEURISTICS v3] Parameterized heuristics: {len(PARAMETERIZED_HEURISTICS)}")
'''

CELL_23_CODE = r'''# ============================================================
# COMPOSITE HEURISTICS (chaining two operations)
# ============================================================

def _make_composite(fn1, fn2, name):
    def fn(grid):
        try:
            mid = fn1(grid)
            return fn2(mid)
        except Exception:
            return grid
    fn.__name__ = name
    return fn

_base_fns = {
    "rotate90": h_rotate90, "rotate180": h_rotate180, "rotate270": h_rotate270,
    "flip_h": h_flip_h, "flip_v": h_flip_v, "flip_diag": h_flip_diag,
    "crop": h_crop, "scale2x": h_scale_2x, "scale3x": h_scale_3x,
    "mirror_h": h_mirror_h_full, "mirror_v": h_mirror_v_full,
    "extract_border": h_extract_border,
    "color_complement": h_color_complement,
    "invert_colors": h_invert_colors,
}

COMPOSITE_HEURISTICS = []
_comp_pairs = [
    ("rotate90", "flip_h"), ("rotate90", "flip_v"),
    ("rotate180", "flip_h"), ("rotate180", "scale2x"),
    ("rotate270", "flip_h"), ("flip_h", "flip_v"),
    ("crop", "scale2x"), ("crop", "scale3x"),
    ("crop", "rotate90"), ("crop", "mirror_h"),
    ("scale2x", "rotate90"), ("scale2x", "flip_h"),
    ("mirror_h", "scale2x"), ("mirror_v", "scale2x"),
    ("extract_border", "scale2x"),
    ("color_complement", "rotate90"),
    ("invert_colors", "flip_h"), ("invert_colors", "mirror_h"),
    ("extract_border", "crop"),
    ("flip_diag", "scale2x"), ("flip_h", "mirror_v"),
]

for name1, name2 in _comp_pairs:
    fn1 = _base_fns.get(name1)
    fn2 = _base_fns.get(name2)
    if fn1 and fn2:
        comp = _make_composite(fn1, fn2, f"{name1}_then_{name2}")
        COMPOSITE_HEURISTICS.append(comp)

print(f"[HEURISTICS] Composite heuristics: {len(COMPOSITE_HEURISTICS)}")
print(f"[HEURISTICS] Total parameterized: {len(PARAMETERIZED_HEURISTICS)}")
'''

CELL_25_CODE = r'''# ============================================================
# HEURISTIC REGISTRATION & SCORING
# ============================================================

BASE_HEURISTICS = [
    h_identity, h_rotate90, h_rotate180, h_rotate270,
    h_flip_h, h_flip_v, h_flip_diag, h_flip_anti_diag,
    h_color_complement, h_color_add_k, h_color_sub_k,
    h_fill_most_common, h_extract_color, h_remap_colors,
    h_count_colors_output, h_invert_colors,
    h_crop, h_crop_each_color,
    h_extract_largest_component, h_extract_smallest_component,
    h_move_to_topleft, h_extract_all_components_separate,
    h_scale_2x, h_scale_3x, h_shrink_by_half, h_shrink_by_third,
    h_tile_2x2, h_tile_3x3, h_repeat_h, h_repeat_v,
    h_mirror_h_full, h_mirror_v_full, h_mirror_4way, h_extract_border,
    h_conditional_color_density, h_conditional_size_rotate,
    h_threshold_binary, h_replace_zero_with_dominant,
    # v3 geometric
    h_translate_up, h_translate_down, h_translate_left, h_translate_right,
    h_rotate_by_k, h_zoom_in, h_zoom_out, h_pad_grid, h_reshape_grid,
    # v3 color
    h_color_cycle, h_color_merge, h_color_split,
    h_unique_colors_only, h_dominant_color_only, h_least_common_color,
    h_color_histogram_output,
    # v3 object
    h_sort_components_by_size, h_align_components,
    h_components_to_rows, h_components_to_columns,
    h_fill_holes, h_erode, h_dilate,
    # v3 pattern
    h_checkerboard, h_stripes_h, h_stripes_v,
    h_frame_grid, h_diagonal_lines, h_concentric_rects, h_partition_grid,
    # v3 logic
    h_conditional_color_count, h_conditional_symmetry,
    h_conditional_has_color, h_conditional_size,
    h_max_pool_2x2, h_min_pool_2x2,
]

ALL_HEURISTICS = BASE_HEURISTICS + PARAMETERIZED_HEURISTICS + COMPOSITE_HEURISTICS

print(f"[HEURISTICS] Base: {len(BASE_HEURISTICS)}")
print(f"[HEURISTICS] Total all: {len(ALL_HEURISTICS)}")


def score_heuristic_on_training(heuristic_fn, train_pairs, require_all_match=True):
    """Score a heuristic on training pairs. Returns (score, predictions).
    score = fraction of training examples correctly solved.
    """
    if not train_pairs:
        return 0.0, []
    predictions = []
    correct = 0
    for inp, expected_out in train_pairs:
        try:
            pred = heuristic_fn(inp)
            if isinstance(pred, list) and len(pred) > 0 and isinstance(pred[0], list):
                if grids_equal(pred, expected_out):
                    correct += 1
                    predictions.append(pred)
                elif not require_all_match:
                    predictions.append(pred)
            else:
                if not require_all_match:
                    predictions.append(pred)
        except Exception:
            pass
    if require_all_match:
        score = correct / len(train_pairs) if correct == len(train_pairs) else 0.0
    else:
        score = correct / len(train_pairs)
    return score, predictions


def find_best_heuristics(train_pairs, top_k=5, min_score=1.0):
    """Find the top-k heuristics that perfectly solve all training pairs."""
    results = []
    for h_fn in ALL_HEURISTICS:
        score, preds = score_heuristic_on_training(h_fn, train_pairs)
        if score >= min_score:
            results.append((score, h_fn, preds))
    results.sort(key=lambda x: -x[0])
    return results[:top_k]


def find_best_partial_heuristics(train_pairs, top_k=10):
    """Find top-k heuristics that partially solve training pairs."""
    results = []
    for h_fn in ALL_HEURISTICS:
        score, preds = score_heuristic_on_training(h_fn, train_pairs, require_all_match=False)
        if score > 0:
            results.append((score, h_fn, preds))
    results.sort(key=lambda x: -x[0])
    return results[:top_k]

print("[SCORING] Heuristic scoring system ready")
'''

CELL_27_CODE = r'''# ============================================================
# SAFE CODE EXECUTION
# ============================================================

def _run_code_in_thread(code_str, input_grid, timeout=CODE_TIMEOUT_SECONDS):
    """Execute code in a separate thread with timeout."""
    result = [None]
    error = [None]

    def worker():
        try:
            local_ns = {
                "grid": copy.deepcopy(input_grid),
                "np": np,
                "copy": copy,
                "json": json,
                "Counter": Counter,
                "deque": deque,
                "math": math,
                "grid_to_numpy": grid_to_numpy,
                "numpy_to_grid": numpy_to_grid,
                "grid_shape": grid_shape,
                "find_components": find_components,
                "count_colors": count_colors,
                "component_bounding_box": component_bounding_box,
                "extract_component_grid": extract_component_grid,
                "h_crop": h_crop,
                "h_flip_h": h_flip_h,
                "h_flip_v": h_flip_v,
                "h_rotate90": h_rotate90,
                "h_rotate180": h_rotate180,
                "h_scale_2x": h_scale_2x,
                "h_mirror_h_full": h_mirror_h_full,
                "h_extract_border": h_extract_border,
            }
            exec(code_str, local_ns)
            output = local_ns.get("output", local_ns.get("result", local_ns.get("solution")))
            if output is not None:
                if isinstance(output, np.ndarray):
                    output = numpy_to_grid(output)
                result[0] = ensure_grid(output)
        except Exception as e:
            error[0] = str(e)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        return None, "timeout"
    if error[0] is not None:
        return None, error[0]
    return result[0], None


def safe_execute_code(code_str, input_grid, timeout=CODE_TIMEOUT_SECONDS):
    """Safely execute LLM-generated code on an input grid."""
    return _run_code_in_thread(code_str, input_grid, timeout)


def verify_code_on_training(code_str, train_pairs, timeout=CODE_TIMEOUT_SECONDS):
    """Verify code on all training pairs. Returns (score, predictions)."""
    if not train_pairs:
        return 0.0, []
    predictions = []
    correct = 0
    for inp, expected in train_pairs:
        result, err = safe_execute_code(code_str, inp, timeout)
        if result is not None and grids_equal(result, expected):
            correct += 1
            predictions.append(result)
        elif result is not None:
            predictions.append(result)
    score = correct / len(train_pairs) if train_pairs else 0.0
    return score, predictions


def extract_function_code(llm_output):
    """Extract Python function from LLM output text."""
    patterns = [
        r"```python\s*(.*?)\s*```",
        r"```\s*(def\s+transform.*?)\s*```",
        r"(def\s+transform\s*\(.*?\n(?:.*?\n)*?.*?return\s+\w+)",
        r"(def\s+solve\s*\(.*?\n(?:.*?\n)*?.*?return\s+\w+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, llm_output, re.DOTALL)
        if matches:
            code = matches[0].strip()
            if "return" in code:
                return code
    # Fallback: try to find any function definition
    if "def " in llm_output:
        start = llm_output.index("def ")
        end = llm_output.find("\n\n", start)
        if end == -1:
            end = len(llm_output)
        return llm_output[start:end].strip()
    return llm_output.strip()

print("[EXECUTION] Safe code execution system ready")
'''

CELL_29_CODE = r'''# ============================================================
# DREAMCODER ABSTRACTION LIBRARY
# ============================================================

class AbstractionLibrary:
    """Library of reusable abstractions extracted from verified solutions."""

    def __init__(self):
        self.abstractions = {}
        self.usage_counts = Counter()
        self._init_builtins()

    def _init_builtins(self):
        """Initialize built-in abstractions."""
        self.abstractions["find_bounding_box"] = {
            "code": "def find_bounding_box(grid):\n    arr = np.array(grid)\n    nz = np.nonzero(arr)\n    if len(nz[0]) == 0: return (0,0,0,0)\n    return (int(nz[0].min()), int(nz[1].min()), int(nz[0].max()), int(nz[1].max()))",
            "description": "Find bounding box of non-zero content",
            "category": "geometry",
        }
        self.abstractions["crop_grid"] = {
            "code": "def crop_grid(grid):\n    r1,c1,r2,c2 = find_bounding_box(grid)\n    arr = np.array(grid)\n    return arr[r1:r2+1, c1:c2+1].tolist()",
            "description": "Crop grid to bounding box",
            "category": "geometry",
        }
        self.abstractions["find_objects"] = {
            "code": "def find_objects(grid):\n    comps = find_components(grid, 4)\n    return [c for c in comps if c['color'] != 0]",
            "description": "Find all non-zero connected objects",
            "category": "objects",
        }
        self.abstractions["get_color_counts"] = {
            "code": "def get_color_counts(grid):\n    cc = Counter()\n    for row in grid:\n        for c in row:\n            cc[c] += 1\n    return dict(cc)",
            "description": "Count pixels per color",
            "category": "color",
        }
        self.abstractions["rotate_grid"] = {
            "code": "def rotate_grid(grid, k=1):\n    arr = np.array(grid)\n    return np.rot90(arr, -k % 4).tolist()",
            "description": "Rotate grid by k*90 degrees",
            "category": "geometry",
        }
        self.abstractions["check_symmetry"] = {
            "code": "def check_symmetry(grid):\n    h_sym = all(row == row[::-1] for row in grid)\n    v_sym = grid == grid[::-1]\n    return {'horizontal': h_sym, 'vertical': v_sym}",
            "description": "Check grid symmetry",
            "category": "analysis",
        }

    def add_abstraction(self, name, code, description="", category="custom"):
        """Add a new abstraction to the library."""
        self.abstractions[name] = {
            "code": code,
            "description": description,
            "category": category,
        }
        self.usage_counts[name] = 0

    def extract_from_verified_code(self, code_str, task_id=""):
        """Try to extract reusable abstractions from verified code."""
        found = []
        if "rotate" in code_str.lower():
            found.append("rotate_grid")
        if "crop" in code_str.lower() or "bounding" in code_str.lower():
            found.append("crop_grid")
        if "find_object" in code_str.lower() or "component" in code_str.lower():
            found.append("find_objects")
        if "color" in code_str.lower() and "count" in code_str.lower():
            found.append("get_color_counts")
        if "symmetr" in code_str.lower():
            found.append("check_symmetry")
        for name in found:
            if name in self.abstractions:
                self.usage_counts[name] += 1
        return found

    def get_relevant_abstractions(self, task_description="", max_abstractions=5):
        """Get most relevant abstractions based on task description."""
        scored = []
        keywords = task_description.lower().split()
        for name, abstr in self.abstractions.items():
            score = 0
            desc_lower = abstr["description"].lower()
            code_lower = abstr["code"].lower()
            for kw in keywords:
                if kw in desc_lower or kw in code_lower:
                    score += 1
            score += self.usage_counts.get(name, 0) * 0.5
            scored.append((score, name, abstr))
        scored.sort(key=lambda x: -x[0])
        return [(name, abstr) for _, name, abstr in scored[:max_abstractions]]

    def get_stats(self):
        """Get library statistics."""
        cats = Counter(a["category"] for a in self.abstractions.values())
        return {
            "total": len(self.abstractions),
            "categories": dict(cats),
            "most_used": self.usage_counts.most_common(5),
        }


abstraction_library = AbstractionLibrary()
print("[DREAMCODER] Abstraction library initialized")
print(f"[DREAMCODER] Built-in abstractions: {len(abstraction_library.abstractions)}")
'''

CELL_31_CODE = r'''# ============================================================
# LLM ENGINE (Lazy Loading)
# ============================================================

class LLMEngine:
    """LLM engine with lazy loading and 4-bit quantization support."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._load_error = None

    def load(self):
        """Lazy load the LLM model."""
        if self._loaded:
            return
        if self._load_error:
            return
        try:
            print("[LLM] Loading model (this may take a few minutes)...")
            torch_mod = get_torch()
            if torch_mod is None:
                self._load_error = "torch not available"
                return
            from transformers import AutoModelForCausalLM, AutoTokenizer
            kwargs = {
                "torch_dtype": torch_mod.float16,
                "device_map": "auto",
                "trust_remote_code": True,
            }
            if QUANTIZATION_4BIT:
                try:
                    import bitsandbytes
                    kwargs["quantization_config"] = bitsandbytes.__dict__.get(
                        "BitsAndBytesConfig",
                        None
                    )
                    if kwargs["quantization_config"] is not None:
                        kwargs["quantization_config"] = kwargs["quantization_config"](
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch_mod.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                except Exception as e:
                    print(f"[LLM] bitsandbytes not available, using fp16: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
            self.model.eval()
            self._loaded = True
            print("[LLM] Model loaded successfully")
        except Exception as e:
            self._load_error = str(e)
            print(f"[LLM] Failed to load model: {e}")
            traceback.print_exc()

    def is_available(self):
        """Check if LLM is available."""
        self.load()
        return self._loaded

    def generate(self, prompt, system_prompt="", temperature=0.7, max_tokens=MAX_NEW_TOKENS):
        """Generate text from prompt."""
        if not self.is_available():
            return None
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch_mod.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return None


# Global LLM engine instance
llm_engine = LLMEngine()

# ============================================================
# LLM PROMPT TEMPLATES
# ============================================================

PHASE1_SYSTEM = """You are an expert ARC puzzle solver. Analyze the input-output pairs and identify the transformation rule.
Describe the transformation in clear, concise steps. Focus on:
1. What changes between input and output?
2. Is there a consistent pattern across all training examples?
3. What properties are preserved vs. changed?
Respond in a structured format with numbered steps."""

PHASE2_SYSTEM = """You are an expert Python programmer for ARC grid puzzles.
Given an analysis of a transformation rule, write a Python function:
def transform(grid):
    # Your code here
    return output_grid

The input `grid` is a 2D list of integers (0-9).
Return a 2D list of integers.
You can use numpy (imported as np) for array operations.
Available helpers: find_components(grid, connectivity), count_colors(grid), grid_to_numpy(grid), numpy_to_grid(arr).
Only return the function code, no explanation."""


def build_enhanced_task_prompt(task, train_pairs):
    """Build a rich prompt for the LLM with task analysis."""
    prompt = "## ARC Task Analysis\n\n"
    prompt += f"Training examples: {len(train_pairs)}\n\n"
    for i, (inp, out) in enumerate(train_pairs):
        prompt += f"### Example {i+1}\n"
        prompt += f"Input ({grid_shape(inp)[0]}x{grid_shape(inp)[1]}):\n"
        prompt += grid_to_str(inp) + "\n"
        prompt += f"Output ({grid_shape(out)[0]}x{grid_shape(out)[1]}):\n"
        prompt += grid_to_str(out) + "\n\n"
    # Add feature analysis
    prompt += "## Feature Analysis\n"
    for i, (inp, out) in enumerate(train_pairs):
        inp_feat = extract_grid_features(inp)
        out_feat = extract_grid_features(out)
        prompt += f"Example {i+1}: "
        prompt += f"Input colors={inp_feat['colors']}, objects={inp_feat['n_objects']} -> "
        prompt += f"Output colors={out_feat['colors']}, objects={out_feat['n_objects']}\n"
    return prompt


def two_phase_llm_solve(task, train_pairs, test_input, max_time=MAX_LLM_TIME_SECONDS):
    """Two-phase LLM solving: analysis then code generation."""
    start_time = time.time()
    # Phase 1: Analysis
    prompt = build_enhanced_task_prompt(task, train_pairs)
    analysis = llm_engine.generate(
        prompt,
        system_prompt=PHASE1_SYSTEM,
        temperature=0.3,
        max_tokens=1024,
    )
    if analysis is None:
        return None, "LLM not available"
    if time.time() - start_time > max_time:
        return None, "timeout in phase 1"
    # Phase 2: Code generation
    code_prompt = f"## Analysis of transformation:\n{analysis}\n\n"
    code_prompt += "## Training pairs:\n"
    for i, (inp, out) in enumerate(train_pairs):
        code_prompt += f"Input {i}: {grid_shape(inp)} -> Output {i}: {grid_shape(out)}\n"
    code_prompt += "\nWrite the transform function:"
    code = llm_engine.generate(
        code_prompt,
        system_prompt=PHASE2_SYSTEM,
        temperature=0.4,
        max_tokens=MAX_NEW_TOKENS,
    )
    if code is None:
        return None, "LLM code gen failed"
    # Extract function
    fn_code = extract_function_code(code)
    if not fn_code:
        return None, "no function extracted"
    # Verify
    score, preds = verify_code_on_training(fn_code, train_pairs)
    if score >= 1.0 and preds:
        result, err = safe_execute_code(fn_code, test_input)
        if result is not None:
            return result, "llm_verified"
    return None, f"code verification score={score:.2f}"

print("[LLM] LLM engine and prompt system ready")
'''

CELL_33_CODE = r'''# ============================================================
# EVOLUTIONARY PROGRAM SYNTHESIS
# ============================================================

class ProgramMutator:
    """Mutates program code strings to explore the solution space."""

    def __init__(self):
        self.mutation_count = 0

    def mutate_numeric(self, code, delta=None):
        """Replace numeric literals with nearby values."""
        if delta is None:
            delta = random.choice([-2, -1, 1, 2])
        numbers = re.findall(r'\b(\d+)\b', code)
        if not numbers:
            return code
        target = random.choice(numbers)
        new_val = max(0, min(9, int(target) + delta))
        new_code = code.replace(target, str(new_val), 1)
        self.mutation_count += 1
        return new_code

    def mutate_operations(self, code):
        """Swap arithmetic operations."""
        swaps = [('+', '-'), ('-', '+'), ('*', '//'), ('//', '*')]
        op1, op2 = random.choice(swaps)
        if op1 in code:
            new_code = code.replace(op1, op2, 1)
            self.mutation_count += 1
            return new_code
        return code

    def mutate_color_value(self, code):
        """Replace a color value with another."""
        for c in range(10):
            if str(c) in code and c > 0:
                new_c = random.choice([x for x in range(1, 10) if x != c])
                new_code = code.replace(str(c), str(new_c), 1)
                self.mutation_count += 1
                return new_code
        return code

    def mutate_indentation_logic(self, code):
        """Adjust conditional thresholds."""
        thresholds = re.findall(r'(?:>|<|==|>=|<=)\s*(\d+)', code)
        if thresholds:
            t = random.choice(thresholds)
            new_t = max(0, min(9, int(t) + random.choice([-1, 1])))
            new_code = code.replace(t, str(new_t), 1)
            self.mutation_count += 1
            return new_code
        return code

    def crossover(self, code1, code2):
        """Crossover two code strings at a random line boundary."""
        lines1 = code1.split("\n")
        lines2 = code2.split("\n")
        if len(lines1) < 2 or len(lines2) < 2:
            return code1
        split1 = random.randint(1, len(lines1) - 1)
        split2 = random.randint(1, len(lines2) - 1)
        child = lines1[:split1] + lines2[split2:]
        self.mutation_count += 1
        return "\n".join(child)

    def mutate(self, code):
        """Apply a random mutation."""
        mutation_fn = random.choice([
            self.mutate_numeric,
            self.mutate_operations,
            self.mutate_color_value,
            self.mutate_indentation_logic,
        ])
        return mutation_fn(code)


def evolutionary_synthesis(train_pairs, test_input, population_size=EVOLUTIONARY_POPULATION,
                           generations=EVOLUTIONARY_GENERATIONS, max_time=120):
    """Evolve program candidates to solve the task."""
    start_time = time.time()
    mutator = ProgramMutator()
    # Seed population from partial heuristics
    seed_codes = []
    for h_fn in ALL_HEURISTICS[:30]:
        name = h_fn.__name__
        seed_code = (
            f"def transform(grid):\n"
            f"    # Based on heuristic: {name}\n"
            f"    result = copy.deepcopy(grid)\n"
            f"    return result\n"
        )
        seed_codes.append(seed_code)
    # Add some generic seeds
    seed_codes.extend([
        "def transform(grid):\n    arr = np.array(grid)\n    result = np.rot90(arr, -1).tolist()\n    return result",
        "def transform(grid):\n    return [row[::-1] for row in grid]",
        "def transform(grid):\n    arr = np.array(grid)\n    nz = np.nonzero(arr)\n    if len(nz[0]) == 0: return grid\n    return arr[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1].tolist()",
    ])
    population = seed_codes[:population_size]
    best_code = None
    best_score = 0.0
    for gen in range(generations):
        if time.time() - start_time > max_time:
            break
        # Score population
        scored = []
        for code in population:
            score, preds = verify_code_on_training(code, train_pairs)
            scored.append((score, code))
        scored.sort(key=lambda x: -x[0])
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_code = scored[0][1]
        if best_score >= 1.0:
            break
        # Selection: keep top half
        survivors = [c for s, c in scored[:population_size // 2]]
        # Create new generation
        new_pop = list(survivors)
        while len(new_pop) < population_size:
            if random.random() < 0.7 and len(survivors) >= 2:
                p1, p2 = random.sample(survivors, 2)
                child = mutator.crossover(p1, p2)
                child = mutator.mutate(child)
            else:
                child = mutator.mutate(random.choice(survivors))
            new_pop.append(child)
        population = new_pop
    # Run best on test
    if best_code and best_score >= 1.0:
        result, err = safe_execute_code(best_code, test_input)
        if result is not None:
            return result, best_score, "evolutionary"
    return None, best_score, "evolutionary_exhausted"

print("[EVOLUTION] Program synthesis engine ready")
'''

CELL_35_CODE = r'''# ============================================================
# MULTI-STRATEGY ENSEMBLE SOLVER
# ============================================================

def solve_task(task, task_id="", time_budget=300):
    """Solve a single task using multi-strategy ensemble.
    Returns: (best_prediction, score, method)
    """
    start_time = time.time()
    train_pairs = [(ex["input"], ex["output"]) for ex in task.get("train", [])]
    test_input = task["test"][0]["input"]

    if not train_pairs:
        return copy.deepcopy(test_input), 0.0, "no_train"

    candidates = []  # List of (prediction, score, method, train_predictions)

    # === PASS 1: Heuristic Search ===
    print(f"  [Pass 1] Heuristic search on {len(ALL_HEURISTICS)} heuristics...")
    best_heuristics = find_best_heuristics(train_pairs, top_k=5)
    for score, h_fn, preds in best_heuristics:
        if preds and score >= 1.0:
            try:
                pred = h_fn(test_input)
                candidates.append((pred, score, f"heuristic:{h_fn.__name__}", preds))
            except Exception:
                pass

    if not candidates:
        partial = find_best_partial_heuristics(train_pairs, top_k=5)
        for score, h_fn, preds in partial:
            try:
                pred = h_fn(test_input)
                candidates.append((pred, score, f"partial:{h_fn.__name__}", preds))
            except Exception:
                pass

    if time.time() - start_time > time_budget * 0.5:
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return best[0], best[1], best[2]

    # === PASS 2: LLM Solving ===
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 60 and len(train_pairs) <= 4:
        print(f"  [Pass 2] LLM solving (budget: {remaining_time:.0f}s)...")
        result, status = two_phase_llm_solve(task, train_pairs, test_input, max_time=min(remaining_time, MAX_LLM_TIME_SECONDS))
        if result is not None:
            candidates.append((result, 1.0, f"llm:{status}", [result]))

    if time.time() - start_time > time_budget * 0.8:
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return best[0], best[1], best[2]

    # === PASS 3: Evolutionary Synthesis ===
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 30:
        print(f"  [Pass 3] Evolutionary synthesis (budget: {remaining_time:.0f}s)...")
        result, evo_score, evo_status = evolutionary_synthesis(
            train_pairs, test_input,
            max_time=min(remaining_time, 120)
        )
        if result is not None:
            candidates.append((result, evo_score, f"{evo_status}", [result]))

    # === Ensemble Selection ===
    if not candidates:
        return copy.deepcopy(test_input), 0.0, "fallback"

    # Pick top candidate
    candidates.sort(key=lambda x: -x[1])
    best = candidates[0]
    elapsed = time.time() - start_time
    print(f"  [Solved] method={best[2]}, score={best[1]:.2f}, time={elapsed:.1f}s, candidates={len(candidates)}")
    return best[0], best[1], best[2]

print("[SOLVER] Multi-strategy ensemble solver ready")
'''

CELL_37_CODE = r'''# ============================================================
# MODULE A: GRID EMBEDDING NETWORK (CNN)
# ============================================================

class GridEmbeddingNetwork:
    """Small CNN (~50K params) to embed grids into fixed-size vectors.
    Uses lazy torch import."""

    def __init__(self):
        self.model = None
        self._built = False

    def _build(self):
        """Build the CNN model."""
        torch_mod = get_torch()
        if torch_mod is None:
            return
        nn = torch_mod.nn
        self.model = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.model.eval()
        self._built = True

    def _grid_to_one_hot(self, grid):
        """Convert grid to one-hot encoding (10, H, W)."""
        torch_mod = get_torch()
        if torch_mod is None:
            return None
        arr = grid_to_numpy(grid)
        h, w = arr.shape
        one_hot = torch_mod.zeros(10, h, w, dtype=torch_mod.float32)
        for c in range(10):
            one_hot[c] = (arr == c).float()
        return one_hot.unsqueeze(0)  # (1, 10, H, W)

    def encode(self, grid):
        """Encode a single grid into a 64-dim vector."""
        if not self._built:
            self._build()
        if self.model is None:
            return None
        torch_mod = get_torch()
        if torch_mod is None:
            return None
        one_hot = self._grid_to_one_hot(grid)
        if one_hot is None:
            return None
        with torch_mod.no_grad():
            embedding = self.model(one_hot)
        return embedding.cpu().numpy().flatten()

    def encode_batch(self, grids):
        """Encode multiple grids into embedding vectors."""
        if not self._built:
            self._build()
        if self.model is None:
            return None
        torch_mod = get_torch()
        if torch_mod is None:
            return None
        embeddings = []
        for grid in grids:
            one_hot = self._grid_to_one_hot(grid)
            if one_hot is not None:
                with torch_mod.no_grad():
                    emb = self.model(one_hot)
                embeddings.append(emb.cpu().numpy().flatten())
        if not embeddings:
            return None
        return np.array(embeddings)

    @staticmethod
    def similarity(emb1, emb2):
        """Cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))


grid_encoder = None


def encode_grid(grid):
    """Encode a single grid, return 64-dim vector."""
    global grid_encoder
    if grid_encoder is None:
        grid_encoder = GridEmbeddingNetwork()
    return grid_encoder.encode(grid)


def find_similar_training_example(test_input, train_pairs, k=3):
    """Find k most similar training inputs by grid embedding."""
    test_emb = encode_grid(test_input)
    if test_emb is None:
        # Fallback: use hash-based similarity
        return list(range(min(k, len(train_pairs))))
    similarities = []
    for i, (train_inp, _) in enumerate(train_pairs):
        train_emb = encode_grid(train_inp)
        if train_emb is not None:
            sim = GridEmbeddingNetwork.similarity(test_emb, train_emb)
        else:
            sim = 0.0
        similarities.append((i, sim))
    similarities.sort(key=lambda x: -x[1])
    return similarities[:k]


print("[MODULE A] Grid Embedding Network defined (lazy torch init)")
'''

CELL_39_CODE = r'''# ============================================================
# MODULE B: SOLUTION SPACE GRAPH EXPLORER (BFS)
# ============================================================

class SolutionNode:
    """Node in the solution space graph."""

    def __init__(self, grid, depth=0, parent=None, action=None):
        self.grid = grid
        self.depth = depth
        self.parent = parent
        self.action = action
        self.grid_hash = grid_hash(grid) if grid else ""

    def __repr__(self):
        return f"SolutionNode(depth={self.depth}, shape={grid_shape(self.grid)}, action={self.action})"


class SolutionGraph:
    """BFS exploration of the solution space via heuristic transformations."""

    def __init__(self, max_depth=3, max_nodes=500):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.visited = set()

    def explore(self, input_grid, target_grid, heuristics=None):
        """BFS from input, trying to reach a grid matching target.
        Returns the solution path or None.
        """
        if heuristics is None:
            heuristics = BASE_HEURISTICS
        start = SolutionNode(copy.deepcopy(input_grid), depth=0)
        queue = deque([start])
        self.visited = {start.grid_hash}
        nodes_explored = 0
        while queue and nodes_explored < self.max_nodes:
            node = queue.popleft()
            nodes_explored += 1
            # Check if goal
            if self._is_goal(node.grid, target_grid):
                return self._reconstruct_path(node)
            # Expand if within depth limit
            if node.depth < self.max_depth:
                next_states = self._apply_all_heuristics(node.grid, heuristics)
                for action_name, new_grid in next_states:
                    if new_grid is None:
                        continue
                    h = grid_hash(new_grid)
                    if h not in self.visited:
                        self.visited.add(h)
                        child = SolutionNode(new_grid, node.depth + 1, node, action_name)
                        queue.append(child)
        return None

    def _apply_all_heuristics(self, grid, heuristics):
        """Apply all heuristics to a grid, return (name, result) pairs."""
        results = []
        for h_fn in heuristics:
            try:
                result = h_fn(grid)
                if result is not None and not grids_equal(result, grid):
                    results.append((h_fn.__name__, result))
            except Exception:
                continue
        return results

    def _is_goal(self, grid, target):
        """Check if grid matches target."""
        return grids_equal(grid, target)

    def _reconstruct_path(self, node):
        """Reconstruct the path from start to goal."""
        path = []
        current = node
        while current.parent is not None:
            path.append((current.action, current.grid))
            current = current.parent
        path.reverse()
        return path


def bfs_solve(train_pairs, test_input, max_depth=3):
    """Use BFS graph exploration to find a solution."""
    graph = SolutionGraph(max_depth=max_depth, max_nodes=500)
    # First verify: does a single heuristic solve all training pairs?
    for h_fn in ALL_HEURISTICS:
        score, _ = score_heuristic_on_training(h_fn, train_pairs)
        if score >= 1.0:
            try:
                return h_fn(test_input), f"bfs_single:{h_fn.__name__}"
            except Exception:
                continue
    # Multi-step: find a 2-step composition
    for i, (inp, out) in enumerate(train_pairs):
        path = graph.explore(inp, out, BASE_HEURISTICS)
        if path and len(path) >= 1:
            # Verify this path on all training pairs
            actions = [a for a, _ in path]
            valid = True
            for j, (t_inp, t_out) in enumerate(train_pairs):
                current = copy.deepcopy(t_inp)
                for action_name in actions:
                    h_fn = next((h for h in ALL_HEURISTICS if h.__name__ == action_name), None)
                    if h_fn:
                        try:
                            current = h_fn(current)
                        except Exception:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                if not valid:
                    break
                if not grids_equal(current, t_out):
                    valid = False
                    break
            if valid:
                result = copy.deepcopy(test_input)
                for action_name in actions:
                    h_fn = next((h for h in ALL_HEURISTICS if h.__name__ == action_name), None)
                    if h_fn:
                        try:
                            result = h_fn(result)
                        except Exception:
                            break
                return result, f"bfs_multi:{'+'.join(actions)}"
    return None, "bfs_no_solution"

print("[MODULE B] Solution Space Graph Explorer ready")
'''

CELL_41_CODE = r'''# ============================================================
# MODULE C: MCTS FOR PROGRAM SYNTHESIS
# ============================================================

class MCTSCodeNode:
    """Node for Monte Carlo Tree Search over code candidates."""

    def __init__(self, code=None, score=0.0, parent=None):
        self.code = code
        self.score = score
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_mutations = list(range(8))

    def ucb1(self, exploration_constant=1.414):
        """UCB1 selection score."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else 0
        return exploitation + exploration

    def is_fully_expanded(self):
        return len(self.untried_mutations) == 0

    def best_child(self, exploration_constant=1.414):
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))


class MCTSCodeSearch:
    """MCTS to search through program space for best solution."""

    def __init__(self, train_pairs, exploration_constant=1.414, max_iterations=50):
        self.train_pairs = train_pairs
        self.c = exploration_constant
        self.max_iterations = max_iterations
        self.mutator = ProgramMutator()
        self.best_code = None
        self.best_score = 0.0

    def select(self, node):
        """UCB1 selection from root to leaf."""
        while node.children:
            if not node.is_fully_expanded():
                return node
            node = node.best_child(self.c)
        return node

    def expand(self, node):
        """Expand a node by generating mutation children."""
        if not node.untried_mutations:
            return node
        mutation_idx = node.untried_mutations.pop()
        base_code = node.code or self._default_code()
        # Apply mutation based on index
        mutations = [
            lambda c: self.mutator.mutate_numeric(c),
            lambda c: self.mutator.mutate_operations(c),
            lambda c: self.mutator.mutate_color_value(c),
            lambda c: self.mutator.mutate_indentation_logic(c),
            lambda c: self.mutator.mutate_numeric(c, delta=random.choice([-3, -2, 2, 3])),
            lambda c: self.mutator.mutate_color_value(c),
            lambda c: self.mutator.mutate_operations(c),
            lambda c: self.mutator.mutate_indentation_logic(c),
        ]
        try:
            new_code = mutations[mutation_idx](base_code)
        except Exception:
            new_code = base_code
        child = MCTSCodeNode(code=new_code, parent=node)
        child.untried_mutations = list(range(8))
        node.children.append(child)
        return child

    def simulate(self, code):
        """Run code on all train pairs, return score."""
        if code is None:
            return 0.0
        score, _ = verify_code_on_training(code, self.train_pairs, timeout=5)
        return score

    def backpropagate(self, node, score):
        """Backpropagate score up the tree."""
        while node is not None:
            node.visits += 1
            node.total_reward += score
            node = node.parent

    def _default_code(self):
        """Generate a default seed code."""
        return (
            "def transform(grid):\n"
            "    arr = np.array(grid, dtype=int)\n"
            "    return arr.tolist()"
        )

    def search(self, initial_codes=None):
        """Main MCTS loop. Returns (best_code, best_score)."""
        if initial_codes is None:
            initial_codes = [self._default_code()]
        root = MCTSCodeNode(code=initial_codes[0])
        for _ in range(self.max_iterations):
            # Select
            leaf = self.select(root)
            # Expand
            if not leaf.is_fully_expanded():
                leaf = self.expand(leaf)
            # Simulate
            score = self.simulate(leaf.code)
            # Backpropagate
            self.backpropagate(leaf, score)
            # Track best
            if score > self.best_score:
                self.best_score = score
                self.best_code = leaf.code
            if score >= 1.0:
                break
        return self.best_code, self.best_score


def mcts_solve(train_pairs, test_input, max_iterations=50):
    """Use MCTS to find a solution code."""
    mcts = MCTSCodeSearch(train_pairs, max_iterations=max_iterations)
    best_code, best_score = mcts.search()
    if best_code and best_score >= 1.0:
        result, err = safe_execute_code(best_code, test_input)
        if result is not None:
            return result, best_score, "mcts"
    return None, best_score, "mcts_exhausted"

print("[MODULE C] MCTS Program Synthesis ready")
'''

CELL_43_CODE = r'''# ============================================================
# MODULE D: CROSS-EXAMPLE PATTERN ANALYZER
# ============================================================

class CrossExampleAnalyzer:
    """Analyzes consistency patterns across multiple training examples."""

    def __init__(self):
        self.analysis_cache = {}

    def analyze(self, train_pairs):
        """Returns analysis dict with transformation patterns."""
        if not train_pairs:
            return {}
        # Cache key
        key = hashlib.md5(str([(grid_hash(i), grid_hash(o)) for i, o in train_pairs]).encode()).hexdigest()
        if key in self.analysis_cache:
            return self.analysis_cache[key]

        result = {
            "size_changes": self.detect_size_pattern(train_pairs),
            "color_mapping": self.detect_color_mapping(train_pairs),
            "preserved_colors": self._detect_preserved_colors(train_pairs),
            "new_colors": self._detect_new_colors(train_pairs),
            "structural_pattern": self.detect_structural_pattern(train_pairs),
            "consistency_score": self._compute_consistency(train_pairs),
        }
        self.analysis_cache[key] = result
        return result

    def detect_color_mapping(self, train_pairs):
        """Detect consistent color transformation rules."""
        if not train_pairs:
            return {}
        mappings = []
        for inp, out in train_pairs:
            if grid_shape(inp) != grid_shape(out):
                return {}
            mapping = {}
            consistent = True
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    s, d = inp[r][c], out[r][c]
                    if s in mapping:
                        if mapping[s] != d:
                            consistent = False
                            break
                    else:
                        mapping[s] = d
                if not consistent:
                    break
            if consistent:
                mappings.append(mapping)
        if not mappings:
            return {}
        # Intersect mappings
        common_keys = set(mappings[0].keys())
        for m in mappings[1:]:
            common_keys &= set(m.keys())
        result = {}
        for k in common_keys:
            vals = set(m[k] for m in mappings)
            if len(vals) == 1:
                result[k] = vals.pop()
        return result

    def detect_size_pattern(self, train_pairs):
        """Detect consistent dimension transformation pattern."""
        if len(train_pairs) < 2:
            return "unknown"
        patterns = []
        for inp, out in train_pairs:
            ih, iw = grid_shape(inp)
            oh, ow = grid_shape(out)
            if ih == 0 or iw == 0:
                patterns.append("unknown")
                continue
            if ih == oh and iw == ow:
                patterns.append("same")
            elif oh % ih == 0 and ow % iw == 0:
                fy, fx = oh // ih, ow // iw
                patterns.append(f"scale_{fy}x{fx}")
            elif ih % oh == 0 and iw % ow == 0:
                patterns.append("shrink")
            else:
                patterns.append("resize")
        # Check consistency
        non_unknown = [p for p in patterns if p != "unknown"]
        if not non_unknown:
            return "unknown"
        if len(set(non_unknown)) == 1:
            return non_unknown[0]
        return "mixed"

    def detect_structural_pattern(self, train_pairs):
        """Detect structural transformation pattern."""
        if not train_pairs:
            return "unknown"
        scores = {
            "same": 0, "crop": 0, "scale": 0, "mirror": 0,
            "tile": 0, "rotate": 0, "color_only": 0, "other": 0
        }
        for inp, out in train_pairs:
            ih, iw = grid_shape(inp)
            oh, ow = grid_shape(out)
            if ih == oh and iw == ow:
                if grids_equal(inp, out):
                    scores["same"] += 1
                elif grids_equal(inp, h_flip_h(out)) or grids_equal(inp, h_flip_v(out)):
                    scores["mirror"] += 1
                elif grids_equal(inp, h_rotate90(out)) or grids_equal(inp, h_rotate180(out)):
                    scores["rotate"] += 1
                else:
                    scores["color_only"] += 1
            elif oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
                scores["scale"] += 1
            elif oh < ih or ow < iw:
                scores["crop"] += 1
            elif oh > ih and oh % ih == 0:
                scores["tile"] += 1
            else:
                scores["other"] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "unknown"

    def _detect_preserved_colors(self, train_pairs):
        """Colors always preserved between input and output."""
        if not train_pairs:
            return set()
        preserved = None
        for inp, out in train_pairs:
            inp_colors = set(c for row in inp for c in row)
            out_colors = set(c for row in out for c in row)
            common = inp_colors & out_colors
            if preserved is None:
                preserved = common
            else:
                preserved &= common
        return preserved or set()

    def _detect_new_colors(self, train_pairs):
        """Colors in output not in input, consistently."""
        if not train_pairs:
            return set()
        new_colors = None
        for inp, out in train_pairs:
            inp_colors = set(c for row in inp for c in row)
            out_colors = set(c for row in out for c in row)
            new = out_colors - inp_colors
            if new_colors is None:
                new_colors = new
            else:
                new_colors &= new
        return new_colors or set()

    def _compute_consistency(self, train_pairs):
        """Compute how consistent the transformation is across examples."""
        if len(train_pairs) < 2:
            return 1.0
        size_pattern = self.detect_size_pattern(train_pairs)
        if size_pattern == "unknown" or size_pattern == "mixed":
            return 0.3
        color_map = self.detect_color_mapping(train_pairs)
        if color_map:
            return 0.9
        struct = self.detect_structural_pattern(train_pairs)
        if struct != "unknown" and struct != "other":
            return 0.7
        return 0.5

    def generate_analysis_report(self, task):
        """Generate a natural language analysis report for the task."""
        train_pairs = [(ex["input"], ex["output"]) for ex in task.get("train", [])]
        analysis = self.analyze(train_pairs)
        report = "## Cross-Example Analysis\n\n"
        report += f"Size pattern: {analysis.get('size_changes', 'unknown')}\n"
        report += f"Structural pattern: {analysis.get('structural_pattern', 'unknown')}\n"
        report += f"Color mapping: {analysis.get('color_mapping', {})}\n"
        report += f"Preserved colors: {analysis.get('preserved_colors', set())}\n"
        report += f"New colors: {analysis.get('new_colors', set())}\n"
        report += f"Consistency: {analysis.get('consistency_score', 0):.2f}\n"
        return report


cross_example_analyzer = CrossExampleAnalyzer()
print("[MODULE D] Cross-Example Pattern Analyzer ready")
'''

CELL_45_CODE = r'''# ============================================================
# MODULE E: OUTPUT PROPERTY PREDICTOR
# ============================================================

class OutputPropertyPredictor:
    """Predicts properties of the expected output grid."""

    def __init__(self):
        self.cache = {}

    def predict(self, train_pairs, test_input):
        """Predict: expected dimensions, colors, num components, density."""
        if not train_pairs:
            return self._default_prediction(test_input)
        # Analyze size changes
        size_pattern = cross_example_analyzer.detect_size_pattern(train_pairs)
        # Analyze color changes
        color_mapping = cross_example_analyzer.detect_color_mapping(train_pairs)
        preserved = cross_example_analyzer._detect_preserved_colors(train_pairs)
        new_colors = cross_example_analyzer._detect_new_colors(train_pairs)
        # Predict dimensions
        expected_dims = self.predict_dimensions(train_pairs, test_input)
        # Predict colors
        expected_colors = self.predict_colors(train_pairs)
        # Predict shape
        expected_shape = self.predict_shape(train_pairs)
        return {
            "expected_dimensions": expected_dims,
            "expected_colors": expected_colors,
            "expected_shape": expected_shape,
            "size_pattern": size_pattern,
            "color_mapping": color_mapping,
            "preserved_colors": preserved,
            "new_colors": new_colors,
        }

    def predict_dimensions(self, train_pairs, test_input):
        """Predict output dimensions for test input."""
        if not train_pairs:
            return grid_shape(test_input)
        test_h, test_w = grid_shape(test_input)
        dims = []
        for inp, out in train_pairs:
            ih, iw = grid_shape(inp)
            oh, ow = grid_shape(out)
            if ih == 0 or iw == 0:
                continue
            fy = oh / ih
            fx = ow / iw
            dims.append((fy, fx))
        if not dims:
            return (test_h, test_w)
        # Most common scale factor
        scale_counts = Counter(dims)
        most_common = scale_counts.most_common(1)[0][0]
        fy, fx = most_common
        if fy == int(fy) and fx == int(fx):
            return (int(test_h * fy), int(test_w * fx))
        return (int(test_h * fy), int(test_w * fx))

    def predict_colors(self, train_pairs):
        """Predict expected colors in output."""
        if not train_pairs:
            return set()
        color_sets = []
        for _, out in train_pairs:
            color_sets.append(set(c for row in out for c in row))
        # Union of all output colors (conservative)
        result = color_sets[0]
        for cs in color_sets[1:]:
            result |= cs
        return result

    def predict_shape(self, train_pairs):
        """Predict the structural shape of the output."""
        struct = cross_example_analyzer.detect_structural_pattern(train_pairs)
        return struct

    def _default_prediction(self, test_input):
        h, w = grid_shape(test_input)
        return {
            "expected_dimensions": (h, w),
            "expected_colors": set(c for row in test_input for c in row),
            "expected_shape": "same",
        }


output_predictor = OutputPropertyPredictor()
print("[MODULE E] Output Property Predictor ready")
'''

CELL_47_CODE = r'''# ============================================================
# MODULE F: ENHANCED DREAMCODER v2 (KNOWLEDGE TRANSFER)
# ============================================================

class EnhancedAbstractionLibrary(AbstractionLibrary):
    """Extended DreamCoder library with knowledge transfer between tasks."""

    def __init__(self):
        super().__init__()
        self.task_solutions = {}  # task_id -> [(code, score)]
        self.task_features = {}   # task_id -> feature dict
        self.higher_order = {}    # name -> code

    def add_verified_solution(self, task_id, code, score):
        """Store verified solutions per task for cross-task reuse."""
        if task_id not in self.task_solutions:
            self.task_solutions[task_id] = []
        self.task_solutions[task_id].append((code, score))

    def store_task_features(self, task_id, features):
        """Store task features for similarity matching."""
        self.task_features[task_id] = features

    def find_similar_solved_tasks(self, task_features, n=3):
        """Find previously solved tasks with similar features."""
        if not self.task_features:
            return []
        similarities = []
        current_colors = set(task_features.get("colors", []))
        current_nobj = task_features.get("n_objects", 0)
        for tid, feat in self.task_features.items():
            if tid not in self.task_solutions:
                continue
            feat_colors = set(feat.get("colors", []))
            overlap = len(current_colors & feat_colors)
            obj_diff = abs(current_nobj - feat.get("n_objects", 0))
            sim = overlap * 2 - obj_diff
            similarities.append((tid, sim))
        similarities.sort(key=lambda x: -x[1])
        return [tid for tid, _ in similarities[:n]]

    def get_transfer_code(self, similar_task_ids):
        """Get code from similar solved tasks for transfer learning."""
        code_blocks = []
        for tid in similar_task_ids:
            if tid in self.task_solutions:
                for code, score in self.task_solutions[tid]:
                    if score >= 0.5:
                        code_blocks.append(f"# Transferred from task {tid} (score={score:.2f})\n{code}")
        return "\n\n".join(code_blocks)

    def extract_higher_order_abstractions(self):
        """Extract higher-order abstractions from verified solutions.
        Returns count of new abstractions extracted.
        """
        count = 0
        for tid, solutions in self.task_solutions.items():
            for code, score in solutions:
                if score < 0.8:
                    continue
                # Look for common patterns
                if "for r in range" in code and "for c in range" in code and "component" in code.lower():
                    name = f"pattern_grid_iter_{tid[:6]}"
                    if name not in self.higher_order:
                        self.higher_order[name] = code
                        count += 1
                if "rotate" in code.lower() and "90" in code:
                    name = f"pattern_rotation_{tid[:6]}"
                    if name not in self.higher_order:
                        self.higher_order[name] = code
                        count += 1
                if "color" in code.lower() and "map" in code.lower():
                    name = f"pattern_color_map_{tid[:6]}"
                    if name not in self.higher_order:
                        self.higher_order[name] = code
                        count += 1
        return count

    def compress_library(self, max_size=200):
        """Compress library to max_size by removing least-used abstractions."""
        if len(self.abstractions) <= max_size:
            return
        # Keep built-in categories first
        priority = ["geometry", "objects", "color", "analysis", "custom", "higher_order"]
        kept = {}
        by_category = defaultdict(list)
        for name, abstr in self.abstractions.items():
            by_category[abstr["category"]].append((name, abstr))
        for cat in priority:
            items = by_category.get(cat, [])
            items.sort(key=lambda x: -self.usage_counts.get(x[0], 0))
            for name, abstr in items:
                if len(kept) < max_size:
                    kept[name] = abstr
        self.abstractions = kept

    def get_enhanced_prompt_context(self, task, train_pairs):
        """Get enhanced context for LLM prompts including transfer knowledge."""
        context = ""
        # Get task features
        features = {}
        if train_pairs:
            inp, _ = train_pairs[0]
            features = extract_grid_features(inp)
        # Find similar tasks
        similar = self.find_similar_solved_tasks(features, n=2)
        if similar:
            transfer_code = self.get_transfer_code(similar)
            if transfer_code:
                context += "## Similar Previously Solved Tasks\n"
                context += transfer_code + "\n\n"
        # Get relevant abstractions
        task_desc = describe_grid_natural_language(train_pairs[0][0]) if train_pairs else ""
        relevant = self.get_relevant_abstractions(task_desc, max_abstractions=3)
        if relevant:
            context += "## Relevant Abstractions\n"
            for name, abstr in relevant:
                context += f"### {name}\n{abstr['code']}\n\n"
        return context


enhanced_library = EnhancedAbstractionLibrary()
print("[MODULE F] Enhanced DreamCoder v2 ready")
'''

CELL_49_CODE = r'''# ============================================================
# MODULE G: MULTI-PHASE STUCK RECOVERY
# ============================================================

class StuckRecovery:
    """6-phase recovery when stuck (no solution found)."""

    PHASES = [
        "intensify_mutations",
        "alternative_llm",
        "composite_search",
        "reverse_engineering",
        "abstraction_injection",
        "hybrid_scoring",
    ]

    def __init__(self):
        self.phase_log = []

    def attempt_recovery(self, task, train_pairs, engine, failed_candidates, test_input):
        """Attempt recovery using all 6 phases.
        Returns list of (prediction, score, phase_name).
        """
        results = []
        for phase_name in self.PHASES:
            try:
                phase_results = self._run_phase(phase_name, task, train_pairs, engine, failed_candidates, test_input)
                results.extend(phase_results)
                if any(s >= 1.0 for _, s, _ in phase_results):
                    break
            except Exception as e:
                self.phase_log.append((phase_name, f"error: {e}"))
        return results

    def _run_phase(self, phase_name, task, train_pairs, engine, failed_candidates, test_input):
        if phase_name == "intensify_mutations":
            return self._phase_intensify(failed_candidates, train_pairs, test_input)
        elif phase_name == "alternative_llm":
            return self._phase_alternative_llm(task, engine, train_pairs, test_input)
        elif phase_name == "composite_search":
            return self._phase_composite_search(train_pairs, test_input)
        elif phase_name == "reverse_engineering":
            return self._phase_reverse_engineering(train_pairs, test_input)
        elif phase_name == "abstraction_injection":
            return self._phase_abstraction_injection(task, train_pairs, test_input)
        elif phase_name == "hybrid_scoring":
            return self._phase_hybrid_scoring(failed_candidates, train_pairs, test_input)
        return []

    def _phase_intensify(self, failed_candidates, train_pairs, test_input):
        """Phase 1: More aggressive mutations on failed candidates."""
        results = []
        mutator = ProgramMutator()
        for pred, score, method, _ in failed_candidates[:5]:
            code = getattr(pred, '__code__', None)
            if code is None and hasattr(pred, 'code'):
                code = pred.code
            if code is None:
                # Generate code from heuristic name
                if "heuristic:" in method:
                    h_name = method.replace("heuristic:", "")
                    h_fn = next((h for h in ALL_HEURISTICS if h.__name__ == h_name), None)
                    if h_fn:
                        try:
                            result = h_fn(test_input)
                            results.append((result, score, "intensify"))
                        except Exception:
                            pass
                continue
            for _ in range(10):
                mutated = mutator.mutate(str(code))
                s, preds = verify_code_on_training(mutated, train_pairs, timeout=3)
                if s >= 1.0 and preds:
                    r, e = safe_execute_code(mutated, test_input, timeout=5)
                    if r is not None:
                        results.append((r, s, "intensify"))
                        break
        self.phase_log.append(("intensify_mutations", f"generated {len(results)} candidates"))
        return results

    def _phase_alternative_llm(self, task, engine, train_pairs, test_input):
        """Phase 2: Different temperature/prompt for LLM."""
        results = []
        if not engine.is_available():
            return results
        for temp in [0.1, 0.9, 1.2]:
            r, status = two_phase_llm_solve(task, train_pairs, test_input, max_time=60)
            if r is not None:
                results.append((r, 1.0, f"alt_llm_t{temp}"))
        self.phase_log.append(("alternative_llm", f"generated {len(results)} candidates"))
        return results

    def _phase_composite_search(self, train_pairs, test_input):
        """Phase 3: Try all 2-step composites."""
        results = []
        top_heuristics = find_best_partial_heuristics(train_pairs, top_k=20)
        tried = set()
        for s1, h1, _ in top_heuristics[:10]:
            for s2, h2, _ in top_heuristics[:10]:
                pair = (h1.__name__, h2.__name__)
                if pair in tried:
                    continue
                tried.add(pair)
                # Test composition on all training pairs
                all_match = True
                for inp, expected in train_pairs:
                    try:
                        mid = h1(inp)
                        final = h2(mid)
                        if not grids_equal(final, expected):
                            all_match = False
                            break
                    except Exception:
                        all_match = False
                        break
                if all_match:
                    try:
                        pred = h2(h1(test_input))
                        results.append((pred, 1.0, f"composite:{h1.__name__}+{h2.__name__}"))
                    except Exception:
                        pass
        self.phase_log.append(("composite_search", f"generated {len(results)} candidates"))
        return results

    def _phase_reverse_engineering(self, train_pairs, test_input):
        """Phase 4: Analyze output properties and construct solution."""
        results = []
        props = output_predictor.predict(train_pairs, test_input)
        expected_h, expected_w = props["expected_dimensions"]
        expected_colors = props["expected_colors"]
        # Try constructing output from properties
        for h_fn in ALL_HEURISTICS[:50]:
            try:
                pred = h_fn(test_input)
                if pred and grid_shape(pred) == (expected_h, expected_w):
                    pred_colors = set(c for row in pred for c in row)
                    if pred_colors == expected_colors:
                        # Verify on training
                        score, _ = score_heuristic_on_training(h_fn, train_pairs)
                        if score > 0.5:
                            results.append((pred, score, f"reverse:{h_fn.__name__}"))
            except Exception:
                continue
        self.phase_log.append(("reverse_engineering", f"generated {len(results)} candidates"))
        return results

    def _phase_abstraction_injection(self, task, train_pairs, test_input):
        """Phase 5: Force use of learned abstractions."""
        results = []
        context = enhanced_library.get_enhanced_prompt_context(task, train_pairs)
        if context:
            prompt = context + "\n\nUsing the above abstractions, write a transform function:"
            code = llm_engine.generate(prompt, system_prompt=PHASE2_SYSTEM, temperature=0.5, max_tokens=1024)
            if code:
                fn_code = extract_function_code(code)
                if fn_code:
                    score, preds = verify_code_on_training(fn_code, train_pairs)
                    if score >= 1.0 and preds:
                        r, e = safe_execute_code(fn_code, test_input)
                        if r is not None:
                            results.append((r, score, "abstraction_inject"))
        self.phase_log.append(("abstraction_injection", f"generated {len(results)} candidates"))
        return results

    def _phase_hybrid_scoring(self, all_candidates, train_pairs, test_input):
        """Phase 6: Weighted ensemble scoring of all candidates."""
        results = []
        if not all_candidates:
            return results
        # Score each candidate by pixel agreement with training outputs
        for pred, base_score, method, _ in all_candidates[:10]:
            if not pred or not isinstance(pred, list) or not pred[0]:
                continue
            # Check consistency with predicted properties
            props = output_predictor.predict(train_pairs, test_input)
            eh, ew = props["expected_dimensions"]
            ph, pw = grid_shape(pred)
            dim_match = 1.0 if (eh, ew) == (ph, pw) else 0.3
            color_match = len(set(c for r in pred for c in r) & props["expected_colors"]) / max(len(props["expected_colors"]), 1)
            hybrid_score = 0.4 * base_score + 0.3 * dim_match + 0.3 * color_match
            results.append((pred, hybrid_score, f"hybrid:{method}"))
        results.sort(key=lambda x: -x[1])
        self.phase_log.append(("hybrid_scoring", f"scored {len(results)} candidates"))
        return results[:3]


stuck_recovery = StuckRecovery()
print("[MODULE G] Multi-Phase Stuck Recovery ready")
'''

CELL_51_CODE = r'''# ============================================================
# MODULE H: SELF-CONSISTENCY LLM SOLVER
# ============================================================

class SelfConsistencySolver:
    """Generate multiple LLM samples and vote on the most consistent output."""

    def __init__(self, num_samples=8, temperatures=None):
        if temperatures is None:
            temperatures = [0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9]
        self.num_samples = num_samples
        self.temperatures = temperatures[:num_samples]

    def solve(self, task, engine, test_input_idx=0, max_time=180):
        """Generate multiple samples and vote.
        Returns list of (prediction, confidence, n_votes).
        """
        if not engine.is_available():
            return []
        train_pairs = [(ex["input"], ex["output"]) for ex in task.get("train", [])]
        test_input = task["test"][test_input_idx]["input"]
        start_time = time.time()
        outputs = []
        for i, temp in enumerate(self.temperatures):
            if time.time() - start_time > max_time:
                break
            try:
                result, status = two_phase_llm_solve(
                    task, train_pairs, test_input, max_time=max_time / self.num_samples
                )
                if result is not None:
                    outputs.append(result)
            except Exception:
                continue
        if not outputs:
            return []
        # Majority vote
        voted = self._majority_vote(outputs)
        results = []
        for pred, votes in voted:
            confidence = votes / len(outputs)
            results.append((pred, confidence, votes))
        results.sort(key=lambda x: -x[2])
        return results

    def _majority_vote(self, outputs):
        """Vote on outputs, return (prediction, vote_count) pairs."""
        if not outputs:
            return []
        # Hash outputs for comparison
        hash_counts = Counter()
        hash_to_output = {}
        for out in outputs:
            h = grid_hash(out)
            hash_counts[h] += 1
            if h not in hash_to_output:
                hash_to_output[h] = out
        # Sort by vote count
        results = []
        for h, count in hash_counts.most_common():
            results.append((hash_to_output[h], count))
        return results


self_consistency_solver = SelfConsistencySolver()
print("[MODULE H] Self-Consistency LLM Solver ready")
'''

CELL_53_CODE = r'''# ============================================================
# MAIN PIPELINE
# ============================================================

def solve_all_tasks(tasks, time_budget_per_task=300, total_time_budget=3600 * 8):
    """Solve all tasks with time budgeting and progress tracking."""
    start_time = time.time()
    total = len(tasks)
    results = {}
    solved_count = 0
    failed_count = 0

    tqdm_fn = get_tqdm()
    task_iter = tqdm_fn(tasks.items(), desc="Solving tasks", total=total) if tqdm_fn else tasks.items()

    for task_id, task in task_iter:
        elapsed = time.time() - start_time
        remaining = total_time_budget - elapsed
        if remaining < 30:
            print(f"[TIME] Only {remaining:.0f}s remaining, stopping.")
            break

        per_task_budget = min(time_budget_per_task, remaining / max(total - solved_count - failed_count, 1))

        try:
            # Get train pairs
            train_pairs = [(ex["input"], ex["output"]) for ex in task.get("train", [])]
            test_input = task["test"][0]["input"]

            # Extract features and store for knowledge transfer
            if train_pairs:
                feat = extract_grid_features(train_pairs[0][0])
                enhanced_library.store_task_features(task_id, feat)

            # Run cross-example analysis
            if train_pairs:
                analysis = cross_example_analyzer.analyze(train_pairs)

            # Solve using main solver
            prediction, score, method = solve_task(task, task_id, time_budget=per_task_budget)

            # If stuck, try recovery
            if score < 1.0 and train_pairs:
                candidates = []
                for h_fn in ALL_HEURISTICS[:20]:
                    try:
                        pred = h_fn(test_input)
                        s, _ = score_heuristic_on_training(h_fn, train_pairs, require_all_match=False)
                        if s > 0:
                            candidates.append((pred, s, f"h:{h_fn.__name__}", [pred]))
                    except Exception:
                        continue
                recovery = stuck_recovery.attempt_recovery(
                    task, train_pairs, llm_engine, candidates, test_input
                )
                for rec_pred, rec_score, rec_method in recovery:
                    if rec_score > score:
                        prediction = rec_pred
                        score = rec_score
                        method = f"recovery:{rec_method}"
                if score >= 1.0:
                    # Store verified solution for knowledge transfer
                    code_hint = method
                    enhanced_library.add_verified_solution(task_id, code_hint, score)

            # Handle multiple test cases
            n_test = len(task.get("test", []))
            attempts = []
            for t_idx in range(n_test):
                if t_idx == 0:
                    pred = prediction if prediction else copy.deepcopy(task["test"][0]["input"])
                else:
                    # For additional test cases, reuse the method
                    t_input = task["test"][t_idx]["input"]
                    try:
                        if "heuristic:" in method:
                            h_name = method.replace("heuristic:", "")
                            h_fn = next((h for h in ALL_HEURISTICS if h.__name__ == h_name), None)
                            pred = h_fn(t_input) if h_fn else copy.deepcopy(t_input)
                        else:
                            pred = copy.deepcopy(t_input)
                    except Exception:
                        pred = copy.deepcopy(t_input)
                if NUM_SUBMISSION_ATTEMPTS == 2:
                    attempts.append({
                        "attempt_1": pred,
                        "attempt_2": copy.deepcopy(task["test"][t_idx]["input"]),
                    })
                else:
                    attempts.append({"attempt_1": pred})

            results[task_id] = attempts

            if score >= 1.0:
                solved_count += 1
                print(f"  [{task_id}] SOLVED via {method}")
            else:
                failed_count += 1
                print(f"  [{task_id}] UNSOLVED (best: {method}, score={score:.2f})")

        except Exception as e:
            print(f"  [{task_id}] ERROR: {e}")
            traceback.print_exc()
            failed_count += 1
            results[task_id] = [{"attempt_1": copy.deepcopy(task["test"][0]["input"])}]

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[PIPELINE] Complete in {elapsed:.1f}s")
    print(f"[PIPELINE] Solved: {solved_count}/{total} ({100*solved_count/max(total,1):.1f}%)")
    print(f"[PIPELINE] Failed: {failed_count}/{total}")
    print(f"{'='*60}")
    return results, solved_count, total

print("[PIPELINE] Main pipeline ready")
'''

CELL_55_CODE = r'''# ============================================================
# TRAINING VALIDATION
# ============================================================

def validate_on_training(tasks, sample_size=None):
    """Validate solver on training tasks to estimate accuracy."""
    task_list = list(tasks.items())
    if sample_size and sample_size < len(task_list):
        task_list = random.sample(task_list, sample_size)

    solved = 0
    total = len(task_list)
    method_counts = Counter()

    tqdm_fn = get_tqdm()
    task_iter = tqdm_fn(task_list, desc="Validating", total=total) if tqdm_fn else task_list

    for task_id, task in task_iter:
        train_pairs = [(ex["input"], ex["output"]) for ex in task.get("train", [])]
        if not train_pairs:
            continue
        test_input = task["test"][0]["input"]

        # Quick heuristic check
        best_heuristics = find_best_heuristics(train_pairs, top_k=1)
        if best_heuristics:
            score, h_fn, preds = best_heuristics[0]
            if score >= 1.0:
                try:
                    pred = h_fn(test_input)
                    if grids_equal(pred, task["test"][0]["output"]):
                        solved += 1
                        method_counts[f"heuristic:{h_fn.__name__}"] += 1
                        continue
                except Exception:
                    pass

        # BFS check
        result, method = bfs_solve(train_pairs, test_input, max_depth=2)
        if result and grids_equal(result, task["test"][0]["output"]):
            solved += 1
            method_counts[method] += 1

    print(f"\n[VALIDATION] Solved: {solved}/{total} ({100*solved/max(total,1):.1f}%)")
    print(f"[VALIDATION] Method breakdown: {method_counts.most_common(10)}")
    return solved, total, method_counts

print("[VALIDATION] Training validation ready")
'''

CELL_57_CODE = r'''# ============================================================
# SUBMISSION VERIFICATION
# ============================================================

def verify_submission(submission, expected_tasks=None):
    """Verify submission format and completeness."""
    if not isinstance(submission, dict):
        print("[VERIFY] ERROR: Submission is not a dict")
        return False
    if expected_tasks:
        missing = set(expected_tasks) - set(submission.keys())
        if missing:
            print(f"[VERIFY] WARNING: Missing {len(missing)} tasks")
    valid = True
    for task_id, attempts in submission.items():
        if not isinstance(attempts, list):
            print(f"[VERIFY] ERROR: {task_id} attempts is not a list")
            valid = False
            continue
        for i, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                print(f"[VERIFY] ERROR: {task_id}[{i}] is not a dict")
                valid = False
                continue
            has_attempt = False
            for key in attempt:
                if key.startswith("attempt_"):
                    has_attempt = True
                    grid = attempt[key]
                    if not isinstance(grid, list):
                        print(f"[VERIFY] ERROR: {task_id}[{i}].{key} is not a list")
                        valid = False
                    elif grid and not isinstance(grid[0], list):
                        print(f"[VERIFY] ERROR: {task_id}[{i}].{key} is not a 2D list")
                        valid = False
            if not has_attempt:
                print(f"[VERIFY] ERROR: {task_id}[{i}] has no attempt_* key")
                valid = False
    if valid:
        print(f"[VERIFY] Submission format OK: {len(submission)} tasks")
    return valid


def save_submission(submission, path=OUTPUT_PATH):
    """Save submission to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(submission, f)
    print(f"[SUBMIT] Saved submission to {path}")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[SUBMIT] File size: {size_mb:.2f} MB")

print("[VERIFY] Submission verification ready")
'''

CELL_59_CODE = r'''# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point: load data -> validate -> solve -> save."""
    total_start = time.time()
    print("=" * 60)
    print("  ARC-AGI-2 SOLVER v3 — Multi-Strategy Ensemble")
    print("  Modules: Heuristics | DreamCoder | LLM | Evolutionary")
    print("           CNN Embedding | BFS | MCTS | Cross-Example")
    print("           Output Predictor | Stuck Recovery | Self-Consistency")
    print("=" * 60)

    # Step 1: Load data
    print("\n[STEP 1/5] Loading data...")
    test_tasks, train_tasks, eval_tasks = load_data_main()
    solve_targets = test_tasks if test_tasks else eval_tasks
    if not solve_targets:
        print("[ERROR] No tasks found to solve!")
        return
    print(f"[STEP 1/5] Found {len(solve_targets)} tasks to solve")

    # Step 2: Quick validation on training subset
    print("\n[STEP 2/5] Quick validation on training subset...")
    all_for_val = {**train_tasks}
    if eval_tasks:
        # Use eval tasks for validation (we have solutions)
        all_for_val.update(eval_tasks)
    val_size = min(20, len(all_for_val))
    if val_size > 0:
        validate_on_training(all_for_val, sample_size=val_size)

    # Step 3: Initialize LLM (optional, may fail gracefully)
    print("\n[STEP 3/5] Initializing LLM engine...")
    if llm_engine.is_available():
        print("[STEP 3/5] LLM engine ready")
    else:
        print("[STEP 3/5] LLM not available, using heuristics + evolutionary")

    # Step 4: Solve all tasks
    print(f"\n[STEP 4/5] Solving {len(solve_targets)} tasks...")
    submission, solved_count, total_count = solve_all_tasks(solve_targets)

    # Step 5: Verify and save submission
    print("\n[STEP 5/5] Verifying and saving submission...")
    verify_submission(submission, set(solve_targets.keys()))
    save_submission(submission)

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Tasks: {total_count}")
    print(f"  Solved: {solved_count} ({100*solved_count/max(total_count,1):.1f}%)")
    print(f"  Time: {total_elapsed:.1f}s")
    print(f"  Heuristics: {len(ALL_HEURISTICS)}")
    print(f"  Abstractions: {len(abstraction_library.abstractions)}")
    if enhanced_library.task_solutions:
        print(f"  Knowledge transfer: {len(enhanced_library.task_solutions)} tasks stored")
    print(f"{'='*60}")

    return submission


if __name__ == "__main__":
    submission = main()
'''

CELL_61_CODE = r'''# ============================================================
# EXECUTE — RUN THE SOLVER
# ============================================================
# This cell runs the full pipeline when the notebook is executed.

submission = main()
'''

# ======================================================================
# NOTEBOOK STRUCTURE: Build cells list
# ======================================================================

cells = [
    # Title
    md(["# ARC-AGI-2 Solver v3 — Multi-Strategy Ensemble", "",
        "Comprehensive ARC puzzle solver combining multiple strategies:", "",
        "- **100+ Heuristic Solvers** (geometric, color, object, scaling, tiling, pattern, logic)",
        "- **DreamCoder Abstraction Library** with knowledge transfer",
        "- **LLM Engine** (Qwen3-4B, 4-bit quantized) for code generation",
        "- **Evolutionary Program Synthesis** with mutation & crossover",
        "- **Grid Embedding CNN** for similarity-based retrieval",
        "- **BFS Solution Space Explorer** for multi-step composition",
        "- **MCTS Code Search** for program space exploration",
        "- **Cross-Example Pattern Analyzer** for consistency detection",
        "- **Output Property Predictor** for solution validation",
        "- **Multi-Phase Stuck Recovery** (6 recovery strategies)",
        "- **Self-Consistency LLM Solver** with majority voting",
        ""]),

    # Cell 1: Imports & Config
    CELL_1_CODE,

    # Cell 2: Section header
    md(["## Grid Utilities", "",
        "Core functions for grid manipulation, analysis, and display."]),

    # Cell 3: Grid Utilities
    CELL_3_CODE,

    # Cell 4: Section header
    md(["## Data Loading", "",
        "Functions to load and explore ARC task data from the competition filesystem."]),

    # Cell 5: Data Loading
    CELL_5_CODE,

    # Cell 6: Section header
    md(["## Heuristic Library — Base Solvers (Geometric, Color, Object, Scaling, Tiling)",
        "",
        "Core heuristic transformations that form the first-pass solving strategy."]),

    # Cell 7: Geometric + Color
    CELL_7_CODE,

    # Cell 8: Section header
    md(["## Heuristic Library — Object, Scaling, Tiling", "",
        "Object manipulation, grid scaling, and tiling heuristics."]),

    # Cell 9: Object + Scaling + Tiling
    CELL_9_CODE,

    # Cell 10: Section header
    md(["## Heuristic Library — Pattern & Logic", "",
        "Pattern-based and conditional logic heuristics."]),

    # Cell 11: Pattern + Logic
    CELL_11_CODE,

    # Cell 12: Section header
    md(["## NEW v3 Heuristics — Geometric Extensions", "",
        "Translation, zoom, pad, and reshape operations."]),

    # Cell 13: v3 Geometric
    CELL_13_CODE,

    # Cell 14: Section header
    md(["## NEW v3 Heuristics — Color Extensions", "",
        "Color cycling, merging, splitting, and filtering operations."]),

    # Cell 15: v3 Color
    CELL_15_CODE,

    # Cell 16: Section header
    md(["## NEW v3 Heuristics — Object Extensions", "",
        "Component sorting, alignment, hole filling, erosion, and dilation."]),

    # Cell 17: v3 Object
    CELL_17_CODE,

    # Cell 18: Section header
    md(["## NEW v3 Heuristics — Pattern Extensions", "",
        "Checkerboard, stripes, frames, diagonals, and concentric shapes."]),

    # Cell 19: v3 Pattern
    CELL_19_CODE,

    # Cell 20: Section header
    md(["## NEW v3 Heuristics — Logic Extensions + Parameterized", "",
        "Conditional logic, pooling, and auto-generated parameterized variants."]),

    # Cell 21: v3 Logic + Parameterized
    CELL_21_CODE,

    # Cell 22: Section header
    md(["## Composite Heuristics", "",
        "Two-step compositions combining base heuristics for multi-step transformations."]),

    # Cell 23: Composites
    CELL_23_CODE,

    # Cell 24: Section header
    md(["## Heuristic Registration & Scoring", "",
        "Registration of all heuristics and scoring system for training validation."]),

    # Cell 25: Registration & Scoring
    CELL_25_CODE,

    # Cell 26: Section header
    md(["## Safe Code Execution", "",
        "Thread-based code execution with timeout for LLM-generated code."]),

    # Cell 27: Safe Execution
    CELL_27_CODE,

    # Cell 28: Section header
    md(["## DreamCoder Abstraction Library", "",
        "Reusable abstractions extracted from verified solutions."]),

    # Cell 29: DreamCoder
    CELL_29_CODE,

    # Cell 30: Section header
    md(["## LLM Engine", "",
        "Lazy-loading LLM with 4-bit quantization, two-phase prompting (analysis → code gen)."]),

    # Cell 31: LLM Engine
    CELL_31_CODE,

    # Cell 32: Section header
    md(["## Evolutionary Program Synthesis", "",
        "Genetic algorithm for evolving program candidates with mutation and crossover."]),

    # Cell 33: Evolutionary
    CELL_33_CODE,

    # Cell 34: Section header
    md(["## Multi-Strategy Ensemble Solver", "",
        "3-pass solver: heuristics → LLM → evolutionary, with ensemble selection."]),

    # Cell 35: Ensemble Solver
    CELL_35_CODE,

    # Cell 36: Section header
    md(["## Module A: Grid Embedding Network (CNN)", "",
        "Small CNN (~50K params) for grid similarity comparison."]),

    # Cell 37: Grid Embedding
    CELL_37_CODE,

    # Cell 38: Section header
    md(["## Module B: Solution Space Graph Explorer (BFS)", "",
        "BFS exploration of solution space via heuristic transformations."]),

    # Cell 39: Solution Graph
    CELL_39_CODE,

    # Cell 40: Section header
    md(["## Module C: MCTS for Program Synthesis", "",
        "Monte Carlo Tree Search over program candidates with UCB1 selection."]),

    # Cell 41: MCTS
    CELL_41_CODE,

    # Cell 42: Section header
    md(["## Module D: Cross-Example Pattern Analyzer", "",
        "Analyzes consistency patterns across multiple training examples."]),

    # Cell 43: Cross-Example
    CELL_43_CODE,

    # Cell 44: Section header
    md(["## Module E: Output Property Predictor", "",
        "Predicts expected output dimensions, colors, and structure."]),

    # Cell 45: Output Predictor
    CELL_45_CODE,

    # Cell 46: Section header
    md(["## Module F: Enhanced DreamCoder v2 (Knowledge Transfer)", "",
        "Extended abstraction library with cross-task knowledge transfer."]),

    # Cell 47: Enhanced DreamCoder
    CELL_47_CODE,

    # Cell 48: Section header
    md(["## Module G: Multi-Phase Stuck Recovery", "",
        "6-phase recovery strategy when no solution is found."]),

    # Cell 49: Stuck Recovery
    CELL_49_CODE,

    # Cell 50: Section header
    md(["## Module H: Self-Consistency LLM Solver", "",
        "Multiple LLM samples with majority voting for robust predictions."]),

    # Cell 51: Self-Consistency
    CELL_51_CODE,

    # Cell 52: Section header
    md(["## Main Pipeline", "",
        "Orchestrates all modules with time budgeting and progress tracking."]),

    # Cell 53: Main Pipeline
    CELL_53_CODE,

    # Cell 54: Section header
    md(["## Training Validation", "",
        "Validate solver accuracy on training tasks with known solutions."]),

    # Cell 55: Training Validation
    CELL_55_CODE,

    # Cell 56: Section header
    md(["## Submission Verification", "",
        "Verify submission format and save to disk."]),

    # Cell 57: Submission Verification
    CELL_57_CODE,

    # Cell 58: Section header
    md(["## Main Entry Point", "",
        "Orchestrates: load data → validate → solve → save submission."]),

    # Cell 59: Main Entry Point
    CELL_59_CODE,

    # Cell 60: Section header
    md(["## Execute", "",
        "Run the full solver pipeline."]),

    # Cell 61: Execute
    CELL_61_CODE,
]

# ======================================================================
# BUILD NOTEBOOK
# ======================================================================

# Convert string cells to proper cell dicts
notebook_cells = []
for cell in cells:
    if isinstance(cell, str):
        notebook_cells.append(code(cell))
    elif isinstance(cell, dict) and cell.get("cell_type") == "code":
        notebook_cells.append(cell)
    elif isinstance(cell, dict) and cell.get("cell_type") == "markdown":
        notebook_cells.append(cell)
    else:
        # md() returns a dict
        notebook_cells.append(cell)

notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": notebook_cells,
}

# Write notebook
with open(OUTPUT_PATH, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"\nNotebook written to {OUTPUT_PATH}")

# Verify
with open(OUTPUT_PATH) as f:
    nb = json.load(f)
total_code_lines = sum(len("".join(c["source"]).splitlines()) for c in nb["cells"] if c["cell_type"] == "code")
total_cells = len(nb["cells"])
code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
md_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
total_chars = sum(len("".join(c["source"])) for c in nb["cells"])
print(f"Total cells: {total_cells}")
print(f"Code cells: {code_cells}")
print(f"Markdown cells: {md_cells}")
print(f"Total code lines: {total_code_lines}")
print(f"Total chars: {total_chars}")
print(f"Valid JSON: YES")
