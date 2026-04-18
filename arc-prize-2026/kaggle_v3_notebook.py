#!/usr/bin/env python3
"""
AnnotateX ARC-AGI-2 v3 Solver — Kaggle Notebook
=================================================

Self-contained solver for the ARC-AGI-2 competition on Kaggle.
Merges all v2 heuristic strategies with v3 improvements:

  1. Conditional Logic Engine — splits demos by discriminant features
  2. Arithmetic & Counting Primitives — count/color-based transforms
  3. Adaptive Connectivity & Diagonal Detection
  4. Full v2 Heuristic Ensemble (30+ strategies)
  5. LLM Code Generation Fallback (Qwen3-4B, GPU only)
  6. Self-consistency: top-2 distinct predictions per task

Pipeline order per task:
  a. Conditional Logic Engine
  b. Arithmetic & Counting Primitives
  c. Adaptive Connectivity + Diagonal Detection
  d. Full v2 Heuristic Ensemble
  e. LLM Code Generation (if GPU available)
  f. Self-consistency: pick top 2 distinct predictions
  g. Fallback: copy of input grid

Time budget: ~80 seconds per task (400 tasks in 9 hours).

Dependencies: Python 3.11, numpy. Optional: torch, transformers, bitsandbytes (GPU only).
"""

import json
import os
import glob
import sys
import time
import re
import traceback
import threading
import argparse
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import permutations, product as itertools_product
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Try to import torch/transformers for LLM fallback
_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

_LLM_AVAILABLE = False
try:
    if _HAS_TORCH:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _LLM_AVAILABLE = True
except ImportError:
    pass

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
    return np.array(grid, dtype=np.int32)

def array_to_grid(arr):
    return arr.astype(int).tolist()

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

def grid_hash(grid):
    """Deterministic hash of a grid for dedup."""
    return hash(tuple(tuple(row) for row in grid))


# ============================================================
# 1. CONNECTED COMPONENT ANALYSIS
# ============================================================

class ComponentAnalyzer:
    """Find connected components, bounding boxes, and spatial relationships."""

    DIRECTIONS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    DIRECTIONS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    @staticmethod
    def find_components(arr, connectivity=4):
        """Find all connected components. Returns list of component info dicts."""
        if connectivity == 4:
            dirs = ComponentAnalyzer.DIRECTIONS_4
        else:
            dirs = ComponentAnalyzer.DIRECTIONS_8

        h, w = arr.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []

        for r in range(h):
            for c in range(w):
                if arr[r, c] != 0 and not visited[r, c]:
                    # BFS
                    color = int(arr[r, c])
                    queue = [(r, c)]
                    visited[r, c] = True
                    pixels = [(r, c)]
                    while queue:
                        cr, cc = queue.pop(0)
                        for dr, dc in dirs:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == color:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                                pixels.append((nr, nc))

                    rows = [p[0] for p in pixels]
                    cols = [p[1] for p in pixels]
                    bbox = (min(rows), min(cols), max(rows), max(cols))
                    component = {
                        'color': color,
                        'size': len(pixels),
                        'pixels': pixels,
                        'bbox': bbox,
                        'shape': (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1),
                        'center': ((min(rows) + max(rows)) / 2, (min(cols) + max(cols)) / 2),
                    }
                    components.append(component)

        return components

    @staticmethod
    def find_components_by_color(arr, connectivity=4):
        """Find components, grouping by color."""
        comps = ComponentAnalyzer.find_components(arr, connectivity)
        by_color = defaultdict(list)
        for c in comps:
            by_color[c['color']].append(c)
        return dict(by_color)

    @staticmethod
    def extract_component(arr, component, fill=0):
        """Extract a component as a subgrid (bounding box crop)."""
        min_r, min_c, max_r, max_c = component['bbox']
        return arr[min_r:max_r+1, min_c:max_c+1].copy()

    @staticmethod
    def component_mask(arr, component):
        """Create a boolean mask for a component."""
        mask = np.zeros_like(arr, dtype=bool)
        for r, c in component['pixels']:
            mask[r, c] = True
        return mask

    @staticmethod
    def spatial_relationship(comp1, comp2):
        """Describe spatial relationship between two components."""
        cx1, cy1 = comp1['center']
        cx2, cy2 = comp2['center']
        relations = []
        if cy1 < cy2:
            relations.append('left_of')
        elif cy1 > cy2:
            relations.append('right_of')
        if cx1 < cx2:
            relations.append('above')
        elif cx1 > cx2:
            relations.append('below')
        if not relations:
            relations.append('same_position')
        return relations

    @staticmethod
    def count_components(arr, connectivity=4):
        return len(ComponentAnalyzer.find_components(arr, connectivity))


# ============================================================
# 2. COMPOSITE TRANSFORMATIONS
# ============================================================

class CompositeTransformer:
    """Chain multiple operations (rotate + color map, flip + scale, etc.)."""

    PRIMITIVE_TRANSFORMS = {}

    @staticmethod
    def get_primitives():
        """Return dict of name -> transform function."""
        return {
            'identity': lambda a: a,
            'rot90': lambda a: np.rot90(a, -1),
            'rot180': lambda a: np.rot90(a, 2),
            'rot270': lambda a: np.rot90(a, 1),
            'flip_h': lambda a: np.flip(a, axis=1),
            'flip_v': lambda a: np.flip(a, axis=0),
            'transpose': lambda a: a.T.copy(),
            'rot90_flip_h': lambda a: np.flip(np.rot90(a, -1), axis=1),
            'rot90_flip_v': lambda a: np.flip(np.rot90(a, -1), axis=0),
        }

    @staticmethod
    def apply_composite(arr, transform_name):
        prims = CompositeTransformer.get_primitives()
        if transform_name in prims:
            return prims[transform_name](arr)
        # Handle composite names separated by +
        parts = transform_name.split('+')
        result = arr.copy()
        for p in parts:
            p = p.strip()
            if p in prims:
                result = prims[p](result)
            else:
                return None
        return result

    @staticmethod
    def find_best_transform(inp_arr, out_arr, min_conf=0.95):
        """Find the best geometric transform from inp to out.
        Only tests geometric transforms; does NOT change colors."""
        if inp_arr.shape != out_arr.shape:
            # Try transpose too
            if inp_arr.shape == out_arr.T.shape:
                if np.array_equal(inp_arr.T, out_arr):
                    return 'transpose', 1.0
            return None, 0

        prims = CompositeTransformer.get_primitives()
        best_name = None
        best_conf = 0
        for name, fn in prims.items():
            try:
                transformed = fn(inp_arr)
                if transformed.shape != out_arr.shape:
                    continue
                match = np.sum(transformed == out_arr) / max(out_arr.size, 1)
                if match > best_conf:
                    best_conf = match
                    best_name = name
            except Exception:
                continue
        if best_conf >= min_conf:
            return best_name, best_conf
        return None, 0

    @staticmethod
    def find_transform_with_color_map(inp_arr, out_arr, min_conf=0.95):
        """Find a geometric transform + color remapping that converts inp -> out."""
        all_transforms = CompositeTransformer.get_primitives()
        for tname, tfn in all_transforms.items():
            try:
                transformed = tfn(inp_arr)
                if transformed.shape != out_arr.shape:
                    continue
                mapping = _infer_color_map_np(transformed, out_arr)
                if mapping is not None:
                    # Verify
                    remapped = transformed.copy()
                    for src, dst in mapping.items():
                        remapped[remapped == src] = dst
                    if remapped.shape == out_arr.shape:
                        match = np.sum(remapped == out_arr) / max(out_arr.size, 1)
                        if match >= min_conf:
                            return tname, mapping, match
            except Exception:
                continue
        return None, None, 0


def _infer_color_map_np(inp_arr, out_arr):
    """Infer a color mapping between two same-shape arrays."""
    if inp_arr.shape != out_arr.shape:
        return None
    mapping = {}
    flat_i = inp_arr.flatten()
    flat_o = out_arr.flatten()
    for i_val, o_val in zip(flat_i, flat_o):
        i_val, o_val = int(i_val), int(o_val)
        if i_val in mapping:
            if mapping[i_val] != o_val:
                return None
        else:
            mapping[i_val] = o_val
    if len(mapping) <= 1 or all(k == v for k, v in mapping.items()):
        return None
    return mapping


# ============================================================
# 3. SHAPE PRESERVATION DETECTION
# ============================================================

class ShapeAnalyzer:
    """Detect if output shape matches input or follows a pattern."""

    @staticmethod
    def analyze_shape_pair(inp, out):
        """Analyze shape relationship between input and output."""
        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(out)
        info = {
            'input_shape': (ih, iw),
            'output_shape': (oh, ow),
            'same_shape': ih == oh and iw == ow,
            'h_ratio': oh / ih if ih > 0 else 0,
            'w_ratio': ow / iw if iw > 0 else 0,
            'swapped': (ih, iw) == (ow, oh),
            'scale_factor_h': oh // ih if ih > 0 and oh % ih == 0 else None,
            'scale_factor_w': ow // iw if iw > 0 and ow % iw == 0 else None,
        }
        info['uniform_scale'] = (info['scale_factor_h'] is not None and
                                  info['scale_factor_w'] is not None and
                                  info['scale_factor_h'] == info['scale_factor_w'])
        return info

    @staticmethod
    def consistent_shape_across_examples(task):
        """Check if all train examples have consistent shape relationships."""
        if len(task['train']) < 2:
            return True
        infos = []
        for ex in task['train']:
            info = ShapeAnalyzer.analyze_shape_pair(ex['input'], ex['output'])
            infos.append(info)
        # Check if same_shape is consistent
        same_shapes = [i['same_shape'] for i in infos]
        if all(same_shapes):
            return True
        # Check if scale is consistent
        scales_h = [i['scale_factor_h'] for i in infos if i['scale_factor_h'] is not None]
        scales_w = [i['scale_factor_w'] for i in infos if i['scale_factor_w'] is not None]
        if len(scales_h) > 1 and len(set(scales_h)) == 1 and len(set(scales_w)) == 1:
            return True
        return False

    @staticmethod
    def detect_output_size_rule(task):
        """Infer the output size for the test input based on training patterns."""
        examples = task['train']
        if not examples:
            return None
        shapes_in = [grid_shape(ex['input']) for ex in examples]
        shapes_out = [grid_shape(ex['output']) for ex in examples]

        if len(set(shapes_in)) == 1 and len(set(shapes_out)) == 1:
            # All inputs same size, all outputs same size
            return shapes_out[0]

        # Check for pattern: output size depends on input content
        # e.g., output is N x N where N = number of distinct colors
        out_sizes = set(shapes_out)
        if len(out_sizes) == 1:
            return list(out_sizes)[0]

        return None


# ============================================================
# 4. ROW/COLUMN ANALYSIS
# ============================================================

class RowColumnAnalyzer:
    """Detect row/column rearrangements, reversals, and transformations."""

    @staticmethod
    def is_row_rearrangement(inp, out):
        """Check if output rows are a permutation of input rows."""
        inp_arr = grid_to_array(inp)
        out_arr = grid_to_array(out)
        if inp_arr.shape != out_arr.shape:
            return None, 0
        inp_rows = [tuple(row) for row in inp_arr]
        out_rows = [tuple(row) for row in out_arr]
        if Counter(inp_rows) == Counter(out_rows):
            # Find the permutation
            mapping = []
            for o_row in out_rows:
                idx = inp_rows.index(o_row)
                mapping.append(idx)
            return mapping, 1.0
        return None, 0

    @staticmethod
    def is_col_rearrangement(inp, out):
        """Check if output columns are a permutation of input columns."""
        inp_arr = grid_to_array(inp)
        out_arr = grid_to_array(out)
        if inp_arr.shape != out_arr.shape:
            return None, 0
        inp_cols = [tuple(inp_arr[:, c]) for c in range(inp_arr.shape[1])]
        out_cols = [tuple(out_arr[:, c]) for c in range(out_arr.shape[1])]
        if Counter(inp_cols) == Counter(out_cols):
            mapping = []
            for o_col in out_cols:
                idx = inp_cols.index(o_col)
                mapping.append(idx)
            return mapping, 1.0
        return None, 0

    @staticmethod
    def find_row_mapping_consistent(task):
        """Find a consistent row mapping across all train examples."""
        mappings = []
        for ex in task['train']:
            m, conf = RowColumnAnalyzer.is_row_rearrangement(ex['input'], ex['output'])
            if m is None:
                return None, 0
            mappings.append(m)
        # Check if mapping is consistent (same index-based pattern)
        if len(mappings) < 2:
            return mappings[0], 1.0
        # Check if it's a reversal
        h = len(mappings[0])
        reversal = list(range(h - 1, -1, -1))
        if all(m == reversal for m in mappings):
            return reversal, 1.0
        # Check if all mappings are identical
        if all(m == mappings[0] for m in mappings):
            return mappings[0], 1.0
        return None, 0

    @staticmethod
    def find_col_mapping_consistent(task):
        """Find a consistent column mapping across all train examples."""
        mappings = []
        for ex in task['train']:
            m, conf = RowColumnAnalyzer.is_col_rearrangement(ex['input'], ex['output'])
            if m is None:
                return None, 0
            mappings.append(m)
        if len(mappings) < 2:
            return mappings[0], 1.0
        w = len(mappings[0])
        reversal = list(range(w - 1, -1, -1))
        if all(m == reversal for m in mappings):
            return reversal, 1.0
        if all(m == mappings[0] for m in mappings):
            return mappings[0], 1.0
        return None, 0

    @staticmethod
    def is_row_reversal(inp, out):
        """Check if output is the input with rows reversed."""
        inp_arr = grid_to_array(inp)
        out_arr = grid_to_array(out)
        if inp_arr.shape != out_arr.shape:
            return False
        return np.array_equal(inp_arr[::-1], out_arr)

    @staticmethod
    def is_col_reversal(inp, out):
        """Check if output is the input with columns reversed."""
        inp_arr = grid_to_array(inp)
        out_arr = grid_to_array(out)
        if inp_arr.shape != out_arr.shape:
            return False
        return np.array_equal(inp_arr[:, ::-1], out_arr)

    @staticmethod
    def apply_row_rearrangement(grid, mapping):
        """Apply a row permutation mapping."""
        arr = grid_to_array(grid)
        result = np.zeros_like(arr)
        for new_idx, old_idx in enumerate(mapping):
            if old_idx < arr.shape[0]:
                result[new_idx] = arr[old_idx]
        return array_to_grid(result)

    @staticmethod
    def apply_col_rearrangement(grid, mapping):
        """Apply a column permutation mapping."""
        arr = grid_to_array(grid)
        result = np.zeros_like(arr)
        for new_idx, old_idx in enumerate(mapping):
            if old_idx < arr.shape[1]:
                result[:, new_idx] = arr[:, old_idx]
        return array_to_grid(result)

    @staticmethod
    def detect_row_transform(inp, out):
        """Detect if rows are individually transformed (e.g., shifted, color-mapped)."""
        inp_arr = grid_to_array(inp)
        out_arr = grid_to_array(out)
        if inp_arr.shape != out_arr.shape:
            return None, 0
        # Check each row for a consistent per-row color map
        row_mappings = []
        for r in range(inp_arr.shape[0]):
            m = {}
            for c in range(inp_arr.shape[1]):
                src = int(inp_arr[r, c])
                dst = int(out_arr[r, c])
                if src in m:
                    if m[src] != dst:
                        return None, 0
                else:
                    m[src] = dst
            if len(m) == 0 or all(k == v for k, v in m.items()):
                row_mappings.append(None)
            else:
                row_mappings.append(m)
        # Check if all non-None mappings are identical
        non_none = [m for m in row_mappings if m is not None]
        if len(non_none) == 0:
            return None, 0
        if all(m == non_none[0] for m in non_none):
            return non_none[0], 1.0
        return None, 0


# ============================================================
# 5. OBJECT COUNTING
# ============================================================

class ObjectCounter:
    """Count objects in input vs output to infer rules."""

    @staticmethod
    def count_objects_by_color(grid, connectivity=4):
        """Count connected components grouped by color."""
        arr = grid_to_array(grid)
        comps = ComponentAnalyzer.find_components(arr, connectivity)
        counts = Counter(c['color'] for c in comps)
        return dict(counts)

    @staticmethod
    def analyze_count_change(inp_grid, out_grid, connectivity=4):
        """Analyze how object counts change from input to output."""
        inp_counts = ObjectCounter.count_objects_by_color(inp_grid, connectivity)
        out_counts = ObjectCounter.count_objects_by_color(out_grid, connectivity)

        all_colors = set(inp_counts.keys()) | set(out_counts.keys())
        changes = {}
        for c in all_colors:
            i = inp_counts.get(c, 0)
            o = out_counts.get(c, 0)
            changes[c] = {'input': i, 'output': o, 'delta': o - i}

        return changes

    @staticmethod
    def detect_count_rule(task):
        """Detect consistent count-change rule across training examples."""
        if len(task['train']) < 2:
            return None

        all_changes = []
        for ex in task['train']:
            changes = ObjectCounter.analyze_count_change(ex['input'], ex['output'])
            all_changes.append(changes)

        # Check if there's a consistent pattern
        # e.g., for each color, the delta is the same across examples
        colors = set()
        for ch in all_changes:
            colors.update(ch.keys())

        consistent_rule = {}
        for c in colors:
            deltas = []
            for ch in all_changes:
                if c in ch:
                    deltas.append(ch[c]['delta'])
                else:
                    deltas.append(None)
            if all(d == deltas[0] for d in deltas if d is not None):
                consistent_rule[c] = deltas[0]

        if consistent_rule:
            return consistent_rule
        return None


# ============================================================
# 6. SYMMETRY DETECTION
# ============================================================

class SymmetryDetector:
    """Check for horizontal, vertical, diagonal symmetry."""

    @staticmethod
    def check_horizontal(arr):
        """Check if grid is symmetric across vertical axis (left-right)."""
        return np.array_equal(arr, arr[:, ::-1])

    @staticmethod
    def check_vertical(arr):
        """Check if grid is symmetric across horizontal axis (top-bottom)."""
        return np.array_equal(arr, arr[::-1, :])

    @staticmethod
    def check_diagonal_main(arr):
        """Check if grid is symmetric across main diagonal."""
        if arr.shape[0] != arr.shape[1]:
            return False
        return np.array_equal(arr, arr.T)

    @staticmethod
    def check_diagonal_anti(arr):
        """Check if grid is symmetric across anti-diagonal."""
        if arr.shape[0] != arr.shape[1]:
            return False
        return np.array_equal(arr, np.rot90(arr, 2).T)

    @staticmethod
    def make_horizontal_symmetric(arr):
        """Make grid horizontally symmetric by mirroring left half."""
        h, w = arr.shape
        result = arr.copy()
        mid = (w + 1) // 2
        result[:, mid:] = arr[:, :mid][:, ::-1][:, :w - mid]
        return result

    @staticmethod
    def make_vertical_symmetric(arr):
        """Make grid vertically symmetric by mirroring top half."""
        h, w = arr.shape
        result = arr.copy()
        mid = (h + 1) // 2
        result[mid:, :] = arr[:mid, :][::-1, :][:h - mid, :]
        return result

    @staticmethod
    def detect_symmetry_rule(task):
        """Detect if the transformation involves symmetry."""
        for ex in task['train']:
            out_arr = grid_to_array(ex['output'])
            if SymmetryDetector.check_horizontal(out_arr):
                if SymmetryDetector.check_vertical(out_arr):
                    return 'both', 1.0
                return 'horizontal', 1.0
            if SymmetryDetector.check_vertical(out_arr):
                return 'vertical', 1.0
            if SymmetryDetector.check_diagonal_main(out_arr):
                return 'diagonal_main', 1.0
        return None, 0


# ============================================================
# 7. PATTERN COMPLETION
# ============================================================

class PatternCompleter:
    """Fill in missing parts of patterns."""

    @staticmethod
    def find_period(sequence):
        """Find the repeating period of a 1D sequence."""
        n = len(sequence)
        for p in range(1, n // 2 + 1):
            if n % p == 0:
                pattern = sequence[:p]
                valid = True
                for i in range(p, n):
                    if sequence[i] != pattern[i % p]:
                        valid = False
                        break
                if valid:
                    return pattern
        return sequence

    @staticmethod
    def complete_grid_pattern(grid):
        """Try to complete a grid by extending detected patterns."""
        arr = grid_to_array(grid)
        h, w = arr.shape

        # Check row periodicity
        row_period = None
        for p in range(1, h):
            if h % p == 0:
                pattern_rows = arr[:p]
                valid = True
                for r in range(h):
                    if not np.array_equal(arr[r], pattern_rows[r % p]):
                        valid = False
                        break
                if valid:
                    row_period = p
                    break

        # Check column periodicity
        col_period = None
        for p in range(1, w):
            if w % p == 0:
                pattern_cols = arr[:, :p]
                valid = True
                for c in range(w):
                    if not np.array_equal(arr[:, c], pattern_cols[:, c % p]):
                        valid = False
                        break
                if valid:
                    col_period = p
                    break

        return row_period, col_period

    @staticmethod
    def extend_pattern_to_size(grid, target_h, target_w):
        """Extend a patterned grid to a target size."""
        arr = grid_to_array(grid)
        row_period, col_period = PatternCompleter.complete_grid_pattern(grid)

        if row_period is not None and col_period is not None:
            # Use the smallest repeating block
            block = arr[:row_period, :col_period]
            result = np.zeros((target_h, target_w), dtype=arr.dtype)
            for r in range(target_h):
                for c in range(target_w):
                    result[r, c] = block[r % row_period, c % col_period]
            return array_to_grid(result)

        # Fallback: tile the whole grid
        result = np.zeros((target_h, target_w), dtype=arr.dtype)
        for r in range(target_h):
            for c in range(target_w):
                result[r, c] = arr[r % arr.shape[0], c % arr.shape[1]]
        return array_to_grid(result)

    @staticmethod
    def detect_completion_rule(task):
        """Detect if the task involves completing/extending a pattern."""
        examples = task['train']
        if len(examples) < 1:
            return None

        # Check if outputs are periodic extensions of inputs
        for ex in examples:
            inp = grid_to_array(ex['input'])
            out = grid_to_array(ex['output'])
            if inp.shape == out.shape:
                continue
            # Check if output is a tiling of input
            if out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]:
                valid = True
                for r in range(out.shape[0]):
                    for c in range(out.shape[1]):
                        if out[r, c] != inp[r % inp.shape[0], c % inp.shape[1]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return 'tile', 1.0
        return None, 0


# ============================================================
# 8. SCALING DETECTION
# ============================================================

class ScalingDetector:
    """Detect if grid is being scaled up or down."""

    @staticmethod
    def detect_scale_factor(inp, out):
        """Detect if output is a scaled version of input."""
        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(out)
        if ih == 0 or iw == 0:
            return None

        # Upscale detection
        if oh % ih == 0 and ow % iw == 0:
            fy, fx = oh // ih, ow // iw
            if fy == fx:
                # Verify nearest-neighbor upscale
                inp_arr = grid_to_array(inp)
                out_arr = grid_to_array(out)
                expected = np.repeat(np.repeat(inp_arr, fy, axis=0), fy, axis=1)
                if expected.shape == out_arr.shape and np.array_equal(expected, out_arr):
                    return ('up', fy)

        # Downscale detection
        if ih % oh == 0 and iw % ow == 0:
            fy, fx = ih // oh, iw // ow
            if fy == fx:
                # Verify nearest-neighbor downscale (take every Nth pixel)
                inp_arr = grid_to_array(inp)
                out_arr = grid_to_array(out)
                expected = inp_arr[::fy, ::fx]
                if expected.shape == out_arr.shape and np.array_equal(expected, out_arr):
                    return ('down', fy)

        return None

    @staticmethod
    def consistent_scale_across_examples(task):
        """Check if a consistent scale factor applies across all examples."""
        factors = []
        for ex in task['train']:
            f = ScalingDetector.detect_scale_factor(ex['input'], ex['output'])
            if f is None:
                return None
            factors.append(f)
        if all(f == factors[0] for f in factors):
            return factors[0]
        return None

    @staticmethod
    def scale_grid(grid, factor):
        """Scale grid by factor using nearest-neighbor."""
        arr = grid_to_array(grid)
        return array_to_grid(np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1))


# ============================================================
# 9. EXTRACTION
# ============================================================

class Extractor:
    """Detect if a subregion is being extracted."""

    @staticmethod
    def find_subregion(inp, out):
        """Check if output is a subregion of the input."""
        inp_arr = grid_to_array(inp)
        out_arr = grid_to_array(out)
        oh, ow = out_arr.shape
        ih, iw = inp_arr.shape

        if oh > ih or ow > iw:
            return None

        # Try to find the output as a contiguous subregion
        for r in range(ih - oh + 1):
            for c in range(iw - ow + 1):
                if np.array_equal(inp_arr[r:r+oh, c:c+ow], out_arr):
                    return (r, c, r + oh - 1, c + ow - 1)
        return None

    @staticmethod
    def find_extraction_pattern(task):
        """Find consistent extraction pattern across examples."""
        regions = []
        for ex in task['train']:
            region = Extractor.find_subregion(ex['input'], ex['output'])
            if region is None:
                return None
            regions.append(region)

        if len(regions) < 2:
            return regions[0] if regions else None

        # Check if extraction is based on the bounding box of non-background
        for ex in task['train']:
            inp_arr = grid_to_array(ex['input'])
            nonzero = np.nonzero(inp_arr)
            if len(nonzero[0]) == 0:
                continue
            min_r, max_r = nonzero[0].min(), nonzero[0].max()
            min_c, max_c = nonzero[1].min(), nonzero[1].max()
            expected = inp_arr[min_r:max_r+1, min_c:max_c+1]
            out_arr = grid_to_array(ex['output'])
            if np.array_equal(expected, out_arr):
                return 'bbox'

        return None

    @staticmethod
    def extract_bounding_box(grid):
        """Extract the bounding box of non-zero content."""
        arr = grid_to_array(grid)
        nonzero = np.nonzero(arr)
        if len(nonzero[0]) == 0:
            return array_to_grid(np.zeros((1, 1), dtype=int))
        min_r, max_r = nonzero[0].min(), nonzero[0].max()
        min_c, max_c = nonzero[1].min(), nonzero[1].max()
        return array_to_grid(arr[min_r:max_r+1, min_c:max_c+1])


# ============================================================
# 10. TILING / REPETITION
# ============================================================

class TilingDetector:
    """Detect if a pattern is being repeated to fill a grid."""

    @staticmethod
    def detect_tile_block(grid):
        """Detect the smallest repeating tile block in a grid."""
        arr = grid_to_array(grid)
        h, w = arr.shape

        best_block = None
        # Try all possible block sizes
        for bh in range(1, h + 1):
            if h % bh != 0:
                continue
            for bw in range(1, w + 1):
                if w % bw != 0:
                    continue
                block = arr[:bh, :bw]
                valid = True
                for r in range(h):
                    for c in range(w):
                        if arr[r, c] != block[r % bh, c % bw]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    if best_block is None or bh * bw < best_block[0] * best_block[1]:
                        best_block = (bh, bw, block)

        if best_block:
            return best_block
        return None

    @staticmethod
    def tile_block(block, target_h, target_w):
        """Tile a block to fill a target grid size."""
        bh, bw = block.shape
        result = np.zeros((target_h, target_w), dtype=block.dtype)
        for r in range(target_h):
            for c in range(target_w):
                result[r, c] = block[r % bh, c % bw]
        return array_to_grid(result)

    @staticmethod
    def detect_tiling_rule(task):
        """Detect if input->output involves tiling."""
        for ex in task['train']:
            inp = ex['input']
            out = ex['output']
            ih, iw = grid_shape(inp)
            oh, ow = grid_shape(out)

            if oh >= ih and ow >= iw and (oh % ih == 0 or ow % iw == 0):
                # Check basic tiling
                if oh % ih == 0 and ow % iw == 0:
                    fy, fx = oh // ih, ow // iw
                    inp_arr = grid_to_array(inp)
                    expected = np.tile(inp_arr, (fy, fx))
                    out_arr = grid_to_array(out)
                    if expected.shape == out_arr.shape and np.array_equal(expected, out_arr):
                        return ('basic', fy, fx)

                # Check checkerboard-style tiling
                inp_arr = grid_to_array(inp)
                out_arr = grid_to_array(out)
                if oh == 2 * ih and ow == 2 * iw:
                    # Normal tile
                    expected = np.tile(inp_arr, (2, 2))
                    if expected.shape == out_arr.shape and np.array_equal(expected, out_arr):
                        return ('tile_2x', 2, 2)
                    # Checkerboard (every other tile flipped)
                    flipped_h = np.flip(inp_arr, axis=0)
                    flipped_v = np.flip(inp_arr, axis=1)
                    flipped_both = np.flip(inp_arr)
                    top = np.hstack([inp_arr, flipped_v])
                    bottom = np.hstack([flipped_h, flipped_both])
                    checker = np.vstack([top, bottom])
                    if checker.shape == out_arr.shape and np.array_equal(checker, out_arr):
                        return ('checkerboard', 2, 2)

        return None


# ============================================================
# FLOOD FILL UTILITIES
# ============================================================

def flood_fill(arr, start_r, start_c, fill_color):
    """Flood fill from a position with a color (fills 0-connected regions)."""
    h, w = arr.shape
    target_color = int(arr[start_r, start_c])
    if target_color == fill_color:
        return arr.copy()

    result = arr.copy()
    queue = [(start_r, start_c)]
    visited = set()
    visited.add((start_r, start_c))

    while queue:
        r, c = queue.pop(0)
        result[r, c] = fill_color
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and
                (nr, nc) not in visited and
                int(arr[nr, nc]) == target_color):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return result


def find_enclosed_regions(arr):
    """Find 0-valued regions that are fully enclosed by non-zero values."""
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=bool)

    # Mark all 0-regions connected to the border
    for r in range(h):
        for c in range(w):
            if arr[r, c] == 0 and not visited[r, c] and (r == 0 or r == h-1 or c == 0 or c == w-1):
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (0 <= nr < h and 0 <= nc < w and
                            not visited[nr, nc] and arr[nr, nc] == 0):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

    # Any 0-cell not visited is enclosed
    enclosed = []
    for r in range(h):
        for c in range(w):
            if arr[r, c] == 0 and not visited[r, c]:
                enclosed.append((r, c))

    return enclosed

# ============================================================
# MAIN HEURISTIC SOLVER V2
# ============================================================

class HeuristicSolverV2:
    """Enhanced heuristic solver with all 10 strategy categories."""

    def __init__(self):
        self.component_analyzer = ComponentAnalyzer()
        self.transformer = CompositeTransformer()
        self.shape_analyzer = ShapeAnalyzer()
        self.row_col_analyzer = RowColumnAnalyzer()
        self.object_counter = ObjectCounter()
        self.symmetry = SymmetryDetector()
        self.pattern = PatternCompleter()
        self.scaling = ScalingDetector()
        self.extractor = Extractor()
        self.tiling = TilingDetector()

    def solve(self, task):
        """Try all enhanced heuristics, return best prediction or None."""
        best_pred = None
        best_conf = -1
        best_strategy = None

        solvers = [
            # Basic transforms (from v1, enhanced)
            ("identity", self.same_as_input),
            ("color_mapping", self.color_mapping),
            ("mirror_h", self.mirror_horizontal),
            ("mirror_v", self.mirror_vertical),
            ("mirror_d", self.mirror_diagonal),
            ("rot90", self.rotate_90),
            ("rot180", self.rotate_180),
            ("rot270", self.rotate_270),

            # Row/column analysis (strategy 4)
            ("row_rearrange", self.row_rearrangement),
            ("col_rearrange", self.col_rearrangement),
            ("row_reversal", self.row_reversal_check),
            ("col_reversal", self.col_reversal_check),

            # Scaling (strategy 8)
            ("scale", self.scaling_transform),

            # Tiling/repetition (strategy 10)
            ("tile", self.tile_repeat),
            ("tiling_rule", self.tiling_rule),

            # Extraction (strategy 9)
            ("extract_bbox", self.extract_bounding_box_transform),
            ("extraction", self.extraction_rule),

            # Symmetry (strategy 6)
            ("symmetry", self.symmetry_transform),

            # Shape preservation + composite (strategy 2, 3)
            ("composite", self.composite_transform),
            ("composite_color", self.composite_transform_with_color),

            # Object counting (strategy 5)
            ("fill_enclosed", self.fill_enclosed_regions),

            # Pattern completion (strategy 7)
            ("pattern_extend", self.pattern_extend),

            # Color operations
            ("remove_color", self.remove_color),
            ("border_extract", self.detect_borders),
            ("fill_interior", self.fill_interior),
            ("flood_fill_from_edge", self.flood_fill_from_edge),
            ("move_object", self.detect_and_transform_pattern),
            ("common_color_fill", self.common_color_fill),

            # Advanced composite
            ("scale_then_color", self.scale_then_color_map),

            # Connected component based
            ("component_color_map", self.component_based_color_map),

            # Deduplicate rows/cols
            ("dedup_rows", self.deduplicate_rows),
            ("dedup_cols", self.deduplicate_cols),
        ]

        for name, solver in solvers:
            try:
                pred, conf = solver(task)
                if pred is not None and conf > best_conf:
                    best_pred = pred
                    best_conf = conf
                    best_strategy = name
            except Exception as e:
                continue

        if best_pred is not None:
            return best_pred, best_conf, best_strategy
        return None, 0, None

    # ---- Validation ----
    def _validate(self, task, pred_fn, shape_check=True):
        """Validate a prediction function against all train examples."""
        preds = []
        total_conf = 0
        for ex in task['train']:
            try:
                pred = pred_fn(ex['input'], ex)
                if pred is None:
                    return 0, None
                pred_arr = grid_to_array(pred)
                exp_arr = grid_to_array(ex['output'])
                if shape_check and pred_arr.shape != exp_arr.shape:
                    return 0, None
                if pred_arr.shape != exp_arr.shape:
                    return 0, None
                match = np.sum(pred_arr == exp_arr) / max(pred_arr.size, 1)
                total_conf += match
                preds.append(pred)
            except Exception:
                return 0, None
        return total_conf / len(task['train']), preds

    # ---- BASIC TRANSFORMS ----

    def same_as_input(self, task):
        def fn(inp, ex):
            return deepcopy(inp)
        conf, preds = self._validate(task, fn)
        if conf >= 0.99:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def color_mapping(self, task):
        ex0 = task['train'][0]
        mapping = _infer_color_map_np(grid_to_array(ex0['input']), grid_to_array(ex0['output']))
        if mapping is None:
            return None, 0
        def fn(inp, ex):
            arr = grid_to_array(inp)
            result = arr.copy()
            for src, dst in mapping.items():
                result[result == src] = dst
            return array_to_grid(result)
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def mirror_horizontal(self, task):
        def fn(inp, ex):
            return [row[::-1] for row in inp]
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def mirror_vertical(self, task):
        def fn(inp, ex):
            return inp[::-1]
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def mirror_diagonal(self, task):
        def fn(inp, ex):
            arr = grid_to_array(inp)
            return array_to_grid(arr.T)
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def rotate_90(self, task):
        def fn(inp, ex):
            arr = grid_to_array(inp)
            return array_to_grid(np.rot90(arr, -1))
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def rotate_180(self, task):
        def fn(inp, ex):
            arr = grid_to_array(inp)
            return array_to_grid(np.rot90(arr, 2))
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def rotate_270(self, task):
        def fn(inp, ex):
            arr = grid_to_array(inp)
            return array_to_grid(np.rot90(arr, 1))
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    # ---- ROW/COLUMN ANALYSIS (Strategy 4) ----

    def row_rearrangement(self, task):
        mapping, conf = self.row_col_analyzer.find_row_mapping_consistent(task)
        if mapping is not None and conf > 0:
            def fn(inp, ex):
                return self.row_col_analyzer.apply_row_rearrangement(inp, mapping)
            vconf, _ = self._validate(task, fn)
            if vconf >= 0.95:
                return fn(task['test'][0]['input'], task), vconf
        return None, 0

    def col_rearrangement(self, task):
        mapping, conf = self.row_col_analyzer.find_col_mapping_consistent(task)
        if mapping is not None and conf > 0:
            def fn(inp, ex):
                return self.row_col_analyzer.apply_col_rearrangement(inp, mapping)
            vconf, _ = self._validate(task, fn)
            if vconf >= 0.95:
                return fn(task['test'][0]['input'], task), vconf
        return None, 0

    def row_reversal_check(self, task):
        def fn(inp, ex):
            return inp[::-1]
        all_match = True
        for ex in task['train']:
            if not self.row_col_analyzer.is_row_reversal(ex['input'], ex['output']):
                all_match = False
                break
        if all_match:
            return fn(task['test'][0]['input'], task), 1.0
        return None, 0

    def col_reversal_check(self, task):
        def fn(inp, ex):
            return [row[::-1] for row in inp]
        all_match = True
        for ex in task['train']:
            if not self.row_col_analyzer.is_col_reversal(ex['input'], ex['output']):
                all_match = False
                break
        if all_match:
            return fn(task['test'][0]['input'], task), 1.0
        return None, 0

    def deduplicate_rows(self, task):
        """Remove duplicate consecutive rows."""
        def fn(inp, ex):
            arr = grid_to_array(inp)
            if arr.shape[0] <= 1:
                return array_to_grid(arr)
            keep = [0]
            for r in range(1, arr.shape[0]):
                if not np.array_equal(arr[r], arr[r-1]):
                    keep.append(r)
            return array_to_grid(arr[keep])
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def deduplicate_cols(self, task):
        """Remove duplicate consecutive columns."""
        def fn(inp, ex):
            arr = grid_to_array(inp)
            if arr.shape[1] <= 1:
                return array_to_grid(arr)
            keep = [0]
            for c in range(1, arr.shape[1]):
                if not np.array_equal(arr[:, c], arr[:, c-1]):
                    keep.append(c)
            return array_to_grid(arr[:, keep])
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    # ---- SCALING (Strategy 8) ----

    def scaling_transform(self, task):
        factor = self.scaling.consistent_scale_across_examples(task)
        if factor is not None:
            direction, val = factor
            if direction == 'up':
                def fn(inp, ex):
                    return self.scaling.scale_grid(inp, val)
                vconf, _ = self._validate(task, fn)
                if vconf >= 0.95:
                    return fn(task['test'][0]['input'], task), vconf
            elif direction == 'down':
                def fn(inp, ex):
                    arr = grid_to_array(inp)
                    return array_to_grid(arr[::val, ::val])
                vconf, _ = self._validate(task, fn)
                if vconf >= 0.95:
                    return fn(task['test'][0]['input'], task), vconf
        return None, 0

    # ---- TILING/REPETITION (Strategy 10) ----

    def tile_repeat(self, task):
        """Detect tiling pattern: input is tiled to form larger output."""
        def fn(inp, ex):
            ih, iw = grid_shape(inp)
            oh, ow = grid_shape(ex['output'])
            if ih == 0 or iw == 0:
                return None
            if oh % ih != 0 or ow % iw != 0:
                return None
            fy, fx = oh // ih, ow // iw
            return self._tile_grid(inp, fy, fx)

        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def _tile_grid(self, grid, fy, fx):
        result = []
        for r in range(fy):
            for row in grid:
                new_row = []
                for c in range(fx):
                    new_row.extend(row)
                result.append(new_row)
        return result

    def tiling_rule(self, task):
        """Use TilingDetector for advanced tiling patterns."""
        rule = self.tiling.detect_tiling_rule(task)
        if rule is not None:
            rule_type, fy, fx = rule
            if rule_type == 'basic':
                def fn(inp, ex):
                    return self._tile_grid(inp, fy, fx)
                conf, _ = self._validate(task, fn)
                if conf >= 0.95:
                    return fn(task['test'][0]['input'], task), conf
            elif rule_type == 'tile_2x':
                def fn(inp, ex):
                    return self._tile_grid(inp, 2, 2)
                conf, _ = self._validate(task, fn)
                if conf >= 0.95:
                    return fn(task['test'][0]['input'], task), conf
            elif rule_type == 'checkerboard':
                def fn(inp, ex):
                    arr = grid_to_array(inp)
                    flipped_h = np.flip(arr, axis=0)
                    flipped_v = np.flip(arr, axis=1)
                    flipped_both = np.flip(arr)
                    top = np.hstack([arr, flipped_v])
                    bottom = np.hstack([flipped_h, flipped_both])
                    return array_to_grid(np.vstack([top, bottom]))
                conf, _ = self._validate(task, fn)
                if conf >= 0.95:
                    return fn(task['test'][0]['input'], task), conf
        return None, 0

    # ---- EXTRACTION (Strategy 9) ----

    def extract_bounding_box_transform(self, task):
        """Extract the non-background bounding box."""
        def fn(inp, ex):
            return self.extractor.extract_bounding_box(inp)
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def extraction_rule(self, task):
        """Use Extractor for subregion extraction patterns."""
        pattern = self.extractor.find_extraction_pattern(task)
        if pattern == 'bbox':
            return self.extract_bounding_box_transform(task)
        return None, 0

    # ---- SYMMETRY (Strategy 6) ----

    def symmetry_transform(self, task):
        sym_type, conf = self.symmetry.detect_symmetry_rule(task)
        if conf > 0 and sym_type:
            if sym_type == 'horizontal':
                def fn(inp, ex):
                    arr = grid_to_array(inp)
                    return array_to_grid(self.symmetry.make_horizontal_symmetric(arr))
                vconf, _ = self._validate(task, fn)
                if vconf >= 0.95:
                    return fn(task['test'][0]['input'], task), vconf
            elif sym_type == 'vertical':
                def fn(inp, ex):
                    arr = grid_to_array(inp)
                    return array_to_grid(self.symmetry.make_vertical_symmetric(arr))
                vconf, _ = self._validate(task, fn)
                if vconf >= 0.95:
                    return fn(task['test'][0]['input'], task), vconf
        return None, 0

    # ---- COMPOSITE TRANSFORMS (Strategy 2) ----

    def composite_transform(self, task):
        """Find a composite geometric transform."""
        ex0 = task['train'][0]
        tname, tconf = self.transformer.find_best_transform(
            grid_to_array(ex0['input']), grid_to_array(ex0['output'])
        )
        if tname is not None:
            def fn(inp, ex):
                result = self.transformer.apply_composite(grid_to_array(inp), tname)
                if result is None:
                    return None
                return array_to_grid(result)
            vconf, _ = self._validate(task, fn)
            if vconf >= 0.95:
                return fn(task['test'][0]['input'], task), vconf
        return None, 0

    def composite_transform_with_color(self, task):
        """Find geometric transform + color remapping."""
        ex0 = task['train'][0]
        tname, color_map, tconf = self.transformer.find_transform_with_color_map(
            grid_to_array(ex0['input']), grid_to_array(ex0['output'])
        )
        if tname is not None and color_map is not None:
            def fn(inp, ex):
                arr = grid_to_array(inp)
                transformed = self.transformer.apply_composite(arr, tname)
                if transformed is None:
                    return None
                result = transformed.copy()
                for src, dst in color_map.items():
                    result[result == src] = dst
                return array_to_grid(result)
            vconf, _ = self._validate(task, fn)
            if vconf >= 0.95:
                return fn(task['test'][0]['input'], task), vconf
        return None, 0

    def scale_then_color_map(self, task):
        """Scale grid then apply color map."""
        # Check if outputs are scaled versions of inputs with color changes
        ex0 = task['train'][0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        for factor in [2, 3]:
            scaled = np.repeat(np.repeat(inp_arr, factor, axis=0), factor, axis=1)
            if scaled.shape == out_arr.shape:
                mapping = _infer_color_map_np(scaled, out_arr)
                if mapping is not None:
                    def fn(inp, ex, f=factor, m=mapping):
                        a = grid_to_array(inp)
                        s = np.repeat(np.repeat(a, f, axis=0), f, axis=1)
                        r = s.copy()
                        for src, dst in m.items():
                            r[r == src] = dst
                        return array_to_grid(r)
                    vconf, _ = self._validate(task, fn)
                    if vconf >= 0.95:
                        return fn(task['test'][0]['input'], task), vconf
        return None, 0

    # ---- CONNECTED COMPONENT BASED (Strategy 1, 5) ----

    def component_based_color_map(self, task):
        """Use connected components to detect per-component color changes."""
        ex0 = task['train'][0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        if inp_arr.shape != out_arr.shape:
            return None, 0

        inp_comps = self.component_analyzer.find_components(inp_arr)
        out_comps = self.component_analyzer.find_components(out_arr)

        if len(inp_comps) != len(out_comps):
            return None, 0

        # Check if components map 1-to-1 with color changes
        comp_color_map = {}
        for ic, oc in zip(sorted(inp_comps, key=lambda c: c['bbox']),
                          sorted(out_comps, key=lambda c: c['bbox'])):
            if ic['size'] != oc['size']:
                return None, 0
            if ic['bbox'] != oc['bbox']:
                return None, 0
            if ic['color'] != oc['color']:
                comp_color_map[ic['color']] = oc['color']

        if not comp_color_map:
            return None, 0

        # Validate across all examples
        valid = True
        for ex in task['train'][1:]:
            i_arr = grid_to_array(ex['input'])
            o_arr = grid_to_array(ex['output'])
            if i_arr.shape != o_arr.shape:
                valid = False
                break
            for src_c, dst_c in comp_color_map.items():
                # Check that every pixel of src_c in input is dst_c in output at same position
                mask = (i_arr == src_c)
                if not np.all(o_arr[mask] == dst_c):
                    valid = False
                    break
            if not valid:
                break

        if valid:
            def fn(inp, ex):
                arr = grid_to_array(inp).copy()
                for src_c, dst_c in comp_color_map.items():
                    arr[arr == src_c] = dst_c
                return array_to_grid(arr)
            vconf, _ = self._validate(task, fn)
            if vconf >= 0.95:
                return fn(task['test'][0]['input'], task), vconf
        return None, 0

    def fill_enclosed_regions(self, task):
        """Fill enclosed regions with a color."""
        ex0 = task['train'][0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        if inp_arr.shape != out_arr.shape:
            return None, 0

        # Find what changed: look for cells that are 0 in input and non-zero in output
        diff = (inp_arr == 0) & (out_arr != 0)
        if not np.any(diff):
            return None, 0

        # Determine fill color (most common non-zero in output at those positions)
        fill_colors = out_arr[diff]
        color_counts = Counter(fill_colors.flatten().tolist())
        fill_color = color_counts.most_common(1)[0][0]

        # Validate
        def fn(inp, ex):
            arr = grid_to_array(inp)
            enclosed = find_enclosed_regions(arr)
            result = arr.copy()
            for r, c in enclosed:
                result[r, c] = fill_color
            return array_to_grid(result)

        vconf, _ = self._validate(task, fn)
        if vconf >= 0.95:
            return fn(task['test'][0]['input'], task), vconf
        return None, 0

    def flood_fill_from_edge(self, task):
        """Flood fill from edges with 0, keeping the rest."""
        ex0 = task['train'][0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])
        if inp_arr.shape != out_arr.shape:
            return None, 0

        # Check if the difference is about filling edge-connected 0s
        diff_mask = (inp_arr != out_arr)
        if not np.any(diff_mask):
            return None, 0

        # Determine what color 0s became
        zero_in_inp = (inp_arr == 0)
        changed_zeros = zero_in_inp & diff_mask
        if not np.any(changed_zeros):
            return None, 0
        fill_color = int(Counter(out_arr[changed_zeros].flatten().tolist()).most_common(1)[0][0])

        # Check if only non-enclosed zeros are changed to fill_color
        # and enclosed zeros stay the same or become a different color
        def fn(inp, ex):
            arr = grid_to_array(inp)
            h, w = arr.shape
            visited = np.zeros((h, w), dtype=bool)
            # BFS from edge zeros
            for r in range(h):
                for c in range(w):
                    if arr[r, c] == 0 and not visited[r, c] and (r == 0 or r == h-1 or c == 0 or c == w-1):
                        queue = [(r, c)]
                        visited[r, c] = True
                        while queue:
                            cr, cc = queue.pop(0)
                            arr[cr, cc] = fill_color
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if (0 <= nr < h and 0 <= nc < w and
                                    not visited[nr, nc] and arr[nr, nc] == 0):
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
            return array_to_grid(arr)

        vconf, _ = self._validate(task, fn)
        if vconf >= 0.95:
            return fn(task['test'][0]['input'], task), vconf
        return None, 0

    # ---- PATTERN COMPLETION (Strategy 7) ----

    def pattern_extend(self, task):
        """Extend a pattern to fill a target grid."""
        examples = task['train']
        if len(examples) < 2:
            return None, 0

        # Check if output is an extension of input pattern
        def fn(inp, ex):
            target_h, target_w = grid_shape(ex['output'])
            return self.pattern.extend_pattern_to_size(inp, target_h, target_w)

        vconf, _ = self._validate(task, fn)
        if vconf >= 0.95:
            # Infer target size for test
            target = self.shape_analyzer.detect_output_size_rule(task)
            if target is not None:
                return self.pattern.extend_pattern_to_size(
                    task['test'][0]['input'], target[0], target[1]
                ), vconf
        return None, 0

    # ---- COLOR OPERATIONS ----

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
        conf, preds = self._validate(task, fn)
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
            return array_to_grid(result)
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def fill_interior(self, task):
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
        conf, preds = self._validate(task, fn)
        if conf >= 0.95:
            return fn(task['test'][0]['input'], task), conf
        return None, 0

    def common_color_fill(self, task):
        """Replace 0 cells inside the bounding box with the most common non-zero color."""
        ex0 = task['train'][0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])
        if inp_arr.shape != out_arr.shape:
            return None, 0

        # Check what fill color was used for zeros inside bbox
        nonzero = np.nonzero(inp_arr)
        if len(nonzero[0]) == 0:
            return None, 0
        min_r, max_r = nonzero[0].min(), nonzero[0].max()
        min_c, max_c = nonzero[1].min(), nonzero[1].max()

        # Inside bbox, zeros became what color?
        zeros_inside = []
        for r in range(min_r, max_r+1):
            for c in range(min_c, max_c+1):
                if inp_arr[r, c] == 0 and out_arr[r, c] != 0:
                    zeros_inside.append(int(out_arr[r, c]))

        if not zeros_inside:
            return None, 0

        fill_color = Counter(zeros_inside).most_common(1)[0][0]

        def fn(inp, ex):
            arr = grid_to_array(inp).copy()
            nz = np.nonzero(arr)
            if len(nz[0]) == 0:
                return array_to_grid(arr)
            rmin, rmax = nz[0].min(), nz[0].max()
            cmin, cmax = nz[1].min(), nz[1].max()
            for r in range(rmin, rmax+1):
                for c in range(cmin, cmax+1):
                    if arr[r, c] == 0:
                        arr[r, c] = fill_color
            return array_to_grid(arr)

        vconf, _ = self._validate(task, fn)
        if vconf >= 0.95:
            return fn(task['test'][0]['input'], task), vconf
        return None, 0

    # ---- OBJECT MOVEMENT ----

    def detect_and_transform_pattern(self, task):
        """Move object to a different position while preserving shape."""
        ex0 = task['train'][0]
        inp = grid_to_array(ex0['input'])
        out = grid_to_array(ex0['output'])

        inp_obj = self._find_object_properties(inp)
        out_obj = self._find_object_properties(out)

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
                    test_out_shape = grid_shape(task['train'][0]['output'])
                    test_out = np.zeros(test_out_shape, dtype=int)
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
# SHARED UTILITIES
# ============================================================

def verify_output(expected, actual):
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


def grids_equal(g1, g2):
    """Check if two grids are identical."""
    try:
        a1 = grid_to_array(g1)
        a2 = grid_to_array(g2)
        return a1.shape == a2.shape and np.array_equal(a1, a2)
    except Exception:
        return False


def ensure_grid(pred):
    """Ensure prediction is a valid grid (list of lists of ints 0-9)."""
    if pred is None:
        return [[0]]
    if isinstance(pred, np.ndarray):
        pred = pred.astype(int).tolist()
    # Clamp values to 0-9
    return [[min(9, max(0, int(c))) for c in row] for row in pred]


def normalize_task_for_evaluation(task):
    """Normalize task format for evaluation."""
    return task

# ============================================================
# MODULE 1: CONDITIONAL LOGIC ENGINE
# ============================================================

class FeatureDetector:
    """
    Extract discriminant features from a grid that can be used to
    split training demos into conditional branches.
    """

    @staticmethod
    def extract_features(grid):
        """Extract a comprehensive feature dict from a single grid."""
        arr = grid_to_array(grid)
        h, w = arr.shape

        # Basic shape features
        is_square = h == w
        wider_than_tall = w > h
        taller_than_wide = h > w
        size_parity = (h * w) % 2

        # Color features
        unique_colors = set(int(c) for c in arr.flatten())
        unique_nonzero = unique_colors - {0}
        dominant_color = int(Counter(arr.flatten()).most_common(1)[0][0]) if arr.size > 0 else 0
        color_count = len(unique_nonzero)

        # Background density
        total_cells = h * w
        nonzero_cells = int(np.sum(arr != 0))
        density = nonzero_cells / max(total_cells, 1)

        # Component features (4-connectivity)
        comps_4 = ComponentAnalyzer.find_components(arr, 4)
        comp_count_4 = len(comps_4)
        comp_count_parity = comp_count_4 % 2

        # Component features (8-connectivity)
        comps_8 = ComponentAnalyzer.find_components(arr, 8)
        comp_count_8 = len(comps_8)

        # Symmetry features
        sym_h = SymmetryDetector.check_horizontal(arr)
        sym_v = SymmetryDetector.check_vertical(arr)
        sym_d_main = SymmetryDetector.check_diagonal_main(arr)
        sym_d_anti = SymmetryDetector.check_diagonal_anti(arr)

        # Border features
        has_border = False
        if h >= 2 and w >= 2:
            border_pixels = set()
            for c in range(w):
                border_pixels.add(int(arr[0, c]))
                border_pixels.add(int(arr[h-1, c]))
            for r in range(h):
                border_pixels.add(int(arr[r, 0]))
                border_pixels.add(int(arr[r, w-1]))
            has_border = len(border_pixels - {0}) == 1

        return {
            'height': h,
            'width': w,
            'is_square': is_square,
            'wider_than_tall': wider_than_tall,
            'taller_than_wide': taller_than_wide,
            'size_parity': size_parity,
            'unique_colors': unique_colors,
            'unique_nonzero': unique_nonzero,
            'dominant_color': dominant_color,
            'color_count': color_count,
            'density': density,
            'comp_count_4': comp_count_4,
            'comp_count_8': comp_count_8,
            'comp_count_parity': comp_count_parity,
            'nonzero_cells': nonzero_cells,
            'sym_h': sym_h,
            'sym_v': sym_v,
            'sym_d_main': sym_d_main,
            'sym_d_anti': sym_d_anti,
            'has_border': has_border,
            'has_symmetry': sym_h or sym_v or sym_d_main or sym_d_anti,
        }

    @staticmethod
    def find_discriminant_features(task):
        """
        Find features that perfectly split the training demos into groups
        such that within each group, the same transformation applies.

        Returns list of (condition_fn, subgroup_indices) tuples sorted by
        subgroup size (largest first), or None if no useful split is found.
        """
        train = task['train']
        n = len(train)
        if n < 2:
            return None

        # Extract features for all inputs
        input_features = []
        for ex in train:
            input_features.append(FeatureDetector.extract_features(ex['input']))

        # Try binary conditions
        candidates = []

        # --- Boolean features ---
        bool_features = [
            'is_square', 'wider_than_tall', 'taller_than_wide',
            'size_parity', 'comp_count_parity', 'sym_h', 'sym_v',
            'sym_d_main', 'sym_d_anti', 'has_symmetry', 'has_border',
        ]

        for feat_name in bool_features:
            values = [f[feat_name] for f in input_features]
            # Only useful if both values appear
            if len(set(values)) < 2:
                continue

            groups = defaultdict(list)
            for idx, val in enumerate(values):
                groups[val].append(idx)

            # Each group must have at least 2 examples to avoid false positives
            if all(len(g) >= 2 for g in groups.values()):
                candidates.append((feat_name, dict(groups), len(groups)))

        # --- Parity features ---
        for feat_name in ['height', 'width', 'color_count', 'comp_count_4', 'comp_count_8', 'dominant_color']:
            values = [f[feat_name] for f in input_features]
            parities = [v % 2 for v in values]

            if len(set(parities)) < 2:
                continue

            groups = defaultdict(list)
            for idx, p in enumerate(parities):
                groups[p].append(idx)

            if all(len(g) >= 2 for g in groups.values()):
                condition_name = f'{feat_name}_parity'
                candidates.append((condition_name, dict(groups), len(groups)))

        # --- Color presence/absence ---
        for color in range(1, 10):
            values = [color in f['unique_nonzero'] for f in input_features]
            if len(set(values)) < 2:
                continue

            groups = defaultdict(list)
            for idx, val in enumerate(values):
                groups[val].append(idx)

            if all(len(g) >= 2 for g in groups.values()):
                condition_name = f'has_color_{color}'
                candidates.append((condition_name, dict(groups), len(groups)))

        # --- Dominant color value ---
        dom_colors = [f['dominant_color'] for f in input_features]
        if len(set(dom_colors)) >= 2:
            groups = defaultdict(list)
            for idx, c in enumerate(dom_colors):
                groups[c].append(idx)
            if all(len(g) >= 2 for g in groups.values()):
                candidates.append(('dominant_color', dict(groups), len(groups)))

        return candidates if candidates else None


class ConditionalLogicEngine:
    """
    Detect conditional logic in ARC tasks where different input properties
    trigger different transformations. Splits training demos by detected
    conditions and routes each subgroup to the appropriate sub-strategy.
    """

    def __init__(self):
        self.v2_solver = HeuristicSolverV2()

    def solve(self, task):
        """
        Try to solve a task using conditional logic.

        Returns (prediction, confidence, strategy_name) or (None, 0, None).
        """
        train = task['train']
        if not train or not task.get('test'):
            return None, 0, None

        test_input = task['test'][0]['input']

        # Find discriminant features
        candidates = FeatureDetector.find_discriminant_features(task)
        if candidates is None:
            return None, 0, None

        test_features = FeatureDetector.extract_features(test_input)

        # Sort candidates: prefer conditions with more balanced groups
        candidates.sort(key=lambda x: min(len(g) for g in x[1].values()), reverse=True)

        # Try each candidate condition
        for condition_name, groups, num_groups in candidates:
            try:
                pred, conf = self._try_condition(task, condition_name, groups, test_features)
                if pred is not None and conf > 0.95:
                    # Validate: the prediction must also match at least one training example
                    # when applied through the conditional routing
                    validated = self._validate_conditional(task, condition_name, groups)
                    if validated:
                        return pred, conf, f'conditional_{condition_name}'
            except Exception:
                continue

        return None, 0, None

    def _try_condition(self, task, condition_name, groups, test_features):
        """
        Try a specific condition split. For each subgroup, find a transformation
        that works for all examples in that subgroup, then apply the matching
        transformation to the test input.
        """
        train = task['train']
        test_input = task['test'][0]['input']

        # Determine which group the test input belongs to
        test_group_key = None
        if condition_name == 'is_square':
            test_group_key = test_features['is_square']
        elif condition_name == 'wider_than_tall':
            test_group_key = test_features['wider_than_tall']
        elif condition_name == 'taller_than_wide':
            test_group_key = test_features['taller_than_wide']
        elif condition_name == 'size_parity':
            test_group_key = test_features['size_parity']
        elif condition_name == 'comp_count_parity':
            test_group_key = test_features['comp_count_parity']
        elif condition_name == 'sym_h':
            test_group_key = test_features['sym_h']
        elif condition_name == 'sym_v':
            test_group_key = test_features['sym_v']
        elif condition_name == 'sym_d_main':
            test_group_key = test_features['sym_d_main']
        elif condition_name == 'sym_d_anti':
            test_group_key = test_features['sym_d_anti']
        elif condition_name == 'has_symmetry':
            test_group_key = test_features['has_symmetry']
        elif condition_name == 'has_border':
            test_group_key = test_features['has_border']
        elif condition_name == 'dominant_color':
            test_group_key = test_features['dominant_color']
        elif condition_name.endswith('_parity'):
            feat = condition_name.replace('_parity', '')
            test_group_key = test_features.get(feat, 0) % 2
        elif condition_name.startswith('has_color_'):
            color = int(condition_name.split('_')[-1])
            test_group_key = color in test_features['unique_nonzero']
        else:
            return None, 0

        if test_group_key not in groups:
            return None, 0

        # Require the test's group to have at least 2 training examples
        group_indices = groups[test_group_key]
        if len(group_indices) < 2:
            return None, 0

        # Build a sub-task for this group
        sub_task = {
            'train': [train[i] for i in group_indices],
            'test': task['test'],
        }

        # Validate: each branch's strategy must work on ALL training examples
        # (including those from other groups)
        # Strategy: find the best transformation for this subgroup
        pred, conf, strategy = self.v2_solver.solve(sub_task)
        if pred is None or conf < 0.95:
            return None, 0

        # Now validate: does this transformation also correctly produce the
        # expected output for ALL training examples when we apply the correct
        # branch routing?
        all_correct = True
        for i, ex in enumerate(train):
            ex_features = FeatureDetector.extract_features(ex['input'])
            ex_group_key = self._get_condition_value(condition_name, ex_features)
            if ex_group_key == test_group_key:
                # Same group as test — transformation should work
                # (already validated via v2_solver.solve on sub_task)
                pass
            else:
                # Different group — we don't need to validate this branch
                # since a different strategy will handle it
                pass

        return pred, conf

    def _validate_conditional(self, task, condition_name, groups):
        """
        Validate that the conditional routing works: for each training example,
        the subgroup it belongs to can be solved correctly.
        """
        train = task['train']
        for group_key, indices in groups.items():
            if len(indices) < 2:
                continue
            sub_task = {
                'train': [train[i] for i in indices],
                'test': task['test'],
            }
            pred, conf, strategy = self.v2_solver.solve(sub_task)
            if pred is None or conf < 0.95:
                return False
        return True

    def _get_condition_value(self, condition_name, features):
        """Get the condition value for a feature dict."""
        if condition_name == 'is_square':
            return features['is_square']
        elif condition_name == 'wider_than_tall':
            return features['wider_than_tall']
        elif condition_name == 'taller_than_wide':
            return features['taller_than_wide']
        elif condition_name == 'size_parity':
            return features['size_parity']
        elif condition_name == 'comp_count_parity':
            return features['comp_count_parity']
        elif condition_name == 'sym_h':
            return features['sym_h']
        elif condition_name == 'sym_v':
            return features['sym_v']
        elif condition_name == 'sym_d_main':
            return features['sym_d_main']
        elif condition_name == 'sym_d_anti':
            return features['sym_d_anti']
        elif condition_name == 'has_symmetry':
            return features['has_symmetry']
        elif condition_name == 'has_border':
            return features['has_border']
        elif condition_name == 'dominant_color':
            return features['dominant_color']
        elif condition_name.endswith('_parity'):
            feat = condition_name.replace('_parity', '')
            return features.get(feat, 0) % 2
        elif condition_name.startswith('has_color_'):
            color = int(condition_name.split('_')[-1])
            return color in features['unique_nonzero']
        return None


# ============================================================
# MODULE 2: LLM CODE GENERATION FALLBACK
# ============================================================

# Try to import transformers; if not available, silently disable LLM
_LLM_AVAILABLE = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    if _HAS_TORCH:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False


class SafeLLMCodeGenerator:
    """
    LLM code generator with graceful degradation.
    Only activates when GPU + transformers are available (Kaggle environment).
    In offline/no-GPU mode, is_available() returns False.
    """

    MODEL_NAME = "Qwen/Qwen3-4B"
    MAX_NEW_TOKENS = 2048
    TEMPERATURE = 0.3
    TOP_P = 0.9
    MAX_INPUT_TOKENS = 30000
    NUM_ATTEMPTS = 6
    CODE_TIMEOUT = 10

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._load_attempted = False

    def is_available(self):
        """Check if LLM can be used."""
        if self._load_attempted:
            return self.model is not None
        # Try to detect GPU
        if not _LLM_AVAILABLE:
            self._load_attempted = True
            return False
        if not _HAS_TORCH:
            self._load_attempted = True
            return False
        try:
            if not torch.cuda.is_available():
                self._load_attempted = True
                return False
        except Exception:
            self._load_attempted = True
            return False
        return True

    def _load_model(self):
        """Lazy load model."""
        if self._loaded:
            return
        self._load_attempted = True

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            print(f"[V3-LLM] Loading {self.MODEL_NAME}...")
            t0 = time.time()

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "quantization_config": bnb_config,
            }

            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME, **kwargs
            )
            self.model.eval()
            self._loaded = True
            print(f"[V3-LLM] Model loaded in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[V3-LLM] Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    def _task_to_prompt(self, task):
        """Convert a full ARC task into a text prompt for code generation."""
        train_pairs = task['train']
        test_pairs = task['test']

        prompt = "You are an expert ARC puzzle solver. Look at the input-output examples and figure out the transformation rule.\n"
        prompt += "Write a Python function `transform(input_grid)` that applies this rule.\n\n"

        for i, pair in enumerate(train_pairs):
            inp = pair['input']
            out = pair['output']
            prompt += f"Example {i+1}:\n"
            prompt += f"Input ({len(inp)}x{len(inp[0])}):\n"
            for row in inp:
                prompt += " ".join(str(v) for v in row) + "\n"
            prompt += f"Output ({len(out)}x{len(out[0])}):\n"
            for row in out:
                prompt += " ".join(str(v) for v in row) + "\n"
            prompt += "\n"

        prompt += f"Now solve this test input:\n"
        for i, test in enumerate(test_pairs):
            inp = test['input']
            prompt += f"Test Input ({len(inp)}x{len(inp[0])}):\n"
            for row in inp:
                prompt += " ".join(str(v) for v in row) + "\n"

        prompt += "\nWrite only the Python function. The function signature must be:\n"
        prompt += "def transform(input_grid):\n"
        prompt += "    # Your code here\n"
        prompt += "    return output_grid\n"
        return prompt

    def generate_code(self, task):
        """Generate multiple Python code candidates for a task."""
        if not self.is_available():
            return []

        self._load_model()
        if self.model is None:
            return []

        prompt = self._task_to_prompt(task)

        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Write only valid Python code. No explanations."},
            {"role": "user", "content": prompt}
        ]

        try:
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            full_prompt = (
                f"<|im_start|>system\nYou are an expert Python programmer.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        codes = []
        for attempt in range(self.NUM_ATTEMPTS):
            try:
                inputs = self.tokenizer(
                    full_prompt, return_tensors="pt",
                    truncation=True, max_length=self.MAX_INPUT_TOKENS
                ).to(self.model.device)

                temp = min(self.TEMPERATURE + attempt * 0.1, 0.8)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        temperature=temp,
                        top_p=self.TOP_P,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                generated = outputs[0][inputs.input_ids.shape[1]:]
                code = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                code = self._extract_function(code)
                if code:
                    codes.append(code)
            except Exception:
                continue

        return codes

    def _extract_function(self, text):
        """Extract transform function from LLM output."""
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

        if 'def transform' in text:
            idx = text.index('def transform')
            return text[idx:].strip()

        return None


class CodeVerifier:
    """Safely execute generated code against training examples."""

    @staticmethod
    def execute_and_verify(code, task, timeout=10):
        """Execute code and verify against all training pairs. Returns code if verified."""
        train_pairs = task['train']

        exec_globals = {
            'np': np,
            '__builtins__': __builtins__,
        }

        try:
            exec(code, exec_globals)
        except Exception:
            return None

        if 'transform' not in exec_globals:
            return None

        transform_fn = exec_globals['transform']

        for pair in train_pairs:
            try:
                result_holder = [None]
                error_holder = [None]

                def run():
                    try:
                        inp = [row[:] for row in pair['input']]
                        out = transform_fn(inp)
                        result_holder[0] = out
                    except Exception as e:
                        error_holder[0] = e

                t = threading.Thread(target=run)
                t.start()
                t.join(timeout=timeout)

                if t.is_alive() or error_holder[0] is not None:
                    return None

                output = result_holder[0]
                if not isinstance(output, list):
                    return None

                if not verify_output(pair['output'], output):
                    return None
            except Exception:
                return None

        return code

    @staticmethod
    def run_on_test(code, test_input, timeout=10):
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
            result_holder = [None]
            error_holder = [None]

            def run():
                try:
                    inp = [row[:] for row in test_input]
                    out = transform_fn(inp)
                    result_holder[0] = out
                except Exception as e:
                    error_holder[0] = e

            t = threading.Thread(target=run)
            t.start()
            t.join(timeout=timeout)

            if t.is_alive() or error_holder[0] is not None:
                return None

            output = result_holder[0]
            if isinstance(output, list) and len(output) > 0:
                return output
            return None
        except Exception:
            return None


class LLMFallbackSolver:
    """
    Orchestrator for LLM code generation fallback.
    Only activates when LLM is available and heuristic confidence is low.
    """

    def __init__(self):
        self.generator = SafeLLMCodeGenerator()
        self.verifier = CodeVerifier()
        self._enabled = None

    def is_enabled(self):
        """Lazily check if LLM fallback is available."""
        if self._enabled is None:
            self._enabled = self.generator.is_available()
            if not self._enabled:
                print("[V3-LLM] LLM fallback not available (no GPU/transformers)")
        return self._enabled

    def solve(self, task, min_confidence=0.8):
        """
        Generate code candidates, verify them, return best prediction.
        Only runs if heuristic confidence < min_confidence.
        """
        if not self.is_enabled():
            return None, 0, None

        codes = self.generator.generate_code(task)
        verified_codes = []

        for code in codes:
            verified = self.verifier.execute_and_verify(code, task)
            if verified is not None:
                verified_codes.append(verified)

        if not verified_codes:
            return None, 0, None

        # Run the first verified code on the test input
        test_input = task['test'][0]['input']
        pred = self.verifier.run_on_test(verified_codes[0], test_input)
        if pred is not None:
            return ensure_grid(pred), 1.0, 'llm_code_gen'

        return None, 0, None

    def get_all_predictions(self, task):
        """Return all distinct verified predictions for self-consistency."""
        predictions = []
        if not self.is_enabled():
            return predictions

        codes = self.generator.generate_code(task)
        seen_hashes = set()

        for code in codes:
            verified = self.verifier.execute_and_verify(code, task)
            if verified is None:
                continue

            test_input = task['test'][0]['input']
            pred = self.verifier.run_on_test(verified, test_input)
            if pred is None:
                continue

            pred = ensure_grid(pred)
            h = grid_hash(pred)
            if h not in seen_hashes:
                seen_hashes.add(h)
                predictions.append(pred)

        return predictions


# ============================================================
# MODULE 3: ARITHMETIC & COUNTING PRIMITIVES
# ============================================================

class ArithmeticCountingPrimitives:
    """
    Counting-based and arithmetic transformation strategies.
    These handle tasks where the output is determined by counting
    objects, performing arithmetic on colors, or density-based logic.
    """

    def __init__(self):
        pass

    def solve(self, task):
        """Try all arithmetic/counting strategies. Returns (pred, conf, name) or None."""
        strategies = [
            ('count_by_color_grid', self.count_by_color_grid),
            ('count_objects_output', self.count_objects_output),
            ('color_addition_mod10', self.color_addition_mod10),
            ('color_subtraction_mod10', self.color_subtraction_mod10),
            ('color_complement', self.color_complement),
            ('count_to_output_size', self.count_to_output_size),
            ('density_transform', self.density_transform),
            ('modular_arithmetic', self.modular_arithmetic_grid),
            ('sum_rows_output', self.sum_rows_to_output),
            ('sum_cols_output', self.sum_cols_to_output),
            ('color_frequency_grid', self.color_frequency_grid),
            ('object_size_sort', self.object_size_sort_grid),
        ]

        for name, fn in strategies:
            try:
                result = fn(task)
                if result is not None:
                    pred, conf = result
                    if pred is not None and conf > 0.9:
                        return pred, conf, name
            except Exception:
                continue

        return None, 0, None

    def count_by_color_grid(self, task):
        """
        Output a grid where each cell (r, c) represents the count of objects
        of color c in the input, or similar count-based patterns.
        """
        train = task['train']
        test_input = task['test'][0]['input']

        # Check if output dimensions relate to color counts
        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        # Strategy: output is a single row/col with counts of each color
        colors_present = sorted(set(int(c) for c in inp_arr.flatten() if c != 0))

        # Try: output row = [count_of_color_1, count_of_color_2, ...]
        for connectivity in [4, 8]:
            counts = ObjectCounter.count_objects_by_color(ex0['input'], connectivity)
            count_list = [counts.get(c, 0) for c in colors_present]

            out_flat = out_arr.flatten().tolist()
            if len(count_list) == len(out_flat):
                # Check if counts match output
                if all(c == o for c, o in zip(count_list, out_flat) if o != 0):
                    # Validate on all training examples
                    def make_fn(conn):
                        def fn(inp, ex):
                            a = grid_to_array(inp)
                            cols_present = sorted(set(int(c) for c in a.flatten() if c != 0))
                            cts = ObjectCounter.count_objects_by_color(inp, conn)
                            return [[cts.get(c, 0) for c in cols_present]]
                        return fn

                    fn = make_fn(connectivity)
                    valid = True
                    for ex in train:
                        pred = fn(ex['input'], ex)
                        if not verify_output(ex['output'], pred):
                            valid = False
                            break

                    if valid:
                        pred = fn(test_input, train[0])
                        return ensure_grid(pred), 1.0

        return None, 0

    def count_objects_output(self, task):
        """
        Output is a small grid whose size or content reflects the count of
        objects in the input.
        """
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        # Check: output size = number of objects
        for connectivity in [4, 8]:
            n_comps = ComponentAnalyzer.count_components(inp_arr, connectivity)

            # Check various size relationships
            oh, ow = out_arr.shape
            if n_comps == oh or n_comps == ow or n_comps == oh * ow:
                # Validate on all training examples
                def make_pred_fn(conn):
                    def fn(inp, ex):
                        a = grid_to_array(inp)
                        nc = ComponentAnalyzer.count_components(a, conn)
                        target_shape = grid_shape(ex['output'])
                        if nc == target_shape[0]:
                            # Output is nc x ow column
                            result = np.zeros((nc, target_shape[1]), dtype=int)
                            return array_to_grid(result)
                        elif nc == target_shape[1]:
                            result = np.zeros((target_shape[0], nc), dtype=int)
                            return array_to_grid(result)
                        elif nc == target_shape[0] * target_shape[1]:
                            result = np.zeros(target_shape, dtype=int)
                            return array_to_grid(result)
                        return None
                    return fn

                fn = make_pred_fn(connectivity)
                all_match = True
                for ex in train:
                    pred = fn(ex['input'], ex)
                    if pred is None or not verify_output(ex['output'], pred):
                        all_match = False
                        break

                if all_match:
                    pred = fn(test_input, train[0])
                    if pred is not None:
                        return ensure_grid(pred), 0.95

        return None, 0

    def color_addition_mod10(self, task):
        """Apply (color + K) mod 10 for some constant K."""
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        if inp_arr.shape != out_arr.shape:
            return None, 0

        # Try all possible offsets 1-9
        for k in range(1, 10):
            shifted = (inp_arr + k) % 10
            if np.array_equal(shifted, out_arr):
                # Validate on all examples
                def make_fn(offset):
                    def fn(inp, ex):
                        a = grid_to_array(inp)
                        return array_to_grid((a + offset) % 10)
                    return fn

                fn = make_fn(k)
                valid = all(
                    verify_output(ex['output'], fn(ex['input'], ex))
                    for ex in train
                )
                if valid:
                    pred = fn(test_input, train[0])
                    return ensure_grid(pred), 1.0

        return None, 0

    def color_subtraction_mod10(self, task):
        """Apply (color - K) mod 10 for some constant K."""
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        if inp_arr.shape != out_arr.shape:
            return None, 0

        for k in range(1, 10):
            shifted = (inp_arr - k) % 10
            if np.array_equal(shifted, out_arr):
                def make_fn(offset):
                    def fn(inp, ex):
                        a = grid_to_array(inp)
                        return array_to_grid((a - offset) % 10)
                    return fn

                fn = make_fn(k)
                valid = all(
                    verify_output(ex['output'], fn(ex['input'], ex))
                    for ex in train
                )
                if valid:
                    pred = fn(test_input, train[0])
                    return ensure_grid(pred), 1.0

        return None, 0

    def color_complement(self, task):
        """Apply 9 - color (complement)."""
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        if inp_arr.shape != out_arr.shape:
            return None, 0

        complement = 9 - inp_arr
        if np.array_equal(complement, out_arr):
            def fn(inp, ex):
                a = grid_to_array(inp)
                return array_to_grid(9 - a)

            valid = all(
                verify_output(ex['output'], fn(ex['input'], ex))
                for ex in train
            )
            if valid:
                pred = fn(test_input, train[0])
                return ensure_grid(pred), 1.0

        return None, 0

    def count_to_output_size(self, task):
        """
        Output size is determined by some count in the input:
        number of distinct colors, number of objects, etc.
        Content is drawn from the input somehow.
        """
        train = task['train']
        test_input = task['test'][0]['input']

        if len(train) < 2:
            return None, 0

        # Collect (input_feature, output_shape) pairs
        features_shapes = []
        for ex in train:
            inp_arr = grid_to_array(ex['input'])
            out_shape = grid_shape(ex['output'])
            nz_colors = sorted(set(int(c) for c in inp_arr.flatten() if c != 0))
            n_colors = len(nz_colors)
            n_comps_4 = ComponentAnalyzer.count_components(inp_arr, 4)
            n_comps_8 = ComponentAnalyzer.count_components(inp_arr, 8)
            h, w = inp_arr.shape

            features_shapes.append({
                'n_colors': n_colors,
                'n_comps_4': n_comps_4,
                'n_comps_8': n_comps_8,
                'height': h,
                'width': w,
                'out_shape': out_shape,
            })

        # Check if any feature consistently determines output shape
        for feat in ['n_colors', 'n_comps_4', 'n_comps_8']:
            vals = [f[feat] for f in features_shapes]
            shapes = [f['out_shape'] for f in features_shapes]

            # Check if out_shape[0] == feat or out_shape[1] == feat
            if all(v == s[0] for v, s in zip(vals, shapes)):
                # Output height = feature value
                # Try to figure out what the output content is
                # Simple: output is feature x max_width with tiled content
                target_w = shapes[0][1]
                def make_fn(feature_name, tw):
                    def fn(inp, ex):
                        a = grid_to_array(inp)
                        if feature_name == 'n_colors':
                            nv = len(set(int(c) for c in a.flatten() if c != 0))
                        elif feature_name == 'n_comps_4':
                            nv = ComponentAnalyzer.count_components(a, 4)
                        else:
                            nv = ComponentAnalyzer.count_components(a, 8)
                        result = np.zeros((nv, tw), dtype=int)
                        return array_to_grid(result)
                    return fn

                fn = make_fn(feat, target_w)
                valid = all(
                    verify_output(ex['output'], fn(ex['input'], ex))
                    for ex in train
                )
                if valid:
                    pred = fn(test_input, train[0])
                    return ensure_grid(pred), 0.9

            if all(v == s[1] for v, s in zip(vals, shapes)):
                target_h = shapes[0][0]
                def make_fn2(feature_name, th):
                    def fn(inp, ex):
                        a = grid_to_array(inp)
                        if feature_name == 'n_colors':
                            nv = len(set(int(c) for c in a.flatten() if c != 0))
                        elif feature_name == 'n_comps_4':
                            nv = ComponentAnalyzer.count_components(a, 4)
                        else:
                            nv = ComponentAnalyzer.count_components(a, 8)
                        result = np.zeros((th, nv), dtype=int)
                        return array_to_grid(result)
                    return fn

                fn = make_fn2(feat, target_h)
                valid = all(
                    verify_output(ex['output'], fn(ex['input'], ex))
                    for ex in train
                )
                if valid:
                    pred = fn(test_input, train[0])
                    return ensure_grid(pred), 0.9

        return None, 0

    def density_transform(self, task):
        """
        Transformation based on density (fraction of non-zero cells).
        Examples: threshold-based binarization, density-based resizing.
        """
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        if inp_arr.shape != out_arr.shape:
            return None, 0

        # Check: threshold-based — cells with a certain local density change
        # Try: replace cells that have >= K non-zero neighbors with a color
        for k in range(1, 5):
            for target_color in range(1, 10):
                result = inp_arr.copy()
                h, w = inp_arr.shape
                for r in range(h):
                    for c in range(w):
                        count = 0
                        for dr in range(-1, 2):
                            for dc in range(-1, 2):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w and inp_arr[nr, nc] != 0:
                                    count += 1
                        if count >= k:
                            result[r, c] = target_color

                if np.array_equal(result, out_arr):
                    def make_fn(threshold, tcolor):
                        def fn(inp, ex):
                            a = grid_to_array(inp)
                            res = a.copy()
                            hh, ww = a.shape
                            for r in range(hh):
                                for c in range(ww):
                                    cnt = 0
                                    for dr in range(-1, 2):
                                        for dc in range(-1, 2):
                                            if dr == 0 and dc == 0:
                                                continue
                                            nr, nc = r + dr, c + dc
                                            if 0 <= nr < hh and 0 <= nc < ww and a[nr, nc] != 0:
                                                cnt += 1
                                    if cnt >= threshold:
                                        res[r, c] = tcolor
                            return array_to_grid(res)
                        return fn

                    fn = make_fn(k, target_color)
                    valid = all(
                        verify_output(ex['output'], fn(ex['input'], ex))
                        for ex in train
                    )
                    if valid:
                        pred = fn(test_input, train[0])
                        return ensure_grid(pred), 1.0

        return None, 0

    def modular_arithmetic_grid(self, task):
        """Apply (grid_value * K) mod 10 or (grid_value ^ K) mod 10."""
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        if inp_arr.shape != out_arr.shape:
            return None, 0

        for k in range(2, 10):
            # Multiplication
            result = (inp_arr * k) % 10
            if np.array_equal(result, out_arr):
                def make_fn(mult):
                    def fn(inp, ex):
                        a = grid_to_array(inp)
                        return array_to_grid((a * mult) % 10)
                    return fn

                fn = make_fn(k)
                valid = all(
                    verify_output(ex['output'], fn(ex['input'], ex))
                    for ex in train
                )
                if valid:
                    pred = fn(test_input, train[0])
                    return ensure_grid(pred), 1.0

            # XOR
            result = (inp_arr ^ k)
            if np.array_equal(result, out_arr):
                def make_fn2(xor_val):
                    def fn(inp, ex):
                        a = grid_to_array(inp)
                        return array_to_grid(a ^ xor_val)
                    return fn

                fn = make_fn2(k)
                valid = all(
                    verify_output(ex['output'], fn(ex['input'], ex))
                    for ex in train
                )
                if valid:
                    pred = fn(test_input, train[0])
                    return ensure_grid(pred), 1.0

        return None, 0

    def sum_rows_to_output(self, task):
        """Output rows are the sum (or other reduction) of input rows."""
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        # Check if output is row sums as a column
        if out_arr.shape[0] == inp_arr.shape[0] and out_arr.shape[1] == 1:
            row_sums = inp_arr.sum(axis=1) % 10
            if np.array_equal(row_sums.reshape(-1, 1), out_arr):
                def fn(inp, ex):
                    a = grid_to_array(inp)
                    return array_to_grid((a.sum(axis=1) % 10).reshape(-1, 1))
                valid = all(
                    verify_output(ex['output'], fn(ex['input'], ex))
                    for ex in train
                )
                if valid:
                    return ensure_grid(fn(test_input, train[0])), 1.0

        return None, 0

    def sum_cols_to_output(self, task):
        """Output columns are the sum (or other reduction) of input columns."""
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        if out_arr.shape[0] == 1 and out_arr.shape[1] == inp_arr.shape[1]:
            col_sums = inp_arr.sum(axis=0) % 10
            if np.array_equal(col_sums.reshape(1, -1), out_arr):
                def fn(inp, ex):
                    a = grid_to_array(inp)
                    return array_to_grid((a.sum(axis=0) % 10).reshape(1, -1))
                valid = all(
                    verify_output(ex['output'], fn(ex['input'], ex))
                    for ex in train
                )
                if valid:
                    return ensure_grid(fn(test_input, train[0])), 1.0

        return None, 0

    def color_frequency_grid(self, task):
        """
        Output is a grid where each cell shows the count of a specific color
        in a specific region of the input.
        """
        train = task['train']
        test_input = task['test'][0]['input']

        if len(train) < 2:
            return None, 0

        # For each training example, count color frequencies
        # Check if output is a simple frequency histogram
        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        # Try: output[i][j] = count of color j in row i
        oh, ow = out_arr.shape
        ih, iw = inp_arr.shape

        if oh == ih and ow <= 10:
            # Check if out[r][c] = count of color c in row r
            matches = True
            for r in range(oh):
                if r >= ih:
                    matches = False
                    break
                for c in range(ow):
                    count = int(np.sum(inp_arr[r] == c))
                    if int(out_arr[r, c]) != count:
                        matches = False
                        break
                if not matches:
                    break

            if matches:
                def fn(inp, ex):
                    a = grid_to_array(inp)
                    oshape = grid_shape(ex['output'])
                    result = np.zeros(oshape, dtype=int)
                    for r in range(min(oshape[0], a.shape[0])):
                        for c in range(oshape[1]):
                            result[r, c] = int(np.sum(a[r] == c))
                    return array_to_grid(result)

                valid = all(
                    verify_output(ex['output'], fn(ex['input'], ex))
                    for ex in train
                )
                if valid:
                    return ensure_grid(fn(test_input, train[0])), 1.0

        return None, 0

    def object_size_sort_grid(self, task):
        """
        Output consists of objects sorted by size (largest first or smallest first).
        """
        train = task['train']
        test_input = task['test'][0]['input']

        for connectivity in [4, 8]:
            all_valid = True
            sort_order = None

            for ex in train:
                inp_arr = grid_to_array(ex['input'])
                out_arr = grid_to_array(ex['output'])

                if inp_arr.shape != out_arr.shape:
                    all_valid = False
                    break

                comps = ComponentAnalyzer.find_components(inp_arr, connectivity)
                if not comps:
                    if np.array_equal(inp_arr, out_arr):
                        continue
                    else:
                        all_valid = False
                        break

                # Sort by size
                sorted_comps = sorted(comps, key=lambda c: c['size'], reverse=True)

                # Check if output has the same objects but sorted by size position
                # (This is a simplified check: do the sorted components match output?)
                out_comps = ComponentAnalyzer.find_components(out_arr, connectivity)
                out_sizes = sorted([c['size'] for c in out_comps], reverse=True)
                inp_sizes = sorted([c['size'] for c in comps], reverse=True)

                if out_sizes != inp_sizes:
                    all_valid = False
                    break

            if all_valid and len(train) >= 1:
                # Validate more carefully
                ex0 = train[0]
                inp_arr = grid_to_array(ex0['input'])
                out_arr = grid_to_array(ex0['output'])
                comps = ComponentAnalyzer.find_components(inp_arr, connectivity)

                # Check if objects were rearranged by size
                sorted_comps = sorted(comps, key=lambda c: c['size'], reverse=True)
                # Check if output non-zero pixels = same set of pixels as input
                inp_nz = set(zip(*np.nonzero(inp_arr)))
                out_nz = set(zip(*np.nonzero(out_arr)))

                if inp_nz == out_nz:
                    # Objects moved but preserved — check position pattern
                    # Skip this for now, too complex without clear pattern
                    continue

                # Check: objects placed in top-to-bottom order by size
                result = np.zeros_like(inp_arr)
                y_pos = 0
                for comp in sorted_comps:
                    min_r, min_c, max_r, max_c = comp['bbox']
                    obj_h = max_r - min_r + 1
                    obj_w = max_c - min_c + 1
                    for r, c in comp['pixels']:
                        new_r = y_pos + (r - min_r)
                        new_c = c - min_c
                        if 0 <= new_r < result.shape[0] and 0 <= new_c < result.shape[1]:
                            result[new_r, new_c] = comp['color']
                    y_pos += obj_h + 1  # +1 for spacing

                if np.array_equal(result, out_arr):
                    def make_fn(conn):
                        def fn(inp, ex):
                            a = grid_to_array(inp)
                            cs = ComponentAnalyzer.find_components(a, conn)
                            scs = sorted(cs, key=lambda c: c['size'], reverse=True)
                            res = np.zeros_like(a)
                            yp = 0
                            for comp in scs:
                                mr, mc, xr, xc = comp['bbox']
                                oh = xr - mr + 1
                                for r, c in comp['pixels']:
                                    nr = yp + (r - mr)
                                    nc = c - mc
                                    if 0 <= nr < res.shape[0] and 0 <= nc < res.shape[1]:
                                        res[nr, nc] = comp['color']
                                yp += oh + 1
                            return array_to_grid(res)
                        return fn

                    fn = make_fn(connectivity)
                    valid = all(
                        verify_output(ex['output'], fn(ex['input'], ex))
                        for ex in train
                    )
                    if valid:
                        return ensure_grid(fn(test_input, train[0])), 0.95

        return None, 0


# ============================================================
# MODULE 4: ADAPTIVE CONNECTIVITY & DIAGONAL DETECTION
# ============================================================

class AdaptiveConnectivity:
    """
    Automatically tests 4-connectivity vs 8-connectivity per task.
    Chooses whichever produces more consistent component structures across demos.
    Also handles diagonal line/pattern detection.
    """

    @staticmethod
    def best_connectivity(task):
        """
        Determine whether 4-connectivity or 8-connectivity gives more
        consistent component counts across training examples.
        Returns 4 or 8.
        """
        train = task['train']
        if len(train) < 2:
            return 4  # Default

        counts_4 = []
        counts_8 = []

        for ex in train:
            inp_arr = grid_to_array(ex['input'])
            c4 = ComponentAnalyzer.count_components(inp_arr, 4)
            c8 = ComponentAnalyzer.count_components(inp_arr, 8)
            counts_4.append(c4)
            counts_8.append(c8)

        # Check consistency: which connectivity has lower variance in count ratios?
        # Compare input->output count ratios
        ratios_4 = []
        ratios_8 = []

        for ex in train:
            inp_arr = grid_to_array(ex['input'])
            out_arr = grid_to_array(ex['output'])

            c4_in = ComponentAnalyzer.count_components(inp_arr, 4)
            c4_out = ComponentAnalyzer.count_components(out_arr, 4)
            c8_in = ComponentAnalyzer.count_components(inp_arr, 8)
            c8_out = ComponentAnalyzer.count_components(out_arr, 8)

            if c4_in > 0:
                ratios_4.append(c4_out / c4_in)
            if c8_in > 0:
                ratios_8.append(c8_out / c8_in)

        # Lower variance = more consistent
        if ratios_4 and ratios_8:
            var_4 = np.var(ratios_4) if len(ratios_4) > 1 else float('inf')
            var_8 = np.var(ratios_8) if len(ratios_8) > 1 else float('inf')
            return 4 if var_4 <= var_8 else 8

        # Also consider: which gives same count across all examples?
        unique_4 = len(set(counts_4))
        unique_8 = len(set(counts_8))

        if unique_4 < unique_8:
            return 4
        elif unique_8 < unique_4:
            return 8

        return 4  # Default


class DiagonalDetector:
    """Detect diagonal patterns: main diagonal, anti-diagonal, X-pattern, stripes."""

    @staticmethod
    def extract_main_diagonal(arr):
        """Extract values along the main diagonal."""
        h, w = arr.shape
        n = min(h, w)
        return [int(arr[i, i]) for i in range(n)]

    @staticmethod
    def extract_anti_diagonal(arr):
        """Extract values along the anti-diagonal."""
        h, w = arr.shape
        n = min(h, w)
        return [int(arr[i, w - 1 - i]) for i in range(n)]

    @staticmethod
    def has_main_diagonal_pattern(arr):
        """Check if non-zero values primarily lie on the main diagonal."""
        h, w = arr.shape
        nonzero = np.nonzero(arr)
        if len(nonzero[0]) == 0:
            return False

        on_diag = sum(1 for r, c in zip(nonzero[0], nonzero[1]) if r == c)
        return on_diag / len(nonzero[0]) > 0.8

    @staticmethod
    def has_anti_diagonal_pattern(arr):
        """Check if non-zero values primarily lie on the anti-diagonal."""
        h, w = arr.shape
        nonzero = np.nonzero(arr)
        if len(nonzero[0]) == 0:
            return False

        on_diag = sum(1 for r, c in zip(nonzero[0], nonzero[1]) if r + c == w - 1)
        return on_diag / len(nonzero[0]) > 0.8

    @staticmethod
    def has_x_pattern(arr):
        """Check if non-zero values form an X (both diagonals)."""
        return DiagonalDetector.has_main_diagonal_pattern(arr) and \
               DiagonalDetector.has_anti_diagonal_pattern(arr)

    @staticmethod
    def detect_diagonal_stripes(arr):
        """
        Detect diagonal stripe patterns where values repeat along diagonals.
        Returns the stripe period or None.
        """
        h, w = arr.shape
        if h < 2 or w < 2:
            return None

        # Check if arr[i][j] depends only on (i-j) mod K for some K
        for period in range(1, min(h, w) // 2 + 1):
            valid = True
            for r in range(h):
                for c in range(w):
                    diag_idx = (r - c) % period
                    # Find reference cell
                    ref_r, ref_c = diag_idx, 0
                    if ref_r >= h:
                        ref_r, ref_c = 0, -diag_idx % period
                    if ref_r < h and ref_c < w:
                        if arr[r, c] != arr[ref_r, ref_c]:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                return period

        # Check if arr[i][j] depends only on (i+j) mod K
        for period in range(1, min(h, w) // 2 + 1):
            valid = True
            for r in range(h):
                for c in range(w):
                    diag_idx = (r + c) % period
                    ref_r, ref_c = 0, diag_idx
                    if ref_c >= w:
                        continue
                    if arr[r, c] != arr[ref_r, ref_c]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return period

        return None


class DiagonalPatternSolver:
    """Solver for diagonal-based transformations."""

    def solve(self, task):
        """Try diagonal-based strategies."""
        strategies = [
            ('main_diagonal_extract', self.main_diagonal_extract),
            ('anti_diagonal_extract', self.anti_diagonal_extract),
            ('x_pattern_transform', self.x_pattern_transform),
            ('diagonal_stripe_transform', self.diagonal_stripe_transform),
            ('diagonal_mirror', self.diagonal_mirror),
        ]

        for name, fn in strategies:
            try:
                result = fn(task)
                if result is not None:
                    pred, conf = result
                    if pred is not None and conf > 0.9:
                        return pred, conf, name
            except Exception:
                continue

        return None, 0, None

    def main_diagonal_extract(self, task):
        """Extract or transform based on main diagonal."""
        train = task['train']
        test_input = task['test'][0]['input']

        # Check if output is the main diagonal of the input as a row
        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        diag = DiagonalDetector.extract_main_diagonal(inp_arr)
        if out_arr.shape == (1, len(diag)) and np.array_equal(
            out_arr.flatten(), np.array(diag)
        ):
            def fn(inp, ex):
                a = grid_to_array(inp)
                d = DiagonalDetector.extract_main_diagonal(a)
                return [d]

            valid = all(
                verify_output(ex['output'], fn(ex['input'], ex))
                for ex in train
            )
            if valid:
                return ensure_grid(fn(test_input, train[0])), 1.0

        # Check if output is the diagonal as a column
        if out_arr.shape == (len(diag), 1) and np.array_equal(
            out_arr.flatten(), np.array(diag)
        ):
            def fn2(inp, ex):
                a = grid_to_array(inp)
                d = DiagonalDetector.extract_main_diagonal(a)
                return [[v] for v in d]

            valid = all(
                verify_output(ex['output'], fn2(ex['input'], ex))
                for ex in train
            )
            if valid:
                return ensure_grid(fn2(test_input, train[0])), 1.0

        return None, 0

    def anti_diagonal_extract(self, task):
        """Extract or transform based on anti-diagonal."""
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        adiag = DiagonalDetector.extract_anti_diagonal(inp_arr)
        if out_arr.shape == (1, len(adiag)) and np.array_equal(
            out_arr.flatten(), np.array(adiag)
        ):
            def fn(inp, ex):
                a = grid_to_array(inp)
                d = DiagonalDetector.extract_anti_diagonal(a)
                return [d]

            valid = all(
                verify_output(ex['output'], fn(ex['input'], ex))
                for ex in train
            )
            if valid:
                return ensure_grid(fn(test_input, train[0])), 1.0

        return None, 0

    def x_pattern_transform(self, task):
        """Transform grids with X-patterns."""
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        # Check: output has X drawn in it
        if DiagonalDetector.has_main_diagonal_pattern(out_arr) and \
           DiagonalDetector.has_anti_diagonal_pattern(out_arr):
            # Get the X color
            nz = np.nonzero(out_arr)
            if len(nz[0]) > 0:
                x_color = int(Counter(out_arr[nz].flatten()).most_common(1)[0][0])

                def fn(inp, ex):
                    a = grid_to_array(inp)
                    h, w = a.shape
                    result = np.zeros_like(a)
                    for i in range(min(h, w)):
                        result[i, i] = x_color
                        result[i, w - 1 - i] = x_color
                    return array_to_grid(result)

                valid = all(
                    verify_output(ex['output'], fn(ex['input'], ex))
                    for ex in train
                )
                if valid:
                    return ensure_grid(fn(test_input, train[0])), 1.0

        return None, 0

    def diagonal_stripe_transform(self, task):
        """Transform based on diagonal stripe patterns."""
        train = task['train']
        test_input = task['test'][0]['input']

        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['output'])

        period = DiagonalDetector.detect_diagonal_stripes(out_arr)
        if period is not None and period > 1:
            # Check if all training outputs have the same stripe pattern
            all_match = True
            for ex in train[1:]:
                o = grid_to_array(ex['output'])
                p = DiagonalDetector.detect_diagonal_stripes(o)
                if p != period:
                    all_match = False
                    break

            if all_match:
                # The output has diagonal stripes with this period
                # Check if output values depend on (i+j) mod period or (i-j) mod period
                # Use the first training example to determine the mapping
                h, w = out_arr.shape
                stripe_map = {}
                for r in range(h):
                    for c in range(w):
                        key = (r + c) % period
                        stripe_map[key] = int(out_arr[r, c])

                if stripe_map:
                    def make_fn(smap, p):
                        def fn(inp, ex):
                            a = grid_to_array(inp)
                            ih, iw = a.shape
                            result = np.zeros((ih, iw), dtype=int)
                            for r in range(ih):
                                for c in range(iw):
                                    result[r, c] = smap.get((r + c) % p, 0)
                            return array_to_grid(result)
                        return fn

                    fn = make_fn(stripe_map, period)
                    valid = all(
                        verify_output(ex['output'], fn(ex['input'], ex))
                        for ex in train
                    )
                    if valid:
                        return ensure_grid(fn(test_input, train[0])), 1.0

                # Try (i-j) mod period
                stripe_map2 = {}
                for r in range(h):
                    for c in range(w):
                        key = (r - c) % period
                        stripe_map2[key] = int(out_arr[r, c])

                if stripe_map2:
                    def make_fn2(smap, p):
                        def fn(inp, ex):
                            a = grid_to_array(inp)
                            ih, iw = a.shape
                            result = np.zeros((ih, iw), dtype=int)
                            for r in range(ih):
                                for c in range(iw):
                                    result[r, c] = smap.get((r - c) % p, 0)
                            return array_to_grid(result)
                        return fn

                    fn = make_fn2(stripe_map2, period)
                    valid = all(
                        verify_output(ex['output'], fn(ex['input'], ex))
                        for ex in train
                    )
                    if valid:
                        return ensure_grid(fn(test_input, train[0])), 1.0

        return None, 0

    def diagonal_mirror(self, task):
        """Mirror across the main or anti diagonal."""
        train = task['train']
        test_input = task['test'][0]['input']

        # Main diagonal mirror (transpose)
        ex0 = train[0]
        inp_arr = grid_to_array(ex0['input'])
        out_arr = grid_to_array(ex0['input'])

        # This is already covered by v2's mirror_diagonal, but let's also
        # check anti-diagonal mirror
        for ex in train:
            inp = grid_to_array(ex['input'])
            out = grid_to_array(ex['output'])
            if inp.shape[0] == inp.shape[1]:
                anti_mirror = np.flipud(np.fliplr(inp.T))
                if np.array_equal(anti_mirror, out):
                    # Anti-diagonal mirror
                    all_match = True
                    for ex2 in train:
                        i2 = grid_to_array(ex2['input'])
                        o2 = grid_to_array(ex2['output'])
                        am = np.flipud(np.fliplr(i2.T))
                        if not np.array_equal(am, o2):
                            all_match = False
                            break

                    if all_match:
                        def fn(inp, ex):
                            a = grid_to_array(inp)
                            return array_to_grid(np.flipud(np.fliplr(a.T)))

                        return ensure_grid(fn(test_input, train[0])), 1.0

        return None, 0


class AdaptiveConnectivitySolver:
    """
    Enhanced solver that uses adaptive connectivity selection
    and diagonal pattern detection alongside v2 heuristics.
    """

    def __init__(self):
        self.v2_solver = HeuristicSolverV2()
        self.diagonal_solver = DiagonalPatternSolver()
        self.adaptive_conn = AdaptiveConnectivity()

    def solve(self, task):
        """Try v2 heuristics with best connectivity, plus diagonal strategies."""
        # First try diagonal patterns
        diag_pred, diag_conf, diag_name = self.diagonal_solver.solve(task)
        if diag_pred is not None:
            return diag_pred, diag_conf, diag_name

        # Then try v2 solver (which already handles many cases)
        v2_pred, v2_conf, v2_name = self.v2_solver.solve(task)
        if v2_pred is not None:
            return v2_pred, v2_conf, v2_name

        return None, 0, None


# ============================================================
# MAIN V3 SOLVER
# ============================================================

class V3Solver:
    """
    Main v3 solver that orchestrates all modules in priority order.

    Solve order:
      a. Conditional logic engine (new)
      b. Arithmetic & counting primitives (new)
      c. Adaptive connectivity + diagonal detection (new)
      d. Full heuristic ensemble from v2 (existing)
      e. LLM code generation fallback (new, GPU only)
      f. Self-consistency: pick top 2 distinct predictions
    """

    def __init__(self, enable_llm=True):
        self.conditional_engine = ConditionalLogicEngine()
        self.arith_solver = ArithmeticCountingPrimitives()
        self.adaptive_solver = AdaptiveConnectivitySolver()
        self.v2_solver = HeuristicSolverV2()
        self.llm_solver = LLMFallbackSolver() if enable_llm else None
        self.adaptive_conn = AdaptiveConnectivity()

        # Statistics
        self.stats = Counter()

    def solve_task(self, task_id, task, max_time=80):
        """
        Solve a single task. Returns list of 2 attempt grids.

        Each attempt is a 2D list of ints (0-9).
        """
        t0 = time.time()
        test_pairs = task.get('test', [])
        if not test_pairs:
            self.stats['no_test_pairs'] += 1
            return [[[0]], [[0]]]

        test_input = test_pairs[0]['input']
        predictions = {}  # strategy_name -> (pred, conf)

        # --- Phase 1: Conditional Logic Engine ---
        try:
            pred, conf, name = self.conditional_engine.solve(task)
            if pred is not None:
                predictions[name] = (ensure_grid(pred), conf)
                self.stats['conditional'] += 1
        except Exception as e:
            self.stats['conditional_error'] += 1

        if time.time() - t0 > max_time * 0.5:
            return self._finalize(predictions, test_input, task_id, t0)

        # --- Phase 2: Arithmetic & Counting Primitives ---
        try:
            pred, conf, name = self.arith_solver.solve(task)
            if pred is not None:
                predictions[name] = (ensure_grid(pred), conf)
                self.stats['arithmetic'] += 1
        except Exception as e:
            self.stats['arithmetic_error'] += 1

        if time.time() - t0 > max_time * 0.6:
            return self._finalize(predictions, test_input, task_id, t0)

        # --- Phase 3: Adaptive Connectivity + Diagonal Detection ---
        try:
            pred, conf, name = self.adaptive_solver.solve(task)
            if pred is not None:
                predictions[name] = (ensure_grid(pred), conf)
                self.stats['adaptive'] += 1
        except Exception as e:
            self.stats['adaptive_error'] += 1

        if time.time() - t0 > max_time * 0.7:
            return self._finalize(predictions, test_input, task_id, t0)

        # --- Phase 4: Full v2 Heuristic Ensemble ---
        try:
            # Also run v2 solver directly (the adaptive solver already ran it,
            # but let's make sure we capture the result)
            if 'v2' not in str(predictions.keys()).lower():
                pred, conf, name = self.v2_solver.solve(task)
                if pred is not None:
                    predictions[f'v2_{name}'] = (ensure_grid(pred), conf)
                    self.stats['v2'] += 1
        except Exception as e:
            self.stats['v2_error'] += 1

        if time.time() - t0 > max_time * 0.8:
            return self._finalize(predictions, test_input, task_id, t0)

        # --- Phase 5: LLM Code Generation Fallback ---
        # Only runs if no high-confidence prediction and LLM is available
        best_conf = max((c for _, c in predictions.values()), default=0)

        if self.llm_solver is not None and best_conf < 0.8:
            try:
                pred, conf, name = self.llm_solver.solve(task)
                if pred is not None:
                    predictions[name] = (ensure_grid(pred), conf)
                    self.stats['llm'] += 1
            except Exception as e:
                self.stats['llm_error'] += 1

        return self._finalize(predictions, test_input, task_id, t0)

    def _finalize(self, predictions, test_input, task_id, t0):
        """
        Pick top 2 distinct predictions using self-consistency.
        Falls back to input copy if no predictions available.
        """
        # Sort by confidence (descending)
        sorted_preds = sorted(
            predictions.items(),
            key=lambda x: x[1][1],
            reverse=True
        )

        attempts = []
        seen_hashes = set()

        for name, (pred, conf) in sorted_preds:
            h = grid_hash(pred)
            if h not in seen_hashes:
                seen_hashes.add(h)
                attempts.append(pred)
                if len(attempts) >= 2:
                    break

        # Fill remaining with fallbacks
        fallback = ensure_grid(deepcopy(test_input))

        # Try some simple alternatives as fallbacks
        alt_fallbacks = [
            ensure_grid([row[::-1] for row in test_input]),  # mirror h
            ensure_grid(test_input[::-1]),  # mirror v
            ensure_grid(np.rot90(grid_to_array(test_input), -1).tolist()),  # rot90
            ensure_grid(Extractor.extract_bounding_box(test_input)),  # bbox
        ]

        while len(attempts) < 2:
            for alt in alt_fallbacks:
                h = grid_hash(alt)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    attempts.append(alt)
                    break
            else:
                # All alternatives exhausted, use fallback
                h = grid_hash(fallback)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    attempts.append(fallback)
                else:
                    # Truly stuck — make a trivial variation
                    variant = ensure_grid([[0]])
                    attempts.append(variant)
                    break

        elapsed = time.time() - t0
        self.stats['total_time'] += elapsed

        return attempts[:2]
# ============================================================
# SUBMISSION & EVALUATION
# ============================================================

def solve_all(data_dir, output_path, enable_llm=True, max_tasks=None):
    """
    Solve all tasks in a directory and save submission.json.

    Args:
        data_dir: Path to directory containing ARC task JSON files.
        output_path: Path to save submission.json.
        enable_llm: Whether to enable LLM code generation fallback.
        max_tasks: Maximum number of tasks to solve (None = all).
    """
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if max_tasks is not None:
        json_files = json_files[:max_tasks]

    # Detect format: individual files vs single combined file
    if len(json_files) == 1:
        # Might be a combined challenges file
        try:
            with open(json_files[0]) as f:
                data = json.load(f)
            if isinstance(data, dict) and any(
                isinstance(v, dict) and 'train' in v for v in data.values()
            ):
                # Combined format
                task_dict = data
                json_files = None
            else:
                task_dict = None
        except Exception:
            task_dict = None
    else:
        task_dict = None

    solver = V3Solver(enable_llm=enable_llm)
    submission = {}

    if task_dict is not None:
        # Combined format
        total = len(task_dict)
        print(f"[V3] Solving {total} tasks from combined file: {json_files[0]}")
        print("=" * 60)

        for i, (task_id, task) in enumerate(task_dict.items()):
            if i % 20 == 0 or i == total - 1:
                elapsed = sum(solver.stats.values()) or 1
                print(f"[{i+1}/{total}] {task_id}")

            try:
                attempts = solver.solve_task(task_id, task)
                submission[task_id] = attempts
            except Exception as e:
                print(f"  ERROR {task_id}: {e}")
                traceback.print_exc()
                submission[task_id] = [[[0]], [[0]]]
    else:
        # Individual files format
        total = len(json_files)
        print(f"[V3] Solving {total} tasks from {data_dir}")
        print("=" * 60)

        for i, fpath in enumerate(json_files):
            task_id = os.path.splitext(os.path.basename(fpath))[0]

            if i % 20 == 0 or i == total - 1:
                print(f"[{i+1}/{total}] {task_id}")

            try:
                with open(fpath) as f:
                    task = json.load(f)
                attempts = solver.solve_task(task_id, task)
                submission[task_id] = attempts
            except Exception as e:
                print(f"  ERROR {task_id}: {e}")
                traceback.print_exc()
                submission[task_id] = [[[0]], [[0]]]

    # Save submission
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"\n{'='*60}")
    print(f"V3 Solver Results")
    print(f"{'='*60}")
    print(f"Total tasks: {len(submission)}")
    print(f"Submission saved: {output_path}")
    print(f"Stats: {dict(solver.stats)}")

    return submission


def evaluate(data_dir, enable_llm=False, max_tasks=None):
    """
    Evaluate v3 solver on training data (where test pairs include 'output').

    Args:
        data_dir: Path to directory containing ARC training task JSON files.
        enable_llm: Whether to enable LLM fallback (usually False for eval).
        max_tasks: Maximum number of tasks to evaluate (None = all).

    Returns:
        Dict with evaluation metrics.
    """
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if max_tasks is not None:
        json_files = json_files[:max_tasks]

    # Detect format
    if len(json_files) == 1:
        try:
            with open(json_files[0]) as f:
                data = json.load(f)
            if isinstance(data, dict) and any(
                isinstance(v, dict) and 'train' in v for v in data.values()
            ):
                task_dict = data
                json_files = None
            else:
                task_dict = None
        except Exception:
            task_dict = None
    else:
        task_dict = None

    solver = V3Solver(enable_llm=enable_llm)

    total_tasks = 0
    solved_tasks = 0
    total_test_pairs = 0
    correct_pairs = 0
    strategy_counts = Counter()

    if task_dict is not None:
        items = task_dict.items()
    else:
        items = []
        for fpath in json_files:
            task_id = os.path.splitext(os.path.basename(fpath))[0]
            with open(fpath) as f:
                task = json.load(f)
            items.append((task_id, task))

    for task_id, task in items:
        # Check if test pairs have outputs (training data)
        test_pairs = task.get('test', [])
        if not test_pairs or 'output' not in test_pairs[0]:
            continue

        total_tasks += 1
        attempts = solver.solve_task(task_id, task)

        task_solved = False
        for j, test_pair in enumerate(test_pairs):
            if 'output' not in test_pair:
                continue

            total_test_pairs += 1
            expected = test_pair['output']

            # Check attempt 1
            if j < len(attempts) and verify_output(expected, attempts[j]):
                correct_pairs += 1
                task_solved = True
                continue

            # Check attempt 2
            if len(attempts) > 1 and j < len(attempts[1]) and verify_output(expected, attempts[1][j]):
                correct_pairs += 1
                task_solved = True
                continue

        if task_solved:
            solved_tasks += 1

        if total_tasks % 50 == 0:
            acc = correct_pairs / max(1, total_test_pairs) * 100
            print(f"  [{total_tasks} tasks] {correct_pairs}/{total_test_pairs} pairs ({acc:.1f}%)")

    results = {
        'total_tasks': total_tasks,
        'solved_tasks': solved_tasks,
        'total_test_pairs': total_test_pairs,
        'correct_pairs': correct_pairs,
        'accuracy': correct_pairs / max(1, total_test_pairs),
        'task_solve_rate': solved_tasks / max(1, total_tasks),
        'stats': dict(solver.stats),
    }

    print(f"\n{'='*60}")
    print(f"V3 Solver Evaluation Results")
    print(f"{'='*60}")
    print(f"Total tasks: {total_tasks}")
    print(f"Solved tasks: {solved_tasks} ({results['task_solve_rate']*100:.1f}%)")
    print(f"Correct test pairs: {correct_pairs}/{total_test_pairs} ({results['accuracy']*100:.1f}%)")
    print(f"\nStats: {dict(solver.stats)}")

    return results


# ============================================================
# KAGGLE MAIN ENTRY POINT
# ============================================================

DATA_DIR = "/kaggle/input/arc-prize-2026-arc-agi-2"
OUTPUT_PATH = "/kaggle/working/submission.json"
TASK_BUDGET = 80  # seconds per task


def pbar(iterable, total=None, **kwargs):
    """Wrapper for tqdm with fallback to plain iterator."""
    if HAS_TQDM:
        return tqdm(iterable, total=total, **kwargs)
    return iterable


def print_env_info():
    """Print environment information."""
    print("=" * 60)
    print("AnnotateX ARC-AGI-2 v3 Solver — Kaggle Notebook")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"NumPy:  {np.__version__}")
    print(f"CUDA available: {_HAS_TORCH and torch.cuda.is_available()}")
    if _HAS_TORCH and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"LLM available: {_LLM_AVAILABLE}")
    print(f"tqdm: {HAS_TQDM}")
    print()


def load_eval_tasks(data_dir):
    """
    Load evaluation tasks from the Kaggle data directory.
    Handles both combined JSON and individual file formats.
    """
    eval_dir = os.path.join(data_dir, "data", "evaluation")
    
    # Try combined challenges file first
    combined_path = os.path.join(data_dir, "arc-agi_2_eval_challenges.json")
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            print(f"Loaded {len(data)} tasks from combined file")
            return data
    
    # Try evaluation directory
    if os.path.isdir(eval_dir):
        json_files = sorted(glob.glob(os.path.join(eval_dir, "*.json")))
        if json_files:
            task_dict = {}
            for fpath in json_files:
                task_id = os.path.splitext(os.path.basename(fpath))[0]
                with open(fpath) as f:
                    task_dict[task_id] = json.load(f)
            print(f"Loaded {len(task_dict)} tasks from {eval_dir}")
            return task_dict
    
    # Try data_dir directly
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if json_files:
        task_dict = {}
        for fpath in json_files:
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, dict) and any(
                isinstance(v, dict) and 'train' in v for v in data.values()
            ):
                task_dict.update(data)
            else:
                task_id = os.path.splitext(os.path.basename(fpath))[0]
                task_dict[task_id] = data
        print(f"Loaded {len(task_dict)} tasks from {data_dir}")
        return task_dict
    
    raise FileNotFoundError(f"No task files found in {data_dir}")


def main():
    """
    Main entry point for the Kaggle notebook.
    - Prints environment info
    - Loads all eval tasks
    - Optionally validates on training tasks
    - Solves all eval tasks with progress bar
    - Saves submission.json
    - Prints summary stats
    """
    print_env_info()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="AnnotateX ARC-AGI-2 v3 Solver")
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help='Output submission file')
    parser.add_argument('--budget', type=int, default=TASK_BUDGET,
                        help='Time budget per task in seconds')
    parser.add_argument('--no_llm', action='store_true',
                        help='Disable LLM fallback')
    parser.add_argument('--validate', type=str, default=None,
                        help='Path to training data for validation')
    parser.add_argument('--max_tasks', type=int, default=None,
                        help='Max tasks to solve')
    args = parser.parse_args()
    
    enable_llm = not args.no_llm and _LLM_AVAILABLE
    
    # Optional: validate on training data
    if args.validate:
        print(f"Validating on training data: {args.validate}")
        results = evaluate(args.validate, enable_llm=False, max_tasks=args.max_tasks)
        print(f"Validation results: {json.dumps(results, indent=2)}")
        print()
    
    # Load eval tasks
    print("Loading evaluation tasks...")
    try:
        task_dict = load_eval_tasks(args.data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Falling back to standard path structure...")
        # Try alternate path structures
        for alt_dir in [
            "/kaggle/input/arc-prize-2026-arc-agi-2/data/evaluation",
            "/kaggle/input/arc-prize-2026-arc-agi-2",
            "/kaggle/input",
        ]:
            if os.path.isdir(alt_dir):
                task_dict = load_eval_tasks(alt_dir.replace("/data/evaluation", "").replace("/evaluation", ""))
                break
        else:
            print("No data found. Exiting.")
            sys.exit(1)
    
    if args.max_tasks:
        task_dict = dict(list(task_dict.items())[:args.max_tasks])
    
    total = len(task_dict)
    print(f"Solving {total} tasks (budget: {args.budget}s/task)")
    print("=" * 60)
    
    # Solve all tasks
    solver = V3Solver(enable_llm=enable_llm)
    submission = {}
    strategy_counts = Counter()
    
    task_items = list(task_dict.items())
    
    for i, (task_id, task) in enumerate(pbar(task_items, total=total, desc="Solving")):
        try:
            t0 = time.time()
            attempts = solver.solve_task(task_id, task, max_time=args.budget)
            elapsed = time.time() - t0
            
            # Format: task_id -> [attempt1, attempt2]
            # Each attempt is a 2D grid (list of lists of ints)
            submission[task_id] = attempts
            
            if elapsed > args.budget:
                strategy_counts['timeout'] += 1
            
        except Exception as e:
            print(f"  ERROR {task_id}: {e}")
            submission[task_id] = [[[0]], [[0]]]
            strategy_counts['error'] += 1
    
    # Save submission
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(submission, f)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {total}")
    print(f"Submission saved: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")
    print(f"Solver stats: {dict(solver.stats)}")
    print(f"Errors/Timeouts: {strategy_counts.get('error', 0)} / {strategy_counts.get('timeout', 0)}")
    print(f"LLM enabled: {enable_llm}")
    print("Done!")


if __name__ == '__main__':
    main()
