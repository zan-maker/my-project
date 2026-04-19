#!/usr/bin/env python3
"""
ARC-AGI-2 Solver v2 — AnnotateX Enhanced Methodology
Comprehensive multi-strategy approach with connected components, composite transforms,
shape analysis, row/column operations, object counting, symmetry, pattern completion,
scaling detection, extraction, and tiling/repetition.
"""

import json
import numpy as np
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import permutations, product
from functools import lru_cache

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
# TASK-LEVEL SOLVER WITH MULTIPLE ATTEMPT STRATEGIES
# ============================================================

def solve_task(task):
    """Solve a single ARC task. Returns list of 2 attempt grids."""
    solver = HeuristicSolverV2()
    test_input = task['test'][0]['input']

    # Attempt 1: best heuristic
    pred1, conf1, strategy1 = solver.solve(task)

    # Attempt 2: second-best or alternative strategy
    pred2 = None
    if pred1 is not None:
        # Try some specific alternatives for attempt 2
        alternatives = [
            lambda t: solver.extract_bounding_box_transform(t),
            lambda t: solver.same_as_input(t),
            lambda t: solver.color_mapping(t),
            lambda t: solver.composite_transform(t),
        ]
        best_alt_conf = -1
        for alt_fn in alternatives:
            try:
                alt_pred, alt_conf = alt_fn(task)
                if alt_pred is not None and alt_conf > best_alt_conf and not _grids_equal(alt_pred, pred1):
                    pred2 = alt_pred
                    best_alt_conf = alt_conf
            except Exception:
                continue

    if pred1 is None:
        pred1 = deepcopy(test_input)
    if pred2 is None:
        # Fallback attempts
        pred2 = deepcopy(test_input)

    # Ensure valid format (list of lists of ints)
    pred1 = _ensure_grid(pred1)
    pred2 = _ensure_grid(pred2)

    return [pred1, pred2]


def _grids_equal(g1, g2):
    """Check if two grids are identical."""
    try:
        a1 = grid_to_array(g1)
        a2 = grid_to_array(g2)
        return a1.shape == a2.shape and np.array_equal(a1, a2)
    except Exception:
        return False


def _ensure_grid(pred):
    """Ensure prediction is a valid grid (list of lists of ints)."""
    if pred is None:
        return [[0]]
    if isinstance(pred, np.ndarray):
        return pred.astype(int).tolist()
    return pred


# ============================================================
# SUBMISSION GENERATION
# ============================================================

def generate_submission(test_data, output_path):
    """Generate submission.json from test challenges."""
    submission = {}
    total_tasks = len(test_data)
    solved = 0
    strategy_counts = Counter()

    for i, (task_id, task) in enumerate(test_data.items()):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Processing task {i+1}/{total_tasks} ({task_id})...")

        attempts = solve_task(task)
        submission[task_id] = [
            {"attempt_1": attempts[0], "attempt_2": attempts[1]}
        ]

        if not _grids_equal(attempts[0], task['test'][0]['input']):
            solved += 1

    print(f"\n{'='*50}")
    print(f"ARC-AGI-2 Solver v2 Results")
    print(f"{'='*50}")
    print(f"Total tasks: {total_tasks}")
    print(f"Non-trivial predictions: {solved}")
    print(f"Identity fallback: {total_tasks - solved}")

    save_json(output_path, submission)
    print(f"\nSubmission saved to: {output_path}")
    return submission


def validate_on_training(train_challenges_path, train_solutions_path):
    """Validate the heuristic solver on training data."""
    solver = HeuristicSolverV2()
    train_challenges = load_json(train_challenges_path)
    train_solutions = load_json(train_solutions_path)

    solved = 0
    total = len(train_challenges)
    strategy_counts = Counter()

    for task_id, task in train_challenges.items():
        pred, conf, strategy = solver.solve(task)
        if pred is None:
            continue

        if task_id in train_solutions:
            solution = train_solutions[task_id][0]
            pred_arr = grid_to_array(pred)
            sol_arr = grid_to_array(solution)

            if pred_arr.shape == sol_arr.shape and np.array_equal(pred_arr, sol_arr):
                solved += 1
                strategy_counts[strategy] += 1
                print(f"  SOLVED: {task_id} (strategy: {strategy}, conf: {conf:.2f})")

    print(f"\n{'='*50}")
    print(f"Training Validation Results (Solver v2)")
    print(f"{'='*50}")
    print(f"Solved: {solved}/{total} ({100*solved/total:.1f}%)")
    print(f"\nStrategy breakdown:")
    for strat, count in strategy_counts.most_common():
        print(f"  {strat}: {count}")
    return solved


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import sys
    import os

    KAGGLE_PATH = '/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_2_data/'
    LOCAL_PATH = '/home/z/my-project/arc-agi-2-data/'

    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        print("Validating on training data with Solver v2...")
        validate_on_training(
            os.path.join(LOCAL_PATH, 'arc-agi_training_challenges.json'),
            os.path.join(LOCAL_PATH, 'arc-agi_training_solutions.json')
        )
    elif len(sys.argv) > 1 and sys.argv[1] == 'kaggle':
        # Running on Kaggle
        print("Loading test challenges from Kaggle...")
        test_data = load_json(os.path.join(KAGGLE_PATH, 'arc-agi_2_test_challenges.json'))
        print("Generating submission...")
        submission = generate_submission(test_data, '/kaggle/working/submission.json')
        print("Done!")
    else:
        # Local mode
        test_path = os.path.join(LOCAL_PATH, 'arc-agi_test_challenges.json')
        if os.path.exists(test_path):
            print("Loading test challenges...")
            test_data = load_json(test_path)
            print("Generating submission...")
            submission = generate_submission(test_data, os.path.join(LOCAL_PATH, 'submission_v2.json'))
            print("Done!")
        else:
            print("No test data found. Running training validation...")
            validate_on_training(
                os.path.join(LOCAL_PATH, 'arc-agi_training_challenges.json'),
                os.path.join(LOCAL_PATH, 'arc-agi_training_solutions.json')
            )
