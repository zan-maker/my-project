#!/usr/bin/env python3
"""
AnnotateX ARC-AGI-2 Solver v2 — Advanced Heuristic + Shape Analysis
Multi-strategy solver with 25+ pattern detection algorithms.
"""

import json
import numpy as np
from copy import deepcopy
from collections import Counter, defaultdict
import os, glob

print("Loading ARC-AGI-2 data...")
data_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'arc-agi_test_challenges.json' in files:
        data_dir = root
        break
if data_dir is None:
    candidates = glob.glob('/kaggle/input/*/arc-agi_test_challenges.json')
    if candidates:
        data_dir = os.path.dirname(candidates[0])

with open(os.path.join(data_dir, 'arc-agi_test_challenges.json')) as f:
    test_data = json.load(f)
with open(os.path.join(data_dir, 'arc-agi_evaluation_challenges.json')) as f:
    eval_data = json.load(f)
with open(os.path.join(data_dir, 'arc-agi_evaluation_solutions.json')) as f:
    eval_solutions = json.load(f)
print(f"Test: {len(test_data)}, Eval: {len(eval_data)} tasks")

# ============================================================
# GRID UTILITIES
# ============================================================

def g2a(grid):
    return np.array(grid, dtype=int)

def a2g(arr):
    return arr.tolist()

def shape(grid):
    return (len(grid), len(grid[0])) if grid and grid[0] else (0,0)

def unique_colors(grid):
    return set(c for row in grid for c in row)

def most_common_color(grid):
    vals = [c for row in grid for c in row if c != 0]
    return Counter(vals).most_common(1)[0][0] if vals else 0

def crop_to_content(grid):
    arr = g2a(grid)
    nz = np.nonzero(arr)
    if len(nz[0]) == 0:
        return grid
    return arr[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1].tolist()

def pad_grid(grid, h, w, val=0):
    arr = np.full((h, w), val, dtype=int)
    g = g2a(grid)
    gh, gw = g.shape
    arr[:min(gh,h), :min(gw,w)] = g[:min(gh,h), :min(gw,w)]
    return arr.tolist()

def count_objects(grid):
    """Count connected components of non-zero cells."""
    arr = g2a(grid)
    visited = np.zeros_like(arr, dtype=bool)
    objects = []
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if arr[r,c] != 0 and not visited[r,c]:
                obj_color = arr[r,c]
                cells = []
                queue = [(r,c)]
                visited[r,c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr,cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (0<=nr<arr.shape[0] and 0<=nc<arr.shape[1]
                            and not visited[nr,nc] and arr[nr,nc]==obj_color):
                            visited[nr,nc] = True
                            queue.append((nr,nc))
                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                objects.append({
                    'color': obj_color,
                    'cells': cells,
                    'size': len(cells),
                    'bbox': (min(rs), min(cs), max(rs), max(cs))
                })
    return objects

def extract_object_grid(grid, obj):
    r1,c1,r2,c2 = obj['bbox']
    return g2a(grid)[r1:r2+1, c1:c2+1].tolist()


# ============================================================
# ADVANCED HEURISTIC SOLVER v2
# ============================================================

class AdvancedSolver:
    def solve(self, task):
        best_pred = None
        best_conf = -1
        for solver_fn in self._get_solvers():
            try:
                pred, conf = solver_fn(task)
                if pred is not None and conf > best_conf:
                    best_pred = pred
                    best_conf = conf
            except:
                continue
        return best_pred

    def _get_solvers(self):
        return [
            self.identity, self.color_map, self.tile,
            self.mirror_h, self.mirror_v, self.mirror_d,
            self.rot90, self.rot180, self.rot270,
            self.scale2x, self.scale3x,
            self.extract_obj, self.crop_content,
            self.fill_interior, self.border_only,
            self.remove_color, self.replace_color_pair,
            self.compress_rows, self.compress_cols,
            self.move_object, self.relocate_to_corner,
            self.count_to_size, self.sort_objects,
            self.overlay_combine, self.diff_extract,
            self.denoise, self.invert_colors,
        ]

    def _validate(self, task, fn):
        preds = []
        total = 0
        for ex in task['train']:
            try:
                pred = fn(ex['input'], ex, task)
                if pred is None:
                    return 0, None
                pa, ea = g2a(pred), g2a(ex['output'])
                if pa.shape == ea.shape:
                    total += np.sum(pa == ea) / pa.size
                    preds.append(pred)
                else:
                    return 0, None
            except:
                return 0, None
        return total / len(task['train']), preds

    def identity(self, task):
        def fn(inp, ex, t): return deepcopy(inp)
        c, p = self._validate(task, fn)
        return (fn(task['test'][0]['input'], None, task), c) if c >= 0.95 else (None, 0)

    def color_map(self, task):
        e0 = task['train'][0]
        m = self._infer_cmap(e0['input'], e0['output'])
        if not m: return (None, 0)
        def fn(inp, ex, t):
            return [[m.get(c,c) for c in row] for row in inp]
        c, p = self._validate(task, fn)
        return (fn(task['test'][0]['input'], None, task), c) if c >= 0.95 else (None, 0)

    def _infer_cmap(self, inp, out):
        ia, oa = g2a(inp), g2a(out)
        if ia.shape != oa.shape: return None
        m = {}
        for i in range(ia.shape[0]):
            for j in range(ia.shape[1]):
                s,d = int(ia[i,j]), int(oa[i,j])
                if s in m:
                    if m[s] != d: return None
                else:
                    m[s] = d
        return m if m and not all(k==v for k,v in m.items()) else None

    def tile(self, task):
        def fn(inp, ex, t):
            f = self._tile_factor(ex['input'], ex['output'])
            if not f: return None
            fy, fx = (f,f) if isinstance(f,int) else f
            return [c for r in range(fy) for row in inp for c2 in range(fx) for c in row]
        c, p = self._validate(task, fn)
        return (fn(task['test'][0]['input'], None, task), c) if c >= 0.95 else (None, 0)

    def _tile_factor(self, inp, out):
        ih,iw = shape(inp); oh,ow = shape(out)
        if ih==0 or iw==0 or oh%ih or ow%iw: return None
        fy,fx = oh//ih, ow//iw
        return fy if fy==fx and fy>1 else ((fy,fx) if (fy>1 or fx>1) else None)

    def mirror_h(self, task):
        def fn(inp,ex,t): return [row[::-1] for row in inp]
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def mirror_v(self, task):
        def fn(inp,ex,t): return inp[::-1]
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def mirror_d(self, task):
        def fn(inp,ex,t): return g2a(inp).T.tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def rot90(self, task):
        def fn(inp,ex,t): return np.rot90(g2a(inp),-1).tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def rot180(self, task):
        def fn(inp,ex,t): return np.rot90(g2a(inp),2).tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def rot270(self, task):
        def fn(inp,ex,t): return np.rot90(g2a(inp),1).tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def _scale(self, task, f):
        def fn(inp,ex,t): return np.repeat(np.repeat(g2a(inp),f,axis=0),f,axis=1).tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def scale2x(self, task): return self._scale(task,2)
    def scale3x(self, task): return self._scale(task,3)

    def extract_obj(self, task):
        def fn(inp,ex,t): return crop_to_content(inp)
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def crop_content(self, task):
        def fn(inp,ex,t):
            cropped = crop_to_content(inp)
            oh, ow = shape(ex['output'])
            return pad_grid(cropped, oh, ow)
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],task['train'][0],task),c) if c>=0.95 else (None,0)

    def fill_interior(self, task):
        def fn(inp,ex,t):
            arr = g2a(inp).astype(int)
            nz = np.nonzero(arr)
            if len(nz[0])==0: return None
            fc = int(Counter(arr[arr>0].flatten()).most_common(1)[0][0])
            arr[nz[0].min()+1:nz[0].max(), nz[1].min()+1:nz[1].max()] = fc
            return arr.tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def border_only(self, task):
        def fn(inp,ex,t):
            arr = g2a(inp)
            res = np.zeros_like(arr)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if arr[i,j]!=0:
                        bord = False
                        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni,nj=i+di,j+dj
                            if ni<0 or ni>=arr.shape[0] or nj<0 or nj>=arr.shape[1] or arr[ni,nj]==0:
                                bord=True; break
                        if bord: res[i,j]=arr[i,j]
            return res.tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def remove_color(self, task):
        e0 = task['train'][0]
        removed = unique_colors(e0['input']) - unique_colors(e0['output'])
        if len(removed)!=1: return (None,0)
        rc = removed.pop()
        def fn(inp,ex,t): return [[0 if c==rc else c for c in row] for row in inp]
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def replace_color_pair(self, task):
        """Detect 2-color swap (e.g., red<->blue)."""
        e0 = task['train'][0]
        ic, oc = unique_colors(e0['input']), unique_colors(e0['output'])
        if shape(e0['input']) != shape(e0['output']): return (None,0)
        only_in = ic - oc
        only_out = oc - ic
        if len(only_in)==1 and len(only_out)==1:
            m = {only_in.pop(): only_out.pop()}
            def fn(inp,ex,t): return [[m.get(c,c) for c in row] for row in inp]
            c,_ = self._validate(task,fn)
            return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)
        return (None,0)

    def compress_rows(self, task):
        def fn(inp,ex,t): return [row for row in inp if any(c!=0 for c in row)]
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def compress_cols(self, task):
        def fn(inp,ex,t):
            if not inp: return None
            nc = len(inp[0])
            ne = [j for j in range(nc) if any(row[j]!=0 for row in inp)]
            return [[row[j] for j in ne] for row in inp]
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def move_object(self, task):
        e0 = task['train'][0]
        io, oo = count_objects(e0['input']), count_objects(e0['output'])
        if not io or not oo: return (None,0)
        if len(io)!=len(oo): return (None,0)
        # Match objects by color and size
        pairs = []
        for o1 in io:
            for o2 in oo:
                if o1['color']==o2['color'] and o1['size']==o2['size']:
                    pairs.append((o1,o2))
        if len(pairs)!=len(io): return (None,0)
        # Check consistent offset
        offsets = [(o2['bbox'][0]-o1['bbox'][0], o2['bbox'][1]-o1['bbox'][1]) for o1,o2 in pairs]
        if len(set(offsets))!=1: return (None,0)
        offr, offc = offsets[0]
        # Validate on all train
        valid = True
        for ex in task['train'][1:]:
            ti, toi = count_objects(ex['input']), count_objects(ex['output'])
            if len(ti)!=len(toi): valid=False; break
            tp = []
            for o1 in ti:
                for o2 in toi:
                    if o1['color']==o2['color'] and o1['size']==o2['size']:
                        tp.append((o1,o2))
            if len(tp)!=len(ti): valid=False; break
            ts = set((o2['bbox'][0]-o1['bbox'][0], o2['bbox'][1]-o1['bbox'][1]) for o1,o2 in tp)
            if ts != {(offr,offc)}: valid=False; break
        if not valid: return (None,0)
        # Apply to test
        arr = g2a(task['test'][0]['input'])
        out = np.zeros_like(arr)
        for o in count_objects(task['test'][0]['input']):
            for r,c in o['cells']:
                nr,nc = r+offr, c+offc
                if 0<=nr<out.shape[0] and 0<=nc<out.shape[1]:
                    out[nr,nc] = arr[r,c]
        return (out.tolist(), 0.9)

    def relocate_to_corner(self, task):
        """Move object to top-left corner."""
        def fn(inp,ex,t):
            out = g2a(ex['output'])
            result = np.zeros_like(out)
            objs = count_objects(inp)
            if not objs: return None
            o = max(objs, key=lambda x: x['size'])
            og = extract_object_grid(inp, o)
            oh, ow = len(og), len(og[0])
            for r in range(min(oh, result.shape[0])):
                for c in range(min(ow, result.shape[1])):
                    result[r,c] = og[r][c]
            return result.tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],task['train'][0],task),c) if c>=0.95 else (None,0)

    def count_to_size(self, task):
        """Count objects in input and create output grid of that size."""
        def fn(inp,ex,t):
            objs = count_objects(inp)
            cnt = len(objs)
            out = g2a(ex['output'])
            result = np.zeros_like(out)
            if cnt > 0 and cnt <= result.shape[0] * result.shape[1]:
                # Fill with most common color from output
                mc = most_common_color(ex['output'])
                for i in range(min(cnt, result.shape[0])):
                    for j in range(min(1, result.shape[1])):
                        result[i,j] = mc
            return result.tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],task['train'][0],task),c) if c>=0.95 else (None,0)

    def sort_objects(self, task):
        """Sort objects by some property (size, color, position)."""
        def fn(inp,ex,t):
            objs = count_objects(inp)
            if not objs: return None
            out = g2a(ex['output'])
            result = np.zeros_like(out)
            # Try sorting by size ascending
            sorted_objs = sorted(objs, key=lambda x: x['size'])
            cur_r = 0
            for obj in sorted_objs:
                og = extract_object_grid(inp, obj)
                for dr, dc in obj['cells']:
                    orig_r, orig_c = dr - obj['bbox'][0], dc - obj['bbox'][1]
                    if cur_r + orig_r < result.shape[0] and orig_c < result.shape[1]:
                        result[cur_r + orig_r, orig_c] = inp[dr][dc]
                cur_r += obj['bbox'][2] - obj['bbox'][0] + 1
            return result.tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],task['train'][0],task),c) if c>=0.95 else (None,0)

    def overlay_combine(self, task):
        """Overlay two or more objects into one."""
        def fn(inp,ex,t):
            objs = count_objects(inp)
            if len(objs) < 2: return None
            out = g2a(ex['output'])
            result = np.zeros_like(out)
            for obj in objs:
                for r,c in obj['cells']:
                    if r < result.shape[0] and c < result.shape[1]:
                        result[r,c] = obj['color']
            return result.tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],task['train'][0],task),c) if c>=0.95 else (None,0)

    def diff_extract(self, task):
        """Extract the difference between input objects."""
        def fn(inp,ex,t):
            objs = count_objects(inp)
            if len(objs) < 2: return None
            # XOR of objects
            arr = g2a(inp)
            out = g2a(ex['output'])
            result = np.zeros_like(out)
            counts = Counter()
            for obj in objs:
                for r,c in obj['cells']:
                    counts[(r,c)] += 1
            for (r,c), cnt in counts.items():
                if cnt == 1 and r < result.shape[0] and c < result.shape[1]:
                    result[r,c] = arr[r,c]
            return result.tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],task['train'][0],task),c) if c>=0.95 else (None,0)

    def denoise(self, task):
        """Remove small objects (noise)."""
        def fn(inp,ex,t):
            objs = count_objects(inp)
            if not objs: return None
            sizes = [o['size'] for o in objs]
            if len(sizes) < 2: return None
            threshold = min(sizes)
            arr = g2a(inp).copy()
            for obj in objs:
                if obj['size'] <= threshold:
                    for r,c in obj['cells']:
                        arr[r,c] = 0
            return arr.tolist()
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)

    def invert_colors(self, task):
        """Map each non-zero color to another color."""
        e0 = task['train'][0]
        if shape(e0['input']) != shape(e0['output']): return (None,0)
        ia, oa = g2a(e0['input']), g2a(e0['output'])
        mapping = {}
        for i in range(ia.shape[0]):
            for j in range(ia.shape[1]):
                s, d = int(ia[i,j]), int(oa[i,j])
                if s != d:
                    if s in mapping:
                        if mapping[s] != d: return (None,0)
                    else:
                        mapping[s] = d
        if not mapping: return (None,0)
        def fn(inp,ex,t):
            return [[mapping.get(c,c) for c in row] for row in inp]
        c,_ = self._validate(task,fn)
        return (fn(task['test'][0]['input'],None,task),c) if c>=0.95 else (None,0)


# ============================================================
# EVAL VALIDATION (to report accuracy before submission)
# ============================================================

print("\nValidating on evaluation set (solutions available)...")
solver = AdvancedSolver()
eval_solved = 0
eval_total = len(eval_data)

for tid, task in eval_data.items():
    pred = solver.solve(task)
    if pred is not None and tid in eval_solutions:
        sol = eval_solutions[tid]
        if isinstance(sol, list):
            sol = sol[0]
        pa, sa = g2a(pred), g2a(sol)
        if pa.shape == sa.shape and np.array_equal(pa, sa):
            eval_solved += 1

print(f"Eval solved: {eval_solved}/{eval_total} ({100*eval_solved/eval_total:.1f}%)")


# ============================================================
# GENERATE TEST SUBMISSION
# ============================================================

print("\nGenerating test submission...")
submission = {}
test_solved = 0

for i, (task_id, task) in enumerate(test_data.items()):
    if (i+1) % 50 == 0:
        print(f"  Progress: {i+1}/{len(test_data)}...")

    test_input = task['test'][0]['input']
    prediction = solver.solve(task)

    if prediction is not None:
        test_solved += 1
        attempt_1 = prediction
        attempt_2 = deepcopy(test_input)
    else:
        attempt_1 = deepcopy(test_input)
        attempt_2 = deepcopy(test_input)

    submission[task_id] = [{"attempt_1": attempt_1, "attempt_2": attempt_2}]

print(f"\n{'='*50}")
print(f"AnnotateX ARC-AGI-2 Solver v2")
print(f"{'='*50}")
print(f"Eval solved: {eval_solved}/{eval_total} ({100*eval_solved/eval_total:.1f}%)")
print(f"Test tasks: {len(test_data)}")
print(f"Test heuristic-solved: {test_solved} ({100*test_solved/len(test_data):.1f}%)")

with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(submission, f)
print(f"\nSubmission saved!")
