#!/usr/bin/env python3
"""
SmartAgent V2 — Hypothesis-Driven Exploration Agent for ARC-AGI-3 (ARC Prize 2026)

Architecture:
  1. HypothesisBeliefTracker  — forms & tests conjectures about what each action does
  2. SpatialMemory            — 2D explored map, player position, boundary detection
  3. ObjectTracker            — detect non-zero objects, track across frames
  4. SmartClickTargeter       — prioritise clicks on unexplored / nearby / changed objects
  5. StuckDetector            — detect loops, no-progress, recovery strategies
  6. LevelKnowledgeTransfer   — remember mechanics between levels/games
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
import traceback
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("SmartAgentV2")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSTANTS & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GRID_MAX = 64
COLOUR_MAX = 16
BACKGROUND = 0

# Directions: (name, dr, dc)
DIRS_4 = [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]
DIRS_8 = DIRS_4 + [("up_left", -1, -1), ("up_right", -1, 1),
                   ("down_left", 1, -1), ("down_right", 1, 1)]

# Default action-id → likely semantic meaning (tested & overridden at runtime)
_ACTION_LABELS: Dict[int, str] = {
    0: "reset",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "interact",
    6: "click",
    7: "undo",
}


def _clamp(v: int, lo: int = 0, hi: int = GRID_MAX - 1) -> int:
    return max(lo, min(hi, v))


def _manhattan(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r1 - r2) + abs(c1 - c2)


def _chebyshev(r1: int, c1: int, r2: int, c2: int) -> int:
    return max(abs(r1 - r2), abs(c1 - c2))


def _grid_hash(frame_3d: list | None) -> str:
    """MD5 of the last layer of a 3-D frame."""
    try:
        if not frame_3d:
            return "__empty__"
        arr = np.array(frame_3d[-1], dtype=np.int8)
        return hashlib.md5(arr.tobytes()).hexdigest()[:16]
    except Exception:
        return "__err__"


def _frame_to_grid(frame_3d: list | None) -> np.ndarray | None:
    try:
        if frame_3d:
            return np.array(frame_3d[-1], dtype=np.int8)
    except Exception:
        pass
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. HYPOTHESIS BELIEF TRACKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Hypothesis:
    """Single conjecture with confidence tracking."""

    __slots__ = ("desc", "action_id", "confidence", "for_count", "against_count",
                 "verified", "falsified")

    def __init__(self, desc: str, action_id: int = -1):
        self.desc = desc
        self.action_id = action_id
        self.confidence = 1.0
        self.for_count = 0
        self.against_count = 0
        self.verified = False
        self.falsified = False

    def evidence(self, positive: bool) -> None:
        if positive:
            self.for_count += 1
            self.confidence = min(self.confidence + 1.0, 10.0)
            if self.confidence >= 4.0:
                self.verified = True
        else:
            self.against_count += 1
            self.confidence = max(self.confidence - 0.5, -5.0)
            if self.confidence <= -1.0:
                self.falsified = True

    def __repr__(self) -> str:
        tag = "V" if self.verified else ("F" if self.falsified else "?")
        return f"[{tag}] conf={self.confidence:+.1f} {self.desc}"


class HypothesisBeliefTracker:
    """
    Maintains a belief-state over game mechanics.

    * Seeded with priors about every action.
    * Updated after every frame transition.
    * Exposes methods to query which actions still need testing and
      which directions are now verified.
    """

    def __init__(self):
        self.hypotheses: List[_Hypothesis] = []
        self._seeded = False
        # Per-action tallies
        self.effect_counts: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"changed": 0, "unchanged": 0, "level_up": 0, "game_over": 0}
        )
        # Learned movement mapping  action_id → (dr, dc)
        self.movement: Dict[int, Tuple[int, int]] = {}

    # ── seeding ──────────────────────────────────────────────────────

    def seed(self, available_action_ids: List[int]) -> None:
        """Add default hypotheses based on available actions."""
        if self._seeded:
            return
        self._seeded = True

        priors = [
            (1, "ACTION1 causes upward movement or input-1"),
            (2, "ACTION2 causes downward movement or input-2"),
            (3, "ACTION3 causes leftward movement or input-3"),
            (4, "ACTION4 causes rightward movement or input-4"),
            (5, "ACTION5 is interact / action button"),
            (6, "ACTION6 is click / point at coordinate"),
            (7, "ACTION7 is undo last action"),
        ]
        for aid, desc in priors:
            if aid in available_action_ids:
                self.hypotheses.append(_Hypothesis(desc, action_id=aid))

        # Structural priors
        self.hypotheses.extend([
            _Hypothesis("Non-zero cells are obstacles or objects"),
            _Hypothesis("Zero cells are walkable floor"),
            _Hypothesis("Moving into a wall produces no state change"),
            _Hypothesis("Player sprite is visible in the grid"),
            _Hypothesis("Reaching a goal location completes the level"),
        ])
        logger.info("Seeded %d hypotheses", len(self.hypotheses))

    # ── update after observing a transition ─────────────────────────

    def observe(self, action_id: int, prev_hash: str, curr_hash: str,
                prev_levels: int, curr_levels: int,
                is_game_over: bool) -> None:
        """Record the outcome of taking *action_id*."""
        ec = self.effect_counts[action_id]
        changed = prev_hash != curr_hash
        ec["changed" if changed else "unchanged"] += 1
        if curr_levels > prev_levels:
            ec["level_up"] += 1
        if is_game_over:
            ec["game_over"] += 1

        # Update linked hypotheses
        for h in self.hypotheses:
            if h.action_id != action_id:
                continue
            if curr_levels > prev_levels:
                h.evidence(True)
            elif changed:
                h.evidence(True)
            else:
                h.evidence(False)

    # ── queries ──────────────────────────────────────────────────────

    def untested_action_ids(self, available: List[int]) -> List[int]:
        """Return action IDs that haven't been verified or falsified yet."""
        resolved = {h.action_id for h in self.hypotheses
                    if h.verified or h.falsified}
        return [a for a in available if a not in resolved and a != 0]

    def verified_movement_directions(self) -> Dict[int, Tuple[int, int]]:
        return {k: v for k, v in self.movement.items()}

    def summary(self) -> str:
        v = sum(1 for h in self.hypotheses if h.verified)
        f = sum(1 for h in self.hypotheses if h.falsified)
        return f"H:{v}v/{f}f/{len(self.hypotheses)}total  mov={self.movement}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. SPATIAL MEMORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SpatialMemory:
    """
    Maintains a 2-D explored map, player position, boundary / wall
    detection, and frontier-based exploration guidance.
    """

    def __init__(self):
        self.visited: Set[Tuple[int, int]] = set()
        self.walls: Set[Tuple[int, int]] = set()
        self.floors: Set[Tuple[int, int]] = set()
        self.player_pos: Optional[Tuple[int, int]] = None
        self.player_color: int = -1
        self.grid_h: int = 0
        self.grid_w: int = 0

    def reset_level(self) -> None:
        self.visited.clear()
        self.walls.clear()
        self.floors.clear()
        self.grid_h = 0
        self.grid_w = 0
        # keep player_color across levels

    # ── update from grid ─────────────────────────────────────────────

    def update(self, grid: np.ndarray | None,
               prev_grid: np.ndarray | None = None,
               last_action_id: int = -1) -> None:
        if grid is None:
            return
        h, w = grid.shape
        self.grid_h, self.grid_w = h, w

        for r in range(h):
            for c in range(w):
                v = int(grid[r, c])
                (self.walls if v != BACKGROUND else self.floors).add((r, c))

        # Infer player from frame diff
        if prev_grid is not None and last_action_id >= 1:
            self._track_player(grid, prev_grid, last_action_id)

    def _track_player(self, cur: np.ndarray, prev: np.ndarray,
                      action_id: int) -> None:
        h, w = cur.shape
        ph, pw = prev.shape
        disappeared: List[Tuple[int, int, int]] = []
        appeared: List[Tuple[int, int, int]] = []
        for r in range(min(h, ph)):
            for c in range(min(w, pw)):
                cv, pv = int(cur[r, c]), int(prev[r, c])
                if cv != pv:
                    if pv != BACKGROUND:
                        disappeared.append((r, c, pv))
                    if cv != BACKGROUND:
                        appeared.append((r, c, cv))
        if disappeared and appeared:
            old_r, old_c, self.player_color = disappeared[0]
            new_r, new_c, _ = appeared[0]
            self.player_pos = (new_r, new_c)

    # ── exploration helpers ──────────────────────────────────────────

    def mark_visited_around(self, cr: int, cc: int, radius: int = 2) -> None:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                self.visited.add((cr + dr, cc + dc))

    def frontier_cells(self) -> List[Tuple[int, int]]:
        """Cells adjacent to visited that are not walls and not yet visited."""
        frontier: Set[Tuple[int, int]] = set()
        for r, c in self.visited:
            for dr, dc in DIRS_4:
                nr, nc = r + dr[1], c + dc[2]
                if (0 <= nr < self.grid_h and 0 <= nc < self.grid_w
                        and (nr, nc) not in self.visited
                        and (nr, nc) not in self.walls):
                    frontier.add((nr, nc))
        return list(frontier)

    def best_direction_toward_frontier(self) -> Optional[Tuple[int, int]]:
        """(dr, dc) toward the nearest unexplored floor cell, or None."""
        if self.player_pos is None:
            return None
        frontiers = self.frontier_cells()
        if not frontiers:
            return None
        pr, pc = self.player_pos
        best = min(frontiers, key=lambda f: _manhattan(pr, pc, f[0], f[1]))
        dr = _clamp(best[0] - pr, -1, 1)
        dc = _clamp(best[1] - pc, -1, 1)
        return (dr, dc)

    def explore_ratio(self) -> float:
        total = len(self.floors)
        return len(self.visited) / total if total else 0.0

    def summary(self) -> str:
        return (f"pos={self.player_pos}  walls={len(self.walls)}  "
                f"explored={len(self.visited)}/{len(self.floors)} "
                f"({self.explore_ratio():.0%})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. OBJECT TRACKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Obj:
    """A tracked game object."""

    __slots__ = ("oid", "color", "cells", "size", "cr", "cc",
                 "born", "last_seen", "moved", "click_tries")

    def __init__(self, oid: int, cells: List[Tuple[int, int]],
                 color: int, frame: int):
        self.oid = oid
        self.color = color
        self.cells = set(cells)
        self.size = len(cells)
        self.cr = sum(r for r, _ in cells) / max(self.size, 1)
        self.cc = sum(c for _, c in cells) / max(self.size, 1)
        self.born = frame
        self.last_seen = frame
        self.moved = False
        self.click_tries = 0

    def refresh(self, cells: List[Tuple[int, int]], frame: int) -> bool:
        old = (self.cr, self.cc)
        self.cells = set(cells)
        self.size = len(cells)
        self.cr = sum(r for r, _ in cells) / max(self.size, 1)
        self.cc = sum(c for _, c in cells) / max(self.size, 1)
        self.last_seen = frame
        dist = math.hypot(self.cr - old[0], self.cc - old[1])
        self.moved = dist > 0.5
        return self.moved

    def bbox(self) -> Tuple[int, int, int, int]:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return min(rs), min(cs), max(rs), max(cs)

    def centroid_cell(self) -> Tuple[int, int]:
        return int(round(self.cr)), int(round(self.cc))


class ObjectTracker:
    """
    Detects non-zero connected components (objects), tracks them across
    frames using overlap matching, and exposes click-priority helpers.
    """

    def __init__(self):
        self.objects: Dict[int, _Obj] = {}
        self._next_id = 1
        self.new_ids: Set[int] = set()
        self.gone_ids: Set[int] = set()

    def reset_level(self) -> None:
        self.objects.clear()
        self._next_id = 1
        self.new_ids.clear()
        self.gone_ids.clear()

    # ── connected-component detection (4-connected, per-colour) ─────

    @staticmethod
    def _components(grid: np.ndarray) -> List[List[Tuple[int, int]]]:
        h, w = grid.shape
        seen = np.zeros((h, w), dtype=bool)
        comps: List[List[Tuple[int, int]]] = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] == BACKGROUND or seen[r, c]:
                    continue
                col = grid[r, c]
                comp: List[Tuple[int, int]] = []
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for _, dr, dc in DIRS_4:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == col:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                comps.append(comp)
        return comps

    # ── per-frame update ─────────────────────────────────────────────

    def update(self, grid: np.ndarray | None, frame_num: int) -> None:
        if grid is None:
            return
        self.new_ids.clear()
        self.gone_ids.clear()

        comps = self._components(grid)

        # snapshot current objects
        obj_cell_map: Dict[Tuple[int, int], int] = {}
        for oid, obj in self.objects.items():
            for cell in obj.cells:
                obj_cell_map[cell] = oid

        matched_obj: Set[int] = set()
        matched_comp: Set[int] = set()

        for oid, obj in list(self.objects.items()):
            best_idx, best_overlap = -1, 0
            for i, comp in enumerate(comps):
                if i in matched_comp:
                    continue
                overlap = len(obj.cells & set(comp))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            if best_idx >= 0 and best_overlap >= max(1, obj.size * 0.25):
                obj.refresh(comps[best_idx], frame_num)
                matched_obj.add(oid)
                matched_comp.add(best_idx)

        for oid in self.objects:
            if oid not in matched_obj:
                self.gone_ids.add(oid)

        for i, comp in enumerate(comps):
            if i not in matched_comp and comp:
                r0, c0 = comp[0]
                color = int(grid[r0, c0])
                obj = _Obj(self._next_id, comp, color, frame_num)
                self.objects[self._next_id] = obj
                self.new_ids.add(self._next_id)
                self._next_id += 1
                matched_comp.add(i)

    def record_click_on(self, r: int, c: int) -> None:
        """Increment click counter for the object at (r,c)."""
        for obj in self.objects.values():
            if (r, c) in obj.cells:
                obj.click_tries += 1
                return

    def summary(self) -> str:
        return (f"objs={len(self.objects)}  new={len(self.new_ids)}  "
                f"gone={len(self.gone_ids)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. SMART CLICK TARGETER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SmartClickTargeter:
    """
    Scores every non-zero object to produce a ranked click-target list.

    Priority factors (weights tuneable):
      + new / freshly-appeared objects
      + recently-changed objects
      + proximity to the player
      + small distinctive objects (likely interactive items)
      + low prior click-attempt count
    """

    WEIGHTS = {
        "new":          50.0,
        "changed":      30.0,
        "proximity":    20.0,
        "small_bonus":  15.0,
        "medium_bonus":  8.0,
        "fresh_click":  10.0,
    }
    MAX_CANDIDATES = 12

    def __init__(self, obj_tracker: ObjectTracker):
        self.ot = obj_tracker
        self.tried: Set[Tuple[int, int]] = set()

    def reset_level(self) -> None:
        self.tried.clear()

    def rank(self, player_pos: Optional[Tuple[int, int]] = None,
             max_n: int = MAX_CANDIDATES) -> List[Tuple[int, int, float]]:
        """Return [(row, col, score), ...] sorted best-first."""
        candidates: List[Tuple[int, int, float]] = []
        W = self.WEIGHTS

        for oid, obj in self.ot.objects.items():
            cr, cc = obj.centroid_cell()
            if (cr, cc) in self.tried and obj.click_tries >= 3:
                continue
            score = 0.0
            if oid in self.ot.new_ids:
                score += W["new"]
            if obj.moved:
                score += W["changed"]
            if player_pos is not None:
                d = _manhattan(player_pos[0], player_pos[1], cr, cc)
                score += max(0.0, W["proximity"] - d)
            if obj.size <= 4:
                score += W["small_bonus"]
            elif obj.size <= 9:
                score += W["medium_bonus"]
            score += max(0.0, W["fresh_click"] - obj.click_tries * 5)
            if score > 0:
                candidates.append((cr, cc, score))

        candidates.sort(key=lambda t: t[2], reverse=True)
        return candidates[:max_n]

    def mark_tried(self, r: int, c: int) -> None:
        self.tried.add((r, c))
        self.ot.record_click_on(r, c)

    def summary(self) -> str:
        return f"tried_clicks={len(self.tried)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. STUCK DETECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StuckDetector:
    """
    Detects loops (repeated states) and no-progress situations.
    Provides ordered recovery strategies.
    """

    LOOP_THRESHOLD = 3
    NO_PROGRESS_THRESHOLD = 8
    MAX_RECOVERY_STEPS = 10

    def __init__(self):
        self.visit_count: Counter = Counter()
        self.recent: deque = deque(maxlen=24)
        self.no_progress = 0
        self.loop = False
        self.recovery = False
        self._rec_idx = 0
        self._rec_seq: List[int] = []
        self._last_hash: str = ""

    def reset_level(self) -> None:
        self.visit_count.clear()
        self.recent.clear()
        self.no_progress = 0
        self.loop = False
        self.recovery = False
        self._rec_idx = 0
        self._rec_seq.clear()
        self._last_hash = ""

    def tick(self, state_hash: str, level_changed: bool) -> Dict[str, Any]:
        out = {"loop": False, "no_progress": False, "recover": False}

        self.visit_count[state_hash] += 1
        self.recent.append(state_hash)

        if self.visit_count[state_hash] >= self.LOOP_THRESHOLD:
            out["loop"] = True
            self.loop = True

        if state_hash == self._last_hash:
            self.no_progress += 1
        else:
            self.no_progress = 0
            self._last_hash = state_hash

        if self.no_progress >= self.NO_PROGRESS_THRESHOLD:
            out["no_progress"] = True

        if out["loop"] or out["no_progress"]:
            if not self.recovery:
                self._begin_recovery()
            out["recover"] = True

        if level_changed:
            self.recovery = False
            self.loop = False
            self.no_progress = 0

        return out

    def _begin_recovery(self) -> None:
        self.recovery = True
        self._rec_idx = 0
        # Recovery sequence: interact, undo, try each direction, click nearby
        self._rec_seq = [5, 7, 1, 2, 3, 4, 5, 7, 1, 4]
        logger.info("STUCK → entering recovery mode")

    def next_recovery_action(self, grid: np.ndarray | None = None,
                             player_pos: Optional[Tuple[int, int]] = None,
                             obj_tracker: ObjectTracker | None = None) -> Optional[Tuple[int, int, Optional[int], Optional[int]]]:
        """
        Returns (action_id, 0, None, None) for simple actions or
                (6, 0, x, y)               for click actions.

        Returns None when recovery is exhausted.
        """
        if not self.recovery:
            return None
        if self._rec_idx >= len(self._rec_seq):
            self.recovery = False
            return None

        aid = self._rec_seq[self._rec_idx]
        self._rec_idx += 1

        if aid == 6 and grid is not None:
            # Click an untried nearby object
            if obj_tracker and player_pos:
                cands = []
                for obj in obj_tracker.objects.values():
                    if obj.click_tries < 2:
                        cr, cc = obj.centroid_cell()
                        d = _manhattan(player_pos[0], player_pos[1], cr, cc)
                        cands.append((d, cr, cc))
                cands.sort()
                if cands:
                    _, cx, cy = cands[0]
                    return (6, 0, cx, cy)
            # fallback random click
            h, w = grid.shape
            return (6, 0, random.randint(0, w - 1), random.randint(0, h - 1))

        return (aid, 0, None, None)

    def summary(self) -> str:
        tag = "RECOVER" if self.recovery else ("LOOP" if self.loop else "ok")
        return f"stuck={tag}  no_prog={self.no_progress}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. LEVEL KNOWLEDGE TRANSFER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LevelKnowledgeTransfer:
    """
    Stores knowledge that persists across levels (and optionally games):
    verified movement mappings, useful interactions, colours with
    semantic meaning, wall patterns, etc.
    """

    def __init__(self):
        self.movement: Dict[int, Tuple[int, int]] = {}   # aid → (dr, dc)
        self.interact_useful: bool = False
        self.click_useful: bool = False
        self.undo_useful: bool = False
        self.wall_colors: Set[int] = set()
        self.floor_colors: Set[int] = set()
        self.player_colors: Set[int] = set()
        self.goal_colors: Set[int] = set()
        self.danger_colors: Set[int] = set()
        self.keys_collected: Set[int] = set()
        self.per_level_actions: List[int] = []           # actions per completed level
        self._actions_this_level = 0

    def reset_for_new_game(self) -> None:
        """Keep only structural knowledge (colours, movement) across games."""
        self.per_level_actions.clear()
        self._actions_this_level = 0
        self.keys_collected.clear()

    def on_level_complete(self, actions_taken: int) -> None:
        self.per_level_actions.append(actions_taken)
        self._actions_this_level = 0

    def tick(self) -> None:
        self._actions_this_level += 1

    def learn_movement(self, aid: int, dr: int, dc: int) -> None:
        if (dr, dc) != (0, 0):
            self.movement[aid] = (dr, dc)

    def learn_wall_color(self, color: int) -> None:
        self.wall_colors.add(color)

    def learn_floor_color(self, color: int) -> None:
        self.floor_colors.add(color)

    def learn_player_color(self, color: int) -> None:
        self.player_colors.add(color)

    def learn_goal_color(self, color: int) -> None:
        self.goal_colors.add(color)

    def movement_direction_for(self, dr: int, dc: int) -> Optional[int]:
        """Find action_id that moves in direction (dr, dc)."""
        for aid, (mr, mc) in self.movement.items():
            if (mr, mc) == (dr, dc):
                return aid
        return None

    def summary(self) -> str:
        return (f"mov={self.movement}  interact={self.interact_useful}  "
                f"click={self.click_useful}  levels_done={len(self.per_level_actions)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SMART AGENT  (core logic, framework-agnostic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SmartAgent:
    """
    Hypothesis-driven exploration agent for ARC-AGI-3 interactive games.

    Subsystems:
      1. HypothesisBeliefTracker  — conjectures about action semantics
      2. SpatialMemory            — explored map & player tracking
      3. ObjectTracker            — connected-component detection & tracking
      4. SmartClickTargeter       — prioritised click targeting
      5. StuckDetector            — loop / no-progress recovery
      6. LevelKnowledgeTransfer   — cross-level / cross-game memory
    """

    MAX_ACTIONS = 200

    def __init__(self):
        # subsystems
        self.hypo = HypothesisBeliefTracker()
        self.spatial = SpatialMemory()
        self.obj_tracker = ObjectTracker()
        self.clicker = SmartClickTargeter(self.obj_tracker)
        self.stuck = StuckDetector()
        self.knowledge = LevelKnowledgeTransfer()

        # internal state
        self.game_id = ""
        self._prev_hash = ""
        self._prev_levels = 0
        self._prev_action_id = -1
        self._prev_grid: np.ndarray | None = None
        self._reset_sent = False
        self._avail_ids: List[int] = []
        self._action_count = 0
        self._phase = 0          # 0=initial-probe, 1=hypothesis-test, 2=directed, 3=exploit
        self._probe_idx = 0
        self._interact_tested = False
        self._click_tested = False
        self._undo_tested = False

    # ── public API (mirrors Agent base class) ────────────────────────

    def is_done(self, frames: list, latest_frame) -> bool:
        """Return True when the agent should stop."""
        try:
            state = latest_frame.state
            state_name = state.name if hasattr(state, "name") else str(state)
            if state_name == "WIN":
                logger.info("[%s] WIN — stopping.", self.game_id)
                return True
        except Exception:
            pass
        if self._action_count >= self.MAX_ACTIONS:
            logger.info("[%s] MAX_ACTIONS reached (%d).", self.game_id, self.MAX_ACTIONS)
            return True
        return False

    def choose_action(self, frames: list, latest_frame) -> Any:
        """Decide the next GameAction."""
        try:
            return self._decide(frames, latest_frame)
        except Exception as exc:
            logger.error("choose_action error: %s\n%s", exc, traceback.format_exc())
            return self._safe_action(latest_frame)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _state_name(frame) -> str:
        try:
            return frame.state.name if hasattr(frame.state, "name") else str(frame.state)
        except Exception:
            return "?"

    @staticmethod
    def _levels(frame) -> int:
        return getattr(frame, "levels_completed", 0) or 0

    @staticmethod
    def _win_levels(frame) -> int:
        return getattr(frame, "win_levels", 1) or 1

    @staticmethod
    def _avail_action_ids(frame) -> List[int]:
        try:
            aa = getattr(frame, "available_actions", None) or []
            return [a.value if hasattr(a, "value") else int(a) for a in aa]
        except Exception:
            return []

    def _safe_action(self, frame) -> Any:
        """Fallback that always returns a valid GameAction."""
        sn = self._state_name(frame)
        if sn in ("NOT_PLAYED", "GAME_OVER"):
            return self._mk_action(0)
        return self._mk_action(1)

    # ── main decision pipeline ───────────────────────────────────────

    def _decide(self, frames: list, lf) -> Any:
        self._action_count += 1
        grid = _frame_to_grid(lf.frame if hasattr(lf, "frame") else None)
        h = _grid_hash(lf.frame if hasattr(lf, "frame") else None)
        sn = self._state_name(lf)
        lvls = self._levels(lf)
        wlvls = self._win_levels(lf)
        aids = self._avail_action_ids(lf)
        if aids:
            self._avail_ids = aids

        # ── state machine ──
        if sn in ("NOT_PLAYED", "GAME_OVER"):
            self._on_game_over_or_not_started(lf)
            return self._mk_action(0, reasoning=f"state={sn}")

        # ── level transition ──
        if lvls > self._prev_levels and self._prev_levels >= 0:
            self._on_level_complete()

        # ── very first action after reset: seed hypotheses ──
        if not self._reset_sent:
            self._reset_sent = True
            self.hypo.seed(self._avail_ids)
            logger.info("[%s] Game started. Seeded hypotheses.", self.game_id)
            return self._mk_action(1, reasoning="first probe: up")

        # ── update subsystems ──
        self._update_all(grid, h, lf)

        # ── stuck? ──
        lvl_changed = lvls > self._prev_levels
        st = self.stuck.tick(h, lvl_changed)
        self._prev_levels = lvls
        if st["recover"]:
            act = self._stuck_recovery(grid)
            if act is not None:
                self._prev_action_id = act
                return act

        # ── phase-based decision ──
        if self._action_count <= 8:
            action = self._phase_probe()
        elif self._action_count <= 40:
            action = self._phase_hypothesis_test(grid)
        elif self._action_count <= self.MAX_ACTIONS - 10:
            action = self._phase_directed(grid)
        else:
            action = self._phase_exploit(grid)

        self._prev_action_id = action if isinstance(action, int) else -1
        self.knowledge.tick()
        return action

    # ── subsystem updates ────────────────────────────────────────────

    def _update_all(self, grid: np.ndarray | None, h: str, lf) -> None:
        self.spatial.update(grid, self._prev_grid, self._prev_action_id)
        self.obj_tracker.update(grid, self._action_count)
        if grid is not None and self.spatial.player_pos:
            self.spatial.mark_visited_around(*self.spatial.player_pos)

        # hypothesis observation
        if self._prev_hash:
            sn = self._state_name(lf)
            self.hypo.observe(
                self._prev_action_id, self._prev_hash, h,
                self._prev_levels, self._levels(lf),
                sn == "GAME_OVER",
            )

        # learn movement direction
        if self.spatial.player_pos and self._prev_action_id >= 1:
            # cross-reference with hypothesis tracker
            mov = self.hypo.movement
            for aid, (dr, dc) in mov.items():
                self.knowledge.learn_movement(aid, dr, dc)

        self._prev_hash = h
        self._prev_grid = grid.copy() if grid is not None else None

    # ── level / game transitions ─────────────────────────────────────

    def _on_game_over_or_not_started(self, lf) -> None:
        if self._prev_levels > 0:
            self._on_level_complete()
        self._reset_sent = False
        self._phase = 0
        self._probe_idx = 0
        self._prev_hash = ""
        self._prev_grid = None
        self._prev_action_id = -1
        self._interact_tested = False
        self._click_tested = False
        self._undo_tested = False
        self.stuck.reset_level()
        self.spatial.reset_level()
        self.obj_tracker.reset_level()
        self.clicker.reset_level()

    def _on_level_complete(self) -> None:
        self.knowledge.on_level_complete(self._action_count)
        self._action_count = 0
        self._phase = 0
        self._probe_idx = 0
        self._interact_tested = False
        self._click_tested = False
        self._undo_tested = False
        self._reset_sent = False
        self._prev_hash = ""
        self._prev_grid = None
        self.stuck.reset_level()
        self.spatial.reset_level()
        self.obj_tracker.reset_level()
        self.clicker.reset_level()
        logger.info("Level complete! Knowledge so far: %s", self.knowledge.summary())

    # ── PHASE 0: initial probe (try each action once) ───────────────

    def _phase_probe(self) -> Any:
        # Order: up, down, left, right, interact, click, undo
        probe_order = [1, 2, 3, 4, 5, 6, 7]
        while self._probe_idx < len(probe_order):
            aid = probe_order[self._probe_idx]
            self._probe_idx += 1
            if aid in self._avail_ids:
                if aid == 6:
                    # click at a non-zero cell near center
                    return self._smart_click(reasoning="probe click")
                return self._mk_action(aid, reasoning=f"probe {aid}")
        self._phase = 1
        return self._phase_hypothesis_test(self._prev_grid)

    # ── PHASE 1: hypothesis testing ─────────────────────────────────

    def _phase_hypothesis_test(self, grid: np.ndarray | None) -> Any:
        # Test untested actions first
        untested = self.hypo.untested_action_ids(self._avail_ids)
        if untested:
            aid = untested[0]
            if aid == 6:
                return self._smart_click(reasoning="hypothesis test click")
            if aid == 5:
                self._interact_tested = True
                self.knowledge.interact_useful = True
            if aid == 7:
                self._undo_tested = True
            return self._mk_action(aid, reasoning=f"test untested {aid}")

        # Test interact in different situations
        if 5 in self._avail_ids and not self._interact_tested:
            self._interact_tested = True
            return self._mk_action(5, reasoning="test interact")

        # Test undo
        if 7 in self._avail_ids and not self._undo_tested:
            self._undo_tested = True
            return self._mk_action(7, reasoning="test undo")

        self._phase = 2
        return self._phase_directed(grid)

    # ── PHASE 2: directed exploration ────────────────────────────────

    def _phase_directed(self, grid: np.ndarray | None) -> Any:
        if grid is None:
            return self._mk_action(1, reasoning="no grid, default up")

        # 1. Try to move toward frontier
        target_dir = self.spatial.best_direction_toward_frontier()
        if target_dir is not None:
            dr, dc = target_dir
            aid = self.knowledge.movement_direction_for(dr, dc)
            if aid is None:
                # Guess: try standard mapping
                guess_map = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
                aid = guess_map.get((dr, dc), 1)
            if aid in self._avail_ids:
                return self._mk_action(aid,
                                       reasoning=f"toward frontier ({dr},{dc})")

        # 2. Try clicking on high-priority objects
        if 6 in self._avail_ids:
            click = self._smart_click(reasoning="directed click")
            if click is not None:
                return click

        # 3. Try interact (ACTION5)
        if 5 in self._avail_ids:
            return self._mk_action(5, reasoning="directed interact")

        # 4. Random walk
        return self._random_movement(reasoning="random walk")

    # ── PHASE 3: exploit / final push ───────────────────────────────

    def _phase_exploit(self, grid: np.ndarray | None) -> Any:
        # Prioritise actions that previously caused level-ups
        best_aid = -1
        best_score = -1
        for aid in self._avail_ids:
            if aid == 0:
                continue
            ec = self.hypo.effect_counts[aid]
            score = ec["level_up"] * 100 + ec["changed"] * 2 - ec["unchanged"]
            if score > best_score:
                best_score = score
                best_aid = aid

        if best_aid > 0:
            if best_aid == 6:
                return self._smart_click(reasoning="exploit click")
            return self._mk_action(best_aid, reasoning=f"exploit best={best_aid}")

        return self._random_movement(reasoning="exploit random")

    # ── stuck recovery ───────────────────────────────────────────────

    def _stuck_recovery(self, grid: np.ndarray | None) -> Any:
        rec = self.stuck.next_recovery_action(
            grid, self.spatial.player_pos, self.obj_tracker
        )
        if rec is None:
            # Recovery exhausted — try reset
            return self._mk_action(0, reasoning="recovery exhausted, reset")
        aid, _, cx, cy = rec
        if aid == 6 and cx is not None and cy is not None:
            self.clicker.mark_tried(cx, cy)
            return self._mk_action(6, x=cx, y=cy, reasoning="recovery click")
        return self._mk_action(aid, reasoning=f"recovery action {aid}")

    # ── smart click ──────────────────────────────────────────────────

    def _smart_click(self, reasoning: str = "") -> Any:
        if 6 not in self._avail_ids:
            return None
        cands = self.clicker.rank(self.spatial.player_pos)
        if cands:
            r, c, score = cands[0]
            self.clicker.mark_tried(r, c)
            return self._mk_action(6, x=c, y=r, reasoning=f"{reasoning} score={score:.1f}")
        # Fallback: click near player or at random non-zero cell
        grid = self._prev_grid
        if grid is not None:
            h, w = grid.shape
            # pick random non-zero cell
            nonzero = list(zip(*np.where(grid != BACKGROUND)))
            if nonzero:
                idx = random.randint(0, len(nonzero) - 1)
                r, c = int(nonzero[idx][0]), int(nonzero[idx][1])
                self.clicker.mark_tried(r, c)
                return self._mk_action(6, x=c, y=r, reasoning=f"{reasoning} fallback nz")
            return self._mk_action(6, x=w // 2, y=h // 2, reasoning=f"{reasoning} fallback center")
        return self._mk_action(6, x=10, y=10, reasoning=f"{reasoning} hardcoded")

    # ── random movement ──────────────────────────────────────────────

    def _random_movement(self, reasoning: str = "") -> Any:
        move_ids = [a for a in [1, 2, 3, 4] if a in self._avail_ids]
        if move_ids:
            aid = random.choice(move_ids)
            return self._mk_action(aid, reasoning=reasoning)
        return self._mk_action(5, reasoning=reasoning)

    # ── action builder ───────────────────────────────────────────────

    def _mk_action(self, action_id: int, x: int | None = None,
                   y: int | None = None, reasoning: str = "") -> Any:
        """Build a GameAction. Tries arcengine first, falls back to mock."""
        try:
            from arcengine import GameAction
            action = GameAction.from_id(action_id)
            if action_id == 6 and x is not None and y is not None:
                action.set_data({"x": int(x), "y": int(y)})
            if reasoning:
                action.reasoning = reasoning
            return action
        except Exception:
            # If arcengine is not available, return a simple namespace
            class _A:
                pass
            a = _A()
            a.action_id = action_id
            a.x = x
            a.y = y
            a.reasoning = reasoning
            return a

    # ── summary / diagnostics ────────────────────────────────────────

    def diagnostics(self) -> str:
        parts = [
            f"game={self.game_id}",
            f"actions={self._action_count}/{self.MAX_ACTIONS}",
            f"phase={self._phase}",
            self.hypo.summary(),
            self.spatial.summary(),
            self.obj_tracker.summary(),
            self.clicker.summary(),
            self.stuck.summary(),
            self.knowledge.summary(),
        ]
        return "  |  ".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FRAMEWORK ADAPTER  (inherits from Agent, delegates to SmartAgent)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SmartAgentAdapter:
    """
    Adapter that wraps SmartAgent to work with the ARC-AGI-3 Agent base class.

    Usage in agents/templates/smart_agent_v2.py:

        from agents.agent import Agent
        from arcengine import FrameData, GameAction, GameState
        # ... paste SmartAgent + subsystems above ...
        # Then:

        class SmartAgentWrapper(Agent):
            MAX_ACTIONS = 200

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._inner = SmartAgent()
                self._inner.game_id = self.game_id

            def is_done(self, frames, latest_frame):
                return self._inner.is_done(frames, latest_frame)

            def choose_action(self, frames, latest_frame):
                return self._inner.choose_action(frames, latest_frame)
    """

    pass  # Adapter is mixed in at notebook-install time (see notebook)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY-POINT for Kaggle notebook (writes agent file + submission)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_framework_agent_code() -> str:
    """Generate the full Python source that will be placed into the
    ARC-AGI-3 Agents framework's templates directory."""
    # Read this file itself and extract everything except the entry-point
    import inspect
    src = inspect.getsource(SmartAgent)
    # Strip the class definition line (we'll add our own)
    # Actually, return the full file content minus the bottom section
    this_file = os.path.abspath(__file__)
    with open(this_file, "r") as f:
        content = f.read()

    # Find the entry-point section
    marker = "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n#  ENTRY-POINT"
    idx = content.find(marker)
    if idx > 0:
        content = content[:idx]

    return content


def install_agent() -> str:
    """
    Write the agent into the ARC-AGI-3 framework directory.
    Called from the Kaggle notebook.
    Returns the path of the written agent file.
    """
    base_code = _build_framework_agent_code()

    wrapper = '''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WRAPPER — inherits from Agent base class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SmartAgentV2(Agent):
    """ARC-AGI-3 Agent wrapper for SmartAgent V2."""

    MAX_ACTIONS: int = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._inner = SmartAgent()
        self._inner.game_id = self.game_id

    def is_done(self, frames: list, latest_frame: FrameData) -> bool:
        return self._inner.is_done(frames, latest_frame)

    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        return self._inner.choose_action(frames, latest_frame)

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        logger.info("SmartAgentV2 diagnostics: %s", self._inner.diagnostics())
        super().cleanup(*args, **kwargs)
'''
    full_code = base_code + wrapper

    # Add necessary imports at the top
    import_header = '''from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
import traceback
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("SmartAgentV2")

'''
    # Remove duplicate imports from base_code
    lines = full_code.split('\n')
    skip_until_empty = True
    filtered = []
    for line in lines:
        if skip_until_empty:
            if line.strip() == '' or line.startswith('#') or line.startswith('"""') or line.startswith("from __future__") or line.startswith("import") or line.startswith("from collections") or line.startswith("from typing") or line.startswith("import numpy"):
                continue
            skip_until_empty = False
        filtered.append(line)
    full_code = import_header + '\n'.join(filtered)

    return full_code


if __name__ == "__main__":
    # Quick smoke-test
    agent = SmartAgent()
    agent.game_id = "test"
    print("SmartAgent V2 loaded successfully.")
    print(f"  Subsystems: hypo, spatial, obj_tracker, clicker, stuck, knowledge")
    print(f"  MAX_ACTIONS = {agent.MAX_ACTIONS}")
    print(f"  Diagnostics: {agent.diagnostics()}")
