#!/usr/bin/env python3
"""
AnnotateX ARC-AGI-3 Agent v1 - Baseline Agent

A heuristic exploration agent that:
1. Connects to the ARC-AGI-3 environment
2. Explores games using BFS-like systematic exploration
3. Tracks visited states and learns from frame deltas
4. Uses simple heuristics for common game patterns
5. Generates valid submissions via the Swarm orchestration

This agent is designed to run as a Kaggle Notebook submission.
"""

import json
import logging
import math
import os
import random
import sys
import time
from collections import deque
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_v1.log", mode="w"),
    ],
)
logger = logging.getLogger()

# ---------------------------------------------------------------------------
# Ensure the ARC-AGI-3 Agents package is on sys.path (Kaggle layout)
# ---------------------------------------------------------------------------
_AGI3_ROOT = "/kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents"
_WHEELS_DIR = "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
if os.path.isdir(_AGI3_ROOT):
    sys.path.insert(0, _AGI3_ROOT)
if os.path.isdir(_WHEELS_DIR):
    for wh in os.listdir(_WHEELS_DIR):
        if wh.endswith(".whl"):
            sys.path.insert(0, os.path.join(_WHEELS_DIR, wh))

# Now we can import the framework
HAS_FRAMEWORK = False
GameAction = None  # type: ignore[assignment, misc]
GameState = None  # type: ignore[assignment, misc]
FrameData = None  # type: ignore[assignment, misc]
Agent = None  # type: ignore[assignment, misc]
Swarm = None  # type: ignore[assignment, misc]

try:
    from arcengine import FrameData, GameAction, GameState
    from agents.agent import Agent
    from agents.swarm import Swarm
    HAS_FRAMEWORK = True
except ImportError:
    HAS_FRAMEWORK = False
    logger.warning(
        "ARC-AGI-3 framework not found. Running in standalone demo mode."
    )


# ===========================================================================
#  16-Color Palette (ARC Standard)
# ===========================================================================
ARC_PALETTE = [
    (0xFF, 0xFF, 0xFF),  # 0  White
    (0xCC, 0xCC, 0xCC),  # 1  Off-white
    (0x99, 0x99, 0x99),  # 2  Neutral light
    (0x66, 0x66, 0x66),  # 3  Neutral
    (0x33, 0x33, 0x33),  # 4  Off-black
    (0x00, 0x00, 0x00),  # 5  Black
    (0xE5, 0x3A, 0xA3),  # 6  Magenta
    (0xFF, 0x7B, 0xCC),  # 7  Magenta light
    (0xF9, 0x3C, 0x31),  # 8  Red
    (0x1E, 0x93, 0xFF),  # 9  Blue
    (0x88, 0xD8, 0xF1),  # 10 Blue light
    (0xFF, 0xDC, 0x00),  # 11 Yellow
    (0xFF, 0x85, 0x1B),  # 12 Orange
    (0x92, 0x12, 0x31),  # 13 Maroon
    (0x4F, 0xCC, 0x30),  # 14 Green
    (0xA3, 0x56, 0xD6),  # 15 Purple
]


# ===========================================================================
#  Grid Analysis Utilities
# ===========================================================================
def grid_to_str(grid: list[list[list[int]]], max_rows: int = 10) -> str:
    """Pretty-print a 3-D frame grid for logging."""
    lines = []
    for i, layer in enumerate(grid):
        lines.append(f"  Layer {i} ({len(layer)}x{len(layer[0]) if layer else 0}):")
        for r, row in enumerate(layer[:max_rows]):
            lines.append(f"    {row[:60]}")
        if len(layer) > max_rows:
            lines.append(f"    ... ({len(layer) - max_rows} more rows)")
    return "\n".join(lines)


def compute_frame_diff(
    prev: list[list[list[int]]], curr: list[list[list[int]]]
) -> tuple[int, set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    """Compute pixel-level diff between two frame grids.

    Returns (num_changes, appeared, disappeared).
    """
    appeared: set[tuple[int, int, int]] = set()
    disappeared: set[tuple[int, int, int]] = set()

    if not prev or not curr:
        return 0, appeared, disappeared

    for li in range(min(len(prev), len(curr))):
        for r in range(min(len(prev[li]), len(curr[li]))):
            for c in range(min(len(prev[li][r]), len(curr[li][r]))):
                pv, cv = prev[li][r][c], curr[li][r][c]
                if pv != cv:
                    disappeared.add((li, r, c, pv))
                    appeared.add((li, r, c, cv))

    return len(appeared), appeared, disappeared


def find_unique_colors(grid: list[list[list[int]]]) -> set[int]:
    """Get all unique color values in a frame."""
    colors: set[int] = set()
    for layer in grid:
        for row in layer:
            for val in row:
                colors.add(val)
    return colors


def find_color_regions(
    grid: list[list[list[int]]], color: int
) -> list[tuple[int, int, int]]:
    """Find (layer, row, col) positions of a specific color."""
    positions = []
    for li, layer in enumerate(grid):
        for r, row in enumerate(layer):
            for c, val in enumerate(row):
                if val == color:
                    positions.append((li, r, c))
    return positions


def find_bounding_box(
    positions: list[tuple[int, int, int]]
) -> Optional[tuple[int, int, int, int]]:
    """Find bounding box of positions: (min_r, min_c, max_r, max_c)."""
    if not positions:
        return None
    rows = [p[1] for p in positions]
    cols = [p[2] for p in positions]
    return (min(rows), min(cols), max(rows), max(cols))


def manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def extract_player_position(
    grid: list[list[list[int]]]
) -> Optional[tuple[int, int]]:
    """Heuristic: find the player by looking for non-background, non-wall clusters
    that appear in consecutive frames. For now, find the center of the largest
    non-static colored region."""
    if not grid:
        return None
    # Use the first layer
    layer = grid[0]
    if not layer:
        return None

    # Count non-zero, non-background pixels per region
    # Simple heuristic: find the densest cluster of color values != background
    bg_colors = {0, 5, 8}  # Common background colors
    player_candidates = []
    for r, row in enumerate(layer):
        for c, val in enumerate(row):
            if val not in bg_colors:
                player_candidates.append((r, c))

    if not player_candidates:
        return None

    avg_r = sum(p[0] for p in player_candidates) // len(player_candidates)
    avg_c = sum(p[1] for p in player_candidates) // len(player_candidates)
    return (avg_r, avg_c)


# ===========================================================================
#  Exploration State Tracker
# ===========================================================================
class ExplorationState:
    """Tracks what the agent has learned about the current game."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.visited_grids: set[str] = set()  # Hashed grids to detect loops
        self.action_results: dict[str, list[bool]] = {}  # action_name -> [had_effect]
        self.total_moves: int = 0
        self.stuck_counter: int = 0
        self.last_had_effect: bool = True
        self.history: deque[tuple[str, str]] = deque(maxlen=50)  # (action, effect_summary)
        self.levels_completed: int = 0
        self.prev_levels: int = 0
        self.exploration_order: list[str] = []  # Track order of exploration
        self.object_map: dict[str, list[tuple[int, int]]] = {}  # color -> positions

    def grid_hash(self, grid: list[list[list[int]]]) -> str:
        """Create a quick hash of the grid state."""
        if not grid:
            return ""
        # Hash only the first layer for speed
        rows = []
        for row in grid[0]:
            rows.append(tuple(row))
        return hash(tuple(rows))

    def record_action(self, action_name: str, had_effect: bool, summary: str = "") -> None:
        self.action_results.setdefault(action_name, []).append(had_effect)
        self.history.append((action_name, summary))
        self.total_moves += 1
        self.last_had_effect = had_effect
        if had_effect:
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1

    def is_stuck(self, threshold: int = 3) -> bool:
        return self.stuck_counter >= threshold

    def action_success_rate(self, action_name: str) -> float:
        results = self.action_results.get(action_name, [])
        if not results:
            return 0.5
        return sum(results) / len(results)

    def best_move_actions(self) -> list:
        """Return movement action names sorted by success rate."""
        move_names = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        if not self.action_results:
            return move_names
        scored = []
        for name in move_names:
            rate = self.action_success_rate(name)
            scored.append((rate, name))
        scored.sort(reverse=True)
        return [name for _, name in scored]

    def update_object_map(self, grid: list[list[list[int]]]) -> None:
        """Scan the grid for notable objects."""
        self.object_map.clear()
        if not grid:
            return
        layer = grid[0] if grid else []
        color_counts: dict[int, int] = {}
        for row in layer:
            for val in row:
                color_counts[val] = color_counts.get(val, 0) + 1
        # Record positions for non-trivial colors
        for val, count in color_counts.items():
            if val not in {0, 5} and count < 500:  # Skip background/dominant
                positions = find_color_regions(grid, val)
                if positions:
                    bb = find_bounding_box(positions)
                    if bb:
                        center = ((bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2)
                        self.object_map[str(val)] = [center]


# ===========================================================================
#  Baseline Agent
# ===========================================================================
if HAS_FRAMEWORK:

    class AnnotateXAgentV1(Agent):
        """
        Baseline heuristic agent for ARC-AGI-3.

        Strategy:
        1. Always RESET to start
        2. Systematic exploration: try actions in order, tracking what works
        3. Detect stuck states (no grid change after action) and switch strategy
        4. After GAME_OVER, reset and try again
        5. Use frame delta analysis to learn game mechanics
        """

        MAX_ACTIONS: int = 200  # Allow more actions for exploration
        name_override: str = "annotatex-v1"

        # Movement directions in order: up, down, left, right
        DIRECTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]  # type: ignore[attr-defined]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            random.seed(int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000)
            self.exploration = ExplorationState()
            self._last_grid: list[list[list[int]]] = []
            self._prev_grid: list[list[list[int]]] = []
            self._direction_idx: int = 0
            self._reset_count: int = 0
            self._phase: str = "explore"  # explore | exploit | retry
            self._explore_trail: list[GameAction] = []

        @property
        def name(self) -> str:
            return f"{super().name}.{self.name_override}.{self.MAX_ACTIONS}"

        def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
            """Stop when the game is won."""
            if latest_frame.state is GameState.WIN:
                logger.info(
                    f"[WIN] Game {self.game_id} won after {self.action_counter} actions!"
                )
                return True
            return False

        def choose_action(
            self, frames: list[FrameData], latest_frame: FrameData
        ) -> GameAction:
            """Choose the next action using heuristic exploration."""

            # ---- Phase 0: Handle reset states ----
            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                self._reset_count += 1
                self.exploration.reset()
                self._phase = "explore"
                self._direction_idx = 0
                self._last_grid = []

                if latest_frame.state is GameState.GAME_OVER:
                    logger.info(
                        f"[GAME_OVER] Game {self.game_id}, reset #{self._reset_count}"
                    )
                    time.sleep(0.1)  # Avoid timeout

                action = GameAction.RESET
                action.reasoning = {
                    "phase": "reset",
                    "reset_count": self._reset_count,
                    "action": "RESET",
                    "reason": "Game not started or game over, resetting",
                }
                return action

            # ---- Phase 1: Handle full_reset flag ----
            if latest_frame.full_reset:
                self._prev_grid = []
                self._last_grid = latest_frame.frame if latest_frame.frame else []
                action = GameAction.RESET
                action.reasoning = {"phase": "full_reset", "action": "RESET"}
                return action

            # ---- Analyze frame delta ----
            current_grid = latest_frame.frame if latest_frame.frame else []
            had_effect = False

            if self._last_grid and current_grid:
                diff_count, _, _ = compute_frame_diff(self._last_grid, current_grid)
                had_effect = diff_count > 0
            elif current_grid:
                had_effect = True  # First real frame

            # Track level completion
            if latest_frame.levels_completed > self.exploration.levels_completed:
                logger.info(
                    f"[LEVEL UP] {self.game_id}: level "
                    f"{latest_frame.levels_completed}/{latest_frame.win_levels}"
                )
                self.exploration.levels_completed = latest_frame.levels_completed
                self.exploration.reset()
                self._phase = "explore"
                self._direction_idx = 0
                self._last_grid = current_grid
                self._prev_grid = []

                # After level completion, need to continue
                action = GameAction.RESET
                action.reasoning = {
                    "phase": "level_complete",
                    "action": "RESET",
                    "levels_completed": latest_frame.levels_completed,
                }
                return action

            # Record action result
            if self._last_grid:
                last_action_name = "unknown"
                if frames and len(frames) > 1:
                    last_action_name = frames[-1].action_input.id.name if frames[-1].action_input else "unknown"
                self.exploration.record_action(last_action_name, had_effect)

            # Update object map periodically
            if self.action_counter % 5 == 0:
                self.exploration.update_object_map(current_grid)

            # ---- Phase 2: Choose action ----
            action = self._select_action(latest_frame, had_effect)

            # Update grid tracking
            self._prev_grid = self._last_grid
            self._last_grid = current_grid

            # Attach reasoning
            action.reasoning = {
                "phase": self._phase,
                "action": action.name,
                "had_effect": had_effect,
                "stuck": self.exploration.is_stuck(),
                "levels_completed": latest_frame.levels_completed,
                "win_levels": latest_frame.win_levels,
                "action_counter": self.action_counter,
                "objects_found": list(self.exploration.object_map.keys()),
            }

            return action

        def _select_action(
            self, latest_frame: FrameData, had_effect: bool
        ) -> GameAction:
            """Core action selection logic."""

            available = latest_frame.available_actions if latest_frame.available_actions else list(GameAction)
            simple_actions = [a for a in available if a.is_simple() and a != GameAction.RESET]
            complex_actions = [a for a in available if a.is_complex()]

            # Strategy 1: If stuck, try a different action
            if self.exploration.is_stuck(threshold=3):
                self._direction_idx = (self._direction_idx + 1) % len(simple_actions) if simple_actions else 0
                # Try ACTION5 (perform action / interact) if stuck on movement
                if GameAction.ACTION5 in simple_actions:
                    logger.info("[STUCK] Trying ACTION5 (interact)")
                    return GameAction.ACTION5
                # Try ACTION7 (undo) if available
                if GameAction.ACTION7 in simple_actions:
                    logger.info("[STUCK] Trying ACTION7 (undo)")
                    return GameAction.ACTION7
                # Try clicking on interesting objects
                if complex_actions:
                    target = self._find_click_target(latest_frame)
                    if target:
                        return target

            # Strategy 2: Systematic exploration with direction cycling
            best_moves = self.exploration.best_move_actions()
            # Filter to only available actions
            best_available = [a for a in best_moves if a in simple_actions]
            if not best_available:
                best_available = simple_actions

            if best_available:
                # Cycle through directions, preferring ones that had effect
                if had_effect and best_available:
                    # Keep going in the same direction if it works
                    last_idx = self._direction_idx
                    if last_idx < len(best_available):
                        action = best_available[last_idx]
                    else:
                        action = best_available[0]
                else:
                    # Switch direction
                    self._direction_idx = (self._direction_idx + 1) % len(best_available)
                    action = best_available[self._direction_idx]
            elif GameAction.ACTION5 in simple_actions:
                action = GameAction.ACTION5
            elif complex_actions:
                action = self._find_click_target(latest_frame) or GameAction.ACTION5
            else:
                action = random.choice(simple_actions) if simple_actions else GameAction.ACTION1

            return action

        def _find_click_target(self, latest_frame: FrameData) -> Optional[GameAction]:
            """Find an interesting position to click (ACTION6)."""
            grid = latest_frame.frame if latest_frame.frame else []
            if not grid:
                return None

            # Look for non-background, non-wall colored pixels to click on
            layer = grid[0]
            bg_colors = {0, 5, 8, 10}
            candidates = []
            for r, row in enumerate(layer):
                for c, val in enumerate(row):
                    if val not in bg_colors and 0 <= val <= 15:
                        candidates.append((r, c, val))

            if not candidates:
                return None

            # Prefer colors we haven't interacted with much
            # Click on the first interesting candidate
            r, c, val = random.choice(candidates[:20])
            action = GameAction.ACTION6
            action.set_data({"x": c, "y": r})
            logger.info(f"[CLICK] Target: ({c}, {r}), color={val}")
            return action

else:
    # Standalone mode for testing without the framework
    logger.info("Framework not available, providing class stubs.")
    AnnotateXAgentV1 = None  # type: ignore[assignment, misc]


# ===========================================================================
#  Demo / Test Mode (when not running inside Kaggle)
# ===========================================================================
def run_demo() -> None:
    """Run a demonstration of the agent's grid analysis capabilities."""
    logger.info("=" * 60)
    logger.info("AnnotateX ARC-AGI-3 Agent v1 - Demo Mode")
    logger.info("=" * 60)

    # Create a sample grid for testing
    sample_grid = [
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 0, 0, 0, 0, 0],
            [0, 5, 9, 5, 0, 11, 11, 0],
            [0, 0, 5, 0, 0, 11, 14, 0],
            [0, 0, 0, 0, 0, 11, 11, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [6, 6, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ]

    # Test grid analysis
    logger.info("\nSample Grid:")
    logger.info(grid_to_str(sample_grid))

    colors = find_unique_colors(sample_grid)
    logger.info(f"\nUnique colors: {colors}")

    for color in sorted(colors - {0}):
        positions = find_color_regions(sample_grid, color)
        bb = find_bounding_box(positions)
        logger.info(f"  Color {color}: {len(positions)} pixels, bbox: {bb}")

    # Test frame diff
    modified_layer = [row[:] for row in sample_grid[0]]
    modified_layer[1][1] = 9  # Change one pixel
    diff_count, appeared, disappeared = compute_frame_diff(sample_grid, [modified_layer])
    logger.info(f"\nFrame diff: {diff_count} changes")

    # Test exploration state
    exploration = ExplorationState()
    exploration.record_action("ACTION1", True, "Moved up")
    exploration.record_action("ACTION2", False, "Wall")
    exploration.record_action("ACTION2", False, "Wall")
    exploration.record_action("ACTION2", False, "Wall")
    logger.info(f"\nExploration state:")
    logger.info(f"  Stuck: {exploration.is_stuck()}")
    logger.info(f"  ACTION1 success rate: {exploration.action_success_rate('ACTION1'):.0%}")
    logger.info(f"  ACTION2 success rate: {exploration.action_success_rate('ACTION2'):.0%}")
    logger.info(f"  Best moves: {exploration.best_move_actions()}")

    logger.info("\nDemo complete!")


# ===========================================================================
#  Kaggle Notebook Entry Point
# ===========================================================================
def main() -> None:
    """Main entry point for the Kaggle Notebook."""

    if not HAS_FRAMEWORK:
        run_demo()
        return

    # The Swarm orchestrator handles everything:
    # - Gets game list from API
    # - Creates one AnnotateXAgentV1 per game
    # - Runs all agents in parallel threads
    # - Manages scorecards
    logger.info("Starting AnnotateX ARC-AGI-3 Agent v1...")

    try:
        from agents import Swarm

        swarm = Swarm(
            agent="annotatexv1",  # Will be matched by class name
            ROOT_URL=os.environ.get(
                "ARC_ROOT_URL",
                f"{os.environ.get('SCHEME', 'http')}://{os.environ.get('HOST', 'localhost')}:{os.environ.get('PORT', '8001')}",
            ),
            games=[],  # Games populated by Swarm from API
            tags=["annotatex", "v1", "baseline"],
        )
        scorecard = swarm.main()

        if scorecard:
            logger.info("Final Scorecard:")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        run_demo()


if __name__ == "__main__":
    main()
