"""
ARC-AGI-3 Baseline Agent v1
============================
An AnnotateX baseline agent for the ARC Prize 2026 ARC-AGI-3 competition.

This agent implements:
1. Installation of provided competition wheels
2. Connection to the ARC-AGI-3 environment via the Arcade API
3. A heuristic exploration strategy using BFS-inspired exploration
4. Automatic submission generation via the scorecard system

Environment:
- Games are interactive grid-based puzzles (64x64 grids with INT<0,15> values)
- 7 possible actions: RESET, ACTION1 (Up/W), ACTION2 (Down/S),
  ACTION3 (Left/A), ACTION4 (Right/D), ACTION5 (Enter/Space/Delete),
  ACTION6 (Click/Point with x,y), ACTION7 (Undo)
- Each game has multiple levels; complete all levels to WIN
- Scoring: 0-100% per game based on levels completed vs human performance

Strategy:
- Phase 1 (Explore): Systematically try actions to learn game mechanics
- Phase 2 (Solve): Use observed patterns to navigate towards goal
- Smart reset handling and state tracking
"""

import json
import logging
import os
import sys
import time
from collections import deque
from typing import Any, Optional

# ============================================================
# STEP 0: Install competition wheels
# ============================================================
WHEELS_DIR = "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
if os.path.exists(WHEELS_DIR):
    import subprocess
    wheels = [os.path.join(WHEELS_DIR, f) for f in os.listdir(WHEELS_DIR) if f.endswith('.whl')]
    if wheels:
        print(f"Installing {len(wheels)} competition wheels...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *wheels])
        print("Wheels installed successfully.")

# ============================================================
# STEP 1: Setup logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_log.txt", mode="w"),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# STEP 2: Import competition SDK
# ============================================================
from arc_agi import Arcade, OperationMode
from arc_agi.scorecard import EnvironmentScorecard
from arcengine import FrameData, FrameDataRaw, GameAction, GameState

# ============================================================
# STEP 3: Agent Configuration
# ============================================================
MAX_ACTIONS_PER_GAME = 200  # Budget for each game
EXPLORE_PHASE_STEPS = 20    # Steps dedicated to initial exploration
GRID_SIZE = 64               # Grid dimensions
NUM_COLORS = 16              # Possible color values (0-15)


class BaselineAgent:
    """
    A heuristic-based ARC-AGI-3 agent that combines systematic exploration
    with state-tracking to solve interactive game environments.
    """

    def __init__(self):
        self.arc = Arcade()
        self.visited_states: set[str] = set()
        self.action_history: list[dict] = []
        self.current_level: int = 0
        self.total_actions: int = 0

    def _state_hash(self, frame: FrameData) -> str:
        """Create a hash of the current frame for state deduplication."""
        if not frame.frame:
            return ""
        grid = frame.frame[-1] if isinstance(frame.frame, list) else frame.frame
        return str(grid) if grid else ""

    def is_terminal_state(self, frame: FrameData) -> bool:
        """Check if we've reached a terminal state."""
        return frame.state in (GameState.WIN, GameState.GAME_OVER)

    def get_available_actions(self, frame: FrameData) -> list[GameAction]:
        """Get actions available in the current frame, excluding RESET."""
        if hasattr(frame, 'available_actions') and frame.available_actions:
            return [a for a in frame.available_actions if a != GameAction.RESET]
        return [a for a in GameAction if a != GameAction.RESET]

    def analyze_grid(self, frame: FrameData) -> dict:
        """
        Analyze the current grid to extract useful features for decision-making.
        """
        analysis = {
            "state": frame.state.name if hasattr(frame.state, 'name') else str(frame.state),
            "levels_completed": getattr(frame, 'levels_completed', 0),
            "win_levels": getattr(frame, 'win_levels', 1),
            "grid_layers": len(frame.frame) if frame.frame else 0,
            "unique_colors": set(),
            "player_pos": None,
            "target_pos": None,
            "walls_count": 0,
            "floor_count": 0,
        }

        if not frame.frame:
            return analysis

        # Use the last grid layer (most recent state)
        grid = frame.frame[-1] if isinstance(frame.frame[-1], list) else None
        if not grid or not isinstance(grid, list):
            return analysis

        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                val = int(cell) if cell is not None else 0
                analysis["unique_colors"].add(val)
                if val >= 10:  # Higher values likely represent walls/objects
                    analysis["walls_count"] += 1
                elif val >= 1:
                    analysis["floor_count"] += 1

        analysis["unique_colors"] = sorted(analysis["unique_colors"])
        return analysis

    def choose_exploration_action(self, frame: FrameData, step: int) -> GameAction:
        """
        Systematic exploration strategy for the first EXPLORE_PHASE_STEPS steps.
        Tries each action type to learn game mechanics.
        """
        actions = self.get_available_actions(frame)
        if not actions:
            return GameAction.ACTION1

        # Cycle through available actions systematically
        idx = step % len(actions)
        chosen = actions[idx]

        # For complex actions (ACTION6 = Click), try center of grid
        if chosen.is_complex():
            chosen.set_data({"x": GRID_SIZE // 2, "y": GRID_SIZE // 2})
            logger.info(f"Exploration step {step}: {chosen.name} at ({GRID_SIZE//2}, {GRID_SIZE//2})")
        else:
            logger.info(f"Exploration step {step}: {chosen.name}")

        return chosen

    def choose_solving_action(self, frame: FrameData, env: Any) -> GameAction:
        """
        Solving strategy based on observations gathered during exploration.
        Uses a priority-based approach: try movement first, then interaction.
        """
        actions = self.get_available_actions(frame)
        if not actions:
            return GameAction.RESET

        analysis = self.analyze_grid(frame)
        state_hash = self._state_hash(frame)

        # Avoid revisiting the same state
        if state_hash in self.visited_states and len(actions) > 1:
            # Try a different action
            last_action = self.action_history[-1]["action"] if self.action_history else None
            alternative = [a for a in actions if a.name != last_action]
            if alternative:
                chosen = alternative[0]
            else:
                chosen = actions[0]
        else:
            # Priority: movement actions (ACTION1-4) over interaction (ACTION5-7)
            movement_actions = [a for a in actions if a in (
                GameAction.ACTION1, GameAction.ACTION2,
                GameAction.ACTION3, GameAction.ACTION4
            )]
            if movement_actions:
                chosen = movement_actions[0]
            else:
                chosen = actions[0]

        # For complex actions, estimate a click position based on analysis
        if chosen.is_complex():
            # Try clicking near areas with interesting features
            x = GRID_SIZE // 2
            y = GRID_SIZE // 2
            chosen.set_data({"x": x, "y": y})

        logger.info(f"Solving action: {chosen.name} | State: {analysis['state']} | "
                     f"Levels: {analysis['levels_completed']}/{analysis['win_levels']} | "
                     f"Colors: {analysis['unique_colors']}")
        return chosen

    def play_game(self, game_id: str, scorecard_id: str) -> dict:
        """Play a single game and return results."""
        logger.info(f"=== Starting game: {game_id} ===")

        env = self.arc.make(game_id, scorecard_id=scorecard_id)
        self.visited_states = set()
        self.action_history = []
        self.current_level = 0

        results = {
            "game_id": game_id,
            "actions_taken": 0,
            "levels_completed": 0,
            "final_state": "UNKNOWN",
            "exploration_done": False,
        }

        try:
            # Initial observation
            obs = env.observation_space
            if obs:
                frame = self._convert_frame(obs)
                logger.info(f"Initial state: {frame.state.name if hasattr(frame.state, 'name') else frame.state}")
        except Exception as e:
            logger.warning(f"Could not get initial observation: {e}")

        for step in range(MAX_ACTIONS_PER_GAME):
            if step == 0:
                action = GameAction.RESET
            elif step < EXPLORE_PHASE_STEPS:
                # Use initial frame for exploration
                try:
                    obs = env.observation_space
                    frame = self._convert_frame(obs) if obs else FrameData(levels_completed=0)
                except Exception:
                    frame = FrameData(levels_completed=0)
                action = self.choose_exploration_action(frame, step)
            else:
                # Switch to solving mode
                results["exploration_done"] = True
                try:
                    obs = env.observation_space
                    frame = self._convert_frame(obs) if obs else FrameData(levels_completed=0)
                except Exception:
                    frame = FrameData(levels_completed=0)

                if self.is_terminal_state(frame):
                    results["final_state"] = frame.state.name if hasattr(frame.state, 'name') else str(frame.state)
                    results["levels_completed"] = getattr(frame, 'levels_completed', 0)
                    logger.info(f"Game ended: {results['final_state']}")
                    break

                action = self.choose_solving_action(frame, env)

            # Execute action
            try:
                data = action.action_data.model_dump() if hasattr(action, 'action_data') else {}
                raw = env.step(action, data=data, reasoning={})
                new_frame = self._convert_frame(raw)

                # Track state
                state_hash = self._state_hash(new_frame)
                self.visited_states.add(state_hash)
                self.action_history.append({
                    "step": step,
                    "action": action.name,
                    "state": new_frame.state.name if hasattr(new_frame.state, 'name') else str(new_frame.state),
                    "levels": getattr(new_frame, 'levels_completed', 0),
                })

                results["actions_taken"] = step + 1
                results["levels_completed"] = getattr(new_frame, 'levels_completed', 0)
                results["final_state"] = new_frame.state.name if hasattr(new_frame.state, 'name') else str(new_frame.state)

                logger.info(f"Step {step}: {action.name} -> {results['final_state']} "
                           f"(levels={results['levels_completed']})")

                if self.is_terminal_state(new_frame):
                    if new_frame.state == GameState.GAME_OVER:
                        # Try resetting to continue
                        if step < MAX_ACTIONS_PER_GAME - 5:
                            logger.info("GAME_OVER detected, attempting reset...")
                    elif new_frame.state == GameState.WIN:
                        logger.info("Game WON!")
                        break

            except Exception as e:
                logger.error(f"Error executing action {action.name}: {e}")
                # Try reset on error
                try:
                    env.step(GameAction.RESET, data={}, reasoning={})
                except Exception:
                    break

        logger.info(f"=== Game {game_id} finished: {results['final_state']}, "
                    f"{results['levels_completed']} levels, {results['actions_taken']} actions ===")
        return results

    def _convert_frame(self, raw: FrameDataRaw | None) -> FrameData:
        """Convert raw frame data to FrameData."""
        if raw is None:
            return FrameData(levels_completed=0)
        return FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame] if raw.frame else [],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )

    def run(self):
        """Main entry point: play all available games."""
        logger.info("=" * 60)
        logger.info("AnnotateX ARC-AGI-3 Agent v1 Starting")
        logger.info("=" * 60)

        start_time = time.time()

        # Get available games
        games = self._get_game_list()
        logger.info(f"Found {len(games)} games: {games}")

        if not games:
            logger.error("No games found! Check environment configuration.")
            return

        # Open scorecard for tracking
        scorecard_id = self.arc.open_scorecard(tags=["annotatex", "agent_v1", "baseline"])
        logger.info(f"Scorecard ID: {scorecard_id}")

        all_results = []

        for game_id in games:
            try:
                result = self.play_game(game_id, scorecard_id)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to play game {game_id}: {e}")
                all_results.append({
                    "game_id": game_id,
                    "error": str(e),
                    "actions_taken": 0,
                    "levels_completed": 0,
                    "final_state": "ERROR",
                })

        # Close scorecard and submit
        scorecard = self.arc.close_scorecard(scorecard_id)
        if scorecard:
            logger.info("--- FINAL SCORECARD ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))

        # Summary
        elapsed = time.time() - start_time
        total_levels = sum(r.get("levels_completed", 0) for r in all_results)
        total_actions = sum(r.get("actions_taken", 0) for r in all_results)

        logger.info("=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info(f"  Games played: {len(all_results)}")
        logger.info(f"  Total levels completed: {total_levels}")
        logger.info(f"  Total actions taken: {total_actions}")
        logger.info(f"  Time elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)

        # Print per-game results
        for r in all_results:
            status = r.get("final_state", "UNKNOWN")
            icon = "WIN" if status == "WIN" else "OK" if "levels" in str(r.get("levels_completed", 0)) else "FAIL"
            logger.info(f"  [{icon}] {r['game_id']}: {r.get('levels_completed', 0)} levels, "
                       f"{r.get('actions_taken', 0)} actions, state={status}")

        return all_results

    def _get_game_list(self) -> list[str]:
        """Get the list of available games."""
        try:
            import requests
            session = requests.Session()
            session.headers.update({
                "X-API-Key": os.getenv("ARC_API_KEY", ""),
                "Accept": "application/json",
            })

            scheme = os.environ.get("SCHEME", "https")
            host = os.environ.get("HOST", "three.arcprize.org")
            port = os.environ.get("PORT", "443")

            if (scheme == "http" and str(port) == "80") or (scheme == "https" and str(port) == "443"):
                url = f"{scheme}://{host}"
            else:
                url = f"{scheme}://{host}:{port}"

            r = session.get(f"{url}/api/games", timeout=10)
            if r.status_code == 200:
                return [g["game_id"] for g in r.json()]
            else:
                logger.warning(f"Failed to get games: {r.status_code}")
        except Exception as e:
            logger.warning(f"Could not fetch game list: {e}")

        # Fallback: try to get games from the Arcade object
        try:
            return list(self.arc.games) if hasattr(self.arc, 'games') else []
        except Exception:
            return []


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    agent = BaselineAgent()
    agent.run()
