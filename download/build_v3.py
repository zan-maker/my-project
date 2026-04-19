import json

# ============================================================
# Build the full V3 agent code as a Python string
# ============================================================

agent_code = r'''"""
SmartAgent V3 — Advanced Exploration Agent for ARC-AGI-3

Major improvements over V2:
1. CNN Frame Encoder & State Transition Predictor (PyTorch)
2. Explicit State Graph Explorer
3. Monte Carlo Tree Search (MCTS)
4. Temporal Pattern Detection
5. Improved Goal Inference
6. Better Cross-Level Knowledge Transfer
7. Improved Multi-Phase Stuck Recovery
8. Smart Click Selector v3

Kept from V2:
- HypothesisTracker (enhanced)
- SpatialMemory (kept)
- ObjectTracker (kept)
- StuckDetector (enhanced)
- LevelProgressionManager (enhanced)
"""

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

logger = logging.getLogger("SmartAgentV3")

# Try importing torch for CNN; fall back gracefully
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    logger.info("PyTorch available — CNN predictor enabled")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch NOT available — CNN predictor disabled")


# ============================================================
# Configuration
# ============================================================
class Config:
    MAX_ACTIONS = 250
    GRID_MAX = 64
    COLOURS = 16
    LOOP_THRESHOLD = 3
    NO_PROGRESS_THRESHOLD = 6
    EXPLORE_FRONTIER_MAX_DEPTH = 12
    CLICK_CANDIDATE_LIMIT = 15
    RANDOM_EXPLORATION_PROB = 0.04
    MIN_OBJECT_SIZE = 1
    BACKGROUND_COLOR = 0
    HYPOTHESIS_CONFIDENCE_INCREMENT = 1.0
    HYPOTHESIS_CONFIDENCE_DECREMENT = 0.5
    # V3 new configs
    MCTS_SIMULATIONS = 15
    MCTS_ROLLOUT_DEPTH = 8
    MCTS_FIRST_N_ACTIONS = 5
    CNN_TRAIN_MIN_SAMPLES = 40
    CNN_TRAIN_STEPS = 8
    CNN_LR = 0.001
    STATE_GRAPH_MAX_NODES = 2000
    TEMPORAL_WINDOW = 10
    GOAL_INFERENCE_CONFIDENCE_THRESHOLD = 0.6
    STUCK_RECOVERY_PHASES = 6


# ============================================================
# Utility Functions
# ============================================================
def grid_hash(frame_data):
    """Compute a hash of a grid frame for state comparison."""
    try:
        if not frame_data:
            return "__empty__"
        last_grid = frame_data[-1] if isinstance(frame_data, list) else frame_data
        arr = np.array(last_grid, dtype=np.int8)
        return hashlib.md5(arr.tobytes()).hexdigest()[:16]
    except Exception:
        return "__error__"


def grid_to_np(frame_data):
    """Convert frame data to a numpy array."""
    try:
        if not frame_data:
            return None
        last_grid = frame_data[-1] if isinstance(frame_data, list) else frame_data
        return np.array(last_grid, dtype=np.float32)
    except Exception:
        return None


def grid_diff(grid_a, grid_b):
    """Compute the difference between two grids, normalized to [-1, 1]."""
    try:
        if grid_a is None or grid_b is None:
            return None
        diff = grid_b.astype(np.float32) - grid_a.astype(np.float32)
        return diff / max(np.max(np.abs(diff)), 1.0)
    except Exception:
        return None


def manhattan_distance(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)


# ============================================================
# CNN Frame Encoder & State Transition Predictor (NEW)
# ============================================================
if TORCH_AVAILABLE:
    class FrameEncoder(nn.Module):
        """
        Small CNN that takes two consecutive 64x64 frames (2 channels)
        and predicts the next frame diff + value estimate.
        Architecture (~90K params):
          Conv2d(2,32,5,pad=2)->ReLU
          Conv2d(32,64,3,pad=1)->ReLU
          Conv2d(64,64,3,pad=1)->ReLU
          Conv2d(64,1,3,pad=1)  -> predicted diff
          Parallel: GlobalAvgPool -> FC(64,32) -> ReLU -> FC(32,1) -> value
        """
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(2, 32, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.diff_head = nn.Conv2d(64, 1, 3, padding=1)
            self.value_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            """x: (B, 2, H, W) -> (diff_pred, value)"""
            features = self.encoder(x)
            diff_pred = self.diff_head(features)
            value = self.value_head(features)
            return diff_pred, value

    class CNNDummy:
        """Fallback when torch is not available."""
        def __init__(self):
            self._trainable = False

        def train_step(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return None, 0.0

        @property
        def trainable(self):
            return False


class StateTransitionPredictor:
    """Online CNN predictor that learns frame transitions as actions are taken."""

    def __init__(self):
        self.model = FrameEncoder() if TORCH_AVAILABLE else CNNDummy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.CNN_LR) if TORCH_AVAILABLE else None
        self.samples = []  # (prev_grid, next_grid, reward)
        self._train_count = 0
        self._trainable = TORCH_AVAILABLE

    @property
    def trainable(self):
        return self._trainable and len(self.samples) >= Config.CNN_TRAIN_MIN_SAMPLES

    def add_sample(self, prev_grid, next_grid, reward=0.0):
        """Store a transition sample for later training."""
        if prev_grid is None or next_grid is None:
            return
        # Downsample to 32x32 for efficiency
        try:
            pg = self._downsample(prev_grid)
            ng = self._downsample(next_grid)
            self.samples.append((pg, ng, float(reward)))
            if len(self.samples) > 500:
                self.samples = self.samples[-400:]
        except Exception:
            pass

    def _downsample(self, grid):
        """Downsample grid to 32x32 for CNN input."""
        h, w = grid.shape[:2]
        if h > 32 or w > 32:
            step_r = max(1, h // 32)
            step_c = max(1, w // 32)
            return grid[:h - h % 32:step_r, :w - w % 32:step_c]
        return grid

    def train_step(self):
        """Do a few gradient steps on collected samples."""
        if not self.trainable or not TORCH_AVAILABLE:
            return
        self._train_count += 1
        if self._train_count % 5 != 0:
            return
        try:
            self.model.train()
            n = min(len(self.samples), 32)
            batch_idx = random.sample(range(len(self.samples)), n)
            prev_batch = []
            next_batch = []
            reward_batch = []
            for idx in batch_idx:
                pg, ng, rew = self.samples[idx]
                prev_batch.append(pg)
                next_batch.append(ng)
                reward_batch.append(rew)
            prev_t = torch.tensor(np.array(prev_batch), dtype=torch.float32)  # (B, H, W)
            next_t = torch.tensor(np.array(next_batch), dtype=torch.float32)
            reward_t = torch.tensor(reward_batch, dtype=torch.float32)
            # Normalize grids to [0, 1]
            prev_t = prev_t / max(Config.COLOURS, 1)
            next_t = next_t / max(Config.COLOURS, 1)
            diff_target = next_t - prev_t  # (B, H, W)
            diff_target = diff_target.unsqueeze(1)  # (B, 1, H, W)
            input_t = torch.cat([prev_t, next_t], dim=1)  # (B, 2, H, W)
            self.optimizer.zero_grad()
            diff_pred, value_pred = self.model(input_t)
            diff_pred = diff_pred.squeeze(1)
            diff_loss = nn.functional.mse_loss(diff_pred, diff_target)
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), reward_t)
            loss = diff_loss + 0.1 * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.eval()
        except Exception as e:
            logger.debug(f"CNN train error: {e}")

    def predict(self, prev_grid, action_idx):
        """Predict next frame diff and value for a given action."""
        if not self.trainable:
            return None, 0.0
        try:
            self.model.eval()
            pg = self._downsample(prev_grid) if prev_grid is not None else None
            if pg is None:
                return None, 0.0
            pg_t = torch.tensor(pg / max(Config.COLOURS, 1), dtype=torch.float32).unsqueeze(0)
            # Use a zero dummy as second channel (no knowledge of next frame)
            zero_t = torch.zeros_like(pg_t)
            input_t = torch.cat([pg_t, zero_t], dim=1).unsqueeze(0)  # (1, 2, H, W)
            with torch.no_grad():
                diff_pred, value_pred = self.model(input_t)
            return diff_pred.squeeze().numpy(), value_pred.item()
        except Exception as e:
            logger.debug(f"CNN predict error: {e}")
            return None, 0.0

    def reset(self):
        self.samples.clear()
        self._train_count = 0


# ============================================================
# State Graph Explorer (NEW)
# ============================================================
class StateGraph:
    """
    Maintains a graph of visited states with transitions.
    Used for BFS to find unexplored paths and detect cycles.
    """

    def __init__(self, max_nodes=Config.STATE_GRAPH_MAX_NODES):
        self.nodes = {}         # hash -> {data, frame_hash, visits, is_win}
        self.edges = defaultdict(dict)  # from_hash -> {action_name: to_hash}
        self.reverse_edges = defaultdict(set)  # to_hash -> {from_hash}
        self.win_states = set()  # hashes of states that led to level win
        self.solution_paths = [] # list of action sequences that won levels
        self.current_hash = ""
        self.max_nodes = max_nodes
        self.dead_ends = set()

    def add_state(self, state_hash, frame_data=None):
        """Add a state node to the graph."""
        if len(self.nodes) >= self.max_nodes:
            return
        if state_hash not in self.nodes:
            self.nodes[state_hash] = {
                "data": frame_data,
                "visits": 0,
                "is_win": False,
                "children_explored": set(),
            }
        self.nodes[state_hash]["visits"] += 1

    def add_transition(self, from_hash, action_name, to_hash):
        """Record a transition between states."""
        if from_hash in self.nodes:
            self.nodes[from_hash]["children_explored"].add(action_name)
        self.edges[from_hash][action_name] = to_hash
        self.reverse_edges[to_hash].add(from_hash)

    def mark_win(self, state_hash):
        """Mark a state as a win state."""
        if state_hash in self.nodes:
            self.nodes[state_hash]["is_win"] = True
            self.win_states.add(state_hash)

    def save_solution_path(self, actions):
        """Save a successful action sequence."""
        if actions:
            self.solution_paths.append(list(actions))

    def find_unexplored_from(self, current_hash):
        """Find an action that leads to an unexplored state from current."""
        if current_hash not in self.edges:
            return None
        explored = self.nodes.get(current_hash, {}).get("children_explored", set())
        # Return first action whose result hasn't been explored deeply
        for action_name, to_hash in self.edges[current_hash].items():
            if to_hash in self.dead_ends:
                continue
            if to_hash not in self.nodes or self.nodes[to_hash]["visits"] < 2:
                return action_name
        return None

    def find_path_to_win(self, start_hash):
        """BFS to find a path from start_hash to any win state."""
        if not self.win_states or start_hash not in self.nodes:
            return None
        queue = deque([(start_hash, [])])
        visited = {start_hash}
        while queue:
            node_hash, path = queue.popleft()
            if node_hash in self.win_states:
                return path if path else None
            for action_name, to_hash in self.edges.get(node_hash, {}).items():
                if to_hash not in visited and to_hash not in self.dead_ends:
                    visited.add(to_hash)
                    queue.append((to_hash, path + [action_name]))
        return None

    def detect_cycle(self, state_hash):
        """Check if the current state is part of a cycle."""
        visited = set()
        queue = deque([state_hash])
        while queue:
            node = queue.popleft()
            if node in visited:
                return True
            visited.add(node)
            for to_hash in self.edges.get(node, {}).values():
                if to_hash == state_hash:
                    return True
                queue.append(to_hash)
        return False

    def prune_dead_ends(self):
        """Remove states with no unexplored neighbors."""
        for node_hash in list(self.nodes.keys()):
            if node_hash in self.win_states:
                continue
            has_unexplored = False
            for to_hash in self.edges.get(node_hash, {}).values():
                if to_hash not in self.dead_ends:
                    has_unexplored = True
                    break
            if not has_unexplored and node_hash in self.edges:
                self.dead_ends.add(node_hash)

    def reset_for_level(self):
        self.nodes.clear()
        self.edges.clear()
        self.reverse_edges.clear()
        self.dead_ends.clear()
        self.current_hash = ""

    def get_stats(self):
        return f"Graph: {len(self.nodes)} nodes, {sum(len(e) for e in self.edges.values())} edges, {len(self.win_states)} wins, {len(self.dead_ends)} dead-ends"


# ============================================================
# Monte Carlo Tree Search (NEW)
# ============================================================
class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(self, state_hash, parent=None, action=None):
        self.state_hash = state_hash
        self.parent = parent
        self.action = action  # action that led to this state
        self.children = {}    # action_name -> MCTSNode
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = []

    def ucb1(self, exploration=1.414):
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration_term

    def best_child(self, exploration=1.414):
        """Select the child with highest UCB1."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: c.ucb1(exploration))

    def most_visited_child(self):
        """Select the child with most visits (for final action choice)."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: c.visits)


class MCTS:
    """
    Simple Monte Carlo Tree Search for action selection.
    Used during early exploration to quickly learn action meanings.
    """

    def __init__(self, simulations=Config.MCTS_SIMULATIONS, rollout_depth=Config.MCTS_ROLLOUT_DEPTH):
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.root = None
        self._action_set = []

    def search(self, state_hash, available_actions, simulate_fn):
        """
        Run MCTS from current state.
        simulate_fn(state_hash, action_name) -> (next_state_hash, reward)
        """
        self._action_set = [a.name if hasattr(a, "name") else str(a)
                           for a in available_actions
                           if not (hasattr(a, "name") and a.name == "RESET")]
        if not self._action_set:
            return None

        self.root = MCTSNode(state_hash)
        self.root.untried_actions = list(self._action_set)

        for _ in range(self.simulations):
            node = self._select(self.root)
            if node is None:
                break
            node = self._expand(node, available_actions)
            reward = self._simulate(node, simulate_fn)
            self._backpropagate(node, reward)

        # Return the most visited child's action
        best = self.root.most_visited_child()
        if best is None and self.root.children:
            best = list(self.root.children.values())[0]
        return best.action if best else None

    def _select(self, node):
        """Select a node to expand using UCB1."""
        while node.untried_actions == [] and node.children:
            node = node.best_child()
            if node is None:
                return None
        return node

    def _expand(self, node, available_actions):
        """Expand a node by trying an untried action."""
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        try:
            next_hash, reward = self._simulate_fn_safe(node.state_hash, action)
            child = MCTSNode(next_hash, parent=node, action=action)
            child.untried_actions = [a for a in self._action_set if a != action]
            node.children[action] = child
            # Update reward based on expansion
            child.wins += reward
            child.visits += 1
            return child
        except Exception:
            return node

    def _simulate_fn_safe(self, state_hash, action_name):
        """Safe wrapper for simulation (returns dummy if no simulate_fn)."""
        # This will be overridden by the agent with actual simulation
        fake_next = hashlib.md5(f"{state_hash}_{action_name}".encode()).hexdigest()[:16]
        return fake_next, 0.5

    def _simulate(self, node, simulate_fn):
        """Random rollout from a node."""
        self._current_simulate_fn = simulate_fn
        total_reward = 0.0
        current_hash = node.state_hash
        for _ in range(self.rollout_depth):
            action = random.choice(self._action_set) if self._action_set else None
            if action is None:
                break
            try:
                current_hash, reward = simulate_fn(current_hash, action)
                total_reward += reward
            except Exception:
                break
        return total_reward

    def _backpropagate(self, node, reward):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent


# ============================================================
# Temporal Pattern Detector (NEW)
# ============================================================
class TemporalDetector:
    """
    Detects periodic patterns and progress indicators in frame sequences.
    """

    def __init__(self, window=Config.TEMPORAL_WINDOW):
        self.window = window
        self.hash_history = deque(maxlen=window * 3)
        self.change_magnitudes = deque(maxlen=window)
        self._oscillation_detected = False
        self._progress_detected = False
        self._oscillation_period = 0
        self._unique_states_in_window = set()

    def update(self, frame_hash, grid_a=None, grid_b=None):
        """Update with new frame observation."""
        self.hash_history.append(frame_hash)
        self._unique_states_in_window.add(frame_hash)

        # Compute change magnitude between consecutive frames
        if grid_a is not None and grid_b is not None:
            try:
                h = min(grid_a.shape[0], grid_b.shape[0])
                w = min(grid_a.shape[1], grid_b.shape[1])
                diff = np.sum(np.abs(grid_a[:h, :w] - grid_b[:h, :w]))
                total = h * w
                self.change_magnitudes.append(diff / max(total, 1))
            except Exception:
                self.change_magnitudes.append(0.0)
        else:
            self.change_magnitudes.append(0.0)

        # Detect oscillation (states repeating with period 2)
        self._detect_oscillation()
        # Detect progress (unique states increasing)
        self._detect_progress()

    def _detect_oscillation(self):
        """Check for alternating states (period 2 oscillation)."""
        if len(self.hash_history) < 4:
            self._oscillation_detected = False
            return
        recent = list(self.hash_history)
        checks = 0
        oscillating = 0
        for i in range(0, min(len(recent) - 2, 10), 2):
            checks += 1
            if recent[i] == recent[i + 2]:
                oscillating += 1
        self._oscillation_detected = checks > 0 and oscillating / checks > 0.7
        if self._oscillation_detected:
            self._oscillation_period = 2

    def _detect_progress(self):
        """Check if the agent is visiting new states (making progress)."""
        recent_hashes = list(self.hash_history)[-self.window:]
        unique_recent = len(set(recent_hashes))
        self._progress_detected = unique_recent > len(recent_hashes) * 0.5

    def is_oscillating(self):
        return self._oscillation_detected

    def is_progressing(self):
        return self._progress_detected

    def get_avg_change_magnitude(self):
        if not self.change_magnitudes:
            return 0.0
        return float(np.mean(list(self.change_magnitudes)))

    def is_stagnant(self):
        """Agent is stagnant if not oscillating and not progressing."""
        return not self._oscillation_detected and not self._progress_detected

    def reset(self):
        self.hash_history.clear()
        self.change_magnitudes.clear()
        self._oscillation_detected = False
        self._progress_detected = False
        self._oscillation_period = 0
        self._unique_states_in_window.clear()

    def get_stats(self):
        return (f"Temporal: osc={self._oscillation_detected}, prog={self._progress_detected}, "
                f"avg_change={self.get_avg_change_magnitude():.4f}")


# ============================================================
# Hypothesis Tracker (kept from V2, enhanced)
# ============================================================
class Hypothesis:
    def __init__(self, description, action_name=""):
        self.description = description
        self.action_name = action_name
        self.confidence = 1.0
        self.evidence_for = 0
        self.evidence_against = 0
        self.verified = False
        self.falsified = False

    def update(self, positive):
        if positive:
            self.evidence_for += 1
            self.confidence += Config.HYPOTHESIS_CONFIDENCE_INCREMENT
            if self.confidence >= 4.0:
                self.verified = True
        else:
            self.evidence_against += 1
            self.confidence -= Config.HYPOTHESIS_CONFIDENCE_DECREMENT
            if self.confidence <= -1.0:
                self.falsified = True


class HypothesisTracker:
    def __init__(self):
        self.hypotheses = []
        self.action_effects = defaultdict(lambda: {"changed_state": 0, "no_change": 0, "level_up": 0, "game_over": 0})
        self._initial_hypotheses_added = False
        # V3: Track which actions caused level completion
        self.level_winning_actions = []
        self.action_meaning = {}  # action_name -> {"direction": (dr,dc), "type": "move"/"interact"}

    def add_initial_hypotheses(self, available_actions):
        if self._initial_hypotheses_added:
            return
        default_beliefs = {
            "ACTION1": "ACTION1 moves player up / triggers input 1",
            "ACTION2": "ACTION2 moves player down / triggers input 2",
            "ACTION3": "ACTION3 moves player left / triggers input 3",
            "ACTION4": "ACTION4 moves player right / triggers input 4",
            "ACTION5": "ACTION5 is an interact/action button (space/enter)",
            "ACTION6": "ACTION6 clicks/points at a coordinate",
            "ACTION7": "ACTION7 is undo",
        }
        for act in available_actions:
            name = act.name if hasattr(act, "name") else str(act)
            if name in default_beliefs:
                self.hypotheses.append(Hypothesis(default_beliefs[name], action_name=name))
        self.hypotheses.append(Hypothesis("Non-zero cells represent walls/obstacles"))
        self.hypotheses.append(Hypothesis("Zero cells represent walkable floor/empty space"))
        self.hypotheses.append(Hypothesis("Moving into walls has no effect"))
        self.hypotheses.append(Hypothesis("The player entity is visible in the grid"))
        self.hypotheses.append(Hypothesis("Reaching certain locations causes level completion"))
        # V3: Additional hypotheses
        self.hypotheses.append(Hypothesis("ACTION5 triggers level completion"))
        self.hypotheses.append(Hypothesis("Reaching a specific cell triggers level completion"))
        self.hypotheses.append(Hypothesis("Collecting all objects of a color triggers level completion"))
        self._initial_hypotheses_added = True
        logger.info(f"Initialized {len(self.hypotheses)} hypotheses")

    def record_action_result(self, action_name, prev_hash, curr_hash, prev_levels, curr_levels, state):
        effects = self.action_effects[action_name]
        changed = prev_hash != curr_hash
        if changed:
            effects["changed_state"] += 1
        else:
            effects["no_change"] += 1
        if curr_levels > prev_levels:
            effects["level_up"] += 1
            self.level_winning_actions.append(action_name)
            for h in self.hypotheses:
                if h.action_name == action_name and not h.falsified:
                    h.update(True)
        else:
            for h in self.hypotheses:
                if h.action_name == action_name and changed:
                    h.update(True)
                elif h.action_name == action_name and not changed:
                    h.update(False)

    def get_best_untested_action(self, available_actions):
        tested_actions = {h.action_name for h in self.hypotheses if h.verified or h.falsified}
        for act in available_actions:
            name = act.name if hasattr(act, "name") else str(act)
            if name not in tested_actions and name != "RESET":
                return name
        return None

    def get_best_winning_action(self):
        """Return the action most associated with level wins."""
        if not self.level_winning_actions:
            return None
        counts = Counter(self.level_winning_actions)
        return counts.most_common(1)[0][0]

    def record_action_meaning(self, action_name, direction=None, action_type=None):
        """Record what an action does based on observation."""
        if action_name not in self.action_meaning:
            self.action_meaning[action_name] = {}
        if direction is not None:
            self.action_meaning[action_name]["direction"] = direction
        if action_type is not None:
            self.action_meaning[action_name]["type"] = action_type

    def get_summary(self):
        verified = sum(1 for h in self.hypotheses if h.verified)
        falsified = sum(1 for h in self.hypotheses if h.falsified)
        testing = sum(1 for h in self.hypotheses if not h.verified and not h.falsified)
        return f"Hypotheses: {verified} verified, {falsified} falsified, {testing} testing, {len(self.level_winning_actions)} wins recorded"


# ============================================================
# Spatial Memory (kept from V2)
# ============================================================
class SpatialMemory:
    def __init__(self):
        self.visited = set()
        self.wall_cells = set()
        self.floor_cells = set()
        self.player_pos = None
        self.player_color = None
        self.grid_height = 0
        self.grid_width = 0
        self.movement_mapping = {}
        self.changed_cells_history = deque(maxlen=5)

    def reset_for_level(self):
        self.visited.clear()
        self.wall_cells.clear()
        self.floor_cells.clear()
        self.grid_height = 0
        self.grid_width = 0
        self.changed_cells_history.clear()

    def update(self, grid, prev_grid=None, prev_action=None):
        if grid is None:
            return
        h, w = grid.shape
        self.grid_height = h
        self.grid_width = w
        for r in range(h):
            for c in range(w):
                val = int(grid[r, c])
                if val != Config.BACKGROUND_COLOR:
                    self.wall_cells.add((r, c))
                else:
                    self.floor_cells.add((r, c))
        if prev_grid is not None and prev_action is not None:
            changed_cells = []
            min_h, min_w = min(h, prev_grid.shape[0]), min(w, prev_grid.shape[1])
            for r in range(min_h):
                for c in range(min_w):
                    if grid[r, c] != prev_grid[r, c]:
                        changed_cells.append((r, c, int(grid[r, c]), int(prev_grid[r, c])))
            self.changed_cells_history.append(changed_cells)
            if changed_cells:
                self._infer_movement_direction(prev_action, changed_cells)

    def _infer_movement_direction(self, action, changed_cells):
        if not changed_cells:
            return
        disappeared = [(r, c, v) for r, c, nv, v in changed_cells if v != Config.BACKGROUND_COLOR]
        appeared = [(r, c, nv) for r, c, nv, v in changed_cells if nv != Config.BACKGROUND_COLOR]
        if disappeared and appeared:
            old_pos = (disappeared[0][0], disappeared[0][1])
            new_pos = (appeared[0][0], appeared[0][1])
            dr = new_pos[0] - old_pos[0]
            dc = new_pos[1] - old_pos[1]
            if (dr, dc) != (0, 0):
                self.movement_mapping[action] = (dr, dc)
                self.player_color = disappeared[0][2]
                self.player_pos = new_pos

    def mark_area_visited(self, center_r, center_c, radius=2):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                self.visited.add((center_r + dr, center_c + dc))

    def get_frontier_cells(self):
        frontier = []
        for r, c in self.visited:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.grid_height and 0 <= nc < self.grid_width
                        and (nr, nc) not in self.visited and (nr, nc) not in self.wall_cells):
                    frontier.append((nr, nc))
        return list(set(frontier))

    def get_unexplored_direction(self, pos=None):
        if pos is None:
            pos = self.player_pos
        if pos is None:
            return None
        frontiers = self.get_frontier_cells()
        if not frontiers:
            return None
        best_dir = None
        best_dist = float("inf")
        for fr, fc in frontiers:
            dist = manhattan_distance(pos[0], pos[1], fr, fc)
            if dist < best_dist:
                best_dist = dist
                dr = fr - pos[0]
                dc = fc - pos[1]
                best_dir = (dr, dc)
        if best_dir is None:
            return None
        dr, dc = best_dir
        best_action = None
        best_score = float("inf")
        for action, (mr, mc) in self.movement_mapping.items():
            score = abs(mr - dr) + abs(mc - dc)
            if score < best_score:
                best_score = score
                best_action = action
        return best_action

    def get_explore_ratio(self):
        total = len(self.floor_cells)
        if total == 0:
            return 0.0
        return len(self.visited) / total

    def get_recently_changed_cells(self):
        """Get cells that changed in the most recent frame transition."""
        if self.changed_cells_history:
            return self.changed_cells_history[-1]
        return []


# ============================================================
# Object Tracker (kept from V2)
# ============================================================
class TrackedObject:
    def __init__(self, obj_id, cells, color, frame_num):
        self.obj_id = obj_id
        self.color = color
        self.cells = set(cells)
        self.size = len(cells)
        self.centroid_r = sum(r for r, c in cells) / max(len(cells), 1)
        self.centroid_c = sum(c for r, c in cells) / max(len(cells), 1)
        self.first_seen = frame_num
        self.last_seen = frame_num
        self.changed_recently = False
        self.interaction_attempts = 0

    def update_position(self, cells, frame_num):
        old_cr, old_cc = self.centroid_r, self.centroid_c
        self.cells = set(cells)
        self.size = len(cells)
        self.centroid_r = sum(r for r, c in cells) / max(len(cells), 1)
        self.centroid_c = sum(c for r, c in cells) / max(len(cells), 1)
        self.last_seen = frame_num
        dist = math.sqrt((self.centroid_r - old_cr) ** 2 + (self.centroid_c - old_cc) ** 2)
        self.changed_recently = dist > 0.5
        return self.changed_recently


class ObjectTracker:
    def __init__(self):
        self.objects = {}
        self.next_obj_id = 1
        self.disappeared_objects = set()
        self.new_objects = set()

    def reset_for_level(self):
        self.objects.clear()
        self.next_obj_id = 1
        self.disappeared_objects.clear()
        self.new_objects.clear()

    def _find_connected_components(self, grid):
        if grid is None:
            return []
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != Config.BACKGROUND_COLOR and not visited[r, c]:
                    component = []
                    queue = deque([(r, c)])
                    visited[r, c] = True
                    color = grid[r, c]
                    while queue:
                        cr, cc = queue.popleft()
                        component.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color):
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                    components.append(component)
        return components

    def update(self, grid, frame_num):
        if grid is None:
            return
        self.new_objects.clear()
        self.disappeared_objects.clear()
        components = self._find_connected_components(grid)
        matched_obj_ids = set()
        matched_comp_ids = set()
        for obj_id, obj in self.objects.items():
            best_comp_idx = -1
            best_overlap = 0
            for i, comp in enumerate(components):
                if i in matched_comp_ids:
                    continue
                overlap = len(obj.cells & set(comp))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_comp_idx = i
            if best_comp_idx >= 0 and best_overlap >= max(1, obj.size * 0.3):
                obj.update_position(components[best_comp_idx], frame_num)
                matched_obj_ids.add(obj_id)
                matched_comp_ids.add(best_comp_idx)
        for obj_id in self.objects:
            if obj_id not in matched_obj_ids:
                self.disappeared_objects.add(obj_id)
        for i, comp in enumerate(components):
            if i not in matched_comp_ids and len(comp) >= Config.MIN_OBJECT_SIZE:
                r0, c0 = comp[0]
                color = int(grid[r0, c0])
                obj = TrackedObject(self.next_obj_id, comp, color, frame_num)
                self.objects[self.next_obj_id] = obj
                self.new_objects.add(self.next_obj_id)
                self.next_obj_id += 1

    def get_click_candidates(self, player_pos=None, max_candidates=10, tried_clicks=None):
        if tried_clicks is None:
            tried_clicks = set()
        candidates = []
        for obj_id, obj in self.objects.items():
            if obj.interaction_attempts > 4:
                continue
            cr, cc = int(obj.centroid_r), int(obj.centroid_c)
            if (cr, cc) in tried_clicks:
                continue
            score = 0.0
            if obj_id in self.new_objects:
                score += 50.0
            if obj.changed_recently:
                score += 30.0
            if player_pos is not None:
                dist = manhattan_distance(player_pos[0], player_pos[1], cr, cc)
                score += max(0, 20 - dist * 0.5)
            if obj.size <= 4:
                score += 20.0
            elif obj.size <= 9:
                score += 12.0
            score += max(0, 10 - obj.interaction_attempts * 3)
            if score > 0:
                candidates.append((cr, cc, score))
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:max_candidates]

    def get_object_summary(self):
        return f"Objects: {len(self.objects)} tracked, {len(self.new_objects)} new, {len(self.disappeared_objects)} disappeared"


# ============================================================
# Stuck Detector (enhanced from V2 — multi-phase recovery)
# ============================================================
class StuckDetector:
    """Enhanced stuck detection with 6-phase recovery strategy."""

    PHASE_UNDO = 0
    PHASE_UNTESTED_SIMPLE = 1
    PHASE_STATE_GRAPH = 2
    PHASE_SMART_CLICK = 3
    PHASE_RANDOM_ESCAPE = 4
    PHASE_RESET = 5

    def __init__(self):
        self.state_visit_counts = Counter()
        self.no_progress_count = 0
        self.loop_detected = False
        self.stuck_count = 0
        self.recovery_mode = False
        self.recovery_phase = 0
        self.recovery_action_idx = 0
        self.last_progress_hash = None
        self.recovery_success = defaultdict(int)  # phase -> success count

    def reset_for_level(self):
        self.state_visit_counts.clear()
        self.no_progress_count = 0
        self.loop_detected = False
        self.stuck_count = 0
        self.recovery_mode = False
        self.recovery_phase = 0
        self.recovery_action_idx = 0
        self.last_progress_hash = None

    def update(self, state_hash, levels_completed, prev_levels):
        status = {
            "loop_detected": False, "no_progress": False,
            "should_recover": False, "progress_made": levels_completed > prev_levels,
            "stagnant": False,
        }
        self.state_visit_counts[state_hash] += 1
        if self.state_visit_counts[state_hash] >= Config.LOOP_THRESHOLD:
            status["loop_detected"] = True
            self.loop_detected = True
        if state_hash == self.last_progress_hash:
            self.no_progress_count += 1
        else:
            self.no_progress_count = 0
            self.last_progress_hash = state_hash
        if self.no_progress_count >= Config.NO_PROGRESS_THRESHOLD:
            status["no_progress"] = True
            status["stagnant"] = True
        if status["loop_detected"] or status["no_progress"]:
            self.stuck_count += 1
            if self.stuck_count >= 2:
                status["should_recover"] = True
                if not self.recovery_mode:
                    self.recovery_mode = True
                    self.recovery_phase = self.PHASE_UNDO
                    self.recovery_action_idx = 0
                    logger.info(f"Starting multi-phase recovery (phase {self.recovery_phase})")
        if status["progress_made"]:
            if self.recovery_mode:
                self.recovery_success[self.recovery_phase] += 1
            self.recovery_mode = False
            self.stuck_count = 0
            self.recovery_phase = 0
        return status

    def get_preferred_phase(self):
        """Return the phase that has been most successful."""
        if not self.recovery_success:
            return self.PHASE_UNDO
        return max(self.recovery_success, key=self.recovery_success.get)

    def advance_phase(self):
        """Move to next recovery phase."""
        self.recovery_phase += 1
        self.recovery_action_idx = 0
        if self.recovery_phase > Config.STUCK_RECOVERY_PHASES:
            self.recovery_mode = False


# ============================================================
# Level Progression Manager (enhanced — per-game knowledge base)
# ============================================================
class LevelProgressionManager:
    """Enhanced cross-level knowledge transfer with per-game knowledge base."""

    def __init__(self):
        self.current_level = 0
        self.levels_completed = 0
        self.win_levels = 1
        self.cross_level_knowledge = {
            "movement_verified": False,
            "interact_useful": False,
            "click_useful": False,
        }
        # V3: Per-game knowledge base
        self.game_knowledge = {
            "action_meanings": {},       # action_name -> what it does
            "win_triggers": [],          # list of (action_name, context) that caused wins
            "level_win_actions": [],     # action sequences that won each level
            "object_interactions": {},   # obj_color -> what happened when clicked
            "movement_actions": [],      # list of actions that move the player
            "interact_actions": [],      # list of actions that interact
            "typical_win_action": None,  # most common winning action
            "level_patterns": [],        # patterns observed across levels
        }
        self.actions_since_level_start = 0
        self.level_action_efficiency = []
        self.current_level_actions = []  # track actions in current level

    def update(self, levels_completed, win_levels):
        info = {"new_level": False, "level_completed": False, "game_won": False}
        if levels_completed > self.levels_completed:
            info["level_completed"] = True
            info["new_level"] = True
            self.level_action_efficiency.append(self.actions_since_level_start)
            self.game_knowledge["level_win_actions"].append(list(self.current_level_actions))
            self.actions_since_level_start = 0
            self.current_level_actions.clear()
        self.levels_completed = levels_completed
        self.current_level = levels_completed
        self.win_levels = win_levels
        if levels_completed >= win_levels and win_levels > 0:
            info["game_won"] = True
        return info

    def record_knowledge(self, key, value=True):
        self.cross_level_knowledge[key] = value

    def record_action_meaning(self, action_name, meaning):
        """Record what an action does in this game."""
        if action_name not in self.game_knowledge["action_meanings"]:
            self.game_knowledge["action_meanings"][action_name] = []
        self.game_knowledge["action_meanings"][action_name].append(meaning)

    def record_win_trigger(self, action_name, context=""):
        """Record an action that triggered a level win."""
        self.game_knowledge["win_triggers"].append({"action": action_name, "context": context})
        # Update typical win action
        counts = Counter(t["action"] for t in self.game_knowledge["win_triggers"])
        self.game_knowledge["typical_win_action"] = counts.most_common(1)[0][0]

    def record_movement_action(self, action_name):
        if action_name not in self.game_knowledge["movement_actions"]:
            self.game_knowledge["movement_actions"].append(action_name)

    def record_interact_action(self, action_name):
        if action_name not in self.game_knowledge["interact_actions"]:
            self.game_knowledge["interact_actions"].append(action_name)

    def get_suggested_first_actions(self):
        """Based on past levels, suggest actions to try first in a new level."""
        suggestions = []
        # If we know a typical winning action, try it early
        if self.game_knowledge["typical_win_action"]:
            suggestions.append(self.game_knowledge["typical_win_action"])
        # Add movement actions first for exploration
        for a in self.game_knowledge["movement_actions"]:
            if a not in suggestions:
                suggestions.append(a)
        # Add interact actions
        for a in self.game_knowledge["interact_actions"]:
            if a not in suggestions:
                suggestions.append(a)
        return suggestions

    def increment_actions(self):
        self.actions_since_level_start += 1

    def track_action(self, action_name):
        self.current_level_actions.append(action_name)

    def get_summary(self):
        return (f"Level {self.current_level}/{self.win_levels}, "
                f"{len(self.level_action_efficiency)} levels completed, "
                f"{len(self.game_knowledge['win_triggers'])} wins recorded")


# ============================================================
# Goal Inference (NEW)
# ============================================================
class GoalInference:
    """
    Infers the likely win condition for the current level
    based on observations and past level patterns.
    """

    def __init__(self):
        self.known_win_conditions = []
        self.current_hypothesis = None
        self.current_confidence = 0.0
        self._observed_states = []

    def add_win_observation(self, pre_win_state_hash, pre_win_grid, action_taken):
        """Record what happened right before a level was won."""
        self.known_win_conditions.append({
            "state_hash": pre_win_state_hash,
            "grid_snapshot": self._grid_signature(pre_win_grid),
            "action": action_taken,
        })

    def _grid_signature(self, grid):
        """Create a compact signature of a grid state."""
        if grid is None:
            return None
        try:
            unique_colors = set(np.unique(grid).tolist())
            total_nonzero = int(np.sum(grid != 0))
            color_counts = Counter(int(v) for v in grid.flatten() if v != 0)
            return {
                "unique_colors": unique_colors,
                "total_nonzero": total_nonzero,
                "color_counts": dict(color_counts.most_common()),
                "shape": grid.shape,
            }
        except Exception:
            return None

    def infer_goal(self, current_grid, level_manager):
        """Infer the most likely win condition for the current level."""
        if not self.known_win_conditions:
            self.current_hypothesis = "explore"
            self.current_confidence = 0.0
            return "explore"

        # Check if current state matches any known pre-win state pattern
        current_sig = self._grid_signature(current_grid)
        if current_sig is None:
            self.current_hypothesis = "explore"
            return "explore"

        # Pattern matching against known win conditions
        best_match = None
        best_score = 0.0

        for wc in self.known_win_conditions:
            score = self._match_score(current_sig, wc.get("grid_snapshot"))
            if score > best_score:
                best_score = score
                best_match = wc

        # Check for common patterns
        patterns = self._detect_common_patterns(current_grid, level_manager)
        if patterns:
            for pattern, conf in patterns:
                if conf > best_score:
                    best_score = conf
                    best_match = {"action": pattern}

        if best_match and best_score > 0.3:
            self.current_hypothesis = best_match.get("action", "unknown")
            self.current_confidence = best_score
        else:
            self.current_hypothesis = "explore"
            self.current_confidence = 0.0

        return self.current_hypothesis

    def _match_score(self, sig_a, sig_b):
        """Score similarity between two grid signatures."""
        if sig_a is None or sig_b is None:
            return 0.0
        score = 0.0
        # Similar number of nonzero cells
        if sig_a.get("total_nonzero") and sig_b.get("total_nonzero"):
            ratio = min(sig_a["total_nonzero"], sig_b["total_nonzero"]) / max(sig_a["total_nonzero"], sig_b["total_nonzero"], 1)
            score += ratio * 0.3
        # Similar color composition
        colors_a = set(sig_a.get("unique_colors", set()))
        colors_b = set(sig_b.get("unique_colors", set()))
        if colors_a and colors_b:
            jaccard = len(colors_a & colors_b) / len(colors_a | colors_b) if (colors_a | colors_b) else 0
            score += jaccard * 0.4
        # Similar shape
        if sig_a.get("shape") == sig_b.get("shape"):
            score += 0.3
        return min(score, 1.0)

    def _detect_common_patterns(self, grid, level_manager):
        """Detect common win condition patterns in current grid."""
        patterns = []
        try:
            # Pattern: all objects gone (collected)
            total_nonzero = int(np.sum(grid != 0))
            if total_nonzero == 0 or total_nonzero <= 2:
                patterns.append(("all_collected", 0.8))

            # Pattern: single color remaining
            unique_colors = set(np.unique(grid).tolist()) - {0}
            if len(unique_colors) == 1:
                patterns.append(("single_color_remaining", 0.6))

            # Pattern: player at target (check if any win was triggered by ACTION5 near objects)
            if level_manager.game_knowledge.get("typical_win_action") == "ACTION5":
                patterns.append(("interact_near_object", 0.5))

            # Pattern: specific location (check if wins correlate with certain areas)
            if self.known_win_conditions:
                # Check if wins happen after specific number of actions
                avg_actions = np.mean([len(lwa) for lwa in level_manager.game_knowledge.get("level_win_actions", [[]])[-3:]] or [0])
                current_actions = level_manager.actions_since_level_start
                if avg_actions > 0 and current_actions >= avg_actions * 0.8:
                    patterns.append(("likely_close_to_win", 0.4))
        except Exception:
            pass
        return patterns

    def get_confidence(self):
        return self.current_confidence

    def reset_level(self):
        self.current_hypothesis = None
        self.current_confidence = 0.0

    def get_stats(self):
        return f"Goal: {self.current_hypothesis} (conf={self.current_confidence:.2f}), {len(self.known_win_conditions)} known wins"


# ============================================================
# Smart Click Selector v3 (enhanced)
# ============================================================
class SmartClickSelector:
    """
    Enhanced ACTION6 target selection with priority scoring.
    V3 adds: frontier cells, recently changed cells, novel colored cells.
    """

    def __init__(self):
        self.tried_clicks = set()
        self.click_history = []  # (x, y, action_name, changed)
        self._novel_colors = set()

    def reset_for_level(self):
        self.tried_clicks.clear()
        self.click_history.clear()
        self._novel_colors.clear()

    def update_novel_colors(self, grid):
        """Track colors seen in the grid."""
        if grid is None:
            return
        try:
            self._novel_colors = set(int(v) for v in np.unique(grid).tolist() if v != 0)
        except Exception:
            pass

    def select_target(self, grid, spatial_memory, object_tracker, temporal_detector=None):
        """
        Select the best (x, y) target for ACTION6 using priority scoring.
        Returns (x, y, score) or None.
        """
        candidates = []

        # 1. Unexplored frontier cells (highest priority)
        frontier = spatial_memory.get_frontier_cells()
        for r, c in frontier[:20]:
            if (r, c) not in self.tried_clicks:
                candidates.append((r, c, 100.0, "frontier"))

        # 2. Cells adjacent to recently changed cells (likely interactive)
        recently_changed = spatial_memory.get_recently_changed_cells()
        for r, c, nv, ov in recently_changed:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < spatial_memory.grid_height and 0 <= nc < spatial_memory.grid_width
                        and (nr, nc) not in self.tried_clicks):
                    candidates.append((nr, nc, 85.0, "near_changed"))

        # 3. Object centroids (from object tracker)
        obj_candidates = object_tracker.get_click_candidates(
            spatial_memory.player_pos,
            max_candidates=Config.CLICK_CANDIDATE_LIMIT,
            tried_clicks=self.tried_clicks,
        )
        for r, c, score in obj_candidates:
            candidates.append((r, c, score, "object"))

        # 4. Small isolated objects (high interaction probability)
        if grid is not None:
            try:
                for obj_id, obj in object_tracker.objects.items():
                    if obj.size <= 3 and obj.interaction_attempts < 2:
                        cr, cc = int(obj.centroid_r), int(obj.centroid_c)
                        if (cr, cc) not in self.tried_clicks:
                            candidates.append((cr, cc, 75.0, "small_object"))
            except Exception:
                pass

        # 5. Cells near player position (efficient to reach)
        if spatial_memory.player_pos is not None:
            pr, pc = spatial_memory.player_pos
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = pr + dr, pc + dc
                    if (0 <= nr < spatial_memory.grid_height and 0 <= nc < spatial_memory.grid_width
                            and (nr, nc) not in self.tried_clicks
                            and grid is not None and grid[nr, nc] != 0):
                        dist = manhattan_distance(pr, pc, nr, nc)
                        candidates.append((nr, nc, max(0, 60 - dist * 5), "near_player"))

        # 6. Novel colored cells
        if grid is not None:
            try:
                for r in range(min(spatial_memory.grid_height, 64)):
                    for c in range(min(spatial_memory.grid_width, 64)):
                        if grid[r, c] != 0 and (r, c) not in self.tried_clicks:
                            color = int(grid[r, c])
                            if color in self._novel_colors:
                                candidates.append((r, c, 50.0, "novel_color"))
            except Exception:
                pass

        if not candidates:
            return None

        # Deduplicate by position, keep highest score
        best_by_pos = {}
        for r, c, score, source in candidates:
            if (r, c) not in best_by_pos or score > best_by_pos[(r, c)][1]:
                best_by_pos[(r, c)] = (score, source)

        # Sort by score descending
        sorted_candidates = sorted(best_by_pos.items(), key=lambda x: x[1][0], reverse=True)

        r, c = sorted_candidates[0][0]
        score, source = sorted_candidates[0][1]
        self.tried_clicks.add((r, c))
        return (c, r, score)  # Note: ACTION6 uses (x, y) = (col, row)

    def record_click_result(self, x, y, changed_state):
        """Record whether a click had an effect."""
        self.click_history.append((x, y, changed_state))


# ============================================================
# Main SmartAgent V3 (core logic)
# ============================================================
class SmartAgent:
    """V3 Agent with CNN predictor, State Graph, MCTS, Temporal Detection, Goal Inference."""

    def __init__(self):
        self.game_id = ""
        self.frames = []
        self.action_counter = 0

        # V2 subsystems (kept)
        self.hypothesis_tracker = HypothesisTracker()
        self.spatial_memory = SpatialMemory()
        self.object_tracker = ObjectTracker()
        self.stuck_detector = StuckDetector()
        self.level_manager = LevelProgressionManager()

        # V3 new subsystems
        self.cnn_predictor = StateTransitionPredictor()
        self.state_graph = StateGraph()
        self.mcts = MCTS()
        self.temporal_detector = TemporalDetector()
        self.goal_inference = GoalInference()
        self.click_selector = SmartClickSelector()

        # State tracking
        self.prev_frame_hash = ""
        self.prev_grid = None
        self.prev_action_name = ""
        self.prev_levels_completed = 0
        self.prev_state_hash = ""
        self.tried_clicks = set()
        self.click_count = 0
        self.click_effect_count = 0
        self._available_actions_cache = []
        self._started = False
        self._initial_reset_done = False
        self._explore_phase = 0
        self._direction_test_idx = 0
        self._direction_test_order = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        self._interact_tested = False
        self._undo_tested = False
        self._phase_actions_count = 0
        self.action_history = []
        self._level_action_buffer = []  # actions taken in current level
        self._pre_win_state = None
        self._pre_win_hash = ""

    def _get_grid(self, frame):
        try:
            if hasattr(frame, "frame") and frame.frame:
                return grid_to_np(frame.frame)
        except Exception:
            pass
        return None

    def _make_action(self, action_name, x=None, y=None, reasoning=""):
        from arcengine import GameAction
        action_map = {
            "RESET": GameAction.RESET, "ACTION1": GameAction.ACTION1,
            "ACTION2": GameAction.ACTION2, "ACTION3": GameAction.ACTION3,
            "ACTION4": GameAction.ACTION4, "ACTION5": GameAction.ACTION5,
            "ACTION6": GameAction.ACTION6, "ACTION7": GameAction.ACTION7,
        }
        action = action_map.get(action_name, GameAction.ACTION5)
        if action_name == "ACTION6" and x is not None and y is not None:
            action.set_data({"x": int(x), "y": int(y)})
        if reasoning:
            action.reasoning = reasoning
        return action

    def is_done(self, frames, latest_frame):
        try:
            from arcengine import GameState
            if latest_frame.state == GameState.WIN:
                logger.info(f"[{self.game_id}] Game WON!")
                return True
        except Exception:
            pass
        if self.action_counter >= Config.MAX_ACTIONS:
            return True
        return False

    def choose_action(self, frames, latest_frame):
        try:
            return self._choose_impl(frames, latest_frame)
        except Exception as e:
            logger.error(f"Error in choose_action: {e}\n{traceback.format_exc()}")
            try:
                from arcengine import GameState
                if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                    return self._make_action("RESET", reasoning="fallback reset")
            except Exception:
                pass
            return self._make_action("ACTION1", reasoning="error fallback")

    def _choose_impl(self, frames, latest_frame):
        self.frames = frames
        current_grid = self._get_grid(latest_frame)
        current_hash = grid_hash(latest_frame.frame) if hasattr(latest_frame, "frame") else ""
        levels = getattr(latest_frame, "levels_completed", 0)
        win_levels = getattr(latest_frame, "win_levels", 1)
        available = getattr(latest_frame, "available_actions", []) or []
        if available:
            self._available_actions_cache = available

        level_info = self.level_manager.update(levels, win_levels)
        if level_info["new_level"]:
            self._on_new_level(latest_frame)
        if level_info["game_won"]:
            return self._make_action("RESET", reasoning="game won")

        try:
            from arcengine import GameState
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self._initial_reset_done = False
                self._on_new_level(latest_frame)
                return self._make_action("RESET", reasoning=f"state={latest_frame.state}")
        except Exception:
            pass

        if not self._initial_reset_done:
            self._initial_reset_done = True
            self.hypothesis_tracker.add_initial_hypotheses(self._available_actions_cache or available)
            return self._make_action("ACTION1", reasoning="first exploration")

        self._update_subsystems(latest_frame, current_grid, current_hash)

        # Update state graph
        self.state_graph.add_state(current_hash, current_grid)
        if self.prev_state_hash and self.prev_action_name:
            self.state_graph.add_transition(self.prev_state_hash, self.prev_action_name, current_hash)

        # Update temporal detector
        self.temporal_detector.update(current_hash, self.prev_grid, current_grid)

        # Train CNN predictor periodically
        self.cnn_predictor.train_step()

        stuck_status = self.stuck_detector.update(current_hash, levels, self.prev_levels_completed)
        self.prev_levels_completed = levels

        # V3: Check for oscillation — try different action to break it
        if self.temporal_detector.is_oscillating() and self._explore_phase >= 1:
            action = self._break_oscillation(available)
            if action is not None:
                self._record_action(action, latest_frame)
                return action

        # Check for level completion — record pre-win state
        if level_info.get("level_completed"):
            self.goal_inference.add_win_observation(
                self._pre_win_hash, self._pre_win_state, self.prev_action_name
            )
            self.state_graph.mark_win(current_hash)
            self.state_graph.save_solution_path(self._level_action_buffer)
            self.level_manager.record_win_trigger(self.prev_action_name)

        # Save pre-win state for next frame
        self._pre_win_state = current_grid.copy() if current_grid is not None else None
        self._pre_win_hash = current_hash

        if stuck_status["should_recover"] or self.stuck_detector.recovery_mode:
            action = self._handle_stuck_recovery_v3(available, current_grid, current_hash)
            if action is not None:
                self._record_action(action, latest_frame)
                return action

        action = self._select_action_v3(latest_frame, current_grid, available, current_hash)
        self._record_action(action, latest_frame)
        return action

    def _on_new_level(self, frame):
        logger.info(f"[{self.game_id}] New level, resetting subsystems")
        self.spatial_memory.reset_for_level()
        self.object_tracker.reset_for_level()
        self.stuck_detector.reset_for_level()
        self.state_graph.reset_for_level()
        self.temporal_detector.reset()
        self.goal_inference.reset_level()
        self.click_selector.reset_for_level()
        self.tried_clicks.clear()
        self._explore_phase = 0
        self._direction_test_idx = 0
        self._interact_tested = False
        self._undo_tested = False
        self._phase_actions_count = 0
        self.prev_frame_hash = ""
        self.prev_grid = None
        self.prev_action_name = ""
        self.prev_state_hash = ""
        self._level_action_buffer.clear()
        self._pre_win_state = None
        self._pre_win_hash = ""
        if self.spatial_memory.movement_mapping:
            self.level_manager.record_knowledge("movement_verified", True)
            for action, (dr, dc) in self.spatial_memory.movement_mapping.items():
                self.level_manager.record_movement_action(action)
                self.hypothesis_tracker.record_action_meaning(action, direction=(dr, dc), action_type="move")

    def _update_subsystems(self, frame, grid, current_hash):
        if grid is None:
            return
        self.spatial_memory.update(grid, self.prev_grid, self.prev_action_name)
        self.click_selector.update_novel_colors(grid)
        if len(self.frames) % 2 == 0:
            self.object_tracker.update(grid, len(self.frames))
        if self.spatial_memory.player_pos:
            r, c = self.spatial_memory.player_pos
            self.spatial_memory.mark_area_visited(r, c)
        if self.prev_frame_hash and self.prev_action_name:
            prev_levels = self.prev_levels_completed
            curr_levels = getattr(frame, "levels_completed", 0)
            self.hypothesis_tracker.record_action_result(
                self.prev_action_name, self.prev_frame_hash, current_hash,
                prev_levels, curr_levels, getattr(frame, "state", None)
            )
            # Feed CNN predictor
            self.cnn_predictor.add_sample(
                self.prev_grid, grid,
                reward=1.0 if curr_levels > prev_levels else 0.0
            )
        self.prev_frame_hash = current_hash
        self.prev_grid = grid.copy() if grid is not None else None

    def _break_oscillation(self, available):
        """Try a different action to break oscillation pattern."""
        simple = [a.name for a in available if a.is_simple() and a.name != "RESET"]
        if not simple:
            return None
        # Pick action least recently used
        recent_actions = [h["action_name"] for h in self.action_history[-6:] if h.get("action_name")]
        for a in simple:
            if a not in recent_actions:
                return self._make_action(a, reasoning="break oscillation")
        # Pick random non-recent action
        non_recent = [a for a in simple if a not in recent_actions[-2:]]
        if non_recent:
            return self._make_action(random.choice(non_recent), reasoning="break oscillation (random)")
        return None

    def _handle_stuck_recovery_v3(self, available, current_grid, current_hash):
        """
        Enhanced 6-phase stuck recovery:
        0. UNDO (revert last action)
        1. Untested simple actions
        2. State graph unexplored paths
        3. Smart click targets
        4. Random escape actions
        5. RESET the level
        """
        phase = self.stuck_detector.recovery_phase

        if phase == StuckDetector.PHASE_UNDO:
            self.stuck_detector.advance_phase()
            return self._make_action("ACTION7", reasoning="recovery phase 0: undo")

        elif phase == StuckDetector.PHASE_UNTESTED_SIMPLE:
            simple = [a.name for a in available if a.is_simple() and a.name != "RESET"]
            tested = set(h["action_name"] for h in self.action_history[-20:])
            untested = [a for a in simple if a not in tested]
            if untested:
                action_name = untested[self.stuck_detector.recovery_action_idx % len(untested)]
                self.stuck_detector.recovery_action_idx += 1
                if self.stuck_detector.recovery_action_idx >= len(untested):
                    self.stuck_detector.advance_phase()
                return self._make_action(action_name, reasoning=f"recovery phase 1: untested {action_name}")
            self.stuck_detector.advance_phase()

        elif phase == StuckDetector.PHASE_STATE_GRAPH:
            action_name = self.state_graph.find_unexplored_from(current_hash)
            if action_name is not None:
                # Check if there is a known path to win
                win_path = self.state_graph.find_path_to_win(current_hash)
                if win_path:
                    self.stuck_detector.advance_phase()
                    return self._make_action(win_path[0], reasoning=f"recovery phase 2: path to win")
                self.stuck_detector.advance_phase()
                return self._make_action(action_name, reasoning="recovery phase 2: state graph")
            self.state_graph.prune_dead_ends()
            self.stuck_detector.advance_phase()

        elif phase == StuckDetector.PHASE_SMART_CLICK:
            target = self.click_selector.select_target(
                current_grid, self.spatial_memory, self.object_tracker, self.temporal_detector
            )
            if target is not None:
                x, y, score = target
                self.stuck_detector.recovery_action_idx += 1
                if self.stuck_detector.recovery_action_idx >= 3:
                    self.stuck_detector.advance_phase()
                return self._make_action("ACTION6", x=x, y=y, reasoning=f"recovery phase 3: smart click ({x},{y})")
            self.stuck_detector.advance_phase()

        elif phase == StuckDetector.PHASE_RANDOM_ESCAPE:
            simple = [a.name for a in available if a.is_simple() and a.name != "RESET"]
            if simple:
                self.stuck_detector.recovery_action_idx += 1
                if self.stuck_detector.recovery_action_idx >= 3:
                    self.stuck_detector.advance_phase()
                return self._make_action(random.choice(simple), reasoning="recovery phase 4: random escape")
            # Try random click
            if current_grid is not None:
                h, w = current_grid.shape[:2]
                x = random.randint(0, min(w - 1, 63))
                y = random.randint(0, min(h - 1, 63))
                self.stuck_detector.recovery_action_idx += 1
                if self.stuck_detector.recovery_action_idx >= 3:
                    self.stuck_detector.advance_phase()
                return self._make_action("ACTION6", x=x, y=y, reasoning="recovery phase 4: random click")
            self.stuck_detector.advance_phase()

        elif phase == StuckDetector.PHASE_RESET:
            self.stuck_detector.recovery_mode = False
            return self._make_action("RESET", reasoning="recovery phase 5: full reset")

        return self._make_action("ACTION1", reasoning="recovery fallback")

    def _select_action_v3(self, frame, grid, available, current_hash):
        """V3 action selection with MCTS, goal inference, and cross-level knowledge."""
        self._phase_actions_count += 1
        self.level_manager.increment_actions()

        # Small random exploration chance
        if random.random() < Config.RANDOM_EXPLORATION_PROB:
            simple = [a.name for a in available if a.is_simple() and a.name != "RESET"]
            if simple:
                return self._make_action(random.choice(simple), reasoning="random exploration")

        # Phase 0: Initial exploration (test all actions)
        if self._explore_phase == 0:
            return self._phase_initial_exploration_v3(available, grid)

        # Phase 1: MCTS-guided exploration (first N actions)
        if self._explore_phase == 1:
            if self._phase_actions_count <= Config.MCTS_FIRST_N_ACTIONS:
                action = self._try_mcts_selection(available, current_hash)
                if action is not None:
                    return action
            return self._phase_directed_exploration_v3(frame, grid, available, current_hash)

        # Phase 2: Directed exploitation with goal inference
        if self._explore_phase == 2:
            return self._phase_exploitation_v3(frame, grid, available, current_hash)

        return self._make_action("ACTION1", reasoning="default fallback")

    def _phase_initial_exploration_v3(self, available, grid):
        """Enhanced initial exploration with cross-level knowledge."""
        # V3: Try suggested actions from knowledge base first
        if self.level_manager.levels_completed > 0 and self._phase_actions_count == 1:
            suggested = self.level_manager.get_suggested_first_actions()
            if suggested:
                # Try the most likely winning action first
                action_name = suggested[0]
                if any(a.name == action_name for a in available if hasattr(a, "name")):
                    return self._make_action(action_name, reasoning=f"cross-level: try known win action {action_name}")

        # Test movement directions
        if self._direction_test_idx < len(self._direction_test_order):
            action_name = self._direction_test_order[self._direction_test_idx]
            self._direction_test_idx += 1
            return self._make_action(action_name, reasoning=f"initial: test {action_name}")

        # Test interact
        if not self._interact_tested:
            self._interact_tested = True
            return self._make_action("ACTION5", reasoning="initial: test interact")

        # Test undo
        if not self._undo_tested:
            self._undo_tested = True
            return self._make_action("ACTION7", reasoning="initial: test undo")

        # Try smart click
        target = self.click_selector.select_target(grid, self.spatial_memory, self.object_tracker)
        if target is not None:
            x, y, score = target
            self.click_count += 1
            return self._make_action("ACTION6", x=x, y=y, reasoning=f"initial: click ({x},{y}) prio={score:.1f}")

        self._explore_phase = 1
        self._phase_actions_count = 0
        return self._make_action("ACTION1", reasoning="transition to MCTS exploration")

    def _try_mcts_selection(self, available, current_hash):
        """Use MCTS for action selection in early exploration."""
        def simulate_fn(state_hash, action_name):
            fake_next = hashlib.md5(f"{state_hash}_{action_name}_{random.random()}".encode()).hexdigest()[:16]
            # Reward based on past action effectiveness
            effects = self.hypothesis_tracker.action_effects.get(action_name, {})
            changed = effects.get("changed_state", 0)
            total = changed + effects.get("no_change", 0)
            reward = changed / max(total, 1)
            if effects.get("level_up", 0) > 0:
                reward += 1.0
            return fake_next, reward

        action_name = self.mcts.search(current_hash, available, simulate_fn)
        if action_name is not None:
            return self._make_action(action_name, reasoning=f"MCTS: {action_name}")
        return None

    def _phase_directed_exploration_v3(self, frame, grid, available, current_hash):
        """Enhanced directed exploration with state graph and goal inference."""
        # Periodic interact
        if self._phase_actions_count % 10 == 0:
            return self._make_action("ACTION5", reasoning="periodic interact")

        # Periodic undo
        if self._phase_actions_count % 20 == 0:
            return self._make_action("ACTION7", reasoning="periodic undo")

        # Periodic smart click
        if self._phase_actions_count % 7 == 0:
            target = self.click_selector.select_target(grid, self.spatial_memory, self.object_tracker)
            if target is not None:
                x, y, score = target
                self.click_count += 1
                return self._make_action("ACTION6", x=x, y=y, reasoning=f"periodic click ({x},{y}) prio={score:.1f}")

        # V3: Check state graph for unexplored paths
        unexplored = self.state_graph.find_unexplored_from(current_hash)
        if unexplored is not None:
            return self._make_action(unexplored, reasoning=f"state graph: unexplored {unexplored}")

        # V3: Goal inference
        goal = self.goal_inference.infer_goal(grid, self.level_manager)
        if goal != "explore" and self.goal_inference.get_confidence() > Config.GOAL_INFERENCE_CONFIDENCE_THRESHOLD:
            return self._make_action(goal, reasoning=f"goal inference: {goal} (conf={self.goal_inference.get_confidence():.2f})")

        # Frontier-based movement
        frontier_dir = self.spatial_memory.get_unexplored_direction()
        if frontier_dir is not None:
            return self._make_action(frontier_dir, reasoning=f"move to frontier: {frontier_dir}")

        # Untested actions
        untested = self.hypothesis_tracker.get_best_untested_action(available)
        if untested is not None:
            return self._make_action(untested, reasoning=f"test hypothesis: {untested}")

        # Systematic exploration
        if self.spatial_memory.movement_mapping:
            action = self._systematic_explore(available)
            if action is not None:
                return action

        remaining = [a for a in self._direction_test_order if a not in self.spatial_memory.movement_mapping]
        if remaining:
            return self._make_action(remaining[0], reasoning=f"re-test: {remaining[0]}")

        simple = [a for a in available if a.is_simple() and a.name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")]
        if simple:
            return self._make_action(random.choice(simple), reasoning="random movement")

        target = self.click_selector.select_target(grid, self.spatial_memory, self.object_tracker)
        if target is not None:
            x, y, score = target
            self.click_count += 1
            return self._make_action("ACTION6", x=x, y=y, reasoning=f"directed click ({x},{y}) prio={score:.1f}")

        return self._make_action("ACTION1", reasoning="directed fallback")

    def _phase_exploitation_v3(self, frame, grid, available, current_hash):
        """Exploitation phase using learned knowledge and goal inference."""
        # V3: Try known winning action first
        win_action = self.hypothesis_tracker.get_best_winning_action()
        if win_action and self._phase_actions_count % 5 == 0:
            return self._make_action(win_action, reasoning=f"exploit known win action: {win_action}")

        # V3: Use goal inference to guide actions
        goal = self.goal_inference.infer_goal(grid, self.level_manager)
        if goal != "explore" and self.goal_inference.get_confidence() > 0.4:
            return self._make_action(goal, reasoning=f"exploit goal: {goal}")

        # Productive actions
        productive = self._find_productive_actions()
        if productive:
            action_name, score = productive[0]
            return self._make_action(action_name, reasoning=f"exploit: {action_name} (score={score:.1f})")

        # Fall back to directed exploration
        return self._phase_directed_exploration_v3(frame, grid, available, current_hash)

    def _systematic_explore(self, available):
        if not self.spatial_memory.movement_mapping:
            return None
        pos = self.spatial_memory.player_pos
        if pos is None:
            return None
        best_action = None
        best_unvisited_ratio = -1
        for action_name, (dr, dc) in self.spatial_memory.movement_mapping.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if (nr, nc) in self.spatial_memory.wall_cells:
                continue
            unvisited = 0
            total = 0
            for ddr in range(-2, 3):
                for ddc in range(-2, 3):
                    r, c = nr + ddr, nc + ddc
                    if (0 <= r < self.spatial_memory.grid_height and 0 <= c < self.spatial_memory.grid_width):
                        total += 1
                        if (r, c) not in self.spatial_memory.visited:
                            unvisited += 1
            ratio = unvisited / max(total, 1)
            if ratio > best_unvisited_ratio:
                best_unvisited_ratio = ratio
                best_action = action_name
        if best_action:
            return self._make_action(best_action, reasoning=f"systematic ({best_unvisited_ratio:.0%} unvisited)")
        return None

    def _try_smart_click(self, available):
        """Fallback smart click method (delegates to SmartClickSelector)."""
        target = self.click_selector.select_target(
            self.prev_grid, self.spatial_memory, self.object_tracker
        )
        if target is not None:
            x, y, score = target
            self.click_count += 1
            return self._make_action("ACTION6", x=x, y=y, reasoning=f"smart click ({x},{y}) prio={score:.1f}")
        return None

    def _find_productive_actions(self):
        action_scores = defaultdict(float)
        for entry in self.action_history:
            name = entry.get("action_name", "")
            if not name or name == "RESET":
                continue
            score = 0
            if entry.get("changed_state"):
                score += 1.0
            if entry.get("level_up"):
                score += 10.0
            if entry.get("new_object"):
                score += 3.0
            if not entry.get("changed_state"):
                score -= 0.5
            action_scores[name] += score
        scored = [(name, score) for name, score in action_scores.items() if score > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _record_action(self, action, frame):
        action_name = ""
        try:
            action_name = action.name
        except Exception:
            action_name = str(action)
        changed = self.prev_frame_hash != (grid_hash(frame.frame) if hasattr(frame, "frame") else "")
        new_objects = len(self.object_tracker.new_objects) > 0
        level_up = getattr(frame, "levels_completed", 0) > self.prev_levels_completed
        self.action_history.append({
            "step": self.action_counter, "action_name": action_name,
            "changed_state": changed, "new_object": new_objects, "level_up": level_up,
        })
        self._level_action_buffer.append(action_name)
        self.level_manager.track_action(action_name)
        self.prev_action_name = action_name
        self.prev_state_hash = grid_hash(frame.frame) if hasattr(frame, "frame") else ""
        if action_name == "ACTION6" and changed:
            self.click_effect_count += 1
        # Transition to exploitation phase
        if self._explore_phase == 1 and self._phase_actions_count > 25 and self.spatial_memory.get_explore_ratio() > 0.4:
            self._explore_phase = 2
            self._phase_actions_count = 0

    def get_diagnostics(self):
        return {
            "action_counter": self.action_counter,
            "explore_phase": self._explore_phase,
            "hypotheses": self.hypothesis_tracker.get_summary(),
            "spatial": f"visited={len(self.spatial_memory.visited)}, walls={len(self.spatial_memory.wall_cells)}, ratio={self.spatial_memory.get_explore_ratio():.1%}",
            "objects": self.object_tracker.get_object_summary(),
            "stuck": f"loop={self.stuck_detector.loop_detected}, recovery={self.stuck_detector.recovery_mode}, phase={self.stuck_detector.recovery_phase}",
            "levels": self.level_manager.get_summary(),
            "clicks": f"{self.click_count} tried, {self.click_effect_count} effective",
            "movement": self.spatial_memory.movement_mapping,
            "cnn_predictor": f"trainable={self.cnn_predictor.trainable}, samples={len(self.cnn_predictor.samples)}",
            "state_graph": self.state_graph.get_stats(),
            "temporal": self.temporal_detector.get_stats(),
            "goal": self.goal_inference.get_stats(),
        }


# ============================================================
# StandaloneSmartAgent — inherits from Agent base class
# ============================================================
class StandaloneSmartAgent:
    MAX_ACTIONS = Config.MAX_ACTIONS

    def __init__(self, *args, **kwargs):
        from agents.agent import Agent
        Agent.__init__(self, *args, **kwargs)
        self._smart = SmartAgent()
        self._smart.game_id = self.game_id

    def is_done(self, frames, latest_frame):
        try:
            from arcengine import GameState
            if latest_frame.state == GameState.WIN:
                logger.info(f"[{self.game_id}] Game WON!")
                return True
        except Exception:
            pass
        if self.action_counter >= self.MAX_ACTIONS:
            logger.info(f"[{self.game_id}] Reached MAX_ACTIONS ({self.MAX_ACTIONS})")
            return True
        return False

    def choose_action(self, frames, latest_frame):
        self._smart.frames = frames
        self._smart.action_counter = self.action_counter
        try:
            action = self._smart.choose_action(frames, latest_frame)
        except Exception as e:
            logger.error(f"[{self.game_id}] Error: {e}")
            from arcengine import GameState, GameAction
            try:
                if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                    return GameAction.RESET
            except Exception:
                pass
            action = GameAction.ACTION1
        self.action_counter += 1
        if self.action_counter % 25 == 0:
            diag = self._smart.get_diagnostics()
            logger.info(f"[{self.game_id}] Diag: {diag}")
        return action

    @property
    def name(self):
        try:
            base = super().name
        except Exception:
            base = self.game_id
        return f"{base}.smart_v3.{self.MAX_ACTIONS}"


print("SmartAgent V3 loaded successfully")
print(f"MAX_ACTIONS = {Config.MAX_ACTIONS}")
print(f"TORCH_AVAILABLE = {TORCH_AVAILABLE}")
print(f"CNN Predictor: {'enabled' if TORCH_AVAILABLE else 'disabled (no torch)'}")
print(f"MCTS: {Config.MCTS_SIMULATIONS} simulations, depth={Config.MCTS_ROLLOUT_DEPTH}")
print(f"State Graph: max {Config.STATE_GRAPH_MAX_NODES} nodes")
print(f"Stuck Recovery: {Config.STUCK_RECOVERY_PHASES + 1} phases")
print(f"Components: FrameEncoder, StateGraph, MCTS, TemporalDetector,")
print(f"  HypothesisTracker, SpatialMemory, ObjectTracker, StuckDetector,")
print(f"  LevelProgressionManager, GoalInference, SmartClickSelector")
'''

# ============================================================
# Build notebook JSON
# ============================================================

cells = []

# Cell 0: Title markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# SmartAgent V3 — Advanced Exploration Agent for ARC-AGI-3\n",
        "\n",
        "## Overview\n",
        "\n",
        "This notebook implements **SmartAgent V3**, a major upgrade over V2 with 8 new/enhanced modules including CNN-based state prediction, state graph exploration, Monte Carlo Tree Search, temporal pattern detection, goal inference, and improved knowledge transfer.\n",
        "\n",
        "### Key Improvements over V2\n",
        "\n",
        "| Module | V2 | V3 |\n",
        "|--------|----|----|\n",
        "| **State Prediction** | None | CNN Frame Encoder (~90K params, PyTorch) |\n",
        "| **State Exploration** | Random walk | Explicit State Graph with BFS |\n",
        "| **Action Selection** | Heuristic phases | MCTS (UCB1) for early exploration |\n",
        "| **Pattern Detection** | Loop detection only | Temporal Pattern Detection (oscillation, progress) |\n",
        "| **Goal Understanding** | None | Goal Inference with pattern matching |\n",
        "| **Knowledge Transfer** | Basic boolean flags | Per-game knowledge base with win triggers |\n",
        "| **Stuck Recovery** | Single linear sequence | 6-phase recovery with preference learning |\n",
        "| **Click Targeting** | Object centroid scoring | 6-category priority queue with frontier detection |\n",
        "| Hypothesis Tracker | 7 hypotheses | 10 hypotheses + action meaning recording |\n",
        "| Action Budget | 200 | 250 |\n",
        "| CNN Training | N/A | Online training after 40 samples, 8 gradient steps |\n",
        "\n",
        "### Architecture\n",
        "\n",
        "```\n",
        "StandaloneSmartAgent (inherits Agent base class)\n",
        "  └── SmartAgent V3 (core logic)\n",
        "        ├── FrameEncoder (CNN) + StateTransitionPredictor  — NEW\n",
        "        ├── StateGraph         — BFS exploration, cycle detection  — NEW\n",
        "        ├── MCTS               — UCB1 tree search                 — NEW\n",
        "        ├── TemporalDetector   — oscillation/progress detection   — NEW\n",
        "        ├── GoalInference      — win condition prediction         — NEW\n",
        "        ├── HypothesisTracker  — enhanced with action meanings\n",
        "        ├── SpatialMemory      — 2D explored map, player tracking\n",
        "        ├── ObjectTracker      — detect/track/categorize objects\n",
        "        ├── StuckDetector      — 6-phase multi-strategy recovery\n",
        "        ├── LevelProgressionMgr — per-game knowledge base\n",
        "        └── SmartClickSelector — 6-category priority targeting\n",
        "```\n",
        "\n",
        "### Competition API\n",
        "\n",
        "- **Actions**: RESET(0), ACTION1-6, ACTION7 (Undo)\n",
        "- **ACTION6** requires `x, y` coordinates (complex action)\n",
        "- **FrameData**: 3D grid `[layer, row, col]`, values 0-15\n",
        "- **Scoring**: RHAE (Relative Human Action Efficiency)\n",
        "- **Max 80 actions** base class default (we override to 250)"
    ]
})

# Cell 1: Install
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install arc-agi from competition wheels\n",
        "!pip install --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv 2>&1 | tail -5"
    ]
})

# Cell 2: submission.parquet
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# CRITICAL: Create submission.parquet for notebook validation\n",
        "import pandas as pd, os\n",
        "_df = pd.DataFrame([{'row_id':'1_0','game_id':'1','end_of_game':True,'score':1.0}])\n",
        "_df.to_parquet('/kaggle/working/submission.parquet', index=False)\n",
        "print('submission.parquet created')"
    ]
})

# Cell 3: Agent Architecture markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Agent Architecture\n",
        "\n",
        "### CNN Frame Encoder & State Transition Predictor\n",
        "- 4-layer CNN (~90K params): `Conv2d(2,32,5)->Conv2d(32,64,3)->Conv2d(64,64,3)->Conv2d(64,1,3)`\n",
        "- Takes two consecutive 64x64 frames (2 channels) → predicts next frame diff\n",
        "- Parallel value head: `GlobalAvgPool → FC(64,32) → FC(32,1)`\n",
        "- Online training: collects transitions, trains after 40+ samples\n",
        "- Loss = MSE(diff) + 0.1 * MSE(value)\n",
        "\n",
        "### State Graph Explorer\n",
        "- Nodes = frame hashes, edges = action transitions\n",
        "- BFS to find unexplored paths and paths to win states\n",
        "- Cycle detection and dead-end pruning\n",
        "- Saves successful action sequences as solution paths\n",
        "\n",
        "### Monte Carlo Tree Search\n",
        "- UCB1 selection for balanced exploration/exploitation\n",
        "- Random rollout simulation for 8 steps\n",
        "- 15 simulations per decision in first 5 actions of each level\n",
        "- Uses hypothesis tracker rewards for simulation guidance\n",
        "\n",
        "### 3-Phase Exploration Pipeline\n",
        "1. **Phase 0 (Initial)**: Test all actions, cross-level knowledge first\n",
        "2. **Phase 1 (MCTS+Directed)**: MCTS for first 5 actions, then frontier exploration\n",
        "3. **Phase 2 (Exploitation)**: Use learned knowledge + goal inference"
    ]
})

# Cell 4: The massive writefile cell
# Split agent_code into lines for the source array
agent_lines = agent_code.rstrip('\n').split('\n')
source_lines = []
for line in agent_lines:
    source_lines.append(line + '\n')
# Remove trailing newline from last line
if source_lines:
    source_lines[-1] = source_lines[-1].rstrip('\n')

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": source_lines
})

# Cell 5: Validation test
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# Validation: Test agent loading and basic functionality\n",
        "# ============================================================\n",
        "import sys, os\n",
        "sys.path.insert(0, '/kaggle/working')\n",
        "\n",
        "print(\"=\" * 60)\n",
        "print(\"SmartAgent V3 — Validation Tests\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "# Test 1: Import the agent\n",
        "print(\"\\n[Test 1] Importing agent...\")\n",
        "try:\n",
        "    from my_agent import SmartAgent, StandaloneSmartAgent, Config, TORCH_AVAILABLE\n",
        "    print(\"  ✓ SmartAgent imported\")\n",
        "    print(\"  ✓ StandaloneSmartAgent imported\")\n",
        "    print(f\"  ✓ TORCH_AVAILABLE = {TORCH_AVAILABLE}\")\n",
        "except Exception as e:\n",
        "    print(f\"  ✗ Import failed: {e}\")\n",
        "\n",
        "# Test 2: Instantiate agent\n",
        "print(\"\\n[Test 2] Instantiating SmartAgent...\")\n",
        "try:\n",
        "    agent = SmartAgent()\n",
        "    agent.game_id = \"test_game\"\n",
        "    print(\"  ✓ SmartAgent created\")\n",
        "    print(f\"  ✓ MAX_ACTIONS = {Config.MAX_ACTIONS}\")\n",
        "except Exception as e:\n",
        "    print(f\"  ✗ Failed: {e}\")\n",
        "\n",
        "# Test 3: Test subsystems\n",
        "print(\"\\n[Test 3] Testing subsystems...\")\n",
        "try:\n",
        "    print(f\"  ✓ CNN Predictor: trainable={agent.cnn_predictor.trainable}\")\n",
        "    print(f\"  ✓ State Graph: {agent.state_graph.get_stats()}\")\n",
        "    print(f\"  ✓ Temporal Detector: {agent.temporal_detector.get_stats()}\")\n",
        "    print(f\"  ✓ Goal Inference: {agent.goal_inference.get_stats()}\")\n",
        "    print(f\"  ✓ Hypothesis Tracker: {agent.hypothesis_tracker.get_summary()}\")\n",
        "    print(f\"  ✓ Stuck Detector phases: {Config.STUCK_RECOVERY_PHASES + 1}\")\n",
        "    print(f\"  ✓ Click Selector: initialized\")\n",
        "except Exception as e:\n",
        "    print(f\"  ✗ Failed: {e}\")\n",
        "\n",
        "# Test 4: Test with mock frame\n",
        "print(\"\\n[Test 4] Testing with mock frame...\")\n",
        "try:\n",
        "    import numpy as np\n",
        "\n",
        "    class MockFrame:\n",
        "        def __init__(self):\n",
        "            self.frame = [np.zeros((64, 64), dtype=np.int8)]\n",
        "            self.state = \"NOT_PLAYED\"\n",
        "            self.levels_completed = 0\n",
        "            self.win_levels = 1\n",
        "            self.available_actions = []\n",
        "\n",
        "    # Test grid utilities\n",
        "    from my_agent import grid_hash, grid_to_np, manhattan_distance\n",
        "    mock_frame = MockFrame()\n",
        "    h = grid_hash(mock_frame.frame)\n",
        "    print(f\"  ✓ grid_hash works: {h[:12]}...\")\n",
        "    g = grid_to_np(mock_frame.frame)\n",
        "    print(f\"  ✓ grid_to_np works: shape={g.shape}\")\n",
        "    print(f\"  ✓ manhattan_distance: {manhattan_distance(0, 0, 3, 4)} = 7\")\n",
        "\n",
        "    # Test is_done\n",
        "    result = agent.is_done([], mock_frame)\n",
        "    print(f\"  ✓ is_done returns: {result}\")\n",
        "\n",
        "except Exception as e:\n",
        "    import traceback\n",
        "    print(f\"  ✗ Failed: {e}\")\n",
        "    traceback.print_exc()\n",
        "\n",
        "# Test 5: State graph\n",
        "print(\"\\n[Test 5] Testing StateGraph...\")\n",
        "try:\n",
        "    from my_agent import StateGraph\n",
        "    sg = StateGraph()\n",
        "    sg.add_state(\"h1\")\n",
        "    sg.add_state(\"h2\")\n",
        "    sg.add_transition(\"h1\", \"ACTION1\", \"h2\")\n",
        "    sg.add_state(\"h3\")\n",
        "    sg.add_transition(\"h2\", \"ACTION5\", \"h3\")\n",
        "    sg.mark_win(\"h3\")\n",
        "    path = sg.find_path_to_win(\"h1\")\n",
        "    print(f\"  ✓ StateGraph: path to win from h1 = {path}\")\n",
        "    unexplored = sg.find_unexplored_from(\"h1\")\n",
        "    print(f\"  ✓ StateGraph: unexplored from h1 = {unexplored}\")\n",
        "    print(f\"  ✓ StateGraph: {sg.get_stats()}\")\n",
        "except Exception as e:\n",
        "    print(f\"  ✗ Failed: {e}\")\n",
        "\n",
        "# Test 6: MCTS\n",
        "print(\"\\n[Test 6] Testing MCTS...\")\n",
        "try:\n",
        "    from my_agent import MCTS\n",
        "    mcts = MCTS(simulations=5, rollout_depth=3)\n",
        "\n",
        "    class MockAction:\n",
        "        def __init__(self, name):\n",
        "            self.name = name\n",
        "        def is_simple(self):\n",
        "            return True\n",
        "\n",
        "    mock_actions = [MockAction(\"ACTION1\"), MockAction(\"ACTION2\"), MockAction(\"ACTION5\")]\n",
        "    result = mcts.search(\"test_state\", mock_actions, lambda s, a: (\"next\", 0.5))\n",
        "    print(f\"  ✓ MCTS selected: {result}\")\n",
        "    print(f\"  ✓ MCTS root visits: {mcts.root.visits if mcts.root else 'N/A'}\")\n",
        "except Exception as e:\n",
        "    print(f\"  ✗ Failed: {e}\")\n",
        "\n",
        "# Test 7: TemporalDetector\n",
        "print(\"\\n[Test 7] Testing TemporalDetector...\")\n",
        "try:\n",
        "    from my_agent import TemporalDetector\n",
        "    td = TemporalDetector()\n",
        "    for i in range(10):\n",
        "        td.update(f\"state_{i % 3}\", np.zeros((4, 4)), np.ones((4, 4)))\n",
        "    print(f\"  ✓ Oscillating: {td.is_oscillating()}\")\n",
        "    print(f\"  ✓ Progressing: {td.is_progressing()}\")\n",
        "    print(f\"  ✓ Stats: {td.get_stats()}\")\n",
        "except Exception as e:\n",
        "    print(f\"  ✗ Failed: {e}\")\n",
        "\n",
        "# Test 8: GoalInference\n",
        "print(\"\\n[Test 8] Testing GoalInference...\")\n",
        "try:\n",
        "    from my_agent import GoalInference\n",
        "    gi = GoalInference()\n",
        "    test_grid = np.zeros((10, 10), dtype=np.float32)\n",
        "    test_grid[5, 5] = 3\n",
        "    gi.add_win_observation(\"pre_win\", test_grid, \"ACTION5\")\n",
        "    goal = gi.infer_goal(test_grid, agent.level_manager)\n",
        "    print(f\"  ✓ Goal inference: {goal}\")\n",
        "    print(f\"  ✓ Stats: {gi.get_stats()}\")\n",
        "except Exception as e:\n",
        "    print(f\"  ✗ Failed: {e}\")\n",
        "\n",
        "# Test 9: SmartClickSelector\n",
        "print(\"\\n[Test 9] Testing SmartClickSelector...\")\n",
        "try:\n",
        "    from my_agent import SmartClickSelector\n",
        "    scs = SmartClickSelector()\n",
        "    test_grid = np.zeros((20, 20), dtype=np.float32)\n",
        "    test_grid[5, 5] = 3\n",
        "    test_grid[10, 10] = 5\n",
        "    scs.update_novel_colors(test_grid)\n",
        "    # Note: full test requires spatial_memory and object_tracker\n",
        "    print(f\"  ✓ SmartClickSelector created and novel colors tracked\")\n",
        "except Exception as e:\n",
        "    print(f\"  ✗ Failed: {e}\")\n",
        "\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"All validation tests completed!\")\n",
        "print(\"=\" * 60)"
    ]
})

# Cell 6: Summary markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Summary\n",
        "\n",
        "SmartAgent V3 brings 8 major improvements:\n",
        "\n",
        "1. **CNN Frame Encoder** — Learns to predict frame transitions, enabling look-ahead planning\n",
        "2. **State Graph** — Systematic BFS exploration with cycle detection and win-path recovery\n",
        "3. **MCTS** — Monte Carlo Tree Search for intelligent early-action selection\n",
        "4. **Temporal Detection** — Identifies oscillation loops vs genuine progress\n",
        "5. **Goal Inference** — Pattern-matches against known win conditions\n",
        "6. **Cross-Level Knowledge** — Per-game knowledge base with action meanings and win triggers\n",
        "7. **6-Phase Stuck Recovery** — Progressive recovery: undo → untested → graph → click → random → reset\n",
        "8. **Smart Click v3** — 6-category priority scoring: frontier, changed, objects, small, near-player, novel\n",
        "\n",
        "The agent is fully self-contained, works offline, and falls back gracefully when PyTorch is unavailable."
    ]
})

# Cell 7: Competition execution
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os, subprocess, time\n",
        "\n",
        "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n",
        "    print('=== COMPETITION MODE ===')\n",
        "    \n",
        "    # Wait for gateway to be ready\n",
        "    for i in range(60):\n",
        "        r = subprocess.run(['curl', '-sf', 'http://gateway:8001/api/games'],\n",
        "                          capture_output=True, timeout=10)\n",
        "        if r.returncode == 0:\n",
        "            print('Gateway ready')\n",
        "            break\n",
        "        time.sleep(5)\n",
        "    \n",
        "    # Setup agent framework\n",
        "    WORK = '/kaggle/working/ARC-AGI-3-Agents'\n",
        "    SRC = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents'\n",
        "    \n",
        "    if os.path.exists(SRC):\n",
        "        subprocess.run(['cp', '-r', SRC, WORK], check=True)\n",
        "        subprocess.run(['cp', '/kaggle/working/my_agent.py',\n",
        "                       f'{WORK}/agents/templates/smart_agent_v3.py'], check=True)\n",
        "        \n",
        "        # Register agent in __init__.py\n",
        "        init_code = '''from typing import Type\n",
        "from dotenv import load_dotenv\n",
        "from .agent import Agent, Playback\n",
        "from .swarm import Swarm\n",
        "from .templates.random_agent import Random\n",
        "from .templates.smart_agent_v3 import StandaloneSmartAgent\n",
        "\n",
        "load_dotenv()\n",
        "\n",
        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\n",
        "    \"random\": Random,\n",
        "    \"smartv3\": StandaloneSmartAgent,\n",
        "}\n",
        "'''\n",
        "        with open(f'{WORK}/agents/__init__.py', 'w') as f:\n",
        "            f.write(init_code)\n",
        "        \n",
        "        # Configure environment\n",
        "        env_content = 'SCHEME=http\\nHOST=gateway\\nPORT=8001\\nARC_API_KEY=***\\nARC_BASE_URL=http://gateway:8001/\\nOPERATION_MODE=online\\nRECORDINGS_DIR=/kaggle/working/server_recording\\n'\n",
        "        with open(f'{WORK}/.env', 'w') as f:\n",
        "            f.write(env_content)\n",
        "        \n",
        "        os.chdir(WORK)\n",
        "        os.environ['MPLBACKEND'] = 'agg'\n",
        "        \n",
        "        # Run the agent on all games\n",
        "        print('Starting SmartAgent V3 on all games...')\n",
        "        result = subprocess.run(\n",
        "            ['python', 'main.py', '--agent', 'smartv3'],\n",
        "            capture_output=True, text=True, timeout=3600\n",
        "        )\n",
        "        \n",
        "        print(f'Exit code: {result.returncode}')\n",
        "        if result.stdout:\n",
        "            print('STDOUT:', result.stdout[-2000:])\n",
        "        if result.stderr:\n",
        "            print('STDERR:', result.stderr[-1000:])\n",
        "    else:\n",
        "        print(f\"Source not found at {SRC}\")\n",
        "        for d in os.listdir('/kaggle/input/'):\n",
        "            print(f\"  {d}\")\n",
        "else:\n",
        "    print('=== DEVELOPMENT MODE (not competition rerun) ===')\n",
        "    print('SmartAgent V3 is prepared.')\n",
        "    print('Submit to competition to run on all games.')\n",
        "    print()\n",
        "    print('Agent components loaded:')\n",
        "    print('  - CNN FrameEncoder + StateTransitionPredictor (PyTorch)')\n",
        "    print('  - StateGraph Explorer (BFS, cycle detection)')\n",
        "    print('  - Monte Carlo Tree Search (UCB1)')\n",
        "    print('  - Temporal Pattern Detector (oscillation, progress)')\n",
        "    print('  - Goal Inference (pattern matching)')\n",
        "    print('  - HypothesisTracker (10 hypotheses + action meanings)')\n",
        "    print('  - SpatialMemory (2D map, player tracking)')\n",
        "    print('  - ObjectTracker (connected components)')\n",
        "    print('  - StuckDetector (6-phase recovery)')\n",
        "    print('  - LevelProgressionManager (per-game knowledge base)')\n",
        "    print('  - SmartClickSelector v3 (6-category priority)')"
    ]
})

# Build the notebook
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": cells
}

output_path = "/home/z/my-project/download/arc_agi3_agent_v3.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to {output_path}")
print(f"Total cells: {len(cells)}")
