
import hashlib, logging, os, random, time, traceback
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnnotateX")


class AnnotateXAgent(Agent):
    """
    AnnotateX ARC-AGI-3 Agent: Hybrid exploration + pattern recognition.
    
    Strategy:
    1. Systematic exploration: visit every cell, try every action
    2. Track visited states and reward signals
    3. Pattern detection: identify game rules from observations
    4. Goal inference: deduce objectives from reward patterns
    5. Exploit: apply learned rules to maximize score
    """

    def __init__(self):
        super().__init__()
        self.visited_states = set()
        self.state_history = []
        self.reward_history = []
        self.grid_memory = {}
        self.action_log = []
        self.game_rules = {}
        self.exploration_queue = deque()
        self.current_goal = None
        self.step_count = 0
        self.max_steps = 500
        
        # Exploration parameters
        self.exploration_budget = 200
        self.explored_actions = set()
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        self.action_types = ['move', 'interact', 'wait']
        
    def get_action(self, frame: FrameData) -> GameAction:
        """Main decision loop."""
        self.step_count += 1
        
        if self.step_count > self.max_steps:
            return self._end_game()
        
        # Phase 1: Explore (first ~40% of budget)
        if self.step_count <= self.exploration_budget * 0.4:
            return self._explore_systematic(frame)
        
        # Phase 2: Pattern detection + directed exploration (~40%)
        elif self.step_count <= self.exploration_budget * 0.8:
            return self._pattern_directed(frame)
        
        # Phase 3: Exploit learned rules (~20%)
        else:
            return self._exploit(frame)
    
    def _get_state_hash(self, frame: FrameData) -> str:
        """Create a hashable representation of the current game state."""
        try:
            if hasattr(frame, 'grid') and frame.grid is not None:
                grid_str = np.array(frame.grid).tobytes()
                return hashlib.md5(grid_str).hexdigest()
            elif hasattr(frame, 'state'):
                return str(frame.state)
            else:
                return str(frame)
        except:
            return str(id(frame))
    
    def _explore_systematic(self, frame: FrameData) -> GameAction:
        """Systematic exploration: try all directions, interactions."""
        state_hash = self._get_state_hash(frame)
        
        if state_hash not in self.visited_states:
            self.visited_states.add(state_hash)
            self.state_history.append({
                'step': self.step_count,
                'state': state_hash,
                'reward': getattr(frame, 'reward', 0)
            })
        
        # Try movement in all directions first
        for dx, dy in self.directions:
            action_key = f"move_{dx}_{dy}"
            if action_key not in self.explored_actions:
                self.explored_actions.add(action_key)
                return GameAction(action_type="move", dx=dx, dy=dy)
        
        # Try interaction with adjacent cells
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            action_key = f"interact_{dx}_{dy}"
            if action_key not in self.explored_actions:
                self.explored_actions.add(action_key)
                return GameAction(action_type="interact", dx=dx, dy=dy)
        
        # Try wait
        if "wait" not in self.explored_actions:
            self.explored_actions.add("wait")
            return GameAction(action_type="wait")
        
        # Reset exploration for new positions
        self.explored_actions = set()
        return self._random_walk(frame)
    
    def _pattern_directed(self, frame: FrameData) -> GameAction:
        """Use observed patterns to guide exploration."""
        state_hash = self._get_state_hash(frame)
        
        # Track rewards to identify goals
        current_reward = getattr(frame, 'reward', 0)
        if self.reward_history and current_reward != self.reward_history[-1]:
            logger.info(f"Reward change: {self.reward_history[-1]} -> {current_reward} at step {self.step_count}")
        self.reward_history.append(current_reward)
        
        # Detect rewarding patterns
        if len(self.reward_history) > 5:
            # Check if certain actions led to reward increases
            recent_rewards = self.reward_history[-10:]
            if any(r > 0 for r in recent_rewards):
                # Find what action caused the reward
                for i in range(len(recent_rewards) - 1):
                    if recent_rewards[i+1] > recent_rewards[i]:
                        # Replay the action that caused reward
                        if i < len(self.action_log):
                            return self.action_log[i]
        
        # Continue systematic exploration with learned bias
        return self._explore_systematic(frame)
    
    def _exploit(self, frame: FrameData) -> GameAction:
        """Exploit learned patterns to maximize score."""
        current_reward = getattr(frame, 'reward', 0)
        
        # If we found rewarding actions, repeat them
        if self.reward_history:
            max_reward = max(self.reward_history)
            if max_reward > 0:
                # Find the best action sequence
                best_idx = self.reward_history.index(max_reward)
                if best_idx < len(self.action_log):
                    logger.info(f"Exploiting: replaying action at step {best_idx} (reward={max_reward})")
                    return self.action_log[best_idx]
        
        # Default: random walk toward unexplored areas
        return self._random_walk(frame)
    
    def _random_walk(self, frame: FrameData) -> GameAction:
        """Random walk for exploration."""
        state_hash = self._get_state_hash(frame)
        if state_hash not in self.visited_states:
            self.visited_states.add(state_hash)
        
        # Prefer unvisited directions
        unvisited = []
        for dx, dy in self.directions:
            action = GameAction(action_type="move", dx=dx, dy=dy)
            action_key = f"move_{dx}_{dy}_{state_hash}"
            if action_key not in self.explored_actions:
                unvisited.append(action)
                self.explored_actions.add(action_key)
        
        if unvisited:
            action = random.choice(unvisited)
        else:
            dx, dy = random.choice(self.directions)
            action = GameAction(action_type="move", dx=dx, dy=dy)
        
        self.action_log.append(action)
        return action
    
    def _end_game(self) -> GameAction:
        """Signal end of game."""
        logger.info(f"Ending game after {self.step_count} steps")
        try:
            return GameAction(action_type="end")
        except:
            return GameAction(action_type="wait")
