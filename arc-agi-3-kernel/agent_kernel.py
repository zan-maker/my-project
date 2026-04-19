#!/usr/bin/env python3
"""
AnnotateX ARC-AGI-3 Agent — Hybrid Exploration & Pattern Recognition Agent
Uses systematic exploration with heuristics for interactive game environments.
"""

import pandas as pd, os

# Create submission.parquet placeholder
_df = pd.DataFrame([{'row_id':'1_0','game_id':'1','end_of_game':True,'score':1.0}])
_df.to_parquet('/kaggle/working/submission.parquet', index=False)
print('submission.parquet created')

# ============================================================
# AGENT CODE (writes to /kaggle/working/my_agent.py)
# ============================================================

AGENT_CODE = r'''
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
'''

# Write agent code
with open('/kaggle/working/my_agent.py', 'w') as f:
    f.write(AGENT_CODE)
print("Agent code written to /kaggle/working/my_agent.py")

# ============================================================
# COMPETITION MODE: Run the agent
# ============================================================

import subprocess

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print('=== COMPETITION MODE ===')
    
    # Wait for gateway
    for i in range(60):
        r = subprocess.run(['curl', '-sf', 'http://gateway:8001/api/games'], 
                          capture_output=True, timeout=10)
        if r.returncode == 0:
            print('Gateway ready')
            break
        time.sleep(5)
    
    # Setup agent
    WORK = '/kaggle/working/ARC-AGI-3-Agents'
    SRC = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents'
    
    if os.path.exists(SRC):
        subprocess.run(['cp', '-r', SRC, WORK], check=True)
        subprocess.run(['cp', '/kaggle/working/my_agent.py', 
                       f'{WORK}/agents/templates/annotatex_agent.py'], check=True)
        
        # Update __init__.py
        init_code = '''from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.annotatex_agent import AnnotateXAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "random": Random,
    "annotatex": AnnotateXAgent,
}
'''
        with open(f'{WORK}/agents/__init__.py', 'w') as f:
            f.write(init_code)
        
        # Setup environment
        env_content = '''SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=***
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
RECORDINGS_DIR=/kaggle/working/server_recording
'''
        with open(f'{WORK}/.env', 'w') as f:
            f.write(env_content)
        
        os.chdir(WORK)
        os.environ['MPLBACKEND'] = 'agg'
        
        # Run agent
        print('Starting AnnotateX agent...')
        result = subprocess.run(
            ['python', 'main.py', '--agent', 'annotatex'],
            capture_output=True, text=True, timeout=3600
        )
        
        print(f'Exit code: {result.returncode}')
        if result.stdout:
            print('STDOUT:', result.stdout[-2000:])
        if result.stderr:
            print('STDERR:', result.stderr[-1000:])
    else:
        print(f"Source not found at {SRC}")
        # List available dirs
        for d in os.listdir('/kaggle/input/'):
            print(f"  {d}")
else:
    print('=== DEVELOPMENT MODE (not competition rerun) ===')
    print('Agent code prepared. Will run when submitted to competition.')
