# ARC-AGI-3 Competition Analysis

## 1. Competition Overview

**ARC Prize 2026 - ARC-AGI-3** is the third iteration of the ARC (Abstraction and Reasoning Corpus) competition, now focused on **interactive reasoning** rather than static puzzle-solving.

Key tagline: *"Create an AI capable of fluid intelligence"*

ARC-AGI-3 is described as **the first interactive reasoning benchmark designed to measure human-like intelligence in AI agents**. Instead of solving static puzzles (like ARC-AGI-1/2), agents must:

1. **Explore novel environments** - perceive what matters
2. **Acquire goals on the fly** - no pre-loaded knowledge or hidden prompts
3. **Build adaptable world models** - learn from experience
4. **Learn continuously** - adapt strategy based on feedback

### What is being measured:
- 100% human-solvable environments
- Skill-acquisition efficiency over time
- Long-horizon planning with sparse feedback
- Experience-driven adaptation across multiple steps

### How intelligence is measured:
> "As long as there is a gap between AI and human learning, we do not have AGI. ARC-AGI-3 makes that gap measurable by testing intelligence across time, not just final answers—capturing planning horizons, memory compression, and the ability to update beliefs as new evidence appears."

### Design Principles:
- Easy for humans to pick up quickly
- No pre-loaded knowledge or hidden prompts
- Clear goals + meaningful feedback
- Novelty that prevents brute-force memorization

---

## 2. Data Format and Structure

### Competition Data Files

The competition data is organized into three main directories:

```
arc-prize-2026-arc-agi-3/
├── ARC-AGI-3-Agents/          # Official agent framework (Python)
│   ├── main.py                 # Entry point for running agents
│   ├── agents/
│   │   ├── agent.py            # Base Agent class (ABC)
│   │   ├── swarm.py            # Multi-game orchestration
│   │   ├── recorder.py         # JSONL gameplay recording
│   │   ├── tracing.py          # AgentOps integration
│   │   ├── __init__.py         # Registers all agent templates
│   │   └── templates/          # Pre-built agent implementations
│   │       ├── random_agent.py
│   │       ├── llm_agents.py
│   │       ├── multimodal.py
│   │       ├── reasoning_agent.py
│   │       ├── smolagents.py
│   │       ├── langgraph_random_agent.py
│   │       ├── langgraph_functional_agent.py
│   │       └── langgraph_thinking/
│   ├── tests/                  # Unit tests (pytest)
│   ├── pyproject.toml          # Dependencies
│   └── llms.txt                # Documentation for LLMs
├── arc_agi_3_wheels/           # Pre-built Python wheels for Kaggle
│   ├── arcengine-0.9.3         # Core game engine
│   ├── arc_agi-0.9.8           # ARC API client library
│   ├── numpy, pillow, matplotlib, etc.
└── environment_files/          # Game environment definitions
    └── ar25/0c556536/
        ├── ar25.py             # Game implementation (sprites, levels, logic)
        └── metadata.json
```

### Game List (23 games confirmed)
`ar25`, `bp35`, `cd82`, `cn04`, `dc22`, `ft09`, `g50t`, `ka59`, `lf52`, `lp85`, `ls20`, `m0r0`, `r11l`, `re86`, `s5i5`, `sb26`, `sc25`, `sk48`, `sp80`, `su15`, `tn36`, `tr87`, `tu93`, `vc33`, `wa30`

---

## 3. Environment API Details

### Core Data Structures

#### FrameData (Observation)
```python
class FrameData:
    game_id: str                    # Game identifier
    frame: list[list[list[int]]]    # 3D grid: [layer, row, col] with INT<0,15>
    state: GameState                # NOT_PLAYED | NOT_FINISHED | GAME_OVER | WIN
    levels_completed: int           # Number of levels beaten
    win_levels: int                 # Total levels to win
    guid: str                       # Session GUID
    full_reset: bool                # Whether this frame is a full reset
    available_actions: list[GameAction]  # Actions available in this frame
    action_input: ActionInput       # The action that produced this frame
```

**Grid specs:**
- Each Grid is a matrix of size `INT<0,63>` by `INT<0,63>`
- Filled with `INT<0,15>` values (16-color palette)
- A Frame contains one or more sequential Grids

#### GameAction (Actions)
```python
class GameAction(Enum):
    RESET    = 0   # Start/restart game (simple)
    ACTION1  = 1   # Move Up / Input 1 / W (simple)
    ACTION2  = 2   # Move Down / Input 2 / S (simple)
    ACTION3  = 3   # Move Left / Input 3 / A (simple)
    ACTION4  = 4   # Move Right / Input 4 / D (simple)
    ACTION5  = 5   # Action / Enter / Spacebar / Delete (simple)
    ACTION6  = 6   # Click / Point - requires x,y coordinates (complex)
    ACTION7  = 7   # Undo (simple)
```

- **Simple actions**: No parameters needed
- **Complex actions**: Require `{"x": Int<0,63>, "y": Int<0,63>}`
- `is_simple()` / `is_complex()` methods for classification

#### GameState
```python
class GameState(Enum):
    NOT_PLAYED    # Game hasn't started yet
    NOT_FINISHED  # Game is active
    GAME_OVER     # Game ended (lost)
    WIN           # Game completed (won)
```

### Agent Interface (Base Class)

```python
class Agent(ABC):
    MAX_ACTIONS: int = 80  # Maximum actions before forced exit

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""

    @abstractmethod
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose which action to take."""
```

### Environment Wrapper (`arc_agi.EnvironmentWrapper`)
- `arc_env.step(action, data, reasoning)` - Execute an action and get `FrameDataRaw`
- `arc_env.observation_space` - Current observation
- Uses `arcengine` library (arcengine-0.9.3)

### Arcade (`arc_agi.Arcade`)
- `arcade.make(game_id, scorecard_id)` - Create an environment for a game
- `arcade.open_scorecard(tags)` - Start a scorecard session
- `arcade.close_scorecard(card_id)` - End session and get results
- `arcade.operation_mode` - ONLINE or LOCAL

### API Communication
- REST API at `three.arcprize.org` (or configurable HOST/PORT)
- Authentication via `X-API-Key` header
- Endpoints:
  - `GET /api/games` - List available games
  - `POST /api/scorecard/open` - Open scorecard
  - `POST /api/scorecard/close` - Close scorecard and get results

### Episodes and Scoring
- An **episode** is one agent playing one game
- **Scorecard** tracks results across all games in a session
- `levels_completed` tracks progress through multi-level games
- `win_levels` is the target number of levels to complete
- `EnvironmentScorecard` aggregates results per game

---

## 4. Evaluation Metric

The evaluation metric appears to be based on:
1. **Games won** (`state == GameState.WIN`)
2. **Levels completed** within each game
3. **Action efficiency** (fewer actions = better)

From the scorecard structure:
- `won`: Number of games won
- `played`: Number of games played
- `total_actions`: Sum of all actions taken
- Each `Card` tracks: `scores`, `states`, `actions`, `resets` per play

A game is won when `state == GameState.WIN`, which requires completing all levels (`levels_completed >= win_levels`).

The 100% benchmark is: "AI agents can beat every game as efficiently as humans."

**Submission**: Agents are submitted via a Google Form at https://forms.gle/wMLZrEFGDh33DhzV9

---

## 5. Key Differences from ARC-AGI-2

| Aspect | ARC-AGI-2 | ARC-AGI-3 |
|--------|-----------|-----------|
| **Format** | Static input/output pairs | Interactive real-time environments |
| **Interface** | JSON grids (input → output) | Action-per-step loop with observations |
| **Knowledge** | All rules visible upfront | Rules must be discovered through interaction |
| **Feedback** | Binary correct/incorrect | Continuous state observation + levels completed |
| **Planning** | Single-step reasoning | Multi-step planning with sparse feedback |
| **Novelty** | Visual pattern recognition | Game mechanics discovery |
| **Scoring** | % of test pairs correct | Games won + action efficiency |
| **API** | Submit prediction grids | Connect agent to live environment |
| **Games** | 400 abstract reasoning tasks | 23+ interactive environments |
| **Evaluation** | Private test set | Replay-based evaluation |

### ARC-AGI-3 introduces:
- **Interactive environments** instead of static puzzles
- **Multi-level games** (e.g., ar25 has 8 levels, ls20 has 6 levels)
- **Energy/steps systems** (e.g., StepCounter limiting moves)
- **Sprite-based rendering** with layers, collisions, and visibility
- **Action space** of 7 actions (RESET + 6 gameplay actions)
- **Real-time feedback** through FrameData observations
- **Scorecard system** for tracking multi-game sessions

---

## 6. Strategy Recommendations

### Tier 1: Immediate Baseline (Random → Heuristic)
1. **Start with random exploration** to understand each game's mechanics
2. **Implement state tracking**: remember which actions led to progress
3. **BFS/DFS exploration**: systematically explore the action space
4. **Pattern matching**: detect common game elements (walls, keys, doors, energy)

### Tier 2: LLM-Based Agents
1. **Text-only LLM agent**: Feed grid as text to GPT-4o-mini/o4-mini (already implemented)
2. **Vision LLM agent**: Convert grids to images for multimodal models (MultiModalLLM template)
3. **Reasoning agent**: Use structured output with hypothesis testing (ReasoningAgent template)
4. **Tool-using agents**: smolagents/LangGraph for code-based reasoning

### Tier 3: Advanced Strategies
1. **Game-specific specialization**: Train/adapt for each game type
2. **Meta-learning**: Learn to learn new games quickly from few examples
3. **Memory and planning**: Build world models, maintain belief states
4. **Exploration strategies**: Curiosity-driven exploration, uncertainty reduction
5. **Hierarchical planning**: High-level goals → low-level actions

### Architecture Recommendations:
1. **Use the official ARC-AGI-3 SDK** (arcengine + arc_agi) for environment interaction
2. **Inherit from `Agent` base class** for consistent interface
3. **Implement `choose_action` and `is_done`** methods
4. **Use `Swarm` for multi-game orchestration**
5. **Record sessions** with `Recorder` for replay analysis
6. **Maximize `levels_completed`** - multi-level games are key
7. **Monitor available_actions** - different games enable different actions
8. **Handle RESET properly** - required at start and after GAME_OVER
9. **Track reasoning** in `action.reasoning` for transparency
10. **Optimize action count** - fewer actions = better efficiency score

### Key Insight for Winning:
The evaluation measures **learning efficiency** - agents that can quickly understand new game mechanics and solve them with fewer actions will score higher. This is fundamentally about **meta-learning** and **rapid adaptation**.
