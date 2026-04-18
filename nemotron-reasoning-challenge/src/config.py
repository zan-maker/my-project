"""
Central configuration for the Nemotron Reasoning Challenge toolkit.
All hyperparameters, paths, and defaults live here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submissions")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


# ---------------------------------------------------------------------------
# Puzzle categories (6-7 known categories from competition)
# ---------------------------------------------------------------------------
PUZZLE_CATEGORIES = [
    "sequence_completion",     # Complete a numeric / alphabetic sequence
    "pattern_transformation",  # Apply a transformation rule to input
    "analogy_reasoning",       # A:B :: C:?  style analogies
    "logic_grid",              # Constraint-satisfaction / grid puzzles
    "causal_inference",        # Cause-effect chain puzzles
    "spatial_reasoning",       # 2-D grid / shape manipulation
    "math_word_problem",       # Arithmetic / algebra word problems
]

CATEGORY_ALIASES: Dict[str, str] = {
    "sequence": "sequence_completion",
    "number sequence": "sequence_completion",
    "letter sequence": "sequence_completion",
    "pattern": "pattern_transformation",
    "transformation": "pattern_transformation",
    "transform": "pattern_transformation",
    "analogy": "analogy_reasoning",
    "a is to b": "analogy_reasoning",
    "grid": "logic_grid",
    "constraint": "logic_grid",
    "sudoku": "logic_grid",
    "causal": "causal_inference",
    "cause": "causal_inference",
    "effect": "causal_inference",
    "spatial": "spatial_reasoning",
    "shape": "spatial_reasoning",
    "rotation": "spatial_reasoning",
    "mirror": "spatial_reasoning",
    "math": "math_word_problem",
    "arithmetic": "math_word_problem",
    "algebra": "math_word_problem",
    "calculation": "math_word_problem",
    "word problem": "math_word_problem",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PuzzleExample:
    """Single input-output example pair within a puzzle."""
    input_str: str
    output_str: str


@dataclass
class Puzzle:
    """A single puzzle instance."""
    puzzle_id: str
    prompt: str                           # Full prompt text with examples
    solution: Optional[str] = None        # Ground-truth (train only)
    examples: List[PuzzleExample] = field(default_factory=list)
    category: Optional[str] = None        # Detected / assigned category
    difficulty: Optional[str] = None      # easy / medium / hard


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------
@dataclass
class SolverConfig:
    """Configuration for the baseline / ensemble solver."""
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    use_quantization: bool = True
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    device_map: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = False              # greedy by default
    num_beam_groups: int = 1
    num_beams: int = 1


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble solver."""
    base_configs: List[SolverConfig] = field(default_factory=list)
    num_samples: int = 15                # K for self-consistency
    temperatures: List[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5, 0.7]
    )
    voting_strategy: str = "majority"    # majority / weighted / rank
    weight_by_temperature: bool = True   # lower temp -> higher weight


@dataclass
class LoRAConfig:
    """PEFT / LoRA hyper-parameters."""
    base_model_name: str = "nvidia/Nemotron-3-Nano-30B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    use_category_adapters: bool = False
    output_dir: str = os.path.join(CHECKPOINT_DIR, "lora")
    logging_dir: str = os.path.join(LOG_DIR, "lora")
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 3
    seed: int = 42


@dataclass
class PromptConfig:
    """Controls how prompts are constructed."""
    style: str = "chain_of_thought"
    include_category_hint: bool = True
    max_examples_in_prompt: int = 3
    system_message: str = (
        "You are an expert logical reasoning assistant. "
        "Analyze the given puzzle carefully, identify the underlying rule or pattern, "
        "and provide the correct answer. Think step-by-step."
    )


PROMPT_STYLES = ["zero_shot", "few_shot", "chain_of_thought"]
