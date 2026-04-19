"""
NVIDIA Nemotron Model Reasoning Challenge - Baseline Solver
Kaggle: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

Strategy:
1. Classify puzzle into one of 6-7 categories
2. Apply category-specific prompt template
3. Generate K samples with self-consistency decoding
4. Majority vote for final answer
"""

import os
import re
import json
import time
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# ============================================================
# Puzzle Category Classifier
# ============================================================

CATEGORIES = [
    "pattern_completion",      # Fill missing element in a pattern
    "rule_application",        # Apply transformation rule to input
    "sequence_prediction",     # Predict next in sequence
    "analogy_solving",         # Solve A:B :: C:D analogy
    "constraint_satisfaction", # Find solution satisfying constraints
    "spatial_reasoning",       # Grid/shape transformations
    "logic_deduction",         # Pure logic puzzles
]

CATEGORY_KEYWORDS = {
    "pattern_completion": ["pattern", "complete", "fill in", "missing", "sequence", "next"],
    "rule_application": ["transform", "apply", "convert", "map", "output", "function"],
    "sequence_prediction": ["sequence", "next", "follows", "series", "continues"],
    "analogy_solving": ["analogy", "similar", "same way", "relationship", "parallel"],
    "constraint_satisfaction": ["constraint", "satisfy", "valid", "allowed", "rules"],
    "spatial_reasoning": ["grid", "move", "rotate", "flip", "shape", "position", "row", "column", "matrix"],
    "logic_deduction": ["logic", "deduce", "infer", "therefore", "because", "given that"],
}


def classify_puzzle(prompt: str) -> str:
    """Classify puzzle into category based on prompt keywords."""
    prompt_lower = prompt.lower()
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        scores[category] = score

    if max(scores.values()) == 0:
        return "logic_deduction"  # Default

    return max(scores, key=scores.get)


# ============================================================
# Prompt Templates (Category-Specific)
# ============================================================

ZERO_SHOT_TEMPLATE = """You are an expert at solving logical reasoning puzzles. You must carefully analyze the given input-output examples to identify the underlying rule or pattern, then apply it to solve the new problem.

{prompt}

Think step by step. First identify the pattern or rule, then apply it carefully. Provide only the final answer in the format shown in the examples."""

FEW_SHOT_COT_TEMPLATE = """You are an expert at solving logical reasoning puzzles. You must carefully analyze the given input-output examples to identify the underlying rule or pattern, then apply it to solve the new problem.

Example approach:
1. Read all input-output pairs carefully
2. Identify what changes between input and output
3. Look for consistent patterns across all pairs
4. Apply the same pattern to the new problem
5. Verify your answer follows the same rule

{prompt}

Now solve this puzzle step by step:
Step 1: Identify the pattern from the examples above
Step 2: Explain the rule clearly
Step 3: Apply the rule to the new input
Step 4: Give the final answer"""

CHAIN_OF_THOUGHT_TEMPLATE = """Solve this reasoning puzzle carefully.

{prompt}

Let me solve this step by step:
1) First, I'll examine each input-output pair to understand the transformation:
"""

SELF_VERIFY_TEMPLATE = """You are a reasoning puzzle solver. Solve the puzzle and verify your answer.

{prompt}

First, identify the rule:
Then apply the rule to get the answer:
Finally, verify: check that applying the same rule to all examples produces the correct outputs.
Answer:"""

CATEGORY_PROMPTS = {
    "pattern_completion": """This is a pattern completion puzzle. Look at the examples carefully to find the repeating pattern or rule that determines the next element.

{prompt}

Identify the pattern, then determine the missing/next element. Show your reasoning step by step.""",

    "rule_application": """This is a rule application puzzle. Study the input-output pairs to discover the transformation rule, then apply it precisely.

{prompt}

1. List each transformation from input to output
2. Find the common rule across all pairs
3. Apply the rule to the test input
4. Final answer:""",

    "sequence_prediction": """This is a sequence prediction puzzle. Analyze the sequence to find what comes next.

{prompt}

1. Identify the sequence type (arithmetic, geometric, recursive, etc.)
2. Determine the pattern/step
3. Predict the next element(s)
4. Answer:""",

    "analogy_solving": """This is an analogy puzzle. Find the relationship in the first pair and apply it to the second pair.

{prompt}

1. What is the relationship between the first pair's input and output?
2. Apply the same relationship to the new input
3. Answer:""",

    "constraint_satisfaction": """This is a constraint satisfaction puzzle. Find a solution that satisfies all given constraints.

{prompt}

1. List all constraints
2. Check which values satisfy each constraint
3. Find the intersection that satisfies ALL constraints
4. Answer:""",

    "spatial_reasoning": """This is a spatial reasoning puzzle. Analyze the grid/shape transformations carefully.

{prompt}

1. Track positions and changes across examples
2. Identify the spatial transformation rule
3. Apply to the test case
4. Answer:""",

    "logic_deduction": """This is a logic deduction puzzle. Use deductive reasoning to find the answer.

{prompt}

1. List all given facts
2. Apply logical deduction step by step
3. Eliminate impossibilities
4. Conclude the answer:""",
}


def get_prompt(prompt_text: str, category: str, style: str = "category") -> str:
    """Get formatted prompt based on category and style."""
    templates = {
        "zero_shot": ZERO_SHOT_TEMPLATE,
        "few_shot_cot": FEW_SHOT_COT_TEMPLATE,
        "cot": CHAIN_OF_THOUGHT_TEMPLATE,
        "self_verify": SELF_VERIFY_TEMPLATE,
        "category": CATEGORY_PROMPTS.get(category, ZERO_SHOT_TEMPLATE),
    }
    template = templates.get(style, ZERO_SHOT_TEMPLATE)
    return template.format(prompt=prompt_text)


# ============================================================
# Self-Consistency Decoder
# ============================================================

def extract_answer(response: str) -> str:
    """Extract the final answer from model response."""
    # Try to find answer after common markers
    markers = ["Answer:", "answer:", "Final answer:", "final answer:", 
               "Result:", "result:", "Output:", "output:",
               "Therefore:", "The answer is", "the answer is"]
    
    response_stripped = response.strip()
    
    for marker in markers:
        idx = response_stripped.rfind(marker)
        if idx >= 0:
            answer = response_stripped[idx + len(marker):].strip()
            # Clean up
            answer = re.sub(r'[.\n].*$', '', answer).strip()
            if answer:
                return answer
    
    # Fallback: return last line or last meaningful token
    lines = response_stripped.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith(('Step', 'The', 'Because', 'Looking', 'I ', 'We ', 'This')):
            return line
    
    return response_stripped.split('\n')[-1].strip()


def self_consistency_vote(responses: List[str]) -> Tuple[str, Counter]:
    """Majority vote across multiple responses."""
    answers = [extract_answer(r) for r in responses]
    counter = Counter(answers)
    best_answer = counter.most_common(1)[0][0]
    return best_answer, counter


# ============================================================
# Baseline Solver (works with any LLM)
# ============================================================

class NemotronSolver:
    """Baseline solver for Nemotron reasoning puzzles.
    
    Can work with:
    - Nemotron-3-Nano-30B via HuggingFace
    - OpenAI-compatible API endpoints
    - Local models via transformers pipeline
    """

    def __init__(self, model_name: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                 temperature: float = 0.3, num_samples: int = 8,
                 prompt_style: str = "category", max_new_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.num_samples = num_samples
        self.prompt_style = prompt_style
        self.max_new_tokens = max_new_tokens
        self.pipeline = None
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load the Nemotron model. For Kaggle, use the provided GPU."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        
        # Use device_map="auto" for multi-GPU or single GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
        )
        print("Model loaded successfully!")

    def generate_response(self, prompt: str) -> str:
        """Generate a single response from the model."""
        if self.pipeline:
            result = self.pipeline(prompt, max_new_tokens=self.max_new_tokens)
            return result[0]['generated_text'][len(prompt):].strip()
        return ""

    def solve(self, puzzle_prompt: str, category: Optional[str] = None) -> Dict:
        """Solve a single puzzle with self-consistency."""
        if category is None:
            category = classify_puzzle(puzzle_prompt)

        formatted_prompt = get_prompt(puzzle_prompt, category, self.prompt_style)

        responses = []
        for i in range(self.num_samples):
            response = self.generate_response(formatted_prompt)
            responses.append(response)

        best_answer, vote_counts = self_consistency_vote(responses)

        return {
            "category": category,
            "answer": best_answer,
            "confidence": vote_counts.most_common(1)[0][1] / len(responses),
            "vote_distribution": dict(vote_counts.most_common()),
            "num_samples": len(responses),
            "all_answers": [extract_answer(r) for r in responses],
        }

    def solve_dataset(self, df: pd.DataFrame, prompt_col: str = "prompt",
                      id_col: str = "puzzle_id") -> pd.DataFrame:
        """Solve all puzzles in a dataset."""
        results = []
        for idx, row in df.iterrows():
            puzzle_id = row[id_col]
            prompt = row[prompt_col]
            print(f"Solving puzzle {puzzle_id} ({idx+1}/{len(df)})...")
            
            result = self.solve(prompt)
            results.append({
                id_col: puzzle_id,
                "solution": result["answer"],
                "category": result["category"],
                "confidence": result["confidence"],
            })
        
        return pd.DataFrame(results)


# ============================================================
# Post-Processing Utilities
# ============================================================

def clean_answer(answer: str) -> str:
    """Clean and normalize answers."""
    # Remove common prefixes
    for prefix in ["Answer:", "answer:", "The answer is", "Output:", "Result:"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove trailing punctuation
    answer = answer.rstrip('.,;:!')
    
    # Normalize whitespace
    answer = ' '.join(answer.split())
    
    return answer


def validate_answer(answer: str, examples: list) -> bool:
    """Basic answer validation against examples format."""
    if not answer:
        return False
    
    # Check length is reasonable
    if len(answer) > 1000:
        return False
    
    return True


# ============================================================
# Main Execution
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Nemotron Reasoning Challenge Solver")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input",
                        help="Directory containing competition data")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output submission file")
    parser.add_argument("--model", type=str, 
                        default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                        help="Model name or path")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--prompt_style", type=str, default="category",
                        choices=["zero_shot", "few_shot_cot", "cot", "self_verify", "category"])
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--analysis_only", action="store_true",
                        help="Only analyze data, don't solve")
    args = parser.parse_args()

    # Load data
    train_path = os.path.join(args.data_dir, "nvidia-nemotron-model-reasoning-challenge", "train.csv")
    test_path = os.path.join(args.data_dir, "nvidia-nemotron-model-reasoning-challenge", "test.csv")

    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        print(f"Training data: {len(train_df)} puzzles")
    else:
        train_df = None
        print("No training data found, will use test data only")

    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"Test data: {len(test_df)} puzzles")
    else:
        print("No test data found!")
        return

    if args.analysis_only:
        # Analysis mode
        print("\n=== Data Analysis ===")
        if train_df is not None:
            print(f"Columns: {train_df.columns.tolist()}")
            print(f"Train shape: {train_df.shape}")
            print(f"Sample prompt:\n{train_df.iloc[0]['prompt'][:500]}")
            print(f"\nSample solution: {train_df.iloc[0].get('solution', 'N/A')}")
            
            # Classify all training puzzles
            if 'prompt' in train_df.columns:
                categories = [classify_puzzle(p) for p in train_df['prompt']]
                cat_counts = Counter(categories)
                print(f"\nCategory distribution:")
                for cat, count in cat_counts.most_common():
                    print(f"  {cat}: {count} ({count/len(categories)*100:.1f}%)")
        return

    # Solve mode
    solver = NemotronSolver(
        model_name=args.model,
        temperature=args.temperature,
        num_samples=args.num_samples,
        prompt_style=args.prompt_style,
        max_new_tokens=args.max_tokens,
    )
    
    solver.load_model()
    
    # Generate submissions
    print("\n=== Solving Test Puzzles ===")
    submission_df = solver.solve_dataset(test_df)
    
    # Post-process
    submission_df['solution'] = submission_df['solution'].apply(clean_answer)
    
    # Save
    submission_df.to_csv(args.output, index=False)
    print(f"\nSubmission saved to {args.output}")
    print(f"Total puzzles solved: {len(submission_df)}")


if __name__ == "__main__":
    main()
