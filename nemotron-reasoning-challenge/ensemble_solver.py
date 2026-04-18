"""
NVIDIA Nemotron Model Reasoning Challenge - Ensemble Solver
Combines multiple strategies: LoRA model + prompting + post-processing.

This is the main submission pipeline that:
1. Classifies puzzles by category
2. Runs LoRA fine-tuned model + zero-shot prompts in parallel
3. Applies self-consistency voting across all outputs
4. Post-processes with category-specific heuristics
5. Generates submission.csv
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import time


# ============================================================
# Import from baseline_solver
# ============================================================

from baseline_solver import (
    classify_puzzle, get_prompt, extract_answer, 
    self_consistency_vote, clean_answer, NemotronSolver,
    CATEGORY_KEYWORDS, CATEGORIES
)


# ============================================================
# Category-Specific Post-Processors
# ============================================================

class PostProcessor:
    """Post-processing rules for different puzzle categories."""
    
    def __init__(self):
        self.rules = {
            "pattern_completion": self._pattern_completion_rules,
            "sequence_prediction": self._sequence_rules,
            "spatial_reasoning": self._spatial_rules,
            "constraint_satisfaction": self._constraint_rules,
        }
    
    def process(self, answer: str, category: str, prompt: str) -> str:
        """Apply category-specific post-processing."""
        processor = self.rules.get(category)
        if processor:
            return processor(answer, prompt)
        return answer
    
    def _pattern_completion_rules(self, answer: str, prompt: str) -> str:
        """Rules for pattern completion puzzles."""
        # If answer contains "option" or letter choices, extract the letter/number
        match = re.search(r'(?:option\s+)?([A-Z])[\.\)]', answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # If answer is a number sequence, extract the final number
        numbers = re.findall(r'\b\d+\b', answer)
        if numbers:
            return numbers[-1]
        
        return answer
    
    def _sequence_rules(self, answer: str, prompt: str) -> str:
        """Rules for sequence prediction."""
        # Extract the last number
        numbers = re.findall(r'\b\d+\b', answer)
        if numbers:
            return numbers[-1]
        
        # Extract last element from a list/comma-separated sequence
        if ',' in answer:
            parts = answer.split(',')
            return parts[-1].strip()
        
        return answer
    
    def _spatial_rules(self, answer: str, prompt: str) -> str:
        """Rules for spatial reasoning puzzles."""
        # Keep grid/matrix format if present
        if '[' in answer or '(' in answer:
            # Try to extract the matrix/grid
            grid_match = re.search(r'(\[.*?\])', answer, re.DOTALL)
            if grid_match:
                return grid_match.group(1)
        
        return answer
    
    def _constraint_rules(self, answer: str, prompt: str) -> str:
        """Rules for constraint satisfaction."""
        # Extract final value (often a number)
        numbers = re.findall(r'\b\d+\b', answer)
        if numbers:
            return numbers[-1]
        
        return answer


# ============================================================
# Multi-Strategy Ensemble
# ============================================================

class EnsembleSolver:
    """Ensemble solver combining LoRA + multiple prompting strategies."""
    
    def __init__(
        self,
        model_name: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        lora_path: Optional[str] = None,
        temperature: float = 0.3,
        num_samples_per_strategy: int = 4,
        prompt_styles: Optional[List[str]] = None,
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self.lora_path = lora_path
        self.temperature = temperature
        self.num_samples = num_samples_per_strategy
        self.prompt_styles = prompt_styles or [
            "category", "few_shot_cot", "self_verify", "zero_shot"
        ]
        self.max_new_tokens = max_new_tokens
        self.solver = None
        self.post_processor = PostProcessor()
    
    def load_model(self):
        """Load the model (with optional LoRA adapter)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        effective_model = self.model_name
        
        if self.lora_path and os.path.exists(self.lora_path):
            from peft import PeftModel
            print(f"Loading base model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"Loading LoRA adapter: {self.lora_path}")
            model = PeftModel.from_pretrained(base_model, self.lora_path)
            model = model.merge_and_unload()
            effective_model = model
        else:
            print(f"Loading model: {effective_model}")
            tokenizer = AutoTokenizer.from_pretrained(
                effective_model, trust_remote_code=True
            )
            effective_model = AutoModelForCausalLM.from_pretrained(
                effective_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        self.solver = NemotronSolver(
            model_name=self.model_name,
            temperature=self.temperature,
            num_samples=1,  # We handle sampling in ensemble
            max_new_tokens=self.max_new_tokens,
        )
        self.solver.tokenizer = tokenizer
        self.solver.model = effective_model
        
        self.pipeline = pipeline(
            "text-generation",
            model=effective_model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
        )
        
        print("Ensemble solver loaded!")
    
    def solve_puzzle(self, puzzle_prompt: str, puzzle_id: str = "") -> Dict:
        """Solve a single puzzle using all strategies + ensemble voting."""
        category = classify_puzzle(puzzle_prompt)
        
        all_answers = []
        strategy_results = {}
        
        for style in self.prompt_styles:
            formatted_prompt = get_prompt(puzzle_prompt, category, style)
            style_answers = []
            
            for _ in range(self.num_samples):
                try:
                    result = self.pipeline(formatted_prompt, max_new_tokens=self.max_new_tokens)
                    response = result[0]['generated_text'][len(formatted_prompt):].strip()
                    answer = extract_answer(response)
                    answer = self.post_processor.process(answer, category, puzzle_prompt)
                    style_answers.append(answer)
                except Exception as e:
                    print(f"  Error with style={style}: {e}")
            
            strategy_results[style] = style_answers
            all_answers.extend(style_answers)
        
        # Global majority vote across ALL strategies
        best_answer, global_votes = self_consistency_vote(all_answers)
        
        # Also get per-strategy consensus
        per_strategy = {}
        for style, answers in strategy_results.items():
            if answers:
                sa, _ = self_consistency_vote(answers)
                per_strategy[style] = sa
            else:
                per_strategy[style] = ""
        
        # Check if all strategies agree (high confidence)
        unique_answers = set(per_strategy.values())
        all_agree = len(unique_answers) == 1
        
        confidence = global_votes.most_common(1)[0][1] / max(len(all_answers), 1)
        if all_agree and len(unique_answers) > 0:
            confidence = min(confidence + 0.2, 1.0)
        
        return {
            "puzzle_id": puzzle_id,
            "solution": clean_answer(best_answer),
            "category": category,
            "confidence": round(confidence, 3),
            "all_strategies_agree": all_agree,
            "per_strategy": per_strategy,
            "global_votes": dict(global_votes.most_common(5)),
        }
    
    def solve_dataset(
        self, 
        test_df: pd.DataFrame, 
        prompt_col: str = "prompt",
        id_col: str = "puzzle_id",
        output_path: str = "submission.csv",
    ) -> pd.DataFrame:
        """Solve all puzzles and generate submission."""
        results = []
        start_time = time.time()
        
        for idx, row in test_df.iterrows():
            puzzle_id = row[id_col]
            prompt = row[prompt_col]
            
            elapsed = time.time() - start_time
            print(f"[{elapsed:.0f}s] Puzzle {puzzle_id} ({idx+1}/{len(test_df)})...")
            
            result = self.solve_puzzle(prompt, puzzle_id)
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                avg_conf = np.mean([r["confidence"] for r in results])
                print(f"  Avg confidence so far: {avg_conf:.3f}")
        
        # Build submission DataFrame
        submission = pd.DataFrame(results)[[id_col, "solution"]]
        submission.to_csv(output_path, index=False)
        
        # Stats
        total_time = time.time() - start_time
        print(f"\n=== Submission Complete ===")
        print(f"Total time: {total_time:.0f}s ({total_time/len(test_df):.1f}s/puzzle)")
        print(f"Avg confidence: {np.mean([r['confidence'] for r in results]):.3f}")
        print(f"All strategies agree: {sum(1 for r in results if r['all_strategies_agree'])}/{len(results)}")
        
        # Category breakdown
        cat_stats = {}
        for r in results:
            cat = r["category"]
            if cat not in cat_stats:
                cat_stats[cat] = []
            cat_stats[cat].append(r["confidence"])
        
        print(f"\nCategory breakdown:")
        for cat, confs in sorted(cat_stats.items()):
            print(f"  {cat}: {len(confs)} puzzles, avg confidence={np.mean(confs):.3f}")
        
        return submission


# ============================================================
# Temperature Sweep Runner
# ============================================================

def temperature_sweep(
    test_df: pd.DataFrame,
    model_name: str,
    temperatures: List[float] = [0.1, 0.2, 0.3, 0.5, 0.7],
    num_samples: int = 8,
    output_dir: str = "sweep_results",
):
    """Run solver at multiple temperatures to find optimal setting."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Temperature: {temp}")
        print(f"{'='*60}")
        
        solver = EnsembleSolver(
            model_name=model_name,
            temperature=temp,
            num_samples_per_strategy=2,  # Fewer for speed
            max_new_tokens=512,
        )
        solver.load_model()
        
        # Solve on a subset for speed
        subset = test_df.head(min(50, len(test_df)))
        sub = solver.solve_dataset(subset, output_path=os.path.join(output_dir, f"temp_{temp}.csv"))
        
        avg_conf = np.mean([r["confidence"] for r in solver.post_processor.rules.__dict__.values()] or [0])
        results.append({"temperature": temp, "avg_confidence": avg_conf})
        
        # Free memory
        del solver
        import torch
        torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "sweep_summary.csv"), index=False)
    
    best_temp = results_df.loc[results_df['avg_confidence'].idxmax(), 'temperature']
    print(f"\nBest temperature: {best_temp}")
    
    return best_temp


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ensemble Solver for Nemotron Challenge")
    parser.add_argument("--model", type=str, default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="/kaggle/input")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--mode", type=str, default="solve",
                        choices=["solve", "sweep", "train_and_solve"])
    args = parser.parse_args()

    # Load test data
    test_path = os.path.join(
        args.data_dir, 
        "nvidia-nemotron-model-reasoning-challenge", 
        "test.csv"
    )
    
    if not os.path.exists(test_path):
        # Try alternative paths
        for alt in ["test.csv", "data/test.csv"]:
            if os.path.exists(alt):
                test_path = alt
                break
    
    test_df = pd.read_csv(test_path)
    print(f"Test data: {len(test_df)} puzzles")

    if args.mode == "solve":
        solver = EnsembleSolver(
            model_name=args.model,
            lora_path=args.lora_path,
            temperature=args.temperature,
            num_samples_per_strategy=args.num_samples,
            max_new_tokens=args.max_tokens,
        )
        solver.load_model()
        solver.solve_dataset(test_df, output_path=args.output)

    elif args.mode == "sweep":
        temperature_sweep(test_df, args.model)

    elif args.mode == "train_and_solve":
        from lora_train import prepare_training_data, train_lora
        
        # Step 1: Prepare data
        train_path = os.path.join(
            args.data_dir,
            "nvidia-nemotron-model-reasoning-challenge",
            "train.csv"
        )
        train_file, val_file = prepare_training_data(train_path, "data")
        
        # Step 2: Train LoRA
        checkpoint_dir = "checkpoints/lora_v1"
        model, tokenizer, trainer = train_lora(
            model_name=args.model,
            train_file=train_file,
            val_file=val_file,
            output_dir=checkpoint_dir,
            lora_r=args.lora_r,
        )
        
        # Step 3: Solve with LoRA
        lora_path = os.path.join(checkpoint_dir, "final")
        solver = EnsembleSolver(
            model_name=args.model,
            lora_path=lora_path,
            temperature=args.temperature,
            num_samples_per_strategy=args.num_samples,
            max_new_tokens=args.max_tokens,
        )
        solver.load_model()
        solver.solve_dataset(test_df, output_path=args.output)


if __name__ == "__main__":
    main()
