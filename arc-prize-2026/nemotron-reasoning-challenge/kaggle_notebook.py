"""
NVIDIA Nemotron Model Reasoning Challenge
Kaggle Competition Notebook
https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

This notebook provides the complete pipeline:
1. Data analysis and puzzle categorization
2. Baseline solver with prompting strategies
3. LoRA fine-tuning
4. Ensemble solver with self-consistency
5. Submission generation
"""

# ============================================================
# CELL 1: Setup and Imports
# ============================================================
# %% [markdown]
# # NVIDIA Nemotron Model Reasoning Challenge
# ## Complete Pipeline: Analysis → Baseline → LoRA → Ensemble

# %%
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')

print("Environment check:")
print(f"  GPU available: {__import__('torch').cuda.is_available()}")
if __import__('torch').cuda.is_available():
    print(f"  GPU: {__import__('torch').cuda.get_device_name(0)}")
    print(f"  VRAM: {__import__('torch').cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Install required packages
# !pip install -q transformers accelerate peft bitsandbytes datasets scikit-learn

# ============================================================
# CELL 2: Data Loading
# ============================================================
# %% [markdown]
# ## 1. Data Loading and Overview

# %%
DATA_DIR = "/kaggle/input/nvidia-nemotron-model-reasoning-challenge"

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

print(f"Training puzzles: {len(train_df)}")
print(f"Test puzzles: {len(test_df)}")
print(f"\nColumns: {train_df.columns.tolist()}")
print(f"\nSample puzzle:")
print(train_df.iloc[0]['prompt'][:500])
print(f"\nSolution: {train_df.iloc[0].get('solution', 'N/A')}")

# ============================================================
# CELL 3: Puzzle Analysis
# ============================================================
# %% [markdown]
# ## 2. Puzzle Category Analysis

# %%
from analyze_puzzles import classify_puzzle, identify_puzzle_patterns, analyze_dataset

# Classify all puzzles
train_categories = [classify_puzzle(p) for p in train_df['prompt']]
test_categories = [classify_puzzle(p) for p in test_df['prompt']]

print("=== Training Set Categories ===")
for cat, count in Counter(train_categories).most_common():
    print(f"  {cat:<25}: {count:>4} ({count/len(train_categories)*100:.1f}%)")

print("\n=== Test Set Categories ===")
for cat, count in Counter(test_categories).most_common():
    print(f"  {cat:<25}: {count:>4} ({count/len(test_categories)*100:.1f}%)")

# Structure patterns
print("\n=== Puzzle Structure Patterns ===")
patterns = identify_puzzle_patterns(train_df)
for k, v in patterns.items():
    print(f"  {k:<30}: {v}")

# Save analysis
train_df['_category'] = train_categories
test_df['_category'] = test_categories
train_df.to_csv("train_with_categories.csv", index=False)
print("\nAnalysis saved.")

# ============================================================
# CELL 4: Baseline Solver - Zero Shot
# ============================================================
# %% [markdown]
# ## 3. Baseline Solver (Zero-Shot Prompting)

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from baseline_solver import (
    get_prompt, extract_answer, self_consistency_vote, 
    clean_answer, NemotronSolver
)

MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("Model loaded!")

# Create pipeline
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    do_sample=True,
    top_p=0.9,
)

# Test on a few training examples
print("\n=== Testing on Training Examples ===")
for i in range(min(3, len(train_df))):
    prompt = train_df.iloc[i]['prompt']
    solution = train_df.iloc[i].get('solution', 'N/A')
    category = train_categories[i]
    
    formatted = get_prompt(prompt, category, "category")
    result = gen_pipeline(formatted, max_new_tokens=256)
    response = result[0]['generated_text'][len(formatted):].strip()
    answer = extract_answer(response)
    
    print(f"\nPuzzle {i+1} ({category}):")
    print(f"  Expected: {solution[:100]}")
    print(f"  Got:      {answer[:100]}")
    print(f"  Match:    {answer.strip().lower() == str(solution).strip().lower()}")

# ============================================================
# CELL 5: Self-Consistency Baseline
# ============================================================
# %% [markdown]
# ## 4. Self-Consistency Decoding

# %%
K_SAMPLES = 8
TEMPERATURE = 0.3

def solve_with_consistency(prompt, category, k=K_SAMPLES, temp=TEMPERATURE):
    """Solve puzzle with self-consistency voting."""
    formatted = get_prompt(prompt, category, "category")
    
    answers = []
    for _ in range(k):
        result = gen_pipeline(
            formatted, 
            max_new_tokens=256,
            temperature=temp,
            do_sample=True,
            top_p=0.9,
        )
        response = result[0]['generated_text'][len(formatted):].strip()
        answer = extract_answer(response)
        answers.append(answer)
    
    best, votes = self_consistency_vote(answers)
    return best, votes, answers

# Evaluate on training set (subset)
print("=== Self-Consistency Evaluation (Training Subset) ===")
eval_size = min(50, len(train_df))
correct = 0

for i in range(eval_size):
    prompt = train_df.iloc[i]['prompt']
    solution = str(train_df.iloc[i].get('solution', ''))
    category = train_categories[i]
    
    best, votes, _ = solve_with_consistency(prompt, category)
    is_correct = clean_answer(best).lower() == clean_answer(solution).lower()
    
    if is_correct:
        correct += 1
    
    if i < 5:
        print(f"\n  Puzzle {i+1}: {'CORRECT' if is_correct else 'WRONG'}")
        print(f"    Expected: {solution[:80]}")
        print(f"    Got:      {best[:80]}")
        print(f"    Votes:    {dict(votes.most_common(3))}")

accuracy = correct / eval_size * 100
print(f"\nBaseline accuracy (train subset, n={eval_size}): {accuracy:.1f}%")

# ============================================================
# CELL 6: LoRA Fine-Tuning
# ============================================================
# %% [markdown]
# ## 5. LoRA Fine-Tuning

# %%
from lora_train import prepare_training_data, train_lora

# Prepare data
print("Preparing training data...")
train_file, val_file = prepare_training_data(
    f"{DATA_DIR}/train.csv", 
    "lora_data",
    val_split=0.1,
)

# Train LoRA
print("\nStarting LoRA fine-tuning...")
model_lora, tokenizer_lora, trainer = train_lora(
    model_name=MODEL_NAME,
    train_file=train_file,
    val_file=val_file,
    output_dir="checkpoints/lora_v1",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_seq_length=2048,
    use_4bit=True,
)

print("LoRA training complete!")

# ============================================================
# CELL 7: Ensemble Solver + Submission
# ============================================================
# %% [markdown]
# ## 6. Ensemble Solver → Submission

# %%
from ensemble_solver import EnsembleSolver, PostProcessor

solver = EnsembleSolver(
    model_name=MODEL_NAME,
    lora_path="checkpoints/lora_v1/final",
    temperature=0.3,
    num_samples_per_strategy=4,
    max_new_tokens=512,
)
solver.load_model()

# Generate submission
submission = solver.solve_dataset(
    test_df,
    output_path="submission.csv",
)

print(f"\nSubmission shape: {submission.shape}")
print(submission.head(10))

# ============================================================
# CELL 8: Temperature Sweep (Optional)
# ============================================================
# %% [markdown]
# ## 7. Temperature Sweep (Optional Optimization)

# %%
# from ensemble_solver import temperature_sweep
# best_temp = temperature_sweep(
#     test_df, 
#     MODEL_NAME,
#     temperatures=[0.1, 0.2, 0.3, 0.5, 0.7],
#     output_dir="sweep_results"
# )
# print(f"Best temperature: {best_temp}")

# Re-run with best temperature if needed

print("\n=== Pipeline Complete ===")
print("Submission saved to: submission.csv")
