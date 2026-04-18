"""
NVIDIA Nemotron Model Reasoning Challenge - Puzzle Analysis
Exploratory data analysis of the reasoning puzzle dataset.
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple


def analyze_dataset(train_csv: str, test_csv: str = None):
    """Comprehensive analysis of the puzzle dataset."""
    
    train_df = pd.read_csv(train_csv)
    print("=" * 70)
    print("NVIDIA Nemotron Model Reasoning Challenge - Data Analysis")
    print("=" * 70)
    
    # Basic stats
    print(f"\n--- Dataset Overview ---")
    print(f"Training puzzles: {len(train_df)}")
    print(f"Columns: {train_df.columns.tolist()}")
    print(f"Column types:\n{train_df.dtypes}")
    
    if test_csv and os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        print(f"Test puzzles: {len(test_df)}")
    
    # Prompt length analysis
    if 'prompt' in train_df.columns:
        train_df['prompt_length'] = train_df['prompt'].str.len()
        train_df['prompt_words'] = train_df['prompt'].str.split().str.len()
        
        print(f"\n--- Prompt Statistics ---")
        print(f"Prompt length (chars):")
        print(f"  Mean: {train_df['prompt_length'].mean():.0f}")
        print(f"  Median: {train_df['prompt_length'].median():.0f}")
        print(f"  Min: {train_df['prompt_length'].min()}")
        print(f"  Max: {train_df['prompt_length'].max()}")
        print(f"  Std: {train_df['prompt_length'].std():.0f}")
        
        print(f"\nPrompt length (words):")
        print(f"  Mean: {train_df['prompt_words'].mean():.0f}")
        print(f"  Median: {train_df['prompt_words'].median():.0f}")
        print(f"  Min: {train_df['prompt_words'].min()}")
        print(f"  Max: {train_df['prompt_words'].max()}")
        
        # Show distribution buckets
        print(f"\nPrompt length distribution:")
        buckets = [0, 500, 1000, 2000, 5000, 10000, float('inf')]
        for i in range(len(buckets) - 1):
            count = ((train_df['prompt_length'] >= buckets[i]) & 
                     (train_df['prompt_length'] < buckets[i+1])).sum()
            print(f"  {buckets[i]:>5}-{buckets[i+1]:>5}: {count:>4} ({count/len(train_df)*100:.1f}%)")
    
    # Solution analysis
    if 'solution' in train_df.columns:
        train_df['solution_length'] = train_df['solution'].str.len()
        
        print(f"\n--- Solution Statistics ---")
        print(f"Solution length (chars):")
        print(f"  Mean: {train_df['solution_length'].mean():.0f}")
        print(f"  Median: {train_df['solution_length'].median():.0f}")
        print(f"  Min: {train_df['solution_length'].min()}")
        print(f"  Max: {train_df['solution_length'].max()}")
        
        # Solution types
        numeric_solutions = train_df['solution'].str.match(r'^\d+$').sum()
        single_char = train_df['solution'].str.match(r'^[A-Za-z0-9]$').sum()
        short_solutions = (train_df['solution_length'] < 50).sum()
        
        print(f"\nSolution type distribution:")
        print(f"  Pure numeric: {numeric_solutions} ({numeric_solutions/len(train_df)*100:.1f}%)")
        print(f"  Single character: {single_char} ({single_char/len(train_df)*100:.1f}%)")
        print(f"  Short (<50 chars): {short_solutions} ({short_solutions/len(train_df)*100:.1f}%)")
        print(f"  Long (>=50 chars): {len(train_df)-short_solutions}")
    
    # Category classification
    print(f"\n--- Puzzle Category Classification ---")
    from baseline_solver import classify_puzzle, CATEGORY_KEYWORDS
    
    categories = [classify_puzzle(p) for p in train_df['prompt']]
    cat_counts = Counter(categories)
    
    print("Category distribution:")
    for cat, count in cat_counts.most_common():
        pct = count / len(categories) * 100
        print(f"  {cat:<25}: {count:>4} ({pct:.1f}%)")
    
    # Keyword frequency analysis
    print(f"\n--- Top Keywords in Prompts ---")
    all_text = ' '.join(train_df['prompt'].str.lower())
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'and', 'or', 'but', 'not', 'that', 'this', 'it', 'as',
                  'input', 'output', 'example', 'following', 'each', 'which'}
    
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    word_counts = Counter(w for w in words if w not in stop_words)
    
    print("Top 30 keywords:")
    for word, count in word_counts.most_common(30):
        print(f"  {word:<20}: {count:>4}")
    
    # Sample puzzles per category
    print(f"\n--- Sample Puzzles per Category ---")
    train_df['_category'] = categories
    for cat in cat_counts.most_common(3):
        cat_name = cat[0]
        sample = train_df[train_df['_category'] == cat_name].iloc[0]
        print(f"\n[{cat_name}] Puzzle {sample.get('puzzle_id', 'N/A')}:")
        prompt_preview = sample['prompt'][:300]
        print(f"  Prompt: {prompt_preview}...")
        if 'solution' in sample:
            print(f"  Solution: {sample['solution'][:100]}")
    
    # Difficulty estimation
    print(f"\n--- Estimated Difficulty ---")
    train_df['_difficulty'] = train_df['prompt_words'] * 0.3 + train_df['solution_length'] * 0.7
    train_df['difficulty_bucket'] = pd.cut(
        train_df['_difficulty'], 
        bins=[0, 50, 200, 500, float('inf')],
        labels=['Easy', 'Medium', 'Hard', 'Very Hard']
    )
    
    print("Difficulty distribution:")
    for bucket in ['Easy', 'Medium', 'Hard', 'Very Hard']:
        count = (train_df['difficulty_bucket'] == bucket).sum()
        print(f"  {bucket:<12}: {count:>4} ({count/len(train_df)*100:.1f}%)")
    
    # Save analysis results
    analysis = {
        "total_puzzles": len(train_df),
        "category_distribution": dict(cat_counts.most_common()),
        "avg_prompt_length": float(train_df['prompt_length'].mean()),
        "avg_solution_length": float(train_df['solution_length'].mean()) if 'solution' in train_df.columns else None,
        "numeric_solution_pct": float(numeric_solutions / len(train_df) * 100) if 'solution' in train_df.columns else None,
        "top_keywords": dict(word_counts.most_common(30)),
    }
    
    with open("analysis_results.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis saved to analysis_results.json")
    
    return train_df, categories


def identify_puzzle_patterns(train_df: pd.DataFrame) -> Dict:
    """Identify common patterns in puzzle structures."""
    
    patterns = {
        "has_grid": 0,
        "has_matrix": 0,
        "has_numbers": 0,
        "has_letters": 0,
        "has_colors": 0,
        "has_shapes": 0,
        "has_input_output_pairs": 0,
        "has_multiple_choice": 0,
        "avg_examples_per_puzzle": 0,
    }
    
    example_counts = []
    
    for prompt in train_df['prompt']:
        lower = prompt.lower()
        
        if 'grid' in lower or 'row' in lower or 'column' in lower:
            patterns["has_grid"] += 1
        if 'matrix' in lower or '[' in prompt:
            patterns["has_matrix"] += 1
        if re.search(r'\d{2,}', prompt):
            patterns["has_numbers"] += 1
        if re.search(r'[A-Z]\s*[\.\)]', prompt):
            patterns["has_letters"] += 1
        if any(c in lower for c in ['red', 'blue', 'green', 'color', 'colour']):
            patterns["has_colors"] += 1
        if any(s in lower for s in ['circle', 'square', 'triangle', 'shape']):
            patterns["has_shapes"] += 1
        if 'input' in lower and 'output' in lower:
            patterns["has_input_output_pairs"] += 1
        if 'option' in lower or 'choice' in lower or '(a)' in lower:
            patterns["has_multiple_choice"] += 1
        
        # Count input-output examples
        example_count = lower.count('input') + lower.count('example')
        example_counts.append(example_count)
    
    n = len(train_df)
    patterns = {k: f"{v}/{n} ({v/n*100:.1f}%)" for k, v in patterns.items()}
    patterns["avg_examples_per_puzzle"] = f"{np.mean(example_counts):.1f}"
    
    return patterns


def main():
    parser = argparse.ArgumentParser(description="Analyze Nemotron puzzle data")
    parser.add_argument("--train", type=str, default="train.csv")
    parser.add_argument("--test", type=str, default=None)
    args = parser.parse_args()

    train_df, categories = analyze_dataset(args.train, args.test)
    
    # Pattern analysis
    print(f"\n--- Puzzle Structure Patterns ---")
    patterns = identify_puzzle_patterns(train_df)
    for k, v in patterns.items():
        print(f"  {k:<30}: {v}")


if __name__ == "__main__":
    main()
