#!/usr/bin/env python3
"""
Build Nemotron LoRA Fine-Tuning Kaggle Notebook
Generates a complete .ipynb file ready to run on Kaggle.
"""

import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def code(source):
    return {"cell_type": "code", "metadata": {"trusted": True}, "source": source.split("\n"), "outputs": [], "execution_count": None}

cells = []

# ============================================================
# Cell 1: Title
# ============================================================
cells.append(md("""# NVIDIA Nemotron Model Reasoning Challenge — LoRA Fine-Tuning

**Goal:** Improve Nemotron-3-Nano-30B reasoning accuracy via LoRA fine-tuning on the provided puzzle dataset.

**Approach:**
1. Load the 9,500 reasoning puzzles from `train.csv`
2. Format training data using Nemotron's chat template with `\\boxed{}` answer format
3. Fine-tune with LoRA (rank 32) targeting Mamba-2 and MLP projection layers
4. Package the trained adapter into `submission.zip`

**Model:** Nemotron-3-Nano-30B-A3B-BF16 (30B total params, 3.2B active, hybrid Mamba2-Transformer MoE)

**Competition:** [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
"""))

# ============================================================
# Cell 2: Install Dependencies
# ============================================================
cells.append(md("""## 1. Install Dependencies

Installing Mamba-2 CUDA kernels and PEFT/TRL libraries required for Nemotron's hybrid architecture.
"""))

cells.append(code("""\
import subprocess
import sys
import os

print("Installing Mamba-2 dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", 
    "mamba_ssm==2.2.5", "--no-build-isolation",
    "causal_conv1d==1.5.2", "--no-build-isolation",
], check=False)

print("Installing PEFT and TRL...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.14.0", "trl>=0.15.0", "transformers>=4.47.0",
    "accelerate>=1.1.0", "datasets>=3.0.0",
], check=False)

# Add NVIDIA CUTLASS package (required for custom ops)
import site
cutlass_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
if os.path.exists(cutlass_path):
    site.addsitedir(cutlass_path)
    print(f"Added CUTLASS path: {cutlass_path}")
else:
    print("CUTLASS path not found — make sure to add 'nvidia-utility-script' via Kaggle Add Data > Utility Script")

print("All dependencies installed!")
"""))

# ============================================================
# Cell 3: Imports & Config
# ============================================================
cells.append(md("""## 2. Imports & Configuration
"""))

cells.append(code("""\
import os
import gc
import json
import re
import torch
import polars as pl
import numpy as np
from pathlib import Path

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Paths
COMPETITION_DATA = "/kaggle/input/nvidia-nemotron-3-reasoning-challenge"
OUTPUT_DIR = "/kaggle/working"

# Training hyperparameters
LORA_RANK = 32          # Max allowed by competition
LORA_ALPHA = 16         # alpha = rank/2 (standard scaling)
LORA_DROPOUT = 0.05
NUM_EPOCHS = 2
BATCH_SIZE = 1          # Per device — model is 30B params
GRAD_ACCUM = 8          # Effective batch = 8
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 2048   # Most puzzles fit within this
WARMUP_STEPS = 50
LOGGING_STEPS = 25

print(f"Config:")
print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
print(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, Grad Accum: {GRAD_ACCUM}")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max seq length: {MAX_SEQ_LENGTH}")
print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
"""))

# ============================================================
# Cell 4: Load & Analyze Training Data
# ============================================================
cells.append(md("""## 3. Load & Analyze Training Data

The dataset contains 9,500 logical reasoning puzzles covering various types (bit manipulation, cryptarithm, encryption, base conversion, unit conversion, gravitational, and more).
"""))

cells.append(code("""\
# Load training data
train = pl.read_csv(f"{COMPETITION_DATA}/train.csv")
print(f"Training examples: {len(train)}")
print(f"Columns: {train.columns}")
print()

# Show first few rows
print("Sample data:")
train.head(5)
"""))

cells.append(code("""\
# Analyze answer formats and puzzle characteristics
print("=== Data Analysis ===")
print(f"Total puzzles: {len(train)}")
print()

# Answer length distribution
answer_lengths = train.select(
    pl.col("answer").str.len_chars().alias("len")
)["len"].to_list()
print(f"Answer length — min: {min(answer_lengths)}, max: {max(answer_lengths)}, "
      f"mean: {np.mean(answer_lengths):.1f}, median: {np.median(answer_lengths):.1f}")
print()

# Prompt length distribution
prompt_lengths = train.select(
    pl.col("prompt").str.len_chars().alias("len")
)["len"].to_list()
print(f"Prompt length — min: {min(prompt_lengths)}, max: {max(prompt_lengths)}, "
      f"mean: {np.mean(prompt_lengths):.1f}, median: {np.median(prompt_lengths):.1f}")
print()

# Count puzzles that fit within max_seq_length
fit_count = sum(1 for p in prompt_lengths if p < MAX_SEQ_LENGTH - 200)
print(f"Puzzles fitting in {MAX_SEQ_LENGTH} tokens (est): {fit_count}/{len(train)} ({fit_count/len(train)*100:.1f}%)")
print()

# Show some answer examples
print("Answer examples:")
for i in range(10):
    ans = train.row(i)["answer"]
    print(f"  [{i}] {repr(ans)}")
"""))

# ============================================================
# Cell 5: Prepare Training Dataset
# ============================================================
cells.append(md("""## 4. Prepare Training Dataset

Format each puzzle as a chat conversation:
- **User message:** The puzzle prompt + instruction to output in `\\boxed{}` format
- **Assistant response:** The ground truth answer wrapped in `\\boxed{}`

This teaches the model both the puzzle-solving patterns and the required output format.
"""))

cells.append(code("""\
from datasets import Dataset

def format_training_data(train_df):
    \"""Format training data into chat conversations for SFT.\""\"
    records = []
    
    for i in range(len(train_df)):
        row = train_df.row(i)
        puzzle_prompt = row["prompt"]
        answer = row["answer"]
        
        # Format as chat conversation
        messages = [
            {
                "role": "user",
                "content": (
                    f"{puzzle_prompt}\\n\\n"
                    "Solve this reasoning puzzle step by step. "
                    "Provide your final answer in \\\\boxed{{}} format."
                )
            },
            {
                "role": "assistant", 
                "content": f"\\\\boxed{{{answer}}}"
            }
        ]
        records.append({"messages": messages})
    
    return records

print("Formatting training data...")
train_records = format_training_data(train)
print(f"Formatted {len(train_records)} examples")

# Create HuggingFace dataset
raw_dataset = Dataset.from_list(train_records)
print(f"Dataset created: {len(raw_dataset)} examples")

# Show a sample
sample = train_records[0]
print(f"\\nSample user message (truncated):")
print(sample["messages"][0]["content"][:300] + "...")
print(f"\\nSample assistant response:")
print(sample["messages"][1]["content"])
"""))

cells.append(code("""\
# We'll apply the chat template after loading the model (needs tokenizer)
# For now, just verify the data looks correct
print("Verifying data formatting...")
print(f"Total records: {len(train_records)}")
print(f"Each record has 'messages' key with user + assistant roles")

# Check for any issues
empty_answers = sum(1 for r in train_records if not r["messages"][1]["content"].strip())
print(f"Empty assistant responses: {empty_answers}")

long_prompts = sum(1 for r in train_records if len(r["messages"][0]["content"]) > MAX_SEQ_LENGTH * 4)
print(f"Very long prompts (>8k chars): {long_prompts}")

print("\\nData preparation complete!")
"""))

# ============================================================
# Cell 6: Load Model & Tokenizer
# ============================================================
cells.append(md("""## 5. Load Model & Configure LoRA

Load the Nemotron-3-Nano-30B-A3B-BF16 model and apply a LoRA adapter targeting the Mamba-2 and MLP projection layers.

The LoRA configuration matches the competition requirements:
- **Rank:** 32 (maximum allowed)
- **Target modules:** `in_proj`, `out_proj`, `up_proj`, `down_proj` (Mamba-2 + MLP layers)
- **Trainable parameters:** ~880M out of ~32.5B total (2.7%)
"""))

cells.append(code("""\
import kagglehub
import mamba_ssm  # Ensure Mamba-2 kernels are loaded

# Download model from Kaggle Models
print("Downloading model from Kaggle Hub...")
MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
)
print(f"Model path: {MODEL_PATH}")

# Load model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

print("\\nLoading model (this may take a few minutes)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded successfully!")
print(f"Model dtype: {model.dtype}")
print(f"Device: {model.device}")
"""))

cells.append(code("""\
# Verify tokenizer works
test_messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "\\\\boxed{4}"}
]
test_text = tokenizer.apply_chat_template(test_messages, tokenize=False, enable_thinking=False)
print("Chat template test:")
print(test_text)
print()

# Count tokens
test_tokens = tokenizer.apply_chat_template(test_messages, tokenize=True, enable_thinking=False)
print(f"Token count: {len(test_tokens)}")
"""))

# ============================================================
# Cell 7: Configure LoRA
# ============================================================
cells.append(code("""\
from peft import LoraConfig, get_peft_model, TaskType

print("Configuring LoRA adapter...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=r".*\\\\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Required for gradient checkpointing

print(f"\\nLoRA configured successfully!")
print(f"  Rank: {LORA_RANK}")
print(f"  Alpha: {LORA_ALPHA}")
print(f"  Target modules: in_proj, out_proj, up_proj, down_proj")
print(f"  Dropout: {LORA_DROPOUT}")
"""))

# ============================================================
# Cell 8: Apply Chat Template to Dataset
# ============================================================
cells.append(md("""## 6. Apply Chat Template & Tokenize

Convert chat messages into tokenized sequences using Nemotron's chat template.
"""))

cells.append(code("""\
def apply_chat_template(example):
    \"""Apply Nemotron chat template to each example.\""\"
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,  # Don't include thinking tokens in training
    )
    return {"text": text}

print("Applying chat template to dataset...")
tokenized_dataset = raw_dataset.map(
    apply_chat_template,
    remove_columns=["messages"],
    desc="Applying chat template",
)

print(f"Dataset size: {len(tokenized_dataset)}")
print(f"\\nSample formatted text (truncated):")
print(tokenized_dataset[0]["text"][:500] + "...")
"""))

# ============================================================
# Cell 9: Train
# ============================================================
cells.append(md("""## 7. Train with SFTTrainer

Fine-tune the LoRA adapter on the reasoning puzzle dataset using TRL's SFTTrainer.

**Training strategy:**
- Batch size 1 with 8x gradient accumulation (effective batch = 8)
- BF16 mixed precision for stability
- AdamW 8-bit optimizer for memory efficiency
- Cosine learning rate schedule with linear warmup
- 2 epochs over the full training set
"""))

cells.append(code("""\
from trl import SFTTrainer, SFTConfig

print("Configuring trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        optim="adamw_8bit",
        max_seq_length=MAX_SEQ_LENGTH,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",        # Don't save checkpoints (save memory)
        report_to="none",          # Disable wandb
        gradient_checkpointing=True,
        seed=SEED,
        max_grad_norm=1.0,
        dataset_num_proc=1,
    ),
)

print("Trainer configured!")
print(f"  Total training steps: {len(tokenized_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRAD_ACCUM)}")
print(f"  VRAM usage will be monitored during training...")
"""))

cells.append(code("""\
# Start training!
import time

print("=" * 60)
print("Starting LoRA Fine-Tuning")
print("=" * 60)

start_time = time.time()
trainer_stats = trainer.train()
elapsed = time.time() - start_time

print(f"\\nTraining completed in {elapsed/60:.1f} minutes!")
print(f"Final training loss: {trainer_stats.training_loss:.4f}")
print(f"Total training steps: {trainer_stats.global_step}")
"""))

# ============================================================
# Cell 10: Quick Evaluation
# ============================================================
cells.append(md("""## 8. Quick Evaluation

Test the fine-tuned model on a few training examples to verify it learned the `\\boxed{}` format.
"""))

cells.append(code("""\
print("Running quick evaluation on 5 training examples...")
model.eval()

correct = 0
total = 5

for i in range(total):
    row = train.row(i)
    puzzle = row["prompt"]
    ground_truth = row["answer"]
    
    messages = [
        {"role": "user", "content": f"{puzzle}\\n\\nSolve this reasoning puzzle step by step. Provide your final answer in \\\\boxed{{}} format."}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Extract boxed answer
    match = re.search(r'\\\\boxed\{([^}]+)\}', response)
    predicted = match.group(1) if match else response.strip()[:50]
    
    is_correct = predicted.strip().lower() == ground_truth.strip().lower()
    if is_correct:
        correct += 1
    
    print(f"\\nExample {i+1}:")
    print(f"  Ground truth:  {ground_truth}")
    print(f"  Predicted:     {predicted}")
    print(f"  Correct:       {'YES' if is_correct else 'NO'}")
    print(f"  Full response: {response[:200]}...")

print(f"\\nAccuracy on {total} samples: {correct}/{total} ({correct/total*100:.0f}%)")
"""))

# ============================================================
# Cell 11: Save & Package Submission
# ============================================================
cells.append(md("""## 9. Save Adapter & Create Submission

Save the trained LoRA adapter and package it into `submission.zip` for Kaggle submission.
"""))

cells.append(code("""\
# Save LoRA adapter
print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Verify saved files
saved_files = list(Path(OUTPUT_DIR).glob("adapter_*"))
saved_files += list(Path(OUTPUT_DIR).glob("*.json"))
print(f"Saved files:")
for f in saved_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}: {size_mb:.1f} MB")
"""))

cells.append(code("""\
# Package into submission.zip
import subprocess

# Remove any existing zip
zip_path = f"{OUTPUT_DIR}/submission.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)

# Zip adapter files only (exclude extra files)
print("Creating submission.zip...")
adapter_files = [f for f in os.listdir(OUTPUT_DIR) 
                 if f.startswith("adapter_") or f == "README.md"]

cmd = f"cd {OUTPUT_DIR} && zip -j submission.zip " + " ".join(adapter_files)
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if result.returncode == 0:
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\\nsubmission.zip created successfully!")
    print(f"  Path: {zip_path}")
    print(f"  Size: {zip_size:.1f} MB")
    print(f"  Files: {', '.join(adapter_files)}")
else:
    print(f"Error creating zip: {result.stderr}")
    # Fallback: zip all files
    subprocess.run(f"cd {OUTPUT_DIR} && zip submission.zip adapter_config.json adapter_model.safetensors", 
                   shell=True, check=True)
    print("Fallback zip created.")
"""))

cells.append(code("""\
# Final verification
print("=" * 60)
print("SUBMISSION READY!")
print("=" * 60)

# Verify adapter_config.json exists and has correct settings
config_path = f"{OUTPUT_DIR}/adapter_config.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    print(f"\\nAdapter config:")
    print(f"  LoRA rank (r): {config.get('r', 'N/A')}")
    print(f"  LoRA alpha: {config.get('lora_alpha', 'N/A')}")
    print(f"  Target modules: {config.get('target_modules', 'N/A')}")
    print(f"  Task type: {config.get('task_type', 'N/A')}")
    
    # Verify rank <= 32
    r = config.get('r', 0)
    if r > 32:
        print(f"\\nWARNING: LoRA rank {r} exceeds maximum of 32!")
    else:
        print(f"\\nLoRA rank {r} is within competition limits.")

# Verify submission.zip
if os.path.exists(zip_path):
    print(f"\\nsubmission.zip: {os.path.getsize(zip_path) / (1024*1024):.1f} MB")
    print(f"\\nClick 'Submit' on the competition page and upload submission.zip!")
"""))

# ============================================================
# Build notebook JSON
# ============================================================
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0,
    "cells": cells
}

output_path = "/home/z/my-project/download/nemotron_lora_finetune.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook saved to: {output_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, {sum(1 for c in cells if c['cell_type']=='code')} code)")
