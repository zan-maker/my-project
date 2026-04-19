"""
NVIDIA Nemotron Model Reasoning Challenge - LoRA Fine-Tuning
Fine-tune Nemotron-3-Nano-30B with LoRA on reasoning puzzles.

Designed for Kaggle's RTX PRO 6000 Blackwell GPU (48GB+ VRAM).
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path


def prepare_training_data(
    train_csv: str,
    output_dir: str,
    val_split: float = 0.1,
    max_samples: Optional[int] = None,
    system_prompt: str = "You are an expert logical reasoning puzzle solver. Carefully analyze the given examples and provide the correct answer."
):
    """Prepare training data in the format expected by the model."""
    
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} training examples")
    
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
        print(f"Sampled {len(df)} examples")
    
    # Prepare conversation format
    train_data = []
    for _, row in df.iterrows():
        prompt = row['prompt']
        solution = row['solution']
        
        # Format as instruction-response pair
        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f" Solve this reasoning puzzle:\n\n{prompt}\n\nProvide the final answer."},
                {"role": "assistant", "content": solution}
            ]
        }
        train_data.append(example)
    
    # Split into train/val
    np.random.seed(42)
    indices = np.random.permutation(len(train_data))
    val_size = int(len(train_data) * val_split)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_examples = [train_data[i] for i in train_indices]
    val_examples = [train_data[i] for i in val_indices]
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    
    with open(train_path, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    
    with open(val_path, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Training set: {len(train_examples)} examples -> {train_path}")
    print(f"Validation set: {len(val_examples)} examples -> {val_path}")
    
    return train_path, val_path


def setup_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
):
    """Set up LoRA configuration."""
    from peft import LoraConfig, TaskType
    
    if target_modules is None:
        # Default target modules for Nemotron/Llama-style models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )
    return config


def load_model_for_training(
    model_name: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    use_4bit: bool = True,
    use_gradient_checkpointing: bool = True,
):
    """Load model with quantization for memory-efficient training."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Required for gradient checkpointing
    
    print(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer


def train_lora(
    model_name: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    train_file: str = "data/train.jsonl",
    val_file: str = "data/val.jsonl",
    output_dir: str = "checkpoints/lora_baseline",
    # LoRA params
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    # Training params
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    max_seq_length: int = 2048,
    use_4bit: bool = True,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: int = 50,
):
    """Full LoRA training pipeline."""
    import torch
    from transformers import TrainingArguments, Trainer
    from peft import get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    
    # Load model
    model, tokenizer = load_model_for_training(
        model_name=model_name,
        use_4bit=use_4bit,
        use_gradient_checkpointing=True,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    lora_config = setup_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    val_dataset = load_dataset("json", data_files=val_file, split="train")
    
    def tokenize_function(examples):
        # Format as chat template
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    train_tokenized = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["messages"]
    )
    val_tokenized = val_dataset.map(
        tokenize_function, batched=True, remove_columns=["messages"]
    )
    
    # Training arguments
    effective_batch = batch_size * gradient_accumulation_steps
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        report_to="none",  # Disable wandb for Kaggle
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        seed=42,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
    )
    
    # Train
    print(f"\nStarting LoRA training...")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Total steps: ~{len(train_tokenized) * num_epochs // effective_batch}")
    
    trainer.train()
    
    # Save
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    
    return model, tokenizer, trainer


def merge_and_save(lora_path: str, base_model: str, output_path: str):
    """Merge LoRA weights into base model and save."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    
    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Merging LoRA from: {lora_path}")
    model = PeftModel.from_pretrained(base, lora_path)
    model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Merged model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for Nemotron")
    parser.add_argument("--model", type=str, default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_v1")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--merge", type=str, default=None)
    args = parser.parse_args()

    if args.prepare_only:
        prepare_training_data(args.train_csv, "data")
        return

    if args.merge:
        merge_and_save(args.merge, args.model, args.output_dir)
        return

    # Full pipeline
    print("Step 1: Preparing training data...")
    train_file, val_file = prepare_training_data(args.train_csv, "data")
    
    print("\nStep 2: Training LoRA...")
    model, tokenizer, trainer = train_lora(
        model_name=args.model,
        train_file=train_file,
        val_file=val_file,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
