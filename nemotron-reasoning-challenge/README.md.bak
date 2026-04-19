# NVIDIA Nemotron Model Reasoning Challenge

Kaggle: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

## Setup
1. Download data from Kaggle
2. Install dependencies: `pip install -r requirements.txt`
3. Run analysis: `python analyze_puzzles.py`
4. Run baseline: `python baseline_solver.py`

## Approach
- Phase 1: Puzzle category analysis and classification
- Phase 2: Zero-shot prompting with category-specific templates
- Phase 3: LoRA fine-tuning on Nemotron-3-Nano-30B
- Phase 4: Self-consistency + ensemble

## Directory Structure
```
├── data/               # train.csv, test.csv
├── analysis/           # EDA notebooks, category analysis
├── prompts/            # Prompt templates per category
├── solvers/            # Category-specific solvers
├── training/           # LoRA fine-tuning scripts
├── configs/            # Hyperparameter configs
├── submissions/        # Generated submission files
├── baseline_solver.py  # Baseline submission
├── lora_train.py       # LoRA fine-tuning script
└── ensemble_solver.py  # Ensemble pipeline
```

## Key Insights
- 6-7 puzzle categories with distinct transformation rules
- Temperature tuning is critical (0.1-0.7 range)
- LoRA fine-tuning feasible on RTX 6000 Blackwell
- Self-consistency decoding improves accuracy significantly
