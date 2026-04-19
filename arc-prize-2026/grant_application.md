# Kaggle Benchmarks Resource Grant Application

**Project:** AnnotateX — Annotation-Guided Reasoning for Abstract Visual Intelligence  
**PI / Applicant:** Shyam Desigan  
**Date:** June 2026  
**GitHub:** https://github.com/zan-maker/arc-prize-2026  
**ARC-AGI-2 Kernel:** https://www.kaggle.com/code/shyamdesigan/annotatex-arc-agi-2-v2  
**ARC-AGI-3 Agent:** https://www.kaggle.com/code/shyamdesigan/annotatex-arc-agi-3-agent  

---

## 1. Project Title

**AnnotateX-Bench: A Multi-Dimensional LLM Benchmark for Abstract Visual Reasoning and Program Synthesis**

*(77 characters)*

---

## 2. Project Description

*(468 words)*

### The Problem

Abstract reasoning — the ability to discover patterns, infer rules from sparse evidence, and apply them to novel situations — is widely regarded as a cornerstone of human intelligence and a critical milestone toward AGI. The Abstraction and Reasoning Corpus (ARC-AGI), introduced by Chollet (2019) and expanded through the ARC Prize competition series, has become the gold standard for measuring this capability. Yet despite years of effort and millions of dollars in prizes, the best systems still fail on approximately 75% of ARC-AGI-2 tasks. The 2025 competition winner (NVARC) achieved only 24.0% on the private evaluation set, and the top program synthesis approach (SOAR) reached 26.0%. This persistent gap suggests that current LLMs and AI systems fundamentally lack the compositional reasoning and novel-primitive discovery abilities that humans use intuitively.

What makes this gap particularly consequential is that *we do not yet have a standardized, reproducible benchmark that systematically evaluates LLMs on ARC-style tasks with granular error analysis*. Existing evaluations are one-off competition submissions that cannot be replicated or compared across models. There is no public leaderboard that tracks how GPT-4o, Claude, Gemini, Qwen, DeepSeek, and Grok perform on abstract reasoning — broken down by reasoning type (conditional logic, novel primitives, compositional depth, arithmetic, spatial relations). Without this diagnostic visibility, the community cannot measure progress or identify which specific reasoning capabilities need improvement.

### What We Will Build

**AnnotateX-Bench** is a standardized, reproducible benchmark built on the Kaggle Benchmarks SDK that evaluates LLMs on abstract visual reasoning across multiple dimensions. It consists of two complementary evaluation tracks:

**Track 1 — Static Grid Reasoning (ARC-AGI-2 style):** Evaluates LLMs on 400 curated ARC tasks spanning 12 reasoning categories: geometric transforms, color remapping, tiling, scaling, object manipulation, symmetry, pattern completion, counting/arithmetic, conditional logic, multi-step composition, maze/graph traversal, and recursive transformations. Each task is annotated with ground-truth category labels, enabling per-category accuracy reporting. LLMs are evaluated through code generation (write a Python transform function verified against demonstration pairs) and direct visual reasoning (VLM grid prediction), using the kaggle-benchmarks SDK's sandbox execution and LLM-as-Judge capabilities.

**Track 2 — Interactive World Modeling (ARC-AGI-3 style):** Evaluates LLM agents on 23 interactive game environments that test hypothesis-driven exploration, game mechanics discovery, and multi-step planning under sparse feedback. Agents receive frame-by-frame observations and must select actions, measuring whether LLMs can build world models from interaction rather than pattern matching.

### Expected Outcomes

1. A **public leaderboard** tracking abstract reasoning performance across all major LLMs (OpenAI, Anthropic, Google, Qwen, DeepSeek, Groq), updated automatically as new models are released via Kaggle's managed infrastructure.
2. A **diagnostic breakdown** by reasoning category that reveals exactly where each model fails — enabling targeted capability improvement.
3. An **error-categorized dataset** of 400+ ARC tasks with standardized annotations, published openly for the research community.
4. A **technical report** analyzing cross-model reasoning gaps and identifying the most impactful capability deficiencies.

---

## 3. Research Questions

1. **How do frontier LLMs compare on specific abstract reasoning categories?** Current evaluations report a single aggregate score. AnnotateX-Bench will measure per-category accuracy (conditional logic, novel primitives, compositional depth, arithmetic, spatial relations), revealing whether failures are concentrated in particular reasoning types or distributed evenly.

2. **Does scaling LLMs improve abstract reasoning proportionally, or are certain reasoning categories "scaling-resistant"?** By evaluating models across size ranges (e.g., GPT-4o-mini vs. GPT-4o vs. o3, Qwen3-4B vs. Qwen3-72B), we can determine whether conditional logic and novel primitive discovery improve with scale or require architectural innovations.

3. **Can LLM agents build functional world models through interaction?** ARC-AGI-3 tasks require discovering game mechanics from scratch. We test whether LLM agents can form hypotheses about unseen rules, test them through action, and refine their understanding — a capability distinct from pattern matching on static inputs.

4. **Which prompt engineering and code generation strategies are most effective for abstract reasoning?** We compare code generation (LLM writes Python), direct grid prediction (VLM outputs), chain-of-thought reasoning, and multi-agent approaches to identify the most reliable paradigm.

5. **What is the correlation between ARC-AGI performance and performance on other reasoning benchmarks?** By cross-referencing with MMLU, GSM8K, and HumanEval, we investigate whether abstract visual reasoning is a distinct capability or a proxy for general reasoning ability.

---

## 4. Methodology

*(284 words)*

### Benchmark Architecture

AnnotateX-Bench is built on the **Kaggle Benchmarks SDK** (`kaggle-benchmarks`), leveraging its unified LLM interaction API, task decorator system, assertion framework, and parallel evaluation capabilities.

**Task Definition:** Each ARC task is defined as a `@kbench.task` that (a) renders the grid as both a text representation and a PIL image for multimodal models, (b) presents the demonstration pairs as a structured prompt, (c) executes the LLM's generated code in a Docker sandbox against all training examples for verification, and (d) scores the output via exact grid match (boolean) and partial credit (cell-level accuracy).

**Multi-Provider Evaluation:** The SDK's `llm.prompt()` API enables uniform interaction with OpenAI, Anthropic, Google, Qwen, DeepSeek, and Grok models using structured output schemas (Pydantic). Each model receives identical prompts and evaluation conditions, ensuring fair comparison.

**LLM-as-Judge for Reasoning Quality:** For tasks where the exact output is incorrect, a secondary evaluator (GPT-4o or Claude) assesses whether the model's reasoning chain demonstrates correct understanding of the transformation type, even if execution fails. This provides a "reasoning accuracy" metric alongside the strict "output accuracy" metric.

**Error Categorization Pipeline:** Each evaluated task is automatically classified into one of six error categories using the kaggle-benchmarks assertion system: conditional logic failure, novel primitive gap, compositional depth exceeded, perceptual extraction error, scoring/selection error, or output format error. These categories are derived from our published error taxonomy (Desigan, 2026).

**ARC-AGI-3 Agent Evaluation:** Interactive game environments are evaluated by running LLM agents through the official ARC-AGI-3 SDK (`arcengine`), measuring games won, levels completed, actions taken, and hypothesis quality. The kaggle-benchmarks parallel evaluation engine runs multiple game sessions simultaneously for statistical significance.

---

## 5. Resource Requirements

| Resource | Specification | Justification |
|----------|--------------|---------------|
| **GPU Compute (A100 80GB)** | 1,000 GPU-hours | Running Qwen3-72B and other large models for code generation and grid reasoning across 400+ tasks, with multiple attempts per task (K=6–8). Memory-intensive inference for 72B parameter models. |
| **GPU Compute (T4/L4)** | 2,000 GPU-hours | Running smaller models (Qwen3-4B, GPT-4o-mini) for baseline comparisons and iterative benchmark development. |
| **Model API Access** | OpenAI, Anthropic, Google, DeepSeek, Qwen, Grok | Critical for cross-provider evaluation. The grant's model access is the primary enabler — without it, evaluating across 6+ providers is cost-prohibitive. |
| **CPU Compute** | 4,000 CPU-hours | Heuristic solver baselines, sandboxed code execution across thousands of task-model pairs, and ARC-AGI-3 agent simulations. |
| **Storage** | 500 GB | Task datasets (ARC-AGI-2, ARC-AGI-3 environments), model outputs, evaluation logs, agent traces, and leaderboard history. |
| **Managed Infrastructure** | Kaggle Benchmarks hosting | Automated leaderboard updates as new models are released. The grant's managed infrastructure eliminates the engineering burden of maintaining the evaluation pipeline. |
| **Kaggle Engineering Support** | SDK integration | Assistance with kaggle-benchmarks integration for multimodal inputs (grid images), Docker sandbox configuration for code execution, and custom evaluation metric implementation. |

**Total estimated compute:** ~3,000 GPU-hours + 4,000 CPU-hours over 6 months.

---

## 6. Expected Deliverables

| Deliverable | Format | Timeline | Description |
|-------------|--------|----------|-------------|
| **AnnotateX-Bench SDK** | Python package (`pip install annotatex-bench`) | Month 2 | Open-source benchmark package built on kaggle-benchmarks, with task definitions, evaluation logic, and error categorization. |
| **Curated Task Dataset** | JSON + documentation | Month 2 | 400 ARC-AGI-2 tasks with ground-truth category labels, difficulty ratings, and feature annotations. |
| **Public Leaderboard** | Kaggle Benchmarks platform | Month 3 | Live leaderboard tracking all major LLMs on abstract reasoning, with per-category breakdowns and model comparison tools. |
| **Baseline Evaluation Report** | Technical report + Kaggle notebook | Month 3 | Initial evaluation of 8–10 LLMs across all categories, establishing first benchmark results. |
| **ARC-AGI-3 Agent Track** | Kaggle Benchmark + notebooks | Month 4 | Interactive reasoning evaluation across 23 game environments. |
| **Technical Paper** | arXiv preprint | Month 5 | Peer-review-ready paper analyzing cross-model reasoning gaps, scaling trends, and error distributions. |
| **Tutorial Notebooks** | Kaggle notebooks (3–5) | Month 4 | Step-by-step tutorials for using the benchmark, adding new tasks, and interpreting results. |
| **Error-Categorized ARC Dataset** | Open dataset (CC BY 4.0) | Month 3 | Annotated dataset of ARC tasks by failure mode, enabling targeted research on specific reasoning deficiencies. |

---

## 7. Timeline

### Month 1: Foundation
- Integrate ARC-AGI-2 task data into kaggle-benchmarks SDK format
- Implement `@kbench.task` definitions for all 400 curated tasks
- Build grid rendering pipeline (text + image representations)
- Develop sandboxed code execution verifier
- Set up evaluation infrastructure

### Month 2: Core Benchmark
- Complete error categorization pipeline (6 failure types)
- Implement LLM-as-Judge evaluator for reasoning quality assessment
- Run initial evaluations across 4–6 model providers
- Publish curated task dataset with annotations
- Release AnnotateX-Bench SDK v0.1

### Month 3: Public Launch
- Deploy public leaderboard on Kaggle Benchmarks
- Publish baseline evaluation report with 8–10 models
- Release error-categorized ARC dataset
- Write and publish 3 tutorial notebooks
- Submit initial results to Kaggle community for feedback

### Month 4: ARC-AGI-3 Extension
- Implement ARC-AGI-3 interactive agent evaluation track
- Build hypothesis-driven exploration agent baselines
- Evaluate LLM agents across 23 game environments
- Integrate multi-model comparison for agent track
- Update leaderboard with agent reasoning metrics

### Month 5: Analysis & Publication
- Complete cross-model analysis across all tracks
- Write and submit technical paper to arXiv
- Analyze scaling trends (small vs. large models by category)
- Investigate correlation with other reasoning benchmarks
- Prepare supplementary materials

### Month 6: Refinement & Handoff
- Incorporate community feedback
- Add new models released during grant period
- Finalize documentation and API stability
- Transition to community maintenance model
- Submit paper to conference (NeurIPS, ICLR, or ICML)

---

## 8. Related Work & Differentiation

*(199 words)*

**Existing ARC evaluations** are competition submissions (NVARC at 24%, SOAR at 26%) that cannot be replicated or compared across models. They report single aggregate scores without diagnostic breakdown. **AnnotateX-Bench** provides the first standardized, reproducible, multi-model evaluation with per-category error analysis.

**General LLM benchmarks** (MMLU, HumanEval, GSM8K) test knowledge and coding but not abstract visual reasoning — the ability to discover novel transformation rules from minimal evidence. ARC-AGI tasks are specifically designed to resist memorization, making them a uniquely challenging test of generalization. AnnotateX-Bench fills this gap in the LLM evaluation landscape.

**Multi-agent benchmarks** (SwarmBench, MultiAgentBench) evaluate coordination but not cognitive reasoning. AnnotateX-Bench's ARC-AGI-3 track is the first to evaluate whether multi-agent collaboration improves abstract reasoning — measuring emergent intelligence rather than coordination efficiency.

**Kaggle Benchmarks** (Cognitive Abilities by DeepMind) represent the closest existing work, evaluating planning and logical reasoning. AnnotateX-Bench is differentiated by (a) its focus on visual/spatial reasoning rather than natural language reasoning, (b) its code-generation evaluation paradigm that tests program synthesis, (c) its interactive ARC-AGI-3 track testing world model construction, and (d) its fine-grained 6-category error taxonomy that provides diagnostic insights beyond aggregate scores.

---

## 9. Broader Impact

*(148 words)*

AnnotateX-Bench directly addresses one of the most pressing questions in AI: *can current LLMs genuinely reason, or do they only pattern-match?* By providing the first standardized, multi-model evaluation of abstract visual reasoning, we enable the research community to measure progress on a capability that is fundamental to AGI yet almost entirely unmeasured at scale.

The per-category error breakdown has immediate practical value. If we demonstrate that conditional logic reasoning is "scaling-resistant" — that even the largest models fail on it — this redirects research effort toward architectural innovations (neuro-symbolic integration, program synthesis) rather than单纯 scaling. Conversely, if certain categories improve smoothly with scale, the community can allocate resources more efficiently.

Beyond the research community, AnnotateX-Bench's open dataset and tutorials make abstract reasoning evaluation accessible to the 30 million Kaggle users, democratizing a capability assessment that currently requires competition-level engineering. The ARC-AGI-3 interactive track additionally advances our understanding of whether AI agents can learn from experience — a capability with direct applications in robotics, scientific discovery, and autonomous systems.

---

## Appendix: Quick Reference — Key URLs

| Resource | URL |
|----------|-----|
| Application Form | https://services.google.com/fb/forms/kaggle-research-grants-application/ |
| GitHub Repository | https://github.com/zan-maker/arc-prize-2026 |
| ARC-AGI-2 Kernel | https://www.kaggle.com/code/shyamdesigan/annotatex-arc-agi-2-v2 |
| ARC-AGI-3 Agent | https://www.kaggle.com/code/shyamdesigan/annotatex-arc-agi-3-agent |
| Kaggle Benchmarks | https://www.kaggle.com/benchmarks |
| ARC Prize | https://arcprize.org |

---

*Prepared by Shyam Desigan, AnnotateX Research Team — June 2026*
