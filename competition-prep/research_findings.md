# Competition Research & Preparation Plan

## 1. NVIDIA Nemotron Model Reasoning Challenge

**URL:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge  
**Type:** Code Submission (Prediction)  
**Model:** Nemotron-3-Nano-30B (base model provided)  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell GPU on Kaggle  
**Deadline:** TBD (check Kaggle page)

### 1.1 Problem Description
The dataset consists of **logical reasoning puzzles** requiring identification and application of underlying transformation rules — similar to ARC/abstract reasoning. Input-output example pairs are given, and the model must solve a new instance.

### 1.2 Dataset
- **train.csv**: puzzle_id, prompt (full puzzle description with input-output examples + instance to solve), solution
- **test.csv**: puzzle_id, prompt (need to predict solution)
- Puzzles categorize into **6-7 distinct formats/categories** of logical reasoning problems

### 1.3 Allowed Techniques
- Prompting strategies
- Data pipelines
- Lightweight fine-tuning (LoRA on the 30B model)
- Self-consistency / multi-sample decoding
- Temperature tuning
- Post-processing pipelines

### 1.4 Evaluation
- Exact match accuracy on test puzzles
- There was a metric bug fix on Apr 4 that changed evaluation
- Training set solve rate is higher than leaderboard (distribution shift expected)

### 1.5 Strategy & Approach

**Phase 1: Data Analysis (Days 1-2)**
- Download train.csv, analyze puzzle categories
- Identify the 6-7 problem types
- Build category classification pipeline
- Analyze difficulty distribution

**Phase 2: Baseline Solver (Days 3-5)**
- Load Nemotron-3-Nano-30B with HuggingFace Transformers
- Build zero-shot prompt templates per category
- Implement self-consistency decoding (K=8-16 samples)
- Temperature sweep (0.1, 0.3, 0.5, 0.7)
- Submit baseline

**Phase 3: LoRA Fine-Tuning (Days 6-10)**
- Prepare training data: (prompt, solution) pairs from train.csv
- Fine-tune with LoRA on RTX 6000 Blackwell
- Experiment with rank (8, 16, 32), alpha, dropout
- Category-specific LoRA adapters vs single adapter
- Gradient checkpointing for 30B model

**Phase 4: Advanced Techniques (Days 11-15)**
- Chain-of-thought prompting with category-specific reasoning templates
- Multi-stage pipeline: classify category then specialized solver then verify
- Self-consistency with majority voting
- Backtracking / verification loops
- Ensemble of LoRA + prompting approaches

**Phase 5: Optimization (Days 16-20)**
- Hyperparameter sweep
- Post-processing: pattern matching for known puzzle types
- Error analysis on validation split
- Final submission optimization

### 1.6 Key Insights from Discussions
- Temperature matters a lot on the leaderboard (lower is not always better for all categories)
- Some categories are easier to solve with deterministic rules than LLM generation
- "Write your answer from scratch" generation mode used
- Training/test distribution shift exists — do not overfit to training patterns
- LoRA fine-tuning on the 30B model is feasible on the RTX 6000 Blackwell

---

## 2. Gemma 4 Good Hackathon

**URL:** https://www.kaggle.com/competitions/gemma-4-good-hackathon  
**Type:** Hackathon (Project Submission)  
**Model:** Gemma 4 (multimodal + native function calling)  
**Prize:** $200,000  
**Deadline:** May 18, 2026 (Final Submission)  
**Start:** April 2, 2026  
**Participants:** 4,990 Entrants, 41 Teams, 42 Submissions (as of mid-April)

### 2.1 Submission Requirements
- **Working Live Demo**
- **Technical Writeup**
- **Public Code Repo**
- **Video** (2-5 min)

### 2.2 Evaluation Criteria
| Criteria | Points | Description |
|----------|--------|-------------|
| Impact & Vision | 40 pts | Demonstrated in video — real-world problem being solved |
| Technical Execution | 30 pts | Working prototype, code quality, Gemma 4 utilization |
| Originality | 20 pts | Novel approach, not a recycled chatbot/PDF assistant |
| Function Calling & Multimodal | 10 pts | Effective use of Gemma 4's native capabilities |

### 2.3 Key Capabilities of Gemma 4
- **Multimodal**: Text + image understanding
- **Native Function Calling**: Can invoke external APIs/tools
- **Local frontier intelligence**: Runs offline on consumer hardware
- **Agentic workflows**: Multi-step reasoning with tool use

### 2.4 Critical Minerals Project Idea

**Project Title:** MineLens AI — Critical Mineral Prospectivity Mapping using Gemma 4

**Problem Statement:**
The global energy transition depends on critical minerals (lithium, cobalt, rare earths, copper, nickel), but exploration is expensive, slow, and geologically complex. Currently, mineral prospectivity mapping requires specialized geologists, months of analysis, and millions in field work. AI can democratize this process.

**Solution:**
A multimodal AI system powered by Gemma 4 that:
1. **Geological Map Analysis**: Takes satellite imagery, geological survey maps, and geophysical data as input (multimodal)
2. **Function Calling Pipeline**: Calls specialized tools for spectral analysis, terrain classification, and mineral signature detection
3. **Prospectivity Report Generation**: Generates detailed mineral prospectivity reports with confidence scores
4. **Supply Chain Intelligence**: Integrates public data on mining operations, trade flows, and geopolitical risk

**Core Features:**
- **Multimodal Input**: Upload satellite images, geological maps, geochemical surveys
- **Function Calling Tools**:
  - `spectral_analysis(image)` — identify mineral spectral signatures
  - `terrain_classifier(image)` — classify landform types
  - `proximity_search(lat, lon, mineral_type)` — find nearby known deposits
  - `risk_assessment(region, mineral_type)` — geopolitical supply chain risk
  - `report_generator(analysis_results)` — generate prospectivity report
- **Agentic Workflow**: Gemma 4 orchestrates multi-step analysis autonomously
- **Offline Capable**: Can run on field laptops without internet

**Why This Project Wins:**
1. **Impact (40 pts)**: Directly addresses global supply chain security, energy transition, and sustainable resource management — a $2T+ market
2. **Technical Execution (30 pts)**: Leverages ALL Gemma 4 capabilities (multimodal + function calling + agentic)
3. **Originality (20 pts)**: Not a chatbot — it is a specialized geoscience tool with real-world utility
4. **Function Calling (10 pts)**: Heavy use of native function calling for geospatial analysis pipeline

**Technical Stack:**
- Gemma 4 (via HuggingFace / Kaggle Models)
- Python backend (FastAPI)
- Frontend: Next.js for map visualization (Leaflet/Mapbox)
- GeoPandas, Rasterio for geospatial data
- ONNX Runtime for spectral analysis models
- Function calling schema: JSON-based tool definitions

**Datasets:**
- USGS Mineral Resources Data System (MRDS)
- USGS National Minerals Information Center
- ASTER satellite imagery (free, global coverage)
- USGS EarthExplorer (Landsat, Sentinel)
- USGS Mineral Prospectivity Maps
- Critical Mineral Mapping Initiative (CMMI) data

**Alternative Critical Mineral Ideas:**
1. **MineRisk**: Critical mineral supply chain risk assessment using multimodal news analysis + function calling
2. **GeoChat**: Conversational geology assistant that analyzes geological maps/images in real-time
3. **MineralLens**: Multimodal mineral identification from rock photos + prospectivity scoring

---

## 3. Parallel Execution Timeline

```
Week 1 (Apr 18-24):   [Nemotron] Data download + analysis       [Gemma] Project design + repo setup
Week 2 (Apr 25-May 1): [Nemotron] Baseline + prompting          [Gemma] Core pipeline + function calling
Week 3 (May 2-8):     [Nemotron] LoRA fine-tuning               [Gemma] Multimodal + integration
Week 4 (May 9-15):    [Nemotron] Advanced techniques             [Gemma] Demo + writeup + video
Week 5 (May 16-18):   [Nemotron] Final submission                [Gemma] Final submission
```

---

## 4. Key Resources

### Nemotron
- Kaggle Notebooks: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/code
- Model: NVIDIA Nemotron-3-Nano-30B on HuggingFace
- Discussion: puzzle categories, LoRA tips, temperature tuning

### Gemma 4 Good
- Gemma 4 models on Kaggle
- Function calling docs: https://ai.google.dev/gemini-api/docs/function-calling
- Gemma 4 multimodal capabilities
- Evaluation rubric focuses on Impact + Technical Execution + Originality

### Critical Minerals
- USGS MRDS: https://mrdata.usgs.gov/mrds/
- CMMI: https://www.usgs.gov/mission-areas/minerals/critical-mineral-mapping-initiative
- ASTER: https://asterweb.jpl.nasa.gov/
- EarthExplorer: https://earthexplorer.usgs.gov/
